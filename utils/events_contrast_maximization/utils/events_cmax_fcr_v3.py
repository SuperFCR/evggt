import argparse
import time
import numpy as np
import scipy
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
import torch
import os
import matplotlib.pyplot as plt
import cv2 as cv
import json
from event_utils import *
from objectives import *
from warps import *
import matplotlib
import subprocess
import shutil
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

def draw_objective_function(xs, ys, ts, ps, objective, warpfunc, x_range=(-200, 200), y_range=(-200, 200),
        gt=(0,0), show_gt=True, resolution=20, img_size=(180, 240), save_path=None):
    """
    Draw the objective function given by sampling over a range. Depending on the value of resolution, this
    can involve many samples and take some time.
    Parameters:
        xs,ys,ts,ps (numpy array) The event components
        objective (object) The objective function
        warpfunc (object) The warp function
        x_range, y_range (tuple) the range over which to plot the parameters
        gt (tuple) The ground truth
        show_gt (bool) Whether to draw the ground truth in
        resolution (float) The resolution of the sampling
        img_size (tuple) The image sensor size
        save_path (str) Path to save the visualization
    """
    width = x_range[1]-x_range[0]
    height = y_range[1]-y_range[0]
    print("Drawing objective function. Taking {} samples".format((width*height)/resolution))
    imshape = (int(height/resolution+0.5), int(width/resolution+0.5))
    img = np.zeros(imshape)
    for x in range(img.shape[1]):
       for y in range(img.shape[0]):
           params = np.array([x*resolution+x_range[0], y*resolution+y_range[0]])
           img[y,x] = -objective.evaluate_function(params, xs, ys, ts, ps, warpfunc, img_size, blur_sigma=0)
    img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)
    plt.figure(figsize=(10, 8))
    plt.imshow(img, interpolation='bilinear', cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Objective: {objective.name}")
    
    if show_gt:
        xloc = ((gt[0]-x_range[0])/(width))*imshape[1]
        yloc = ((gt[1]-y_range[0])/(height))*imshape[0]
        plt.axhline(y=yloc, color='r', linestyle='--')
        plt.axvline(x=xloc, color='r', linestyle='--')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def optimize_contrast(xs, ys, ts, ps, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None,
        numeric_grads=False, blur_sigma=None, img_size=(180, 240)):
    """
    Optimize contrast for a set of events
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp_function (function) The function with which to warp the events
    objective (objective class object) The objective to optimize
    optimizer (function) The optimizer to use
    x0 (np array) The initial guess for optimization
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor
    blur_sigma (float) Size of the blurring kernel. Blurring the images of warped events can
        have a large impact on the convergence of the optimization.

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    args = (xs, ys, ts, ps, warp_function, img_size, blur_sigma)
    x0 = np.array([0,0])
    if x0 is None:
        x0 = np.zeros(warp_function.dims)
    if numeric_grads:
        argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False)
    else:
        argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=False)
    return argmax

def optimize(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events. Uses optimize_contrast() for the optimiziation, but allows
    blurring schedules for successive optimization iterations.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=None, img_size=img_size)
    return argmax_an

def optimize_r2(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events, finishing with SoE loss.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    soe_obj = soe_objective()
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=None)
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, soe_obj, x0=argmax_an, numeric_grads=numeric_grads, blur_sigma=1.0)
    return argmax_an

'''

def visualize_warped_events(xs, ys, ts, ps, warp_function, params, img_size=(180, 240), save_path=None):
    """
    Visualize the events after warping them with the parameters found by contrast maximization.
    
    Parameters:
        xs, ys, ts, ps (numpy arrays): Event components
        warp_function (object): The warp function to use
        params (numpy array): The parameters to use for warping (e.g., from cmax optimization)
        img_size (tuple): The image sensor size
        save_path (str): Path to save the visualization
    """
    img = np.zeros(img_size, dtype=np.float32)
    xs_warped, ys_warped = warp_function.warp(xs, ys, ts, params)
    
    # Round coordinates to integers and filter out events outside the image
    xs_rounded = np.round(xs_warped).astype(int)
    ys_rounded = np.round(ys_warped).astype(int)
    valid_idx = (xs_rounded >= 0) & (xs_rounded < img_size[1]) & (ys_rounded >= 0) & (ys_rounded < img_size[0])
    
    # Create the image by accumulating events
    for i in range(len(xs_rounded[valid_idx])):
        x, y, p = xs_rounded[valid_idx][i], ys_rounded[valid_idx][i], ps[valid_idx][i]
        # For visualization, we'll add for positive events and subtract for negative ones
        img[y, x] += 1 if p > 0 else -1
    
    # Normalize the image for better visualization
    img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    plt.title(f"Warped Events (params: {params})")
    plt.colorbar(label='Normalized Event Count')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved warped events visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return img

def visualize_original_events(xs, ys, ps, img_size=(180, 240), save_path=None):
    """
    Visualize the original events without warping.
    
    Parameters:
        xs, ys, ps (numpy arrays): Event components
        img_size (tuple): The image sensor size
        save_path (str): Path to save the visualization
    """
    # Create an empty image
    img = np.zeros(img_size, dtype=np.float32)
    
    # Round coordinates to integers and filter out events outside the image
    xs_rounded = np.round(xs).astype(int)
    ys_rounded = np.round(ys).astype(int)
    valid_idx = (xs_rounded >= 0) & (xs_rounded < img_size[1]) & (ys_rounded >= 0) & (ys_rounded < img_size[0])
    
    # Create the image by accumulating events
    for i in range(len(xs_rounded[valid_idx])):
        x, y, p = xs_rounded[valid_idx][i], ys_rounded[valid_idx][i], ps[valid_idx][i]
        # For visualization, we'll add for positive events and subtract for negative ones
        img[y, x] += 1 if p > 0 else -1
    
    # Normalize the image for better visualization
    img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    plt.title("Original Events (No Warping)")
    plt.colorbar(label='Normalized Event Count')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved original events visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return img

'''
def visualize_warped_events(xs, ys, ts, ps, warp_function, params, img_size=(180, 240), save_path=None):
    """
    Visualize warped events as red (ON) and blue (OFF) points, save MP4, and ensure no extra margins.
    """
    x_prime, y_prime, _, _ = warp_function.warp(xs, ys, ts, ps, ts[0], params, compute_grad=False)

    xs_rounded = np.round(x_prime)
    ys_rounded = np.round(y_prime)
    valid_idx = (xs_rounded >= 0) & (xs_rounded < img_size[1]) & (ys_rounded >= 0) & (ys_rounded < img_size[0])

    xs_valid = xs_rounded[valid_idx]
    ys_valid = ys_rounded[valid_idx]
    ts_valid = ts[valid_idx]
    ps_valid = ps[valid_idx]

    frame_dir = save_path.replace('.png', '_frames')
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    total_frames = 100
    ts_min, ts_max = ts_valid.min(), ts_valid.max()
    t_slices = np.linspace(ts_min, ts_max, total_frames+1)

    for i in range(total_frames):
        t_start = t_slices[i]
        t_end = t_slices[i+1]
        mask = (ts_valid >= t_start) & (ts_valid < t_end)

        fig = plt.figure(figsize=(img_size[1]/100, img_size[0]/100), dpi=100)
        ax = plt.gca()
        ax.set_xlim(0, img_size[1])
        ax.set_ylim(img_size[0], 0)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        pos_idx = ps_valid[mask] > 0
        neg_idx = ps_valid[mask] < 0
        plt.scatter(xs_valid[mask][pos_idx], ys_valid[mask][pos_idx], c='red', s=1)
        plt.scatter(xs_valid[mask][neg_idx], ys_valid[mask][neg_idx], c='blue', s=1)

        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100)  # 不加tight
        plt.close()

    print(f"Saved {total_frames} frames into {frame_dir}")

    mp4_save_path = save_path.replace('.png', '.mp4')
    ffmpeg_command = [
        'ffmpeg', '-y', '-r', '60', '-i', os.path.join(frame_dir, 'frame_%04d.png'),
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', mp4_save_path
    ]
    print(f"Running ffmpeg to generate {mp4_save_path}...")
    subprocess.run(ffmpeg_command)
    print(f"🟢 Saved warped events video to {mp4_save_path}")

    # Save final static image
    fig = plt.figure(figsize=(img_size[1]/100, img_size[0]/100), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, img_size[1])
    ax.set_ylim(img_size[0], 0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    pos_idx = ps_valid > 0
    neg_idx = ps_valid < 0
    plt.scatter(xs_valid[pos_idx], ys_valid[pos_idx], c='red', s=1)
    plt.scatter(xs_valid[neg_idx], ys_valid[neg_idx], c='blue', s=1)

    plt.savefig(save_path, dpi=100)
    print(f"Saved single frame visualization to {save_path}")
    plt.close()

def visualize_original_events(xs, ys, ps, img_size=(180, 240), save_path=None):
    """
    Visualize original events as red (ON) and blue (OFF) points, save MP4 with independent frames, no margins.
    """
    num_events = len(xs)
    fake_ts = np.linspace(0, 1, num_events)  # Fake timestamps

    xs_rounded = np.round(xs)
    ys_rounded = np.round(ys)
    valid_idx = (xs_rounded >= 0) & (xs_rounded < img_size[1]) & (ys_rounded >= 0) & (ys_rounded < img_size[0])

    xs_valid = xs_rounded[valid_idx]
    ys_valid = ys_rounded[valid_idx]
    ts_valid = fake_ts[valid_idx]
    ps_valid = ps[valid_idx]

    frame_dir = save_path.replace('.png', '_frames')
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    total_frames = 100
    t_slices = np.linspace(0, 1, total_frames+1)

    for i in range(total_frames):
        t_start = t_slices[i]
        t_end = t_slices[i+1]
        mask = (ts_valid >= t_start) & (ts_valid < t_end)

        fig = plt.figure(figsize=(img_size[1]/100, img_size[0]/100), dpi=100)
        ax = plt.gca()
        ax.set_xlim(0, img_size[1])
        ax.set_ylim(img_size[0], 0)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        pos_idx = ps_valid[mask] > 0
        neg_idx = ps_valid[mask] < 0
        plt.scatter(xs_valid[mask][pos_idx], ys_valid[mask][pos_idx], c='red', s=1)
        plt.scatter(xs_valid[mask][neg_idx], ys_valid[mask][neg_idx], c='blue', s=1)

        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100)  # no tight
        plt.close()

    print(f"Saved {total_frames} original frames into {frame_dir}")

    mp4_save_path = save_path.replace('.png', '.mp4')
    ffmpeg_command = [
        'ffmpeg', '-y', '-r', '60', '-i', os.path.join(frame_dir, 'frame_%04d.png'),
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', mp4_save_path
    ]
    print(f"Running ffmpeg to generate {mp4_save_path}...")
    subprocess.run(ffmpeg_command)
    print(f"🟢 Saved original events video to {mp4_save_path}")

    # 保存最终静态图
    fig = plt.figure(figsize=(img_size[1]/100, img_size[0]/100), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, img_size[1])
    ax.set_ylim(img_size[0], 0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    pos_idx = ps_valid > 0
    neg_idx = ps_valid < 0
    plt.scatter(xs_valid[pos_idx], ys_valid[pos_idx], c='red', s=1)
    plt.scatter(xs_valid[neg_idx], ys_valid[neg_idx], c='blue', s=1)

    plt.savefig(save_path, dpi=100)
    print(f"Saved single frame visualization to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="h5 events path")
    parser.add_argument("--gt", nargs='+', type=float, default=(0,0))
    parser.add_argument("--img_size", nargs='+', type=float, default=(180,240))
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--save_vis", action="store_true", help="Save visualizations")
    parser.add_argument("--start_idx", type=int, default=20000, help="Start index for event slice")
    parser.add_argument("--num_events", type=int, default=100000, help="Number of events to use")
    parser.add_argument("--x_range", nargs='+', type=float, default=(-200, 200), help="X range for visualization")
    parser.add_argument("--y_range", nargs='+', type=float, default=(-200, 200), help="Y range for visualization")
    parser.add_argument("--resolution", type=float, default=20, help="Resolution for objective visualization")
    parser.add_argument("--visualize_warped", action="store_true", help="Visualize warped events")
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if args.save_vis and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load events
    xs, ys, ts, ps = read_h5_event_components(args.path)
    ts = ts - ts[0]
    gt_params = tuple(args.gt)
    img_size = tuple(map(int, args.img_size))
    x_range = tuple(args.x_range)
    y_range = tuple(args.y_range)
    start_idx = args.start_idx
    end_idx = start_idx + args.num_events

    base_filename = os.path.splitext(os.path.basename(args.path))[0]
    
    results_file = os.path.join(args.save_dir, f"{base_filename}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Results for {args.path}\n")
        f.write(f"Events: {start_idx}-{end_idx}\n")
        f.write(f"GT: {gt_params}\n\n")

    # Cut event slice
    xs_slice = xs[start_idx:end_idx]
    ys_slice = ys[start_idx:end_idx]
    ts_slice = ts[start_idx:end_idx]
    ps_slice = ps[start_idx:end_idx]

    # Visualize original events (before any warp)
    if args.visualize_warped:
        original_img_path = os.path.join(args.save_dir, f"{base_filename}_original_events.png")
        visualize_original_events(xs_slice, ys_slice, ps_slice, img_size=img_size, save_path=original_img_path if args.save_vis else None)

    # Set warp function
    warp = linvel_warp()

    # Visualize one objective function (e.g., Variance Objective)
    if args.save_vis:
        variance_obj = variance_objective()
        variance_vis_path = os.path.join(args.save_dir, f"{base_filename}_variance_objective.png")
        draw_objective_function(xs_slice, ys_slice, ts_slice, ps_slice, variance_obj, warp, 
                                 x_range=x_range, y_range=y_range, gt=gt_params, resolution=args.resolution, 
                                 img_size=img_size, save_path=variance_vis_path)

    # Define list of objectives
    objectives = [
        r1_objective(), 
        zhu_timestamp_objective(), 
        variance_objective(), 
        sos_objective(), 
        soe_objective(), 
        moa_objective(),
        isoa_objective(), 
        sosa_objective(), 
        rms_objective()
    ]

    all_results = {}

    for obj in objectives:
        print(f"Optimizing Objective: {obj.name}")

        # Step 1: Optimize to find best warp parameters
        argmax = optimize(xs_slice, ys_slice, ts_slice, ps_slice, warp, obj, numeric_grads=True, img_size=img_size)

        # Step 2: Evaluate loss at optimized parameters and GT
        loss = obj.evaluate_function(argmax, xs_slice, ys_slice, ts_slice, ps_slice, warp, img_size=img_size)
        gtloss = obj.evaluate_function(gt_params, xs_slice, ys_slice, ts_slice, ps_slice, warp, img_size=img_size)

        result_str = "{}:({})= {:.6f}, gt= {:.6f}".format(obj.name, argmax, loss, gtloss)
        print(result_str)
        with open(results_file, 'a') as f:
            f.write(result_str + "\n")

        all_results[obj.name] = {"numeric": {"params": argmax.tolist(), "loss": float(loss)}}

        # Step 3: Visualize warped events after optimization
        if args.visualize_warped:
            warped_img_path = os.path.join(args.save_dir, f"{base_filename}_{obj.name}_warped_events.png")
            visualize_warped_events(xs_slice, ys_slice, ts_slice, ps_slice, warp, argmax, img_size=img_size, save_path=warped_img_path)

        # Step 4: Analytical optimization (only if available)
        if obj.has_derivative:
            print(f"  Trying analytical gradient for {obj.name}")
            argmax_an = optimize(xs_slice, ys_slice, ts_slice, ps_slice, warp, obj, numeric_grads=False, img_size=img_size)
            loss_an = obj.evaluate_function(argmax_an, xs_slice, ys_slice, ts_slice, ps_slice, warp, img_size=img_size)

            result_str_an = "   analytical:{}= {:.6f}".format(argmax_an, loss_an)
            print(result_str_an)
            with open(results_file, 'a') as f:
                f.write(result_str_an + "\n")

            all_results[obj.name]["analytical"] = {"params": argmax_an.tolist(), "loss": float(loss_an)}

            if args.visualize_warped:
                warped_an_img_path = os.path.join(args.save_dir, f"{base_filename}_{obj.name}_warped_events_analytical.png")
                visualize_warped_events(xs_slice, ys_slice, ts_slice, ps_slice, warp, argmax_an, img_size=img_size, save_path=warped_an_img_path)

        # Step 5: Visualize objective function (contour plot) for each objective
        if args.save_vis:
            viz_path = os.path.join(args.save_dir, f"{base_filename}_{obj.name}_objective.png")
            draw_objective_function(xs_slice, ys_slice, ts_slice, ps_slice, obj, warp, 
                                     x_range=x_range, y_range=y_range, gt=gt_params, resolution=args.resolution, 
                                     img_size=img_size, save_path=viz_path)

    # Save all results as JSON
    json_path = os.path.join(args.save_dir, f"{base_filename}_all_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            "metadata": {
                "path": args.path,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "gt_params": gt_params,
                "img_size": img_size
            },
            "results": all_results
        }, f, indent=2)

    print(f"All results saved to {args.save_dir}")
