import glob
import argparse
import os
import h5py
import numpy as np
from event_packagers import *

def extract_npz(npz_paths, output_path, zero_timestamps=False, packager=hdf5_packager):
    ep = packager(output_path)
    first_ts = -1
    total_num_pos, total_num_neg = 0, 0
    last_ts = 0

    # Iterate through all NPZ files
    for npz_path in npz_paths:
        if not os.path.exists(npz_path):
            print("{} does not exist!".format(npz_path))
            continue

        # Load the NPZ file
        with np.load(npz_path) as data:
            xs = data['x']
            ys = data['y']
            ts = data['t']
            ps = data['p']

        # Ensure polarity is boolean
        ps[ps < 0] = 0  # should be [0 or 1]
        ps = ps.astype(bool)

        if first_ts == -1:
            first_ts = ts[0]

        if zero_timestamps:
            ts -= first_ts
        
        last_ts = ts[-1]
        sensor_size = [max(ys) + 1, max(xs) + 1]  # Assuming sensor size is max x and y values

        sum_ps = np.sum(ps)
        total_num_pos += sum_ps
        total_num_neg += len(ps) - sum_ps
        ep.package_events(xs, ys, ts, ps)

    print('Total events written: {} M'.format((total_num_pos + total_num_neg) / 1e6))
    print("Detected sensor size [h={}, w={}]".format(sensor_size[0], sensor_size[1]))

    t0 = 0 if zero_timestamps else first_ts
    ep.add_metadata(total_num_pos, total_num_neg, last_ts - t0, t0, last_ts, num_imgs=0, num_flow=0, sensor_size=sensor_size)

if __name__ == "__main__":
    """
    Tool for converting multiple npz files into a single efficient HDF5 format that can be speedily
    accessed by Python code.
    Input path can be a directory containing npz files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory containing npz files")
    parser.add_argument("--output_dir", default="/tmp/extracted_data", help="Folder where to extract the data")
    parser.add_argument('--zero_timestamps', action='store_true', help='If true, timestamps will be offset to start at 0')
    args = parser.parse_args()

    print('Data will be extracted in folder: {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    npz_paths = sorted(glob.glob(os.path.join(args.path, "*.npz")))
    extract_npz(npz_paths, os.path.join(args.output_dir, "event.h5"), zero_timestamps=args.zero_timestamps)

# python /mnt/workspace/ywb/WorldNew/evggt/utils/events_contrast_maximization/tools/npz_to_h5.py /mnt/workspace/ywb/WorldNew/rpg_vid2e_v2/rpg_vid2e/outputs/event/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/gt_0/ --output_dir /mnt/workspace/ywb/WorldNew/evggt/utils/outputs