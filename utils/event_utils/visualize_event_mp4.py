#!/usr/bin/env python3
"""
Visualize events from an H5 file and create an MP4 video.
Events are visualized with ON events as red and OFF events as blue on a white background.
Handles event timestamps in nanoseconds and ensures proper time alignment.
"""

import argparse
import os
import numpy as np
import h5py
import cv2
from tqdm import tqdm
import sys
from lib.data_formats.read_events import read_h5_event_components

def create_event_frame(xs, ys, ps, img_size):
    """
    Create a frame with white background, ON events as red pixels, OFF events as blue pixels.
    """
    # Start with white background (255 for all channels)
    frame = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255
    
    # Convert to integer coordinates
    xs_int = np.round(xs).astype(int)
    ys_int = np.round(ys).astype(int)
    
    # Filter valid coordinates
    valid = (xs_int >= 0) & (xs_int < img_size[1]) & (ys_int >= 0) & (ys_int < img_size[0])
    xs_valid = xs_int[valid]
    ys_valid = ys_int[valid]
    ps_valid = ps[valid]
    
    # For faster processing, use numpy operations instead of loops
    # For positive events (red)
    pos_mask = ps_valid > 0
    if np.any(pos_mask):
        pos_coords = np.vstack((ys_valid[pos_mask], xs_valid[pos_mask])).T
        for y, x in pos_coords:
            frame[y, x] = [0, 0, 255]  # Red in RGB format
    
    # For negative events (blue)
    neg_mask = ~pos_mask
    if np.any(neg_mask):
        neg_coords = np.vstack((ys_valid[neg_mask], xs_valid[neg_mask])).T
        for y, x in neg_coords:
            frame[y, x] = [255, 0, 0]  # Blue in RGB format
    
    return frame

def visualize_events_to_video(file_path, output_path, fps=30, event_window_sec=0.1, 
                             image_size=None, start_time=None, end_time=None,
                             max_events=None, frame_limit=None):
    """
    Create a video visualization of events from an H5 file.
    
    Parameters:
        file_path: Path to the H5 file containing events
        output_path: Where to save the output video
        fps: Frames per second for the output video
        event_window_sec: Time window in seconds for each frame
        image_size: Custom image size (height, width)
        start_time: Start time in seconds (will be converted to ns)
        end_time: End time in seconds (will be converted to ns)
        max_events: Maximum number of events to process
        frame_limit: Maximum number of frames to generate
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print(f"[INFO] Reading events from {file_path}")
        xs, ys, ts, ps = read_h5_event_components(file_path)
        
        # Check if timestamps are in seconds or nanoseconds
        # Typically, if max timestamp > 1e9, it's likely in nanoseconds
        time_unit_is_ns = np.max(ts) > 1e9
        print(f"[INFO] Detected time unit: {'nanoseconds' if time_unit_is_ns else 'seconds'}")
        
        # Convert window to nanoseconds if timestamps are in ns
        event_window = event_window_sec * (1e9 if time_unit_is_ns else 1.0)
        
        # Convert start/end times to match timestamp units
        if start_time is not None:
            start_time = start_time * (1e9 if time_unit_is_ns else 1.0)
        if end_time is not None:
            end_time = end_time * (1e9 if time_unit_is_ns else 1.0)
        
        # If max_events is specified, limit the number of events
        if max_events is not None and max_events > 0:
            max_idx = min(len(xs), max_events)
            xs, ys = xs[:max_idx], ys[:max_idx]
            ts, ps = ts[:max_idx], ps[:max_idx]
            print(f"[INFO] Limited to {max_idx} events")
        
        # Determine image size if not provided
        if image_size is None:
            height = int(np.max(ys) + 1)
            width = int(np.max(xs) + 1)
            image_size = (height, width)
        
        print(f"[INFO] Image size: {image_size[0]}x{image_size[1]}")
        
        # Apply time limits if specified
        if start_time is not None:
            start_idx = np.searchsorted(ts, start_time)
            xs, ys = xs[start_idx:], ys[start_idx:]
            ts, ps = ts[start_idx:], ps[start_idx:]
            print(f"[INFO] Starting from time {start_time / (1e9 if time_unit_is_ns else 1.0):.6f}s (event index {start_idx})")
        
        if end_time is not None:
            end_idx = np.searchsorted(ts, end_time)
            xs, ys = xs[:end_idx], ys[:end_idx]
            ts, ps = ts[:end_idx], ps[:end_idx]
            print(f"[INFO] Ending at time {end_time / (1e9 if time_unit_is_ns else 1.0):.6f}s (event index {end_idx})")
        
        # Get the total time range
        total_time = ts[-1] - ts[0]
        num_frames = int(total_time / event_window)
        
        print(f"[INFO] Time range: {total_time / (1e9 if time_unit_is_ns else 1.0):.6f}s")
        print(f"[INFO] Time window: {event_window / (1e9 if time_unit_is_ns else 1.0):.6f}s")
        print(f"[INFO] Target frame count: {num_frames}")
        
        # Apply frame limit if specified
        if frame_limit is not None and num_frames > frame_limit:
            event_window = total_time / frame_limit
            num_frames = frame_limit
            print(f"[INFO] Limiting to {frame_limit} frames. Adjusted window to {event_window / (1e9 if time_unit_is_ns else 1.0):.6f}s")
        
        # Create time windows
        time_windows = np.linspace(ts[0], ts[-1] - event_window, num_frames)
        
        # Create temporary directory for debug frames
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Calculate the expected video duration in seconds
        video_duration = (ts[-1] - ts[0]) / (1e9 if time_unit_is_ns else 1.0)
        actual_fps = num_frames / video_duration if video_duration > 0 else fps
        
        print(f"[INFO] Video duration: {video_duration:.2f}s")
        print(f"[INFO] Actual FPS to match time scale: {actual_fps:.2f}")
        print(f"[INFO] Target FPS for output: {fps}")
        
        # Create a video writer
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (image_size[1], image_size[0])
        )
        
        if not video_writer.isOpened():
            raise Exception("Could not open video writer")
        
        # Generate frames
        print(f"[INFO] Generating {len(time_windows)} frames...")
        for i, t_start in enumerate(tqdm(time_windows)):
            t_end = t_start + event_window
            
            # Find events in the current time window
            mask = (ts >= t_start) & (ts < t_end)
            window_xs = xs[mask]
            window_ys = ys[mask]
            window_ps = ps[mask]
            
            # Create frame with white background and colored events
            frame = create_event_frame(window_xs, window_ys, window_ps, image_size)
            
            # Write directly to video
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # For debugging: save the first few frames to check them
            if i < 5:
                debug_path = os.path.join(temp_dir, f"debug_frame_{i}.png")
                cv2.imwrite(debug_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Release video writer
        video_writer.release()
        
        # Calculate actual playback duration
        playback_duration = num_frames / fps
        time_scale_factor = playback_duration / video_duration
        
        print(f"[INFO] Video saved to {output_path}")
        print(f"[INFO] Events time span: {video_duration:.2f}s")
        print(f"[INFO] Video playback time: {playback_duration:.2f}s")
        print(f"[INFO] Time scale factor: {time_scale_factor:.2f}x")
        
        # Clean up debug frames
        print("[INFO] Cleaning up temporary files...")
        for i in range(min(5, len(time_windows))):
            try:
                os.remove(os.path.join(temp_dir, f"debug_frame_{i}.png"))
            except:
                pass
        
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Visualize events from H5 file to MP4 video.')
    parser.add_argument('input', help='Path to H5 file with events')
    parser.add_argument('--output', default='event_visualization.mp4', help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output video')
    parser.add_argument('--window', type=float, default=0.1, help='Time window (in seconds) for each frame')
    parser.add_argument('--width', type=int, default=None, help='Custom width for output video')
    parser.add_argument('--height', type=int, default=None, help='Custom height for output video')
    parser.add_argument('--start_time', type=float, default=None, help='Start time for visualization (in seconds)')
    parser.add_argument('--end_time', type=float, default=None, help='End time for visualization (in seconds)')
    parser.add_argument('--max_events', type=int, default=None, help='Maximum number of events to process')
    parser.add_argument('--frame_limit', type=int, default=None, help='Maximum number of frames to generate')
    
    args = parser.parse_args()
    
    # Set custom image size if provided
    image_size = None
    if args.width is not None and args.height is not None:
        image_size = (args.height, args.width)
    
    # Visualize events and create video
    visualize_events_to_video(
        args.input, 
        args.output, 
        fps=args.fps, 
        event_window_sec=args.window, 
        image_size=image_size,
        start_time=args.start_time,
        end_time=args.end_time,
        max_events=args.max_events,
        frame_limit=args.frame_limit
    )

if __name__ == "__main__":
    main()
