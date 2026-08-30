import argparse
import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def main():
    parser = argparse.ArgumentParser(description="Synchronize IMU data with video using manual offset + automated refinement.")
    parser.add_argument("--video", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\20260830_180600.mp4", help="Path to video file")
    parser.add_argument("--imu-clean", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\cleaned_orientation.csv", help="Path to cleaned IMU orientation CSV")
    parser.add_argument("--imu-accel", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\extracted\TotalAcceleration.csv", help="Path to raw acceleration CSV")
    parser.add_argument("--out-csv", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\frame_roll_lookup.csv", help="Output path for synchronized frame lookup CSV")
    parser.add_argument("--initial-offset", type=float, default=6.606, help="Initial guess for video delay vs IMU (seconds)")
    parser.add_argument("--search-min", type=float, default=6.3, help="Min offset search range (seconds)")
    parser.add_argument("--search-max", type=float, default=6.9, help="Max offset search range (seconds)")
    parser.add_argument("--search-step", type=float, default=0.01, help="Offset search step size (seconds)")
    
    args = parser.parse_args()
    
    print(f"Loading acceleration data from {args.imu_accel}...")
    accel_df = pd.read_csv(args.imu_accel)
    
    time_col = 'seconds_elapsed'
    if time_col not in accel_df.columns:
        if 'time' in accel_df.columns:
            time_col = 'time'
        else:
            raise ValueError("Could not find time column in acceleration CSV")
            
    # Compute magnitude
    accel_df['magnitude'] = np.sqrt(accel_df['x']**2 + accel_df['y']**2 + accel_df['z']**2)
    
    # Find top 5 sharpest spikes (min separation 5 seconds)
    dt = accel_df[time_col].diff().median()
    if pd.isna(dt) or dt <= 0:
        dt = 0.01
    
    dist_samples = int(5.0 / dt)
    
    peaks, properties = find_peaks(accel_df['magnitude'], distance=dist_samples, height=0)
    
    # Sort peaks by height
    top_peak_indices = sorted(peaks, key=lambda idx: accel_df['magnitude'].iloc[idx], reverse=True)[:5]
    top_peak_indices = sorted(top_peak_indices) # Sort by time
    
    imu_spike_times = accel_df.iloc[top_peak_indices][time_col].values
    
    print(f"Found {len(imu_spike_times)} IMU spikes at times (s): {imu_spike_times}")
    
    print(f"Loading video from {args.video}...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
         raise ValueError(f"Could not open video file: {args.video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.3f}, Total Frames: {total_frames}")
    
    video_spike_times = []
    
    # Attempt AUTO-REFINEMENT
    # Extract 2-second window of video frames around each expected spike timestamp
    for i, imu_time in enumerate(imu_spike_times):
        expected_vid_time = imu_time - args.initial_offset
        if expected_vid_time < 1.0 or expected_vid_time > (total_frames/fps) - 1.0:
            print(f"Spike {i+1} at IMU {imu_time:.2f}s is outside video bounds (expected ~{expected_vid_time:.2f}s). Skipping.")
            continue
            
        start_time = expected_vid_time - 1.0
        end_time = expected_vid_time + 1.0
        
        start_frame = max(0, int(start_time * fps))
        end_frame = min(total_frames - 1, int(end_time * fps))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        prev_gray = None
        frame_diffs = []
        frame_times = []
        
        for f in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = np.mean(diff)
                frame_diffs.append(mean_diff)
                frame_times.append(f / fps) # approximate time
            
            prev_gray = gray
            
        if not frame_diffs:
            continue
            
        # Find peak frame-difference timestamp
        peak_idx = np.argmax(frame_diffs)
        vid_spike_time = frame_times[peak_idx]
        video_spike_times.append((imu_time, vid_spike_time))
        print(f"Spike {i+1}: IMU {imu_time:.3f}s -> Expected Vid {expected_vid_time:.3f}s, Found Vid Spike at {vid_spike_time:.3f}s")
        
    cap.release()
    
    if len(video_spike_times) == 0:
        print("Could not find any matching video spikes. Using manual offset.")
        best_offset = args.initial_offset
    else:
        # Search the offset range 6.3-6.9s in 0.01s steps
        search_offsets = np.arange(args.search_min, args.search_max + args.search_step, args.search_step)
        
        best_offset = args.initial_offset
        min_error = float('inf')
        
        for offset in search_offsets:
            error = 0
            for imu_t, vid_t in video_spike_times:
                error += (imu_t - (vid_t + offset)) ** 2
            if error < min_error:
                min_error = error
                best_offset = offset
                
        print(f"\nRefined offset: {best_offset:.3f}s (moved {best_offset - args.initial_offset:.3f}s from manual estimate)")
        print(f"Top 3 spikes report:")
        for j, (imu_t, vid_t) in enumerate(video_spike_times[:3]):
            print(f"  Spike {j+1}: IMU {imu_t:.3f}s, Manual Vid Exp {imu_t - args.initial_offset:.3f}s, Refined Vid Exp {imu_t - best_offset:.3f}s")

    print(f"\nLoading cleaned IMU orientation from {args.imu_clean}...")
    clean_imu_df = pd.read_csv(args.imu_clean)
    
    imu_times = clean_imu_df['timestamp_s'].values
    rolls = clean_imu_df['roll_deg'].values
    pitches = clean_imu_df['pitch_deg'].values
    yaws = clean_imu_df['yaw_deg'].values
    
    interp_roll = interp1d(imu_times, rolls, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_pitch = interp1d(imu_times, pitches, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_yaw = interp1d(imu_times, yaws, kind='linear', bounds_error=False, fill_value=np.nan)
    
    print("Generating frame_roll_lookup.csv...")
    frame_indices = np.arange(total_frames)
    video_times = frame_indices / fps
    synced_imu_times = video_times + best_offset
    
    synced_rolls = interp_roll(synced_imu_times)
    synced_pitches = interp_pitch(synced_imu_times)
    synced_yaws = interp_yaw(synced_imu_times)
    
    lookup_df = pd.DataFrame({
        'frame_idx': frame_indices,
        'video_time_s': video_times,
        'imu_time_s': synced_imu_times,
        'roll_deg': synced_rolls,
        'pitch_deg': synced_pitches,
        'yaw_deg': synced_yaws
    })
    
    lookup_df.to_csv(args.out_csv, index=False)
    print(f"Successfully saved {args.out_csv}")

if __name__ == "__main__":
    main()
