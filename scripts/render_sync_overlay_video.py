import argparse
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import os

def render_sync_overlay(video_path, csv_path, output_path):
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'frame_idx' in df.columns:
        df.set_index('frame_idx', inplace=True)
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 29.938
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process at HALF resolution
    width = orig_width // 2
    height = orig_height // 2
    
    print(f"Source resolution: {orig_width}x{orig_height}, Output resolution: {width}x{height}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    chart_h = 300
    chart_w = width
    chart_y_start = height - chart_h
    history_frames = int(8 * fps)
    
    for i in tqdm(range(total_frames), desc="Rendering Video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (width, height))
        
        roll_deg = np.nan
        if i in df.index:
            roll_deg = df.loc[i, 'roll_deg']
            
        if not np.isnan(roll_deg):
            # 1. Top-left readout
            abs_roll = abs(roll_deg)
            if abs_roll < 5:
                color = (0, 255, 0) # Green
            elif abs_roll <= 15:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 0, 255) # Red
                
            text = f"Roll: {roll_deg:+.1f} deg"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # 3. Center horizon line (semi-transparent green)
            cx, cy = width // 2, height // 2
            line_len = int(width * 0.6)
            angle_rad = math.radians(-roll_deg)
            dx = int(math.cos(angle_rad) * line_len / 2)
            dy = int(math.sin(angle_rad) * line_len / 2)
            
            overlay = frame.copy()
            cv2.line(overlay, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 255, 0), 3)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        else:
            cv2.putText(frame, "Roll: N/A", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
            
        # 2. Bottom strip chart
        chart = np.zeros((chart_h, chart_w, 3), dtype=np.uint8)
        
        start_idx = max(0, i - history_frames)
        valid_idxs = [idx for idx in range(start_idx, i + 1) if idx in df.index and not np.isnan(df.loc[idx, 'roll_deg'])]
        
        max_r = 45.0
        
        if valid_idxs:
            pts = []
            for idx in valid_idxs:
                x = int((idx - start_idx) / history_frames * chart_w)
                r = df.loc[idx, 'roll_deg']
                y = int(chart_h / 2 - (r / max_r) * (chart_h / 2 - 20))
                pts.append((x, y))
                
            if len(pts) > 1:
                pts_np = np.array(pts, np.int32)
                cv2.polylines(chart, [pts_np], isClosed=False, color=(255, 255, 0), thickness=2) # Cyan trace
                
        # Zero-line
        cv2.line(chart, (0, chart_h // 2), (chart_w, chart_h // 2), (255, 255, 255), 1)
        
        # Y-axis marks (-30, 0, 30)
        for r_mark in [-30, 0, 30]:
            y = int(chart_h / 2 - (r_mark / max_r) * (chart_h / 2 - 20))
            cv2.putText(chart, str(r_mark), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.line(chart, (0, y), (10, y), (255, 255, 255), 1)
            
        # Current time marker (vertical line)
        curr_x = int((i - start_idx) / history_frames * chart_w)
        cv2.line(chart, (curr_x, 0), (curr_x, chart_h), (0, 0, 255), 1) # Red marker
        
        # Alpha blend chart into bottom of frame
        frame_roi = frame[chart_y_start:height, 0:width]
        blended = cv2.addWeighted(frame_roi, 0.4, chart, 0.9, 0)
        frame[chart_y_start:height, 0:width] = blended
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render QA overlay video with synced IMU roll data.")
    parser.add_argument("--video", default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\20260830_180600.mp4", help="Input video path")
    parser.add_argument("--csv", default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\frame_roll_lookup.csv", help="Input frame roll lookup CSV")
    parser.add_argument("--output", default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\output\sync_overlay.mp4", help="Output video path")
    
    args = parser.parse_args()
    render_sync_overlay(args.video, args.csv, args.output)
