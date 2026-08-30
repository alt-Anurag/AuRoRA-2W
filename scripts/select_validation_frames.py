import pandas as pd
import numpy as np
import argparse
import os

def select_frames(input_csv, output_csv, bins=10, max_per_bin=5, min_separation_s=2.0, seed=42):
    np.random.seed(seed)
    
    # 1. Load frame_roll_lookup.csv, drop any NaN rows
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['frame_idx', 'video_time_s', 'roll_deg'])
    
    # 2. Divide the roll_deg range into 10 equal-width bins
    df['roll_bin'] = pd.cut(df['roll_deg'], bins=bins, labels=False)
    
    # 3. From each bin, randomly select up to 5 frames
    selected_indices = []
    for bin_idx in range(bins):
        bin_df = df[df['roll_bin'] == bin_idx]
        if len(bin_df) == 0:
            continue
        
        # Shuffle bin dataframe
        bin_df = bin_df.sample(frac=1.0, random_state=seed)
        
        # 4. Ensure minimum temporal separation of 2 seconds between selected frames
        bin_selected = []
        for _, row in bin_df.iterrows():
            if len(bin_selected) >= max_per_bin:
                break
            
            # Check separation against ALL previously selected frames
            valid = True
            for sel_idx in selected_indices + [r.name for r in bin_selected]:
                sel_row = df.loc[sel_idx]
                if abs(row['video_time_s'] - sel_row['video_time_s']) < min_separation_s:
                    valid = False
                    break
            
            if valid:
                bin_selected.append(row)
        
        selected_indices.extend([r.name for r in bin_selected])
    
    selected_df = df.loc[selected_indices].sort_values('video_time_s').reset_index(drop=True)
    
    # 5. Print summary
    print(f"Total frames selected: {len(selected_df)}")
    print("Frames per bin:")
    print(selected_df['roll_bin'].value_counts().sort_index())
    print(f"Roll range covered: {selected_df['roll_deg'].min():.2f} to {selected_df['roll_deg'].max():.2f} degrees")
    
    # 6. Save the selected frames CSV
    out_cols = ['frame_idx', 'video_time_s', 'roll_deg', 'roll_bin']
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    selected_df[out_cols].to_csv(output_csv, index=False)
    print(f"Saved validation frames to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select validation frames")
    parser.add_argument("--input", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\frame_roll_lookup.csv")
    parser.add_argument("--output", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\validation_frames.csv")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max_per_bin", type=int, default=5)
    parser.add_argument("--min_separation_s", type=float, default=2.0)
    
    args = parser.parse_args()
    select_frames(args.input, args.output, args.bins, args.max_per_bin, args.min_separation_s)
