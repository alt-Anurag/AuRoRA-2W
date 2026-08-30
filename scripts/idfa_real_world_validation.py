import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision.ops import deform_conv2d

# Setup paths
sys.path.append(r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\models\PIDNet")
from models.idfa import IDFAModule
from models.pidnet import get_pred_model

def get_args():
    parser = argparse.ArgumentParser(description="Run IDFA real world validation")
    parser.add_argument("--video", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\20260830_180600.mp4")
    parser.add_argument("--frames_csv", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\sensor_log\validation_frames.csv")
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\anura\Desktop\research_2wadas\aurora-2w\output\idfa_qa")
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--img_width", type=int, default=1024)
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load PIDNet stem and IDFAModule
    print("Loading models...")
    
    # Initialize stem (conv1 equivalent)
    stem = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    ).to(device)
    stem.eval()
    
    # IDFA module
    idfa = IDFAModule(in_channels=64, kernel_size=3).to(device)
    idfa.eval()
    
    # Standard directional filters (uncorrected)
    standard_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False).to(device)
    standard_conv.eval()
    
    # Load validation frames
    val_df = pd.read_csv(args.frames_csv)
    print(f"Loaded {len(val_df)} validation frames.")
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video: {args.video}")
        return
    
    results = []
    images_for_contact_sheet = []
    
    # 2. For each validation frame
    for idx, row in val_df.iterrows():
        frame_idx = int(row['frame_idx'])
        roll_deg = float(row['roll_deg'])
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            continue
            
        # Resize to model input size
        frame_resized = cv2.resize(frame, (args.img_width, args.img_height))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.to(device)
        
        roll_tensor = torch.tensor([roll_deg], dtype=torch.float32, device=device)
        
        # Extract features through stem
        features = stem(img_tensor) # (1, 64, H', W')
        
        # Generate geometric prior offsets at real roll_deg
        b, c, h, w = features.shape
        prior_offsets = idfa._generate_geometric_prior(b, h, w, roll_tensor, device)
        
        # Apply standard and corrected directional filters
        out_uncorrected = standard_conv(features)
        
        final_offsets = idfa(features, roll_tensor)
        weight = standard_conv.weight
        out_corrected = deform_conv2d(features, final_offsets, weight, padding=1)
        
        # Compute difference heatmap
        diff = torch.abs(out_corrected - out_uncorrected).mean(dim=1, keepdim=True)
        diff_np = diff.squeeze().cpu().numpy()
        
        diff_norm = cv2.normalize(diff_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (args.img_width, args.img_height))
        
        # Calculate stats
        mean_diff = float(diff_np.mean())
        max_diff = float(diff_np.max())
        results.append({
            'frame_idx': frame_idx,
            'roll_deg': roll_deg,
            'mean_diff': mean_diff,
            'max_diff': max_diff
        })
        
        images_for_contact_sheet.append((frame_resized[:, :, ::-1], heatmap, roll_deg, frame_idx))
    
    cap.release()
    
    if not images_for_contact_sheet:
        print("No images processed successfully.")
        return

    # 3. Create a contact sheet
    num_frames = len(images_for_contact_sheet)
    cols = min(5, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig = plt.figure(figsize=(cols * 4, rows * 6))
    gs = GridSpec(rows * 2, cols, figure=fig)
    
    for i, (img, heat, roll, f_idx) in enumerate(images_for_contact_sheet):
        r = i // cols
        c = i % cols
        
        ax_img = fig.add_subplot(gs[r*2, c])
        ax_img.imshow(img)
        ax_img.set_title(f"Frame: {f_idx} | Roll: {roll:.1f}°")
        ax_img.axis('off')
        
        ax_heat = fig.add_subplot(gs[r*2+1, c])
        ax_heat.imshow(heat[:, :, ::-1])
        ax_heat.axis('off')
    
    plt.tight_layout()
    contact_sheet_path = os.path.join(args.output_dir, "real_world_validation_contact_sheet.png")
    plt.savefig(contact_sheet_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Print summary statistics
    print(f"\nSaved contact sheet to {contact_sheet_path}")
    print("\nSummary Statistics:")
    overall_mean = np.mean([r['mean_diff'] for r in results])
    overall_max = np.max([r['max_diff'] for r in results])
    
    print(f"Overall Mean Feature Difference: {overall_mean:.4f}")
    print(f"Overall Max Feature Difference: {overall_max:.4f}")
    print("\nPer-frame statistics:")
    for r in results:
        print(f"  Frame {r['frame_idx']} (Roll: {r['roll_deg']:.1f}°): Mean Diff = {r['mean_diff']:.4f}, Max Diff = {r['max_diff']:.4f}")

if __name__ == "__main__":
    main()
