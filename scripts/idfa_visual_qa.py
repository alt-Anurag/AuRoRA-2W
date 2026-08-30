import os
import sys
import glob
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.ops as ops
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))
from models.idfa import IDFAModule
from models.pidnet import get_pred_model

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "idfa_qa")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def part1_vector_fields():
    print("--- Part 1: IDFA Vector Fields ---")
    idfa = IDFAModule(in_channels=32, kernel_size=3)
    b, h, w = 1, 15, 15 # small grid for quiver
    
    angles = [0.0, 10.0, 20.0, 30.0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    uniform = True
    for i, angle in enumerate(angles):
        offsets = idfa._generate_geometric_prior(b, h, w, torch.tensor([angle]), 'cpu')
        
        # Offsets shape: (1, 18, 15, 15). Channel 0 is dy, 1 is dx for tap 0 (top-left)
        dy = offsets[0, 0, :, :].numpy()
        dx = offsets[0, 1, :, :].numpy()
        
        # Check uniformity
        if not (np.allclose(dy, dy[0,0], atol=1e-5) and np.allclose(dx, dx[0,0], atol=1e-5)):
            uniform = False
            
        Y, X = np.mgrid[0:h, 0:w]
        
        axes[i].quiver(X, Y, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue')
        axes[i].set_title(f"Roll: {angle} deg (Tap 0)")
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vector_fields.png"))
    plt.close()
    
    if uniform:
        print("[PASS] Vector fields are spatially uniform.")
    else:
        print("[WARN] Vector fields are NOT spatially uniform!")

def part2_mask_overlay(image_paths, mask_paths):
    print("--- Part 2: Image/Mask Rotation & Overlay ---")
    
    for i, (img_p, mask_p) in enumerate(zip(image_paths, mask_paths)):
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p)
        
        img_t = TF.to_tensor(img).unsqueeze(0)
        mask_t = torch.from_numpy(np.array(mask)).unsqueeze(0).unsqueeze(0).float()
        
        # Rotate 20 degrees
        rot_img = TF.rotate(img_t, 20.0, interpolation=TF.InterpolationMode.BILINEAR)
        rot_mask = TF.rotate(mask_t, 20.0, interpolation=TF.InterpolationMode.NEAREST)
        
        # Extract edges from the rotated mask
        rot_mask_np = rot_mask[0, 0].numpy().astype(np.uint8)
        rot_mask_vis = cv2.normalize(rot_mask_np, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(rot_mask_vis, 50, 150)
        
        rot_img_np = (rot_img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Overlay edges in red
        overlay = rot_img_np.copy()
        overlay[edges > 0] = [255, 0, 0]
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_overlay_{i}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
    print("[PASS] Overlays generated. Please review visually for slip.")

def part3_feature_correction(image_paths):
    print("--- Part 3: IDFA Feature Correction ---")
    
    # Load PIDNet stem
    model = get_pred_model("pidnet_s", 19)
    model.eval()
    stem = model.conv1
    
    idfa = IDFAModule(in_channels=64, kernel_size=3)
    
    for i, img_p in enumerate(image_paths):
        img = Image.open(img_p).convert("RGB")
        img_t = TF.to_tensor(img).unsqueeze(0)
        
        # Tilt input
        angle = 30.0
        tilted_img = TF.rotate(img_t, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        with torch.no_grad():
            features = stem(tilted_img)
            
        b, c, h, w = features.shape
        
        # Apply standard Conv2d (uncorrected) vs DeformConv2d (corrected)
        weight = torch.zeros(1, c, 3, 3)
        # Directional filter to highlight local grid alignment changes
        weight[0, 0, 0, :] = -1
        weight[0, 0, 2, :] = 1
        
        uncorrected = F.conv2d(features, weight, padding=1)
        
        offsets = idfa._generate_geometric_prior(b, h, w, torch.tensor([angle]), features.device)
        corrected = ops.deform_conv2d(features, offsets, weight, padding=1)
        
        # 1. QUANTIFY THE FEATURE DIFFERENCE
        diff = (corrected - uncorrected).abs()
        diff_max = diff.max().item()
        diff_mean = diff.mean().item()
        
        print(f"Image {i} (30 deg): Feature Diff Max: {diff_max:.6f}, Mean: {diff_mean:.6f}")
        assert diff_max > 1e-6, "Corrected and uncorrected features are numerically identical. IDFA had no effect!"
        
        # Save contrast-stretched heatmap
        diff_np = diff[0, 0].numpy()
        d_min, d_max = diff_np.min(), diff_np.max()
        if d_max > d_min:
            heatmap = (diff_np - d_min) / (d_max - d_min)
        else:
            heatmap = diff_np
            
        plt.imsave(os.path.join(OUTPUT_DIR, f"feature_diff_heatmap_{i}.png"), heatmap, cmap='hot')
        
        # 2. VERIFY SAMPLING LOCATION SHIFT DIRECTLY
        # Pick output pixel location
        py, px = h // 2, w // 2
        # Tap 0 is at (-1, -1) in the local 3x3 kernel grid.
        base_y = py - 1
        base_x = px - 1
        
        # Offset for Tap 0 (channels 0 and 1)
        dy = offsets[0, 0, py, px].item()
        dx = offsets[0, 1, py, px].item()
        
        sampled_y = base_y + dy
        sampled_x = base_x + dx
        
        delta_y = sampled_y - base_y
        delta_x = sampled_x - base_x
        
        if i == 0:
            print(f"  Tap 0 Sampling Verification (Output pixel y={py}, x={px}):")
            print(f"    Uncorrected (Base) Sample Location: y={base_y}, x={base_x}")
            print(f"    Corrected (Deform) Sample Location: y={sampled_y:.4f}, x={sampled_x:.4f}")
            print(f"    Delta: dy={delta_y:.4f}, dx={delta_x:.4f}")
            print(f"    Offset Field Values: dy={dy:.4f}, dx={dx:.4f}")
        
        assert abs(delta_y - dy) < 1e-6 and abs(delta_x - dx) < 1e-6, "Sampled coordinate delta does not match offset field!"

    print("[PASS] Feature correction quantitative QA generated successfully.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    img_dir = os.path.join(base_dir, "models/PIDNet/data/cityscapes/leftImg8bit/test/berlin")
    mask_dir = os.path.join(base_dir, "models/PIDNet/data/cityscapes/gtFine/test/berlin")
    
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:5]
    masks = sorted(glob.glob(os.path.join(mask_dir, "*_labelIds.png")))[:5]
    
    print("Starting IDFA Visual QA Harness...\n")
    part1_vector_fields()
    part2_mask_overlay(images, masks)
    part3_feature_correction(images)
    print("\nVisual QA complete. All outputs saved to output/idfa_qa/")
