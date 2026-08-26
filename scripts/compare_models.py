import os
import sys
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))
from models.pidnet import get_pred_model as get_pidnet
from models.aurora2w import get_pred_model as get_aurora2w
from run_pidnet_baseline import input_transform, color_map, load_pretrained

def get_segmentation_mask(model, img_tensor, roll=None, device='cuda'):
    model.eval()
    with torch.no_grad():
        if roll is not None:
            # AuRoRA-2W takes roll_angles
            pred = model(img_tensor, roll_angles=torch.tensor([roll], device=device))
        else:
            # Baseline PIDNet
            pred = model(img_tensor)
            
        if isinstance(pred, (list, tuple)):
            pred = pred[-1] 
            
        # Interpolate to original size
        pred = F.interpolate(pred, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        
    return pred

def colorize(pred, img_bgr):
    sv_img = np.zeros_like(img_bgr).astype(np.uint8)
    for i, color in enumerate(color_map):
        sv_img[pred == i] = color
    return cv2.addWeighted(img_bgr, 0.4, sv_img, 0.6, 0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Image and Roll Angle
    img_path = 'samples/high_roll_frame.png'
    with open('samples/roll_info.txt', 'r') as f:
        roll_angle = float(f.read().strip())
        
    print(f"[*] Loaded test image with {roll_angle:.2f} degree lean.")
    
    img_bgr = cv2.imread(img_path)
    img_h, img_w = img_bgr.shape[:2]
    
    img = input_transform(img_bgr)
    img = img.transpose((2, 0, 1)).copy()
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    # 2. Load Models
    print("[*] Initializing Baseline PIDNet...")
    baseline = get_pidnet('pidnet_s', 19).to(device)
    baseline = load_pretrained(baseline, 'models/PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt')
    
    print("[*] Initializing AuRoRA-2W (with IDFA)...")
    aurora = get_aurora2w('pidnet_s', 19).to(device)
    aurora = load_pretrained(aurora, 'models/PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt')
    
    # 3. Inference
    print("[*] Running Baseline inference...")
    pred_base = get_segmentation_mask(baseline, img_tensor, device=device)
    
    print("[*] Running AuRoRA-2W inference...")
    pred_aurora = get_segmentation_mask(aurora, img_tensor, roll=roll_angle, device=device)
    
    # 4. Colorize and Stitch
    vis_base = colorize(pred_base, img_bgr)
    vis_aurora = colorize(pred_aurora, img_bgr)
    
    # Add text labels
    cv2.putText(vis_base, "Baseline PIDNet (No Roll Comp)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(vis_aurora, f"AuRoRA-2W (Roll: {roll_angle:.1f} deg)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    comparison = np.hstack((vis_base, vis_aurora))
    
    cv2.imwrite('samples/comparison.png', comparison)
    print("[+] Successfully saved side-by-side comparison to samples/comparison.png")

if __name__ == '__main__':
    main()
