import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import argparse

# Add PIDNet dir to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))
import models.aurora2w
import datasets
from configs import config, update_config
from utils.criterion import OhemCrossEntropy, BondaryLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="models/PIDNet/configs/cityscapes/pidnet_small_cityscapes.yaml", type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args([])
    update_config(config, args)
    
    config.defrost()
    config.DATASET.ROOT = 'models/PIDNet/data/'
    # Use standard small image size to save memory and speed up
    config.TRAIN.IMAGE_SIZE = [512, 512] 
    config.MODEL.PRETRAINED = 'models/PIDNet/pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar'
    config.freeze()
    
    OUTPUT_DIR = "output/idfa_overfit"
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running smoke test on device: {device} with memory optimizations")
    
    # Empty cache to free up any lingering VRAM
    torch.cuda.empty_cache()
    
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=1.0)
    
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=True)
    batch = next(iter(loader))
    images, labels, bd_gts, _, _ = batch
    
    images = images.to(device)
    labels = labels.to(device)
    bd_gts = bd_gts.to(device)
    
    pretrained_path = config.MODEL.PRETRAINED
    use_pretrained = os.path.isfile(pretrained_path)
    model = models.aurora2w.get_seg_model(config, imgnet_pretrained=use_pretrained).to(device)
    model.train()
    
    criterion_sem = OhemCrossEntropy(ignore_label=255, thres=0.9, min_kept=131072, weight=train_dataset.class_weights).to(device)
    criterion_bd = BondaryLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    print("[*] Pre-rotating batch to simulate fixed roll angles in the -45 to 45 degree range...")
    roll_angles = torch.linspace(-45.0, 45.0, steps=8).to(device)
    
    rot_inputs = torch.stack([TF.rotate(images[i], float(roll_angles[i]), fill=[0.0, 0.0, 0.0]) for i in range(8)])
    rot_labels = torch.stack([TF.rotate(labels[i].unsqueeze(0).float(), float(roll_angles[i]), interpolation=TF.InterpolationMode.NEAREST, fill=[255.0]).squeeze(0).long() for i in range(8)])
    rot_bd = torch.stack([TF.rotate(bd_gts[i].unsqueeze(0), float(roll_angles[i]), interpolation=TF.InterpolationMode.NEAREST, fill=[0.0]).squeeze(0) for i in range(8)])
    
    print("[*] Starting training loop (300 iterations) using micro-batches (accumulated) to prevent OOM on 6GB VRAM...")
    losses = []
    
    MICRO_BATCH_SIZE = 2
    
    for it in range(1, 301):
        optimizer.zero_grad()
        
        total_loss_val = 0.0
        
        # Micro-batching to save memory
        for i in range(0, 8, MICRO_BATCH_SIZE):
            mb_inputs = rot_inputs[i:i+MICRO_BATCH_SIZE]
            mb_labels = rot_labels[i:i+MICRO_BATCH_SIZE]
            mb_bd = rot_bd[i:i+MICRO_BATCH_SIZE]
            mb_angles = roll_angles[i:i+MICRO_BATCH_SIZE]
            
            with torch.amp.autocast('cuda'):
                # APPLYING SIGN FLIP
                mb_idfa_angles = -1.0 * mb_angles
                out = model(mb_inputs, roll_angles=mb_idfa_angles)
                
                ph, pw = out[0].size(2), out[0].size(3)
                h, w = mb_labels.size(1), mb_labels.size(2)
                if ph != h or pw != w:
                    for j in range(len(out)):
                        out[j] = F.interpolate(out[j], size=(h, w), mode='bilinear', align_corners=True)
                        
                loss1 = criterion_sem([out[0]], mb_labels)
                loss2 = criterion_sem([out[1]], mb_labels)
                loss3 = criterion_bd(out[2], mb_bd)
                
                # Scale loss down by number of micro-batches (8/2 = 4)
                mb_loss = (loss1 + loss2 + loss3) / 4.0
            
            scaler.scale(mb_loss).backward()
            total_loss_val += mb_loss.item() * 4.0  # Keep track of true scale
            
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(total_loss_val)
        
        if it % 5 == 0 or it == 1:
            print(f"Iter {it:03d}/300 | Loss: {total_loss_val:.4f}")
    
    print("[*] Generating loss curve plot...")
    plt.figure()
    plt.plot(range(1, 301), losses, label='Total Loss')
    plt.title("IDFA Smoke Test Overfit Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    print("[*] Generating Pred vs GT plots...")
    model.eval()
    
    preds_list = []
    with torch.no_grad():
        for i in range(0, 8, MICRO_BATCH_SIZE):
            mb_inputs = rot_inputs[i:i+MICRO_BATCH_SIZE]
            mb_angles = roll_angles[i:i+MICRO_BATCH_SIZE]
            with torch.amp.autocast('cuda'):
                # APPLYING SIGN FLIP
                mb_idfa_angles = -1.0 * mb_angles
                out = model(mb_inputs, roll_angles=mb_idfa_angles)
                
            ph, pw = out[0].size(2), out[0].size(3)
            h, w = rot_labels.size(1), rot_labels.size(2)
            if ph != h or pw != w:
                out[1] = F.interpolate(out[1], size=(h, w), mode='bilinear', align_corners=True)
            preds_list.append(torch.argmax(out[1], dim=1).cpu())
            
    preds = torch.cat(preds_list, dim=0).numpy()
    gt_labels = rot_labels.cpu().numpy()
    
    fig, axes = plt.subplots(8, 2, figsize=(10, 24))
    for i in range(8):
        gt_vis = gt_labels[i].copy()
        gt_vis[gt_vis == 255] = 19
        
        axes[i, 0].imshow(gt_vis, cmap='tab20', vmin=0, vmax=19)
        axes[i, 0].set_title(f"GT Image {i} (Roll: {roll_angles[i].item():.1f})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(preds[i], cmap='tab20', vmin=0, vmax=19)
        axes[i, 1].set_title(f"Pred Image {i}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_gt.png"))
    plt.close()
    
    final_loss = losses[-1]
    if final_loss < 0.2:
        print(f"\n[PASS] Loss dropped successfully to {final_loss:.4f} (close to zero).")
    else:
        print(f"\n[WARN] Loss is stuck at {final_loss:.4f}. It did NOT drop close to zero.")

if __name__ == "__main__":
    main()
