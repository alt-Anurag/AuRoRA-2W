import os
import sys
import argparse
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))
import models.aurora2w
from configs import config, update_config

# Cityscapes mean/std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Cityscapes color map
color_map = [(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
             (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
             (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
             (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
             (  0, 80,100), (  0,  0,230), (119, 11, 32)]

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1] # BGR to RGB
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained_path):
    print(f"[*] Loading weights from {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    model_dict = model.state_dict()
    
    # The training script saves the AuRoRA2W_FullModel which has a 'model.' prefix
    cleaned_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('model.'):
            cleaned_dict[k[6:]] = v
        else:
            cleaned_dict[k] = v
            
    # Filter matching parameters
    pretrained_dict = {k: v for k, v in cleaned_dict.items() if (k in model_dict and v.shape == model_dict[k].shape)}
    
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/raw_video/03_033.mp4')
    parser.add_argument('--output', default='data/synced/03_033_segmentation_output.mp4')
    args = parser.parse_args()

    # Model Config
    config.defrost()
    config.DATASET.ROOT = 'models/PIDNet/data/'
    config.MODEL.PRETRAINED = 'models/PIDNet/pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar'
    config.TRAIN.IMAGE_SIZE = [512, 512]
    config.freeze()

    print(f"[*] Initializing AuRoRA2W Model...")
    model = models.aurora2w.get_seg_model(config, imgnet_pretrained=False)
    
    # Find latest checkpoint
    final_output_dir = 'output/cityscapes/pidnet_small_cityscapes'
    checkpoints = glob.glob(os.path.join(final_output_dir, 'aurora2w_epoch_*.pt'))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found!")
    latest_cp = max(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    model = load_pretrained(model, latest_cp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    print(f"[*] Processing Video: {total_frames} frames...")
    
    with torch.no_grad():
        # Autocast for mixed precision inference to avoid OOM and speed it up
        with torch.amp.autocast('cuda'):
            for _ in tqdm(range(total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess
                img = input_transform(frame)
                img = img.transpose((2, 0, 1)).copy()
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
                
                # Aurora2W has a roll_angles argument
                roll_angles = torch.zeros(1, device=device)
                pred = model(img_tensor, roll_angles=roll_angles)
                
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                    
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
                pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
                
                # Create segmentation map
                sv_img = np.zeros_like(frame).astype(np.uint8)
                for i, color in enumerate(color_map):
                    sv_img[pred == i] = color
                    
                # Blend with original
                blended = cv2.addWeighted(frame, 0.5, sv_img, 0.5, 0)
                out.write(blended)

    cap.release()
    out.release()
    print(f"[+] Video saved to {args.output}")

if __name__ == '__main__':
    main()
