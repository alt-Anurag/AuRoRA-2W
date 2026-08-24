import os
import sys
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'PIDNet'))
import models.pidnet

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
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Weights not found at {pretrained_path}. Please download them manually.")
    
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    
    model_dict = model.state_dict()
    # PIDNet author weights have a 'model.' prefix due to DataParallel
    if any(k.startswith('model.') for k in pretrained_dict.keys()):
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and v.shape == model_dict[k].shape)}
    
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from {pretrained_path}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pidnet-s', choices=['pidnet-s', 'pidnet-m', 'pidnet-l'])
    parser.add_argument('--weights', default='models/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt')
    parser.add_argument('--image', default='samples/test_frame.png')
    parser.add_argument('--output', default='samples/test_frame_out.png')
    args = parser.parse_args()

    print(f"[*] Initializing {args.model}...")
    model = models.pidnet.get_pred_model(args.model, 19)
    model = load_pretrained(model, args.weights)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"[*] Processing {args.image}...")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise ValueError(f"Could not read image {args.image}")
    
    img_h, img_w = img_bgr.shape[:2]
    
    img = input_transform(img_bgr)
    img = img.transpose((2, 0, 1)).copy()
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        # Handle cases where model outputs a tuple (e.g., train mode or aux branches)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1] 
            
        pred = F.interpolate(pred, size=(img_h, img_w), mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    # Create segmentation map
    sv_img = np.zeros_like(img_bgr).astype(np.uint8)
    for i, color in enumerate(color_map):
        sv_img[pred == i] = color

    # Blend with original
    blended = cv2.addWeighted(img_bgr, 0.5, sv_img, 0.5, 0)
    
    # Save results
    cv2.imwrite(args.output, blended)
    print(f"[+] Saved inference result to {args.output}")

if __name__ == '__main__':
    main()
