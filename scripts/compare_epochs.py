import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))
import models.aurora2w
from configs import config

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
             (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
             (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
             (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
             (  0, 80,100), (  0,  0,230), (119, 11, 32)]

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

config.defrost()
config.DATASET.ROOT = 'models/PIDNet/data/'
config.MODEL.PRETRAINED = 'models/PIDNet/pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar'
config.TRAIN.IMAGE_SIZE = [512, 512]
config.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.aurora2w.get_seg_model(config, imgnet_pretrained=False).to(device)

def load_cp(cp_path):
    pretrained_dict = torch.load(cp_path, map_location='cpu')
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    
    cleaned_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('model.'):
            cleaned_dict[k[6:]] = v
        else:
            cleaned_dict[k] = v
            
    model_dict = model.state_dict()
    final_dict = {k: v for k, v in cleaned_dict.items() if (k in model_dict and v.shape == model_dict[k].shape)}
    model.load_state_dict(final_dict, strict=False)
    model.eval()

# Extract frame 100
cap = cv2.VideoCapture('data/raw_video/01_001.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]

img = input_transform(frame).transpose((2, 0, 1)).copy()
img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
roll_angles = torch.zeros(1, device=device)

epochs = [116, 120, 130, 140, 147]
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Frame 100')
axes[0].axis('off')

with torch.no_grad():
    with torch.amp.autocast('cuda'):
        for i, ep in enumerate(epochs):
            cp_path = f'output/cityscapes/pidnet_small_cityscapes/aurora2w_epoch_{ep}.pt'
            if not os.path.exists(cp_path):
                print(f"Skipping {cp_path}")
                continue
            load_cp(cp_path)
            
            pred = model(img_tensor, roll_angles=roll_angles)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            sv_img = np.zeros_like(frame).astype(np.uint8)
            for j, color in enumerate(color_map):
                sv_img[pred == j] = color
                
            blended = cv2.addWeighted(frame, 0.5, sv_img, 0.5, 0)
            axes[i+1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f'Epoch {ep}')
            axes[i+1].axis('off')

plt.tight_layout()
plt.savefig('C:/Users/anura/.gemini/antigravity-cli/brain/fd9c13c1-d995-4908-b078-78a46d676111/scratch/epoch_comparison.png')
print("Saved comparison image.")
