import urllib.request
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from models.pidnet_baseline import get_pred_model, load_pretrained
import torch.nn.functional as F
import os

model = get_pred_model('pidnet-s', 19)
load_pretrained(model, 'checkpoints/PIDNet_S_Cityscapes_val.pt')
model.eval()

transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

def analyze_image(img_path, name):
    print(f"\n--- Analyzing {name} ({img_path}) ---")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image")
        return
    
    orig_h, orig_w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
    print(f"Input tensor shape: {tensor.shape}")
    print(f"Input tensor min: {tensor.min().item():.3f}, max: {tensor.max().item():.3f}")
    
    with torch.no_grad():
        out = model(tensor)
        print(f"Raw output shape: {out.shape}")
        pred = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        pred_cls = torch.argmax(pred, dim=1).squeeze(0).numpy()
    
    total_pixels = pred_cls.size
    u, c = np.unique(pred_cls, return_counts=True)
    
    print("\nclass_id | class_name | pixel_count | percentage")
    print("-" * 55)
    for uu, cc in zip(u, c):
        pct = (cc / total_pixels) * 100
        cls_name = classes[uu] if uu < len(classes) else f"Unknown({uu})"
        print(f"{uu:8d} | {cls_name:10s} | {cc:11d} | {pct:5.2f}%")

if os.path.exists('real_control.png'):
    analyze_image('real_control.png', 'Control Image')
analyze_image('data/frames/01_019/000001.png', 'MOTOR Frame 000001')
