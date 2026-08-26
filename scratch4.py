import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from models.pidnet_baseline import get_pred_model, load_pretrained
import numpy as np
import torch.nn.functional as F

model = get_pred_model('pidnet-s', 19)
load_pretrained(model, 'checkpoints/PIDNet_S_Cityscapes_val.pt')
model.eval()

transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = cv2.imread('data/frames/01_019/000001.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(rgb, (2048, 1024))
tensor = transform(Image.fromarray(img_resized)).unsqueeze(0)

with torch.no_grad():
    pred = torch.argmax(model(tensor), dim=1).squeeze().numpy()

u, c = np.unique(pred, return_counts=True)
print("Resized 1024x2048 predictions:")
for uu, cc in zip(u, c):
    print(f"Class {uu}: {cc}")
