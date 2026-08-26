import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from models.pidnet_baseline import get_pred_model, load_pretrained
import numpy as np

model = get_pred_model('pidnet-s', 19)
load_pretrained(model, 'checkpoints/PIDNet_S_Cityscapes_val.pt')
model.eval()

transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = cv2.imread('data/frames/01_019/000001.png')
# BGR
tensor_bgr = transform(Image.fromarray(img)).unsqueeze(0)

with torch.no_grad():
    pred_bgr = torch.argmax(model(tensor_bgr), dim=1).squeeze().numpy()

u, c = np.unique(pred_bgr, return_counts=True)
print("BGR predictions:")
for uu, cc in zip(u, c):
    print(f"Class {uu}: {cc}")
