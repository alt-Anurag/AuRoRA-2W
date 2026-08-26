import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from models.pidnet_baseline import get_pred_model, load_pretrained
import numpy as np

model = get_pred_model('pidnet-s', 19)
load_pretrained(model, 'checkpoints/PIDNet_S_Cityscapes_val.pt')
model.train() 

transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = cv2.imread('data/frames/01_019/000001.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = transform(Image.fromarray(rgb)).unsqueeze(0)

with torch.no_grad():
    out = model(tensor)
    if isinstance(out, list):
        out = out[1] # x_ is at index 1 for augment=True
    pred = torch.argmax(out, dim=1).squeeze().numpy()

u, c = np.unique(pred, return_counts=True)
print("Train mode predictions:")
for uu, cc in zip(u, c):
    print(f"Class {uu}: {cc}")
