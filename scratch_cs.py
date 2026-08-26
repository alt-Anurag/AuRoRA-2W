import urllib.request
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from models.pidnet_baseline import get_pred_model, load_pretrained

url = "https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/preparation/frankfurt_000000_000294_leftImg8bit.png"
# Just grab any dummy cityscapes image, if this URL fails I'll catch it.
try:
    urllib.request.urlretrieve(url, "cs_img.png")
except Exception as e:
    print("Download failed:", e)

model = get_pred_model('pidnet-s', 19)
load_pretrained(model, 'checkpoints/PIDNet_S_Cityscapes_val.pt')
model.eval()

transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = cv2.imread('cs_img.png')
if img is not None:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        pred = torch.argmax(model(tensor), dim=1).squeeze().numpy()

    u, c = np.unique(pred, return_counts=True)
    print("Cityscapes predictions:")
    for uu, cc in zip(u, c):
        print(f"Class {uu}: {cc}")
else:
    print("Failed to read image.")
