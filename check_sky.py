import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, 'models/PIDNet')
from models.pidnet import get_pred_model
import torchvision.transforms.functional as TF
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_sky_class(checkpoint_path):
    model = get_pred_model('pidnet-s', 19)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=False)
    model.eval()
    
    # Load first frame
    cap = cv2.VideoCapture('data/raw_video/01_001.mp4')
    ret, frame = cap.read()
    cap.release()
    
    img = frame.astype(np.float32)[:, :, ::-1] / 255.0
    img -= mean
    img /= std
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        roll_angles = torch.zeros(1, device=device)
        pred = model(img_tensor, roll_angles=roll_angles)
        if isinstance(pred, (list, tuple)): pred = pred[0]
        pred = F.interpolate(pred, size=(frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        
    # Check top left corner for sky
    sky_pixels = pred[10:50, 10:50].flatten()
    unique, counts = np.unique(sky_pixels, return_counts=True)
    print(f"{checkpoint_path} Top-Left Majority Class: {unique[np.argmax(counts)]}")

get_sky_class('output/cityscapes/pidnet_small_cityscapes/aurora2w_epoch_116.pt')
get_sky_class('output/cityscapes/pidnet_small_cityscapes/aurora2w_epoch_120.pt')
