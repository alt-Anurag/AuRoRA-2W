"""
Composite Dataset — IDD + RDD2022 + BDD100K
=============================================
Unified dataset that merges three sources into a single PyTorch Dataset:

  Source       | Task                  | Classes
  ─────────────┼───────────────────────┼──────────────────────────────
  IDD          | Drivable seg + Det    | Vehicle, Drivable Area
  RDD2022 India| Detection only        | Pothole (D40), Speed Breaker (D10)
  BDD100K      | Lane seg + Det        | Lane lines, Vehicle

Every sample returns:
  image:       (3, H, W) float tensor — normalised
  roll_angle:  scalar float — IMU roll (0.0 if not available for that source)
  drive_mask:  (H, W) long — drivable area label (255 = ignore)
  lane_mask:   (H, W) long — lane line label     (255 = ignore)
  boxes:       (N, 5) float — [class_id, cx, cy, w, h] normalised YOLO format
               N=0 if no boxes in this sample

NOTE ON PATHS (for Colab):
  Set the three root paths at the top of this file (or pass via config dict).
  Everything else is handled automatically.
"""

import os
import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset


# ─────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Drivable area class IDs (unified across all sources)
DRIVE_BG          = 0
DRIVE_DRIVABLE    = 1
DRIVE_ALT_DRIVABLE = 2
DRIVE_IGNORE      = 255

# Lane class IDs
LANE_BG     = 0
LANE_LINE   = 1
LANE_IGNORE = 255

# Detection class IDs
DET_VEHICLE        = 0
DET_POTHOLE        = 1
DET_SPEED_BREAKER  = 2


def normalize(img_rgb: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB → float32 HWC normalised."""
    img = img_rgb.astype(np.float32) / 255.0
    img -= np.array(MEAN, dtype=np.float32)
    img /= np.array(STD,  dtype=np.float32)
    return img


def to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_hwc.transpose(2, 0, 1))  # CHW


def resize_image(img, target_hw):
    return cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask, target_hw):
    return cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


def scale_boxes(boxes_xyxy: np.ndarray, orig_hw, new_hw) -> np.ndarray:
    """Scale bounding boxes from one resolution to another."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    sx = new_hw[1] / orig_hw[1]
    sy = new_hw[0] / orig_hw[0]
    boxes_xyxy[:, [0, 2]] *= sx
    boxes_xyxy[:, [1, 3]] *= sy
    return boxes_xyxy


def xyxy_to_cxcywh_norm(boxes_xyxy: np.ndarray, H: int, W: int) -> np.ndarray:
    """Convert [x1,y1,x2,y2] pixel to [cx,cy,w,h] normalised 0-1."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Sample dataclass
# ─────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Custom collate: boxes have variable N so we keep them as a list.
    Returns dict of tensors + list-of-tensors for boxes.
    """
    images      = torch.stack([b['image']      for b in batch])
    roll_angles = torch.tensor([b['roll_angle'] for b in batch], dtype=torch.float32)
    drive_masks = torch.stack([b['drive_mask'] for b in batch])
    lane_masks  = torch.stack([b['lane_mask']  for b in batch])
    boxes       = [b['boxes'] for b in batch]           # list of (N_i, 5) tensors
    return {
        'image':      images,
        'roll_angle': roll_angles,
        'drive_mask': drive_masks,
        'lane_mask':  lane_masks,
        'boxes':      boxes,
    }


# ─────────────────────────────────────────────────────────────────
# IDD Dataset Adapter
# ─────────────────────────────────────────────────────────────────

class IDDDataset(Dataset):
    """
    Indian Driving Dataset adapter.

    Expected directory layout (after download from idd.insaan.iiit.ac.in):
      idd_root/
        IDD_Detection/
          Annotations/           ← Pascal VOC XML files
          JPEGImages/            ← .jpg images
        IDD_Segmentation/
          gtFine/
            train/
              <city>/
                <img>_gtFine_labelids.png
          leftImg8bit/
            train/
              <city>/
                <img>_leftImg8bit.png

    IDD segmentation label IDs → our drivable area mapping:
      0  = drivable         → DRIVE_DRIVABLE (1)
      1  = non-drivable     → DRIVE_BG (0)
      7  = out-of-roi       → DRIVE_IGNORE (255)
      everything else       → DRIVE_BG (0)
    """

    # IDD Detection XML class name → our detection class ID
    IDD_DET_CLASS_MAP = {
        'car':          DET_VEHICLE,
        'truck':        DET_VEHICLE,
        'bus':          DET_VEHICLE,
        'motorcycle':   DET_VEHICLE,
        'autorickshaw': DET_VEHICLE,
        'bicycle':      DET_VEHICLE,
        'person':       DET_VEHICLE,   # treat pedestrian as obstacle
        'animal':       DET_VEHICLE,
    }

    # IDD drivable area label mapping (from IDD20k labels)
    IDD_SEG_DRIVABLE_IDS = {0, 1}  # IDD classes 0 (drivable) and 1 (alt_drivable)

    def __init__(self, idd_root: str, split: str = 'train', target_hw=(384, 640)):
        self.target_hw = target_hw
        self.samples = []

        seg_img_root  = Path(idd_root) / 'IDD_Segmentation' / 'leftImg8bit' / split
        seg_lbl_root  = Path(idd_root) / 'IDD_Segmentation' / 'gtFine' / split
        det_img_root  = Path(idd_root) / 'IDD_Detection' / 'JPEGImages'
        det_ann_root  = Path(idd_root) / 'IDD_Detection' / 'Annotations'

        # Segmentation samples
        if seg_img_root.exists():
            for city_dir in sorted(seg_img_root.iterdir()):
                for img_path in sorted(city_dir.glob('*_leftImg8bit.png')):
                    stem = img_path.stem.replace('_leftImg8bit', '')
                    lbl_path = seg_lbl_root / city_dir.name / f'{stem}_gtFine_labelids.png'
                    if lbl_path.exists():
                        self.samples.append({
                            'image': str(img_path),
                            'drive_mask': str(lbl_path),
                            'lane_mask': None,
                            'boxes_xml': None,
                            'source': 'idd_seg',
                        })

        # Detection samples
        if det_img_root.exists():
            for xml_path in sorted(det_ann_root.glob('*.xml')):
                img_path = det_img_root / xml_path.with_suffix('.jpg').name
                if img_path.exists():
                    self.samples.append({
                        'image': str(img_path),
                        'drive_mask': None,
                        'lane_mask': None,
                        'boxes_xml': str(xml_path),
                        'source': 'idd_det',
                    })

        print(f"[IDD] {split}: {len(self.samples)} samples loaded.")

    def _parse_voc_xml(self, xml_path: str, H: int, W: int):
        """Parse Pascal VOC XML → list of [class_id, x1, y1, x2, y2]."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()
            cls_id = self.IDD_DET_CLASS_MAP.get(name, None)
            if cls_id is None:
                continue
            bndbox = obj.find('bndbox')
            x1 = float(bndbox.find('xmin').text)
            y1 = float(bndbox.find('ymin').text)
            x2 = float(bndbox.find('xmax').text)
            y2 = float(bndbox.find('ymax').text)
            boxes.append([cls_id, x1, y1, x2, y2])
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)

    def _parse_drive_mask(self, mask_path: str, H: int, W: int):
        """Read IDD segmentation label → unified drivable mask."""
        raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.full(raw.shape, DRIVE_BG, dtype=np.uint8)
        mask[raw == 0] = DRIVE_DRIVABLE
        mask[raw == 1] = DRIVE_ALT_DRIVABLE
        mask[raw == 7] = DRIVE_IGNORE
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        H, W = self.target_hw

        img_bgr = cv2.imread(s['image'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        img_rgb = resize_image(img_rgb, self.target_hw)

        # Drive mask
        if s['drive_mask']:
            drive_mask = self._parse_drive_mask(s['drive_mask'], orig_h, orig_w)
            drive_mask = resize_mask(drive_mask, self.target_hw)
        else:
            drive_mask = np.full((H, W), DRIVE_IGNORE, dtype=np.uint8)

        # Lane mask — IDD has no lane annotations
        lane_mask = np.full((H, W), LANE_IGNORE, dtype=np.uint8)

        # Boxes
        if s['boxes_xml']:
            raw_boxes = self._parse_voc_xml(s['boxes_xml'], orig_h, orig_w)
            if len(raw_boxes) > 0:
                cls_ids = raw_boxes[:, 0:1]
                xyxy = raw_boxes[:, 1:]
                xyxy = scale_boxes(xyxy, (orig_h, orig_w), self.target_hw)
                cxcywh = xyxy_to_cxcywh_norm(xyxy, H, W)
                boxes = np.concatenate([cls_ids, cxcywh], axis=1)
            else:
                boxes = np.zeros((0, 5), dtype=np.float32)
        else:
            boxes = np.zeros((0, 5), dtype=np.float32)

        return {
            'image':      to_tensor(normalize(img_rgb)),
            'roll_angle': 0.0,           # IDD has no IMU
            'drive_mask': torch.from_numpy(drive_mask).long(),
            'lane_mask':  torch.from_numpy(lane_mask).long(),
            'boxes':      torch.from_numpy(boxes),
        }


# ─────────────────────────────────────────────────────────────────
# RDD2022 India Subset Adapter
# ─────────────────────────────────────────────────────────────────

class RDD2022IndiaDataset(Dataset):
    """
    Road Damage Dataset 2022 — India subset.
    Kaggle: https://www.kaggle.com/datasets/deepsystemsresearch/road-damage-dataset-2022

    Expected layout after unzip:
      rdd_root/
        India/
          train/
            images/   ← .jpg
            labels/   ← YOLO format .txt (class cx cy w h, normalised)

    RDD2022 classes → our detection class IDs:
      0 (D00 longitudinal crack) → ignore
      1 (D10 transverse crack)   → DET_SPEED_BREAKER (2) — approx mapping
      2 (D20 alligator crack)    → ignore
      3 (D40 pothole)            → DET_POTHOLE (1)
    """

    RDD_CLASS_MAP = {
        0: None,            # D00 longitudinal crack — ignore
        1: DET_SPEED_BREAKER,  # D10 transverse crack — approximate speed breaker
        2: None,            # D20 alligator crack — ignore
        3: DET_POTHOLE,     # D40 pothole
    }

    def __init__(self, rdd_root: str, split: str = 'train', target_hw=(384, 640)):
        self.target_hw = target_hw
        img_dir = Path(rdd_root) / 'India' / split / 'images'
        lbl_dir = Path(rdd_root) / 'India' / split / 'labels'
        self.samples = []

        if not img_dir.exists():
            print(f"[RDD2022] Warning: {img_dir} not found. Skipping.")
            return

        for img_path in sorted(img_dir.glob('*.jpg')):
            lbl_path = lbl_dir / img_path.with_suffix('.txt').name
            self.samples.append({'image': str(img_path), 'label': str(lbl_path)})

        print(f"[RDD2022 India] {split}: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        H, W = self.target_hw

        img_bgr = cv2.imread(s['image'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = resize_image(img_rgb, self.target_hw)

        # Parse YOLO-format labels
        boxes = []
        lbl_path = Path(s['label'])
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    rdd_cls = int(parts[0])
                    our_cls = self.RDD_CLASS_MAP.get(rdd_cls, None)
                    if our_cls is None:
                        continue
                    cx, cy, bw, bh = map(float, parts[1:5])
                    boxes.append([our_cls, cx, cy, bw, bh])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)

        return {
            'image':      to_tensor(normalize(img_rgb)),
            'roll_angle': 0.0,
            'drive_mask': torch.full((H, W), DRIVE_IGNORE, dtype=torch.long),
            'lane_mask':  torch.full((H, W), LANE_IGNORE,  dtype=torch.long),
            'boxes':      torch.from_numpy(boxes),
        }


# ─────────────────────────────────────────────────────────────────
# BDD100K Lane Adapter
# ─────────────────────────────────────────────────────────────────

class BDD100KDataset(Dataset):
    """
    Berkeley DeepDrive 100K adapter.
    Download: https://bdd-data.berkeley.edu/

    Expected layout:
      bdd_root/
        images/
          100k/
            train/  ← .jpg images
            val/
        labels/
          lane/
            masks/
              train/  ← binary lane masks (.png, 255=lane, 0=bg)
              val/
          det_20/
            det_train.json   ← BDD100K detection annotations (JSON)
            det_val.json

    BDD100K detection categories → our class IDs:
      'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person' → DET_VEHICLE
    """

    BDD_DET_CLASS_MAP = {
        'car':        DET_VEHICLE,
        'truck':      DET_VEHICLE,
        'bus':        DET_VEHICLE,
        'motorcycle': DET_VEHICLE,
        'bicycle':    DET_VEHICLE,
        'person':     DET_VEHICLE,
        'rider':      DET_VEHICLE,
    }

    def __init__(self, bdd_root: str, split: str = 'train', target_hw=(384, 640)):
        self.target_hw = target_hw
        img_dir  = Path(bdd_root) / 'images' / '100k' / split
        lane_dir = Path(bdd_root) / 'labels' / 'lane' / 'masks' / split
        det_json = Path(bdd_root) / 'labels' / 'det_20' / f'det_{split}.json'

        self.samples = []
        self.det_lookup: Dict[str, list] = {}

        # Load detection JSON
        if det_json.exists():
            with open(det_json) as f:
                det_data = json.load(f)
            for ann in det_data:
                fname = ann['name']
                self.det_lookup[fname] = ann.get('labels', [])

        if not img_dir.exists():
            print(f"[BDD100K] Warning: {img_dir} not found. Skipping.")
            return

        for img_path in sorted(img_dir.glob('*.jpg')):
            lane_mask_path = lane_dir / img_path.with_suffix('.png').name
            self.samples.append({
                'image': str(img_path),
                'lane_mask': str(lane_mask_path) if lane_mask_path.exists() else None,
                'name': img_path.name,
            })

        print(f"[BDD100K] {split}: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        H, W = self.target_hw

        img_bgr = cv2.imread(s['image'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        img_rgb = resize_image(img_rgb, self.target_hw)

        # Lane mask
        if s['lane_mask']:
            raw = cv2.imread(s['lane_mask'], cv2.IMREAD_GRAYSCALE)
            lane_mask = np.zeros(raw.shape, dtype=np.uint8)
            lane_mask[raw > 127] = LANE_LINE
            lane_mask = resize_mask(lane_mask, self.target_hw)
        else:
            lane_mask = np.full((H, W), LANE_IGNORE, dtype=np.uint8)

        # Drivable mask — BDD100K has drivable area masks too (optional path)
        drive_mask = np.full((H, W), DRIVE_IGNORE, dtype=np.uint8)

        # Detection boxes
        boxes = []
        for lbl in self.det_lookup.get(s['name'], []):
            cat = lbl.get('category', '').lower()
            cls_id = self.BDD_DET_CLASS_MAP.get(cat, None)
            if cls_id is None:
                continue
            box2d = lbl.get('box2d', None)
            if box2d is None:
                continue
            x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
            xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            xyxy = scale_boxes(xyxy, (orig_h, orig_w), self.target_hw)
            cxcywh = xyxy_to_cxcywh_norm(xyxy, H, W)
            boxes.append(np.concatenate([[cls_id], cxcywh[0]]))

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)

        return {
            'image':      to_tensor(normalize(img_rgb)),
            'roll_angle': 0.0,
            'drive_mask': torch.from_numpy(drive_mask).long(),
            'lane_mask':  torch.from_numpy(lane_mask).long(),
            'boxes':      torch.from_numpy(boxes),
        }


# ─────────────────────────────────────────────────────────────────
# Composite Dataset Builder
# ─────────────────────────────────────────────────────────────────

def build_composite_dataset(
    idd_root: Optional[str],
    rdd_root: Optional[str],
    bdd_root: Optional[str],
    split: str = 'train',
    target_hw: Tuple[int, int] = (384, 640),
) -> ConcatDataset:
    """
    Build the merged IUR (Indian Unstructured Roads) dataset.

    Only includes sources for which a root path is provided and valid.
    """
    datasets = []

    if idd_root and Path(idd_root).exists():
        datasets.append(IDDDataset(idd_root, split, target_hw))
    else:
        print("[CompositeDataset] IDD root not found — skipping.")

    if rdd_root and Path(rdd_root).exists():
        datasets.append(RDD2022IndiaDataset(rdd_root, split, target_hw))
    else:
        print("[CompositeDataset] RDD2022 root not found — skipping.")

    if bdd_root and Path(bdd_root).exists():
        datasets.append(BDD100KDataset(bdd_root, split, target_hw))
    else:
        print("[CompositeDataset] BDD100K root not found — skipping.")

    if not datasets:
        raise RuntimeError("No dataset roots found! Check your paths.")

    combined = ConcatDataset(datasets)
    print(f"\n[CompositeDataset] Total {split} samples: {len(combined)}")
    return combined
