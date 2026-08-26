"""
AuRoRA-2W-IUR Video Inference
==============================
Runs inference on a video file using a trained AuRoRA-2W-IUR checkpoint.
Overlays drivable area, lane lines, and detection bounding boxes on each frame.

CRITICAL COLOR FIX (ported from Phase 1 test_video.py):
  Our color palette is defined in RGB, but OpenCV uses BGR.
  All colors use color[::-1] when drawing. Without this, lanes turn red.

Usage:
    python test_video_iur.py \
        --video    data/raw_video/clip.mp4 \
        --imu-csv  data/synced/clip_synced.csv \
        --checkpoint outputs/iur_run_01/aurora2w_iur_epoch_99.pt \
        --output   outputs/clip_iur_overlay.mp4
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.aurora2w_iur import get_aurora2w_iur_model


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Drivable area colors (RGB) — will be reversed to BGR for OpenCV
DRIVE_COLORS_RGB = [
    (0,   0,   0),    # 0: background       — transparent
    (0, 255, 102),    # 1: drivable area    — green
    (255, 165,  0),   # 2: alt. drivable    — orange
]

# Lane line color (RGB)
LANE_COLOR_RGB = (0, 200, 255)   # cyan

# Detection class labels and colors (RGB)
DET_CLASSES = ['Vehicle', 'Pothole', 'Speed Breaker']
DET_COLORS_RGB = [
    (255,  50,  50),   # Vehicle       — red
    (255, 220,   0),   # Pothole       — yellow
    (255,  50, 255),   # Speed Breaker — magenta
]


# ─────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────

def preprocess_frame(frame_bgr: np.ndarray, target_hw: tuple) -> torch.Tensor:
    """BGR uint8 frame → normalised float tensor (1, 3, H, W)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (target_hw[1], target_hw[0]))
    img = frame_rgb.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────
# Postprocessing
# ─────────────────────────────────────────────────────────────────

def decode_detections(
    det_preds: list,
    anchors: list,
    conf_thresh: float,
    img_h: int,
    img_w: int,
    num_classes: int = 3,
) -> list:
    """
    Decode raw detection predictions from all FPN scales.

    Returns list of dicts: [{'cls': int, 'conf': float, 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}]
    """
    results = []
    for pred, anch in zip(det_preds, anchors):
        B, _, grid_h, grid_w = pred.shape
        A = len(anch)
        stride_h = img_h / grid_h
        stride_w = img_w / grid_w

        pred = pred.view(B, A, 5 + num_classes, grid_h, grid_w)
        pred_xy  = torch.sigmoid(pred[0, :, :2])   # (A, 2, H, W)
        pred_wh  = pred[0, :, 2:4]                  # (A, 2, H, W)
        pred_obj = torch.sigmoid(pred[0, :, 4])     # (A, H, W)
        pred_cls = torch.sigmoid(pred[0, :, 5:])    # (A, C, H, W)

        for a_idx, (aw, ah) in enumerate(anch):
            for gj in range(grid_h):
                for gi in range(grid_w):
                    obj_conf = pred_obj[a_idx, gj, gi].item()
                    if obj_conf < conf_thresh:
                        continue
                    cls_probs = pred_cls[a_idx, :, gj, gi]
                    cls_id = cls_probs.argmax().item()
                    cls_conf = cls_probs[cls_id].item() * obj_conf

                    if cls_conf < conf_thresh:
                        continue

                    cx = (pred_xy[a_idx, 0, gj, gi].item() + gi) * stride_w
                    cy = (pred_xy[a_idx, 1, gj, gi].item() + gj) * stride_h
                    w  = np.exp(pred_wh[a_idx, 0, gj, gi].item()) * aw
                    h  = np.exp(pred_wh[a_idx, 1, gj, gi].item()) * ah

                    x1 = max(0, cx - w / 2)
                    y1 = max(0, cy - h / 2)
                    x2 = min(img_w, cx + w / 2)
                    y2 = min(img_h, cy + h / 2)

                    results.append({'cls': cls_id, 'conf': cls_conf,
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    # NMS (simple greedy)
    results = nms(results, iou_thresh=0.45)
    return results


def nms(boxes: list, iou_thresh: float = 0.45) -> list:
    """Simple greedy NMS."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b['conf'], reverse=True)
    keep = []
    for b in boxes:
        dominated = False
        for k in keep:
            if k['cls'] != b['cls']:
                continue
            inter_x1 = max(b['x1'], k['x1'])
            inter_y1 = max(b['y1'], k['y1'])
            inter_x2 = min(b['x2'], k['x2'])
            inter_y2 = min(b['y2'], k['y2'])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter   = inter_w * inter_h
            area_b  = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
            area_k  = (k['x2'] - k['x1']) * (k['y2'] - k['y1'])
            iou     = inter / (area_b + area_k - inter + 1e-8)
            if iou > iou_thresh:
                dominated = True
                break
        if not dominated:
            keep.append(b)
    return keep


# ─────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────

def draw_segmentation(frame: np.ndarray, drive_pred: np.ndarray, lane_pred: np.ndarray,
                       orig_h: int, orig_w: int) -> np.ndarray:
    """Overlay drivable area and lane line segmentation on the original frame."""
    overlay = frame.copy()

    # Drivable area
    drive_resized = cv2.resize(drive_pred.astype(np.uint8), (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
    for cls_id, rgb in enumerate(DRIVE_COLORS_RGB):
        if cls_id == 0:
            continue  # skip background
        mask = drive_resized == cls_id
        # CRITICAL FIX from Phase 1: color[::-1] converts RGB → BGR for OpenCV
        overlay[mask] = rgb[::-1]

    # Lane lines
    lane_resized = cv2.resize(lane_pred.astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    # CRITICAL FIX from Phase 1: color[::-1]
    overlay[lane_resized == 1] = LANE_COLOR_RGB[::-1]

    return cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)


def draw_detections(frame: np.ndarray, detections: list,
                     orig_h: int, orig_w: int,
                     model_h: int, model_w: int) -> np.ndarray:
    """Draw bounding boxes and labels on the original frame."""
    sx = orig_w / model_w
    sy = orig_h / model_h

    for det in detections:
        cls_id = det['cls']
        x1 = int(det['x1'] * sx)
        y1 = int(det['y1'] * sy)
        x2 = int(det['x2'] * sx)
        y2 = int(det['y2'] * sy)

        # CRITICAL FIX from Phase 1: color[::-1] RGB → BGR
        color_bgr = DET_COLORS_RGB[cls_id][::-1]
        label = f"{DET_CLASSES[cls_id]} {det['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AuRoRA-2W-IUR Video Inference')
    parser.add_argument('--video',       required=True, help='Input video path')
    parser.add_argument('--checkpoint',  required=True, help='Model checkpoint path')
    parser.add_argument('--output',      default='output_iur.mp4', help='Output video path')
    parser.add_argument('--imu-csv',     default=None,  help='Synced IMU CSV (from extract_roll_pitch.py)')
    parser.add_argument('--img-h',       type=int, default=384)
    parser.add_argument('--img-w',       type=int, default=640)
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Detection confidence threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_hw = (args.img_h, args.img_w)

    # ── Load Model ────────────────────────────────────────────
    print(f"[*] Loading model from {args.checkpoint} ...")
    model = get_aurora2w_iur_model(num_det_classes=3, num_drive_classes=3)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print("[✓] Model loaded.")

    # ── Load IMU CSV ──────────────────────────────────────────
    imu_df = None
    if args.imu_csv and Path(args.imu_csv).exists():
        imu_df = pd.read_csv(args.imu_csv)
        print(f"[*] Loaded IMU data: {len(imu_df)} frames.")
    else:
        print("[!] No IMU CSV provided — using roll_angle=0.0 for all frames.")

    # ── Open Video ────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))

    print(f"[*] Processing {n_frames} frames @ {fps:.0f} fps ...")

    DEFAULT_ANCHORS = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)],
    ]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            for frame_idx in tqdm(range(n_frames)):
                ret, frame = cap.read()
                if not ret:
                    break

                # Get roll angle for this frame
                roll_angle = 0.0
                if imu_df is not None and frame_idx < len(imu_df):
                    roll_angle = float(imu_df.iloc[frame_idx].get('roll_deg', 0.0))

                # Preprocess
                img_tensor = preprocess_frame(frame, target_hw).to(device)
                roll_tensor = torch.tensor([roll_angle], device=device)

                # Forward pass
                outputs = model(img_tensor, roll_tensor)

                # Decode segmentation
                drive_pred = outputs['drive_seg'].argmax(dim=1).squeeze(0).cpu().numpy()
                lane_pred  = outputs['lane_seg'].argmax(dim=1).squeeze(0).cpu().numpy()

                # Decode detections
                det_preds_cpu = [p.cpu() for p in outputs['det']]
                detections = decode_detections(
                    det_preds_cpu, DEFAULT_ANCHORS,
                    conf_thresh=args.conf_thresh,
                    img_h=args.img_h, img_w=args.img_w,
                )

                # Draw everything
                result = draw_segmentation(frame, drive_pred, lane_pred, orig_h, orig_w)
                result = draw_detections(result, detections, orig_h, orig_w, args.img_h, args.img_w)

                # Add roll angle HUD
                cv2.putText(result, f"IMU Roll: {roll_angle:+.1f} deg",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

                out_writer.write(result)

    cap.release()
    out_writer.release()
    print(f"\n[+] Output video saved to: {args.output}")


if __name__ == '__main__':
    main()
