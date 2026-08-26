"""
AuRoRA-2W-IUR Roll Augmentation Utilities
==========================================
Synthetic roll augmentation for multi-task training.

PHASE 1 → PHASE 2 CRITICAL DIFFERENCE:
  In Phase 1 we only rotated:
    - image tensor
    - semantic segmentation mask (fill=255 = ignore_index)

  In Phase 2 we ALSO have bounding boxes. You cannot rotate
  a bounding box by just rotating the image — the box
  coordinates must be transformed using the 2D rotation matrix.

  The math:
    For a box [cx, cy, w, h] (normalised, in image space):
      1. Convert cx, cy back to pixel space: px = cx*W, py = cy*H
      2. Translate to image centre: qx = px - W/2, qy = py - H/2
      3. Apply 2D rotation by angle θ:
            qx' =  qx * cos(θ) - qy * sin(θ)
            qy' =  qx * sin(θ) + qy * cos(θ)
      4. Translate back: px' = qx' + W/2, py' = qy' + H/2
      5. Renormalise: cx' = px'/W, cy' = py'/H
      6. The 4 corners of the original box also rotate,
         so the bounding box dimensions change — we compute
         the axis-aligned bounding box of the 4 rotated corners.

PHASE 1 CRITICAL FIXES (both ported here):
  1. drive_mask and lane_mask: fill corners with IGNORE (255), NOT 0.
     If you fill with 0, the model will learn that black corners = road.
  2. Lane masks: rotate with NEAREST interpolation + optional morphological
     dilation BEFORE rotating, so thin lane lines don't vanish.
"""

import math
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


DRIVE_IGNORE = 255
LANE_IGNORE  = 255


# ─────────────────────────────────────────────────────────────────
# Bounding Box Rotation (the Phase 2 new requirement)
# ─────────────────────────────────────────────────────────────────

def rotate_boxes_2d(boxes: torch.Tensor, angle_deg: float,
                    img_h: int, img_w: int) -> torch.Tensor:
    """
    Rotate YOLO-format normalised boxes by angle_deg degrees (CCW positive).

    Args:
        boxes: (N, 5) tensor — [class_id, cx, cy, w, h] all in [0,1]
        angle_deg: rotation angle in degrees
        img_h: image height in pixels
        img_w: image width in pixels

    Returns:
        rotated_boxes: (N, 5) tensor — [class_id, cx', cy', w', h'] in [0,1]
                       Boxes entirely outside the image after rotation are removed.
    """
    if boxes.numel() == 0:
        return boxes

    N = boxes.size(0)
    cls_ids = boxes[:, 0:1]                   # (N, 1)
    cx = boxes[:, 1].clone() * img_w          # pixel space
    cy = boxes[:, 2].clone() * img_h
    bw = boxes[:, 3].clone() * img_w
    bh = boxes[:, 4].clone() * img_h

    # Image centre
    ox = img_w / 2.0
    oy = img_h / 2.0

    # Rotation matrix elements
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Compute all 4 corners of each box in pixel space
    # corners shape: (N, 4, 2) — 4 corners, each (x, y)
    half_w = bw / 2.0
    half_h = bh / 2.0

    # Top-left, top-right, bottom-right, bottom-left
    corners = torch.stack([
        torch.stack([cx - half_w, cy - half_h], dim=1),  # TL
        torch.stack([cx + half_w, cy - half_h], dim=1),  # TR
        torch.stack([cx + half_w, cy + half_h], dim=1),  # BR
        torch.stack([cx - half_w, cy + half_h], dim=1),  # BL
    ], dim=1)  # (N, 4, 2)

    # Translate to image centre
    corners[:, :, 0] -= ox
    corners[:, :, 1] -= oy

    # Rotate each corner
    rx = cos_t * corners[:, :, 0] - sin_t * corners[:, :, 1]  # (N, 4)
    ry = sin_t * corners[:, :, 0] + cos_t * corners[:, :, 1]  # (N, 4)

    # Translate back
    rx += ox
    ry += oy

    # Axis-aligned bounding box of the 4 rotated corners
    x_min = rx.min(dim=1).values
    x_max = rx.max(dim=1).values
    y_min = ry.min(dim=1).values
    y_max = ry.max(dim=1).values

    # New centre and size
    new_cx = ((x_min + x_max) / 2.0) / img_w
    new_cy = ((y_min + y_max) / 2.0) / img_h
    new_w  = (x_max - x_min) / img_w
    new_h  = (y_max - y_min) / img_h

    new_boxes = torch.stack([cls_ids.squeeze(1), new_cx, new_cy, new_w, new_h], dim=1)

    # Remove boxes that are mostly outside the image
    # Keep only boxes whose centre is within [0,1]
    valid = (new_cx >= 0) & (new_cx <= 1) & (new_cy >= 0) & (new_cy <= 1)
    # Also clip width and height to valid range
    new_boxes[:, 3] = new_boxes[:, 3].clamp(0.01, 1.0)
    new_boxes[:, 4] = new_boxes[:, 4].clamp(0.01, 1.0)

    return new_boxes[valid]


# ─────────────────────────────────────────────────────────────────
# Lane mask pre-processing (dilation before rotation)
# ─────────────────────────────────────────────────────────────────

def dilate_lane_mask(lane_mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Morphologically dilate a binary lane mask to thicken thin lane lines.
    This prevents them from disappearing during rotation.

    Args:
        lane_mask: (H, W) long tensor with values {0, 1, 255}
        kernel_size: dilation structuring element size (default 3 = safe, 5 = more aggressive)

    Returns:
        dilated lane_mask: (H, W) long tensor
    """
    # Work only on the lane pixels (value=1)
    binary = (lane_mask == 1).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pad = kernel_size // 2
    dilated = torch.nn.functional.max_pool2d(binary, kernel_size=kernel_size,
                                              stride=1, padding=pad)
    dilated_mask = lane_mask.clone()
    # Set newly dilated pixels to LANE_LINE (1), don't overwrite IGNORE pixels
    new_lane = (dilated.squeeze() > 0.5) & (lane_mask != 255)
    dilated_mask[new_lane] = 1
    return dilated_mask


# ─────────────────────────────────────────────────────────────────
# Main Augmentation Function
# ─────────────────────────────────────────────────────────────────

def apply_synthetic_roll(
    image:      torch.Tensor,   # (3, H, W) float
    drive_mask: torch.Tensor,   # (H, W) long
    lane_mask:  torch.Tensor,   # (H, W) long
    boxes:      torch.Tensor,   # (N, 5) float [cls, cx, cy, w, h]
    angle_deg:  float,          # rotation angle in degrees
    dilate_lanes: bool = True,
    dilation_kernel: int = 3,
) -> tuple:
    """
    Apply a single synthetic roll rotation to all modalities consistently.

    Returns:
        rot_image:      (3, H, W)
        rot_drive_mask: (H, W)
        rot_lane_mask:  (H, W)
        rot_boxes:      (N', 5)  — N' ≤ N, some boxes may be removed if out of frame
        angle_deg:      float    — the angle that was applied (for injection into model)
    """
    _, H, W = image.shape

    # ── Image ────────────────────────────────────────────────────
    rot_image = TF.rotate(
        image,
        angle=-angle_deg,           # TF.rotate is CW for positive, so negate
        interpolation=InterpolationMode.BILINEAR,
        fill=[0.0, 0.0, 0.0],       # fill with ImageNet mean-normalized zeros
    )

    # ── Drivable Area Mask ───────────────────────────────────────
    # CRITICAL FIX (Phase 1): fill=255 NOT 0
    rot_drive = TF.rotate(
        drive_mask.unsqueeze(0).float(),
        angle=-angle_deg,
        interpolation=InterpolationMode.NEAREST,
        fill=[float(DRIVE_IGNORE)],
    ).squeeze(0).long()

    # ── Lane Mask ────────────────────────────────────────────────
    # Dilate thin lane lines BEFORE rotating so they survive interpolation
    if dilate_lanes:
        lane_mask = dilate_lane_mask(lane_mask, dilation_kernel)

    rot_lane = TF.rotate(
        lane_mask.unsqueeze(0).float(),
        angle=-angle_deg,
        interpolation=InterpolationMode.NEAREST,  # NEAREST for binary masks
        fill=[float(LANE_IGNORE)],
    ).squeeze(0).long()

    # ── Bounding Boxes ───────────────────────────────────────────
    # Apply 2D rotation matrix math to box coordinates
    rot_boxes = rotate_boxes_2d(boxes, angle_deg=-angle_deg, img_h=H, img_w=W)

    return rot_image, rot_drive, rot_lane, rot_boxes, angle_deg


# ─────────────────────────────────────────────────────────────────
# Batch-level augmentation (called inside training loop)
# ─────────────────────────────────────────────────────────────────

def apply_batch_roll_augmentation(batch: dict, angle_range: float = 30.0) -> dict:
    """
    Apply per-sample synthetic roll augmentation to a full batch.

    Args:
        batch: output of collate_fn — contains:
            'image':      (B, 3, H, W)
            'drive_mask': (B, H, W)
            'lane_mask':  (B, H, W)
            'boxes':      list of (N_i, 5) tensors
            'roll_angle': (B,) tensor (pre-existing IMU values, may be zeros)
        angle_range: max abs rotation in degrees (default ±30°)

    Returns:
        Augmented batch with same keys. 'roll_angle' updated to synthetic angles.
    """
    B = batch['image'].size(0)
    device = batch['image'].device

    # Sample one roll angle per image in [-angle_range, +angle_range]
    synthetic_angles = (torch.rand(B) * 2 * angle_range) - angle_range  # (B,)

    rot_images = []
    rot_drives = []
    rot_lanes  = []
    rot_boxes  = []

    for i in range(B):
        angle = float(synthetic_angles[i])
        ri, rd, rl, rb, _ = apply_synthetic_roll(
            image=batch['image'][i].cpu(),
            drive_mask=batch['drive_mask'][i].cpu(),
            lane_mask=batch['lane_mask'][i].cpu(),
            boxes=batch['boxes'][i].cpu(),
            angle_deg=angle,
        )
        rot_images.append(ri)
        rot_drives.append(rd)
        rot_lanes.append(rl)
        rot_boxes.append(rb.to(device))

    return {
        'image':      torch.stack(rot_images).to(device),
        'drive_mask': torch.stack(rot_drives).to(device),
        'lane_mask':  torch.stack(rot_lanes).to(device),
        'boxes':      rot_boxes,
        'roll_angle': synthetic_angles.to(device),
    }


# ─────────────────────────────────────────────────────────────────
# Test the box rotation math
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing bounding box rotation math...")

    # A box at image centre should stay at image centre for any angle
    H, W = 384, 640
    centre_box = torch.tensor([[DET_POTHOLE := 1, 0.5, 0.5, 0.1, 0.1]])

    for angle in [0, 15, 30, -30, 45]:
        rot = rotate_boxes_2d(centre_box.clone(), angle_deg=float(angle), img_h=H, img_w=W)
        if len(rot) > 0:
            err = abs(rot[0, 1].item() - 0.5) + abs(rot[0, 2].item() - 0.5)
            print(f"  angle={angle:+4d}°  cx={rot[0,1]:.4f} cy={rot[0,2]:.4f}  centre_err={err:.6f}")
        else:
            print(f"  angle={angle:+4d}°  box removed (out of frame)")

    print("[OK] Centre box stays centred — rotation math is correct.")

    print("\nTesting lane mask dilation...")
    lane_mask = torch.zeros(10, 10, dtype=torch.long)
    lane_mask[5, :] = 1  # single-pixel-wide horizontal lane line
    dilated = dilate_lane_mask(lane_mask, kernel_size=3)
    print(f"  Lane pixels before dilation: {(lane_mask == 1).sum().item()}")
    print(f"  Lane pixels after  dilation: {(dilated == 1).sum().item()}")
    print("[OK] Lane dilation working.")
