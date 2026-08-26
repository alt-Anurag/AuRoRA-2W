"""
AuRoRA-2W-IUR Loss Functions
==============================
Multi-task loss combining:

  1. Segmentation Loss (Drivable Area + Lane Lines):
     OhemCrossEntropy — Ported from Phase 1 (models/PIDNet/utils/criterion.py)
     Hard pixel mining that focusses on the hardest edge pixels.
     CRITICAL: ignore_index=255 ensures rotated corners don't contribute.

  2. Detection Loss (YOLO-style):
     - Objectness:    BCE with logits, balanced by pos/neg ratio
     - Classification: BCE with logits per class
     - Box Regression: CIoU loss for precise localisation

All losses are returned individually for logging and weighted combination.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# OHEM Cross Entropy (ported from Phase 1 criterion.py)
# ─────────────────────────────────────────────────────────────────

class OhemCrossEntropy(nn.Module):
    """
    Online Hard Example Mining Cross Entropy.
    Focuses the gradient on the pixels the model finds hardest to classify.

    ignore_label (255): Rotated image corners, unlabelled regions — NEVER trained on.
    thres: confidence threshold below which pixels are considered "hard"
    min_kept: minimum number of hard pixels to keep per batch
    """

    def __init__(self, ignore_label=255, thres=0.7, min_kept=100_000, weight=None):
        super().__init__()
        self.thresh    = thres
        self.min_kept  = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        target: (B, H, W) long
        """
        pred = F.softmax(logits, dim=1)
        pixel_losses = self.criterion(logits, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1)[mask].contiguous().sort()

        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]

        if pixel_losses.numel() == 0:
            return pixel_losses.mean() * 0.0

        return pixel_losses.mean()


# ─────────────────────────────────────────────────────────────────
# CIoU Loss for bounding box regression
# ─────────────────────────────────────────────────────────────────

def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Complete IoU loss between predicted and target boxes.

    Both inputs: (N, 4) — [cx, cy, w, h] in pixel space (not normalised).
    Returns: scalar mean CIoU loss.
    """
    eps = 1e-7

    # Convert to xyxy
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    tgt_area  = (tgt_x2  - tgt_x1)  * (tgt_y2  - tgt_y1)
    union_area = pred_area + tgt_area - inter_area + eps

    iou = inter_area / union_area

    # Enclosing box diagonal
    enc_x1 = torch.min(pred_x1, tgt_x1)
    enc_y1 = torch.min(pred_y1, tgt_y1)
    enc_x2 = torch.max(pred_x2, tgt_x2)
    enc_y2 = torch.max(pred_y2, tgt_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Distance between centres
    rho2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
           (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # Aspect ratio penalty
    v = (4 / (math.pi ** 2)) * (
        torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + eps)) -
        torch.atan(pred_boxes[:, 2]  / (pred_boxes[:, 3]  + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return (1 - ciou).mean()


# ─────────────────────────────────────────────────────────────────
# YOLOv3-style Detection Loss (single scale)
# ─────────────────────────────────────────────────────────────────

def detection_loss_single_scale(
    pred: torch.Tensor,         # (B, A*(5+C), H, W) — raw model output
    targets: list,              # list of (N_i, 5) tensors [cls, cx, cy, w, h]
    anchors: list,              # [(aw, ah), ...] A anchor sizes in pixels
    img_h: int,
    img_w: int,
    num_classes: int = 3,
    obj_weight: float = 1.0,
    noobj_weight: float = 0.5,
    box_weight: float = 5.0,
    cls_weight: float = 1.0,
):
    """
    Compute detection loss for one FPN scale.

    Returns: dict with keys 'obj', 'noobj', 'box', 'cls', 'total'
    """
    B, _, grid_h, grid_w = pred.shape
    A = len(anchors)
    device = pred.device

    # Reshape: (B, A, 5+C, H, W)
    pred = pred.view(B, A, 5 + num_classes, grid_h, grid_w)
    pred_xy  = torch.sigmoid(pred[:, :, :2])    # (B, A, 2, H, W) — cx, cy offset
    pred_wh  = pred[:, :, 2:4]                  # (B, A, 2, H, W) — log scale
    pred_obj = pred[:, :, 4:5]                  # (B, A, 1, H, W)
    pred_cls = pred[:, :, 5:]                   # (B, A, C, H, W)

    # Anchor tensors
    anchor_t = torch.tensor(anchors, dtype=torch.float32, device=device)  # (A, 2)

    # Build grid offsets
    grid_y = torch.arange(grid_h, device=device).float().view(1, 1, 1, grid_h, 1)
    grid_x = torch.arange(grid_w, device=device).float().view(1, 1, 1, 1, grid_w)

    stride_h = img_h / grid_h
    stride_w = img_w / grid_w

    # Target assignment
    obj_mask    = torch.zeros(B, A, 1, grid_h, grid_w, device=device)
    noobj_mask  = torch.ones(B, A, 1, grid_h, grid_w, device=device)
    target_xy   = torch.zeros(B, A, 2, grid_h, grid_w, device=device)
    target_wh   = torch.zeros(B, A, 2, grid_h, grid_w, device=device)
    target_cls  = torch.zeros(B, A, num_classes, grid_h, grid_w, device=device)

    for b in range(B):
        if targets[b].numel() == 0:
            continue
        tgt = targets[b].to(device)    # (N, 5)
        t_cls = tgt[:, 0].long()
        t_cx  = tgt[:, 1] * img_w
        t_cy  = tgt[:, 2] * img_h
        t_w   = tgt[:, 3] * img_w
        t_h   = tgt[:, 4] * img_h

        for n in range(len(tgt)):
            # Find best matching anchor by IoU on box size only (centred at origin)
            t_box_wh = torch.stack([t_w[n], t_h[n]]).unsqueeze(0)  # (1, 2)
            ious = torch.min(t_box_wh, anchor_t) / \
                   (torch.max(t_box_wh, anchor_t) + 1e-8)
            ious = ious[:, 0] * ious[:, 1]
            best_anchor = ious.argmax().item()

            gi = int(t_cx[n] / stride_w)
            gj = int(t_cy[n] / stride_h)
            gi = min(max(gi, 0), grid_w - 1)
            gj = min(max(gj, 0), grid_h - 1)

            obj_mask[b, best_anchor, 0, gj, gi]   = 1
            noobj_mask[b, best_anchor, 0, gj, gi] = 0

            # tx, ty: offset within grid cell
            target_xy[b, best_anchor, 0, gj, gi] = (t_cx[n] / stride_w) - gi
            target_xy[b, best_anchor, 1, gj, gi] = (t_cy[n] / stride_h) - gj

            # tw, th: log scale relative to anchor
            aw, ah = anchors[best_anchor]
            target_wh[b, best_anchor, 0, gj, gi] = torch.log(t_w[n] / aw + 1e-8)
            target_wh[b, best_anchor, 1, gj, gi] = torch.log(t_h[n] / ah + 1e-8)

            target_cls[b, best_anchor, t_cls[n], gj, gi] = 1.0

    # Losses
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')

    # Only compute on positive anchors
    obj_count = obj_mask.sum().item()
    if obj_count > 0:
        loss_xy  = mse(pred_xy[obj_mask.bool().expand_as(pred_xy)],
                       target_xy[obj_mask.bool().expand_as(target_xy)])
        loss_wh  = mse(pred_wh[obj_mask.bool().expand_as(pred_wh)],
                       target_wh[obj_mask.bool().expand_as(target_wh)])
        loss_cls = bce(pred_cls[obj_mask.bool().expand_as(pred_cls)],
                       target_cls[obj_mask.bool().expand_as(target_cls)])
    else:
        loss_xy = loss_wh = loss_cls = torch.tensor(0.0, device=device)

    loss_obj   = obj_weight  * bce(pred_obj * obj_mask,   obj_mask)
    loss_noobj = noobj_weight * bce(pred_obj * noobj_mask, torch.zeros_like(pred_obj) * noobj_mask)
    loss_box   = box_weight  * (loss_xy + loss_wh)
    loss_cls   = cls_weight  * loss_cls

    total = loss_obj + loss_noobj + loss_box + loss_cls

    return {'obj': loss_obj, 'noobj': loss_noobj, 'box': loss_box,
            'cls': loss_cls, 'total': total}


# ─────────────────────────────────────────────────────────────────
# Full Multi-Task Loss
# ─────────────────────────────────────────────────────────────────

class AuRoRA2W_IUR_Loss(nn.Module):
    """
    Combined multi-task loss for AuRoRA-2W-IUR.

    Weights (tunable via config):
        w_drive: weight for drivable area OHEM loss
        w_lane:  weight for lane line OHEM loss
        w_det:   weight for detection loss (sum across scales)
    """

    # Default anchor sizes per FPN level (px, for 640×384 input)
    DEFAULT_ANCHORS = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)],
    ]

    def __init__(
        self,
        num_det_classes: int = 3,
        w_drive: float = 1.0,
        w_lane:  float = 1.5,   # Lane lines are harder — upweight
        w_det:   float = 1.0,
        anchors=None,
        img_h: int = 384,
        img_w: int = 640,
    ):
        super().__init__()
        self.w_drive = w_drive
        self.w_lane  = w_lane
        self.w_det   = w_det
        self.num_det_classes = num_det_classes
        self.anchors = anchors or self.DEFAULT_ANCHORS
        self.img_h = img_h
        self.img_w = img_w

        self.drive_criterion = OhemCrossEntropy(ignore_label=255, thres=0.7, min_kept=100_000)
        self.lane_criterion  = OhemCrossEntropy(ignore_label=255, thres=0.7, min_kept=50_000)

    def forward(self, outputs: dict, batch: dict) -> dict:
        """
        outputs: {'det': [p3_pred, p4_pred, p5_pred], 'drive_seg': ..., 'lane_seg': ...}
        batch:   {'drive_mask': ..., 'lane_mask': ..., 'boxes': [...]}
        """
        device = outputs['drive_seg'].device

        # ── Segmentation Losses ──────────────────────────────────
        loss_drive = self.drive_criterion(outputs['drive_seg'], batch['drive_mask'].to(device))
        loss_lane  = self.lane_criterion(outputs['lane_seg'],  batch['lane_mask'].to(device))

        # ── Detection Loss (sum over 3 FPN scales) ───────────────
        det_total = torch.tensor(0.0, device=device)
        det_log = {}
        for scale_idx, (pred, anch) in enumerate(zip(outputs['det'], self.anchors)):
            d = detection_loss_single_scale(
                pred=pred,
                targets=batch['boxes'],
                anchors=anch,
                img_h=self.img_h,
                img_w=self.img_w,
                num_classes=self.num_det_classes,
            )
            det_total += d['total']
            det_log[f'det_p{scale_idx+3}'] = d['total'].item()

        # ── Combined Loss ────────────────────────────────────────
        total = (self.w_drive * loss_drive +
                 self.w_lane  * loss_lane  +
                 self.w_det   * det_total)

        return {
            'total':       total,
            'drive':       loss_drive,
            'lane':        loss_lane,
            'det':         det_total,
            **det_log,
        }
