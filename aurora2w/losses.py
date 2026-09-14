"""FCOS-style detection and independent binary segmentation with missing-label masks."""

import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from .model import flatten_detection, distance_to_boxes


def binary_segmentation_loss(logits, target):
    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0
    truth = target.clamp(0, 1).to(logits)
    bce = F.binary_cross_entropy_with_logits(logits[valid], truth[valid])
    probabilities = logits.sigmoid()[valid]
    truth = truth[valid]
    dice = 1 - (2 * (probabilities * truth).sum() + 1) / (probabilities.sum() + truth.sum() + 1)
    return bce + dice


def assign_targets(points, strides, target, num_classes):
    """Smallest eligible box; center sampling and level ranges in resized pixels."""
    n = len(points)
    classes = points.new_zeros((n, num_classes))
    valid = target["known_classes"].to(points.device)[None].expand(n, -1).clone()
    boxes = target["boxes"].to(points)
    labels = target["labels"].to(points.device)
    assigned = torch.full((n,), -1, device=points.device, dtype=torch.long)
    if len(boxes):
        ltrb = torch.cat((points[:, None] - boxes[None, :, :2], boxes[None, :, 2:] - points[:, None]), dim=-1)
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        near_center = (points[:, None] - center).abs().amax(-1) <= strides[:, None] * 1.5
        max_distance = ltrb.amax(-1)
        low = torch.where(strides == 8, 0, torch.where(strides == 16, 64, 128))
        high = torch.where(strides == 8, 64, torch.where(strides == 16, 128, float("inf")))
        eligible = (ltrb.amin(-1) > 0) & near_center & (max_distance >= low[:, None]) & (max_distance < high[:, None])
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[None].expand(n, -1).clone()
        area[~eligible] = float("inf")
        best_area, best = area.min(1)
        positive = best_area.isfinite()
        assigned[positive] = best[positive]
        classes[positive, labels[best[positive]]] = 1
        # Incompletely annotated classes can supply positives, never implicit negatives.
        valid[positive, labels[best[positive]]] = True
    xy = points.long()
    pixel_valid = target["valid_pixels"].to(points.device)[xy[:, 1], xy[:, 0]]
    ignores = target["ignore_boxes"].to(points)
    if len(ignores):
        inside_ignore = ((points[:, None] >= ignores[None, :, :2]) & (points[:, None] < ignores[None, :, 2:])).all(-1).any(-1)
        pixel_valid &= ~inside_ignore
    valid &= pixel_valid[:, None]
    assigned[~pixel_valid] = -1
    return classes, valid, assigned


def multitask_loss(outputs, targets, weights):
    cls, distances, center, points, strides = flatten_detection(outputs["detection"])
    class_loss, box_loss, center_loss = cls.sum() * 0, distances.sum() * 0, center.sum() * 0
    num_positive = 0
    for b, target in enumerate(targets):
        truth, valid, matched = assign_targets(points, strides, target, cls.shape[-1])
        positive = matched >= 0
        num_positive += int(positive.sum())
        class_loss += (sigmoid_focal_loss(cls[b], truth, alpha=.25, gamma=2, reduction="none") * valid).sum()
        if positive.any():
            gt = target["boxes"].to(points)[matched[positive]]
            target_distance = torch.cat((points[positive] - gt[:, :2], gt[:, 2:] - points[positive]), -1)
            lr, tb = target_distance[:, [0, 2]], target_distance[:, [1, 3]]
            quality = ((lr.amin(-1) / lr.amax(-1).clamp_min(1e-6)) *
                       (tb.amin(-1) / tb.amax(-1).clamp_min(1e-6))).sqrt()
            pred = distance_to_boxes(points[positive], distances[b, positive])
            box_loss += generalized_box_iou_loss(pred, gt, reduction="sum")
            center_loss += F.binary_cross_entropy_with_logits(center[b, positive], quality, reduction="sum")
    scale = max(num_positive, 1)
    losses = {"classification": class_loss / scale, "box": box_loss / scale, "centerness": center_loss / scale}
    for task in ("road", "lane"):
        truth = torch.stack([t[task] for t in targets]).to(cls.device)
        losses[task] = binary_segmentation_loss(outputs[task + "_logits"], truth)
    losses["total"] = sum(losses[k] * weights[k] for k in weights)
    return losses
