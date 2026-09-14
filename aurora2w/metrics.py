"""Transparent metrics for partial labels; use official evaluators for paper tables."""

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import batched_nms, box_iou

from .config import CLASS_NAMES
from .model import flatten_detection, distance_to_boxes


@torch.no_grad()
def decode(outputs, image_size, threshold=.05, max_detections=100):
    cls, distances, centers, points, _ = flatten_detection(outputs["detection"])
    score = (cls.sigmoid() * centers.sigmoid()[..., None]).sqrt()
    all_boxes = distance_to_boxes(points, distances)
    h, w = image_size
    results = []
    for boxes, scores in zip(all_boxes, score):
        locations, labels = torch.where(scores > threshold)
        values = scores[locations, labels]
        # Bound postprocessing latency before NMS.
        order = values.argsort(descending=True)[:1000]
        boxes, labels, values = boxes[locations[order]].clone(), labels[order], values[order]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        nonempty = (boxes[:, 2:] > boxes[:, :2]).all(-1)
        boxes, labels, values = boxes[nonempty], labels[nonempty], values[nonempty]
        keep = batched_nms(boxes, values, labels, .5)[:max_detections]
        results.append({"boxes": boxes[keep].cpu(), "scores": values[keep].cpu(), "labels": labels[keep].cpu()})
    return results


def interpolated_ap(recall, precision):
    return float(np.mean([precision[recall >= r].max(initial=0) for r in np.linspace(0, 1, 101)]))


class Metrics:
    def __init__(self):
        self.confusion = {k: np.zeros((2, 2), dtype=np.int64) for k in ("road", "lane")}
        self.samples = []

    def update(self, output, predictions, targets):
        for task in self.confusion:
            shape = targets[0][task].shape
            labels = (F.interpolate(output[task + "_logits"], size=shape, mode="bilinear", align_corners=False).sigmoid()[:, 0] >= .5).cpu()
            for prediction, target in zip(labels, targets):
                truth = target[task].cpu()
                valid = truth != 255
                encoded = truth[valid] * 2 + prediction[valid].long()
                self.confusion[task] += torch.bincount(encoded, minlength=4).reshape(2, 2).numpy()
        # Retain only sparse detection annotations, not every full-resolution mask.
        for pred, target in zip(predictions, targets):
            xy = (pred["boxes"][:, :2] + pred["boxes"][:, 2:]) / 2
            coords = xy.long()
            valid = target["valid_pixels"]
            keep = valid[coords[:, 1].clamp(0, valid.shape[0]-1), coords[:, 0].clamp(0, valid.shape[1]-1)]
            ignore = target["ignore_boxes"]
            if len(ignore):
                keep &= ~((xy[:, None] >= ignore[None, :, :2]) & (xy[:, None] < ignore[None, :, 2:])).all(-1).any(-1)
            compact_pred = {k: v[keep] for k, v in pred.items()}
            compact_target = {k: target[k] for k in ("boxes", "labels", "known_classes")}
            self.samples.append((compact_pred, compact_target))

    def compute(self):
        result = {"images": len(self.samples), "segmentation": {}, "detection": {}}
        for task, c in self.confusion.items():
            tp, fp, fn = int(c[1, 1]), int(c[0, 1]), int(c[1, 0])
            result["segmentation"][task] = {"foreground_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                                              "f1": 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else None,
                                              "valid_pixels": int(c.sum())}
        mean_aps = []
        for label, name in enumerate(CLASS_NAMES):
            samples = []
            for pred, target in self.samples:
                if not bool(target["known_classes"][label]):
                    continue  # AP needs exhaustive annotations for this class.
                select = pred["labels"] == label
                boxes, scores = pred["boxes"][select], pred["scores"][select]
                truth = target["boxes"][target["labels"] == label]
                samples.append((boxes, scores, truth))
            count = sum(len(s[2]) for s in samples)
            # IoU is independent of the matching threshold. Compute each small
            # per-image/class matrix once, preserving torchvision float32 math.
            # Copies below prevent one threshold's matches mutating another's.
            overlaps = [box_iou(boxes, gt).numpy() for boxes, _, gt in samples]
            hits, predicted = 0, 0
            for image_index, (_, scores, gt) in enumerate(samples):
                used = np.zeros(len(gt), dtype=bool)
                for j in scores.argsort(descending=True).tolist():
                    if float(scores[j]) < .3:
                        continue
                    predicted += 1
                    overlap = overlaps[image_index][j].copy()
                    overlap[used] = -1
                    if len(overlap) and float(overlap.max()) >= .5:
                        hits += 1
                        used[int(overlap.argmax())] = True
            operating = {"precision_at_score_0_3_iou_0_5": hits / predicted if predicted else None,
                         "recall_at_score_0_3_iou_0_5": hits / count if count else None,
                         "false_positives_at_score_0_3_iou_0_5": predicted - hits}
            if not count:
                result["detection"][name] = {"ap50": None, "ap50_95": None, "gt": 0, "labeled_images": len(samples), **operating}
                continue
            aps = []
            # Keep the existing reverse tuple ordering, including score ties.
            detections = sorted([(float(score), i, j) for i, (_, scores, _) in enumerate(samples)
                                 for j, score in enumerate(scores)], reverse=True)
            for threshold in np.linspace(.5, .95, 10):
                used = [np.zeros(len(gt), dtype=bool) for _, _, gt in samples]
                hits = []
                for _, i, j in detections:
                    overlap = overlaps[i][j].copy()
                    overlap[used[i]] = -1
                    if len(overlap) and float(overlap.max()) >= threshold:
                        used[i][int(overlap.argmax())] = True
                        hits.append(1)
                    else:
                        hits.append(0)
                tp = np.cumsum(hits)
                recall = tp / count
                precision = tp / np.arange(1, len(tp) + 1)
                aps.append(interpolated_ap(recall, precision))
            mean_aps.append(aps)
            result["detection"][name] = {"ap50": aps[0], "ap50_95": float(np.mean(aps)), "gt": count, "labeled_images": len(samples), **operating}
        result["map50"] = float(np.mean([a[0] for a in mean_aps])) if mean_aps else None
        result["map50_95"] = float(np.mean(mean_aps)) if mean_aps else None
        result["protocol"] = "101-point AP, max 100 detections/image, class-exhaustive images only; not official COCO evaluation"
        return result
