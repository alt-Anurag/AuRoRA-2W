"""Golden research-metric regression, including score ties and partial coverage."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from aurora2w.config import CLASS_NAMES
from aurora2w.metrics import Metrics


def regression_case():
    rng = np.random.default_rng(9741)
    metric = Metrics()
    metric.confusion["road"] = np.array([[14, 3], [2, 17]], dtype=np.int64)
    metric.confusion["lane"] = np.array([[44, 2], [7, 11]], dtype=np.int64)
    for image in range(6):
        truth, truth_labels, boxes, labels, scores = [], [], [], [], []
        for label in range(len(CLASS_NAMES)):
            x, y = (rng.integers(0, 100, 2) + label * 15).tolist()
            gt = [x, y, x + 20, y + 15]
            if image != 5:
                truth.append(gt)
                truth_labels.append(label)
                if image % 2 == 0:
                    truth.append([x + 8, y, x + 28, y + 15])
                    truth_labels.append(label)
            for delta, score in [(0, .8), (0, .8), (3, .3), (8, .6), (20, .1)]:
                boxes.append([x + delta, y, x + 20 + delta, y + 15])
                labels.append(label)
                scores.append(score)
        known = np.ones(len(CLASS_NAMES), dtype=bool)
        if image in (2, 4):
            known[::2] = False
        metric.samples.append((
            {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "scores": torch.tensor(scores), "labels": torch.tensor(labels)},
            {"boxes": torch.tensor(truth, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.tensor(truth_labels, dtype=torch.long),
             "known_classes": torch.from_numpy(known)},
        ))
    return metric


def test_metric_matches_frozen_preoptimization_results():
    expected = json.loads((Path(__file__).parent / "fixtures" / "metrics_before_iou_cache.json").read_text())
    assert regression_case().compute() == expected


def test_duplicate_detection_and_unknown_image_do_not_change_recall_scope():
    metric = Metrics()
    known = torch.zeros(len(CLASS_NAMES), dtype=torch.bool)
    known[-1] = True
    prediction = {"boxes": torch.tensor([[0., 0, 10, 10], [0., 0, 10, 10]]),
                  "scores": torch.tensor([.9, .9]), "labels": torch.tensor([8, 8])}
    target = {"boxes": torch.tensor([[0., 0, 10, 10]]), "labels": torch.tensor([8]),
              "known_classes": known}
    metric.samples.append((prediction, target))
    metric.samples.append((prediction, {**target, "known_classes": torch.zeros_like(known)}))
    result = metric.compute()["detection"]["pothole"]
    assert result["gt"] == 1 and result["labeled_images"] == 1
    assert result["ap50_95"] == 1
    assert result["precision_at_score_0_3_iou_0_5"] == .5
    assert result["recall_at_score_0_3_iou_0_5"] == 1
    assert result["false_positives_at_score_0_3_iou_0_5"] == 1
