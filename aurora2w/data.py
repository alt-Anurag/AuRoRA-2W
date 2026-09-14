"""Partial-label JSONL dataset: missing annotation is UNKNOWN, never background."""

import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLASS_NAMES
from .geometry import letterbox, normalize_image, roll_matrix, pixel_center_matrix, transform_boxes, warp_mask


def read_manifest(path):
    path = Path(path).resolve()
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["_manifest"] = str(path)
        for key in ("image", "road_mask", "lane_mask"):
            if record.get(key):
                record[key] = str((path.parent / record[key]).resolve())
        for key in ("id", "source", "sequence", "split", "image"):
            if not record.get(key):
                raise ValueError(f"{path}:{line_no}: missing {key}")
        if record["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid split for {record['id']}")
        known = record.get("detection_classes", [])
        if len(set(known)) != len(known) or not set(known).issubset(CLASS_NAMES):
            raise ValueError(f"Invalid exhaustive detection_classes for {record['id']}")
        for obj in record.get("boxes", []):
            if obj["class"] not in CLASS_NAMES:
                raise ValueError(f"Unknown class: {obj['class']}")
        imu = record.get("imu", {})
        values = [imu.get("roll_rad", 0), imu.get("confidence", 0)]
        if not np.isfinite(values).all() or not 0 <= values[1] <= 1:
            raise ValueError(f"Invalid camera IMU for {record['id']}")
        records.append(record)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def audit_records(records, check_hashes=False, image_size=None):
    """Fail on split leakage, malformed boxes/masks, and absent supervision."""
    ids, paths, groups, hashes = set(), {}, {}, {}
    report = {"samples": len(records), "sources": {}, "tasks": {"road": 0, "lane": 0, "detection": 0}}
    report["instances"] = {c: 0 for c in CLASS_NAMES}
    report["exhaustively_labeled_images"] = {c: 0 for c in CLASS_NAMES}
    report["trusted_imu_images"] = 0
    report["splits"] = {}
    report["positive_only_images"] = {c: 0 for c in CLASS_NAMES}
    report["small_box_assignment"] = {"input_size": image_size, "unassigned_instances": {}, "examples": []}
    for r in records:
        key = (r["source"], r["id"])
        if r["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid split in {key}")
        known = r.get("detection_classes", [])
        if len(known) != len(set(known)) or not set(known).issubset(CLASS_NAMES):
            raise ValueError(f"Invalid exhaustive detection_classes in {key}")
        if r.get("data_kind", "unspecified") not in ("real", "synthetic", "unspecified"):
            raise ValueError(f"Invalid data_kind in {key}")
        if key in ids:
            raise ValueError(f"Duplicate record: {key}")
        ids.add(key)
        group = (r["source"], r["sequence"])
        if group in groups and groups[group] != r["split"]:
            raise ValueError(f"Sequence crosses splits: {group}")
        groups[group] = r["split"]
        if r["image"] in paths:
            raise ValueError(f"Image appears more than once: {r['image']}")
        paths[r["image"]] = r["split"]
        image = cv2.imread(r["image"])
        if image is None:
            raise ValueError(f"Unreadable image: {r['image']}")
        h, w = image.shape[:2]
        supervised = bool(r.get("detection_classes") or r.get("boxes"))
        if supervised:
            report["tasks"]["detection"] += 1
        for obj in r.get("boxes", []):
            if obj["class"] not in CLASS_NAMES:
                raise ValueError(f"Unknown class in {key}")
            report["instances"][obj["class"]] += 1
        for name in set(o["class"] for o in r.get("boxes", [])) - set(known):
            report["positive_only_images"][name] += 1
        for name in r.get("detection_classes", []):
            report["exhaustively_labeled_images"][name] += 1
        report["trusted_imu_images"] += r.get("imu", {}).get("confidence", 0) > 0
        for box in [o["xyxy"] for o in r.get("boxes", [])] + r.get("ignore_boxes", []):
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError(f"Invalid box in {key}")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"Out-of-bounds/empty box in {key}: {box}")
        for task in ("road", "lane"):
            if not r.get(task + "_mask"):
                continue
            mask = cv2.imread(r[task + "_mask"], cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (h, w) or not np.all((mask == 0) | (mask == 1) | (mask == 255)):
                raise ValueError(f"{key}: {task} mask must be HxW with values 0,1,255")
            has_labels = bool((mask != 255).any())
            report["tasks"][task] += int(has_labels)
            supervised |= has_labels
        if not supervised:
            raise ValueError(f"No supervised task in {key}")
        if check_hashes:
            # Hash decoded pixels, catching identical images with different encodings.
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image pixels: {key} and {hashes[digest]}")
            hashes[digest] = key
        report["sources"][r["source"]] = report["sources"].get(r["source"], 0) + 1
        split_key = r["source"] + "/" + r["split"]
        report["splits"][split_key] = report["splits"].get(split_key, 0) + 1
        if image_size and r.get("boxes"):
            from .losses import assign_targets
            _, valid_pixels, matrix = letterbox(image, image_size)
            boxes, keep = transform_boxes([o["xyxy"] for o in r["boxes"]], matrix, image_size[1], image_size[0])
            points, strides = [], []
            for stride in (8, 16, 32):
                y, x = torch.meshgrid(torch.arange(image_size[0] // stride), torch.arange(image_size[1] // stride), indexing="ij")
                points.append(torch.stack((x + .5, y + .5), -1).reshape(-1, 2) * stride)
                strides.append(torch.full((x.numel(),), stride))
            target = {"known_classes": torch.tensor([c in known for c in CLASS_NAMES]),
                      "boxes": torch.from_numpy(boxes[keep]),
                      "labels": torch.tensor([CLASS_NAMES.index(o["class"]) for o, k in zip(r["boxes"], keep) if k]),
                      "valid_pixels": torch.from_numpy(valid_pixels.astype(bool)), "ignore_boxes": torch.empty((0, 4))}
            _, _, matched = assign_targets(torch.cat(points), torch.cat(strides), target, len(CLASS_NAMES))
            assigned = set(matched[matched >= 0].tolist())
            for i, label in enumerate(target["labels"]):
                if i not in assigned:
                    name = CLASS_NAMES[label]
                    count = report["small_box_assignment"]["unassigned_instances"]
                    count[name] = count.get(name, 0) + 1
                    if len(report["small_box_assignment"]["examples"]) < 20:
                        report["small_box_assignment"]["examples"].append({"id": r["id"], "source": r["source"], "class": name,
                                                                          "resized_xyxy": target["boxes"][i].tolist()})
    report["warning"] = "Pixel duplicate checks do not detect near duplicates. Small boxes without stride-8 sampling centers need higher input resolution or a separately evaluated detection design; they are not silently enlarged."
    return report


class RoadDataset(Dataset):
    def __init__(self, records, config, training=False):
        self.records, self.config, self.training = records, config, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r, cfg = self.records[index], self.config
        original = cv2.imread(r["image"])
        if original is None:
            raise ValueError(f"Unreadable image: {r['image']}")
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        image, valid, matrix = letterbox(original, cfg["image_size"])
        h, w = image.shape[:2]
        masks = {}
        for task in ("road", "lane"):
            raw = cv2.imread(r[task + "_mask"], cv2.IMREAD_UNCHANGED) if r.get(task + "_mask") else None
            masks[task] = warp_mask(raw, matrix, (h, w)) if raw is not None else np.full((h, w), 255, np.uint8)
        imu = r.get("imu", {})
        roll, confidence = float(imu.get("roll_rad", 0)), float(imu.get("confidence", 0))
        if r.get("upright_reference", False) and confidence == 0:
            roll, confidence = 0., 1.  # Explicit dataset curation assumption, not a measured IMU.
        if self.training:
            angle = math.radians(random.uniform(-cfg["max_roll_degrees"], cfg["max_roll_degrees"]))
            rotation = roll_matrix(w, h, angle)
            image = cv2.warpPerspective(image, pixel_center_matrix(rotation), (w, h), borderValue=(114, 114, 114))
            valid = warp_mask(valid, rotation, (h, w), fill=0)
            masks = {k: warp_mask(v, rotation, (h, w)) for k, v in masks.items()}
            matrix = rotation @ matrix
            roll += angle
            if confidence > 0:
                roll += math.radians(random.gauss(0, cfg["sensor_noise_degrees"]))
            if random.random() < cfg["sensor_dropout"]:
                confidence = 0.
        boxes, keep = transform_boxes([o["xyxy"] for o in r.get("boxes", [])], matrix, w, h)
        labels = np.array([CLASS_NAMES.index(o["class"]) for o in r.get("boxes", [])], dtype=np.int64)
        ignores, ik = transform_boxes(r.get("ignore_boxes", []), matrix, w, h)
        for mask in masks.values():
            mask[valid == 0] = 255
        target = {"boxes": torch.from_numpy(boxes[keep]), "labels": torch.from_numpy(labels[keep]),
                  "ignore_boxes": torch.from_numpy(ignores[ik]),
                  "known_classes": torch.tensor([c in r.get("detection_classes", []) for c in CLASS_NAMES]),
                  "road": torch.from_numpy(masks["road"].astype(np.int64)),
                  "lane": torch.from_numpy(masks["lane"].astype(np.int64)),
                  "valid_pixels": torch.from_numpy(valid.astype(bool)),
                  "id": r["id"], "source": r["source"],
                  "roll_rad": roll, "imu_confidence": confidence}
        return torch.from_numpy(normalize_image(image)), torch.tensor([roll, confidence], dtype=torch.float32), target


def collate(batch):
    images, imu, targets = zip(*batch)
    return torch.stack(images), torch.stack(imu), list(targets)
