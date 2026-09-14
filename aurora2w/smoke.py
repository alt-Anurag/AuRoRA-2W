"""Generate labeled toy scenes and exercise training; NEVER a road-quality benchmark."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import CLASS_NAMES, load_config
from .data import read_manifest
from .engine import train


def make_fixture(destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(8):
        h, w = 128, 192
        rng = np.random.default_rng(i)
        image = rng.integers(100, 160, (h, w, 3), dtype=np.uint8)
        road, lane = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        cv2.fillPoly(road, [np.array([[80, 40], [112, 40], [192, 128], [0, 128]])], 1)
        image[road == 1] = [70, 70, 70]
        cv2.line(lane, (85, 50), (40, 127), 1, 2)
        cv2.line(lane, (108, 50), (155, 127), 1, 2)
        image[lane == 1] = [240, 240, 240]
        x = 74 + i * 2
        cv2.rectangle(image, (x, 52), (x+24, 75), (40, 40, 210), -1)
        cv2.ellipse(image, (90+i, 105), (12, 5), 0, 0, 360, (15, 15, 15), -1)
        for name, array in ((f"image_{i}", image), (f"road_{i}", road), (f"lane_{i}", lane)):
            if not cv2.imwrite(str(root / f"{name}.png"), array):
                raise OSError(name)
        split = "train" if i < 4 else "val" if i < 6 else "test"
        records.append({"id": str(i), "source": "synthetic_test_only", "sequence": f"scene_{i}", "split": split,
                        "image": str(root / f"image_{i}.png"), "road_mask": str(root / f"road_{i}.png"),
                        "lane_mask": str(root / f"lane_{i}.png"), "detection_classes": list(CLASS_NAMES),
                        "upright_reference": True,
                        "boxes": [{"class": "car", "xyxy": [x, 52, x+25, 76]},
                                  {"class": "pothole", "xyxy": [78+i, 100, 103+i, 111]}]})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    writer = cv2.VideoWriter(str(root / "input.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (192, 128))
    if not writer.isOpened():
        raise OSError("Cannot create fixture video")
    for r in records:
        writer.write(cv2.imread(r["image"]))
    writer.release()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/phase2_smoke")
    parser.add_argument("--alignment", choices=["none", "idfa"], default="none")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("Use a fresh smoke output directory")
    manifest = make_fixture(root / "data")
    config = load_config("configs/phase2.json")
    config.update(image_size=[128, 192], channels=32, neck_repeats=1, pretrained_backbone=False,
                  alignment=args.alignment, epochs=1, max_roll_degrees=20)
    path = root / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    import torch
    torch.set_num_threads(2)
    train(config, read_manifest(manifest), root / "run", max_steps=2)


if __name__ == "__main__":
    main()
