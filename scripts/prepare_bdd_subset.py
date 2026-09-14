"""Freeze a representative BDD legacy subset and prepare its real labels.

Full metadata candidate indexes and original raw annotations remain available.
This script performs no model construction, training, optimization or prediction.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random

from aurora2w.native import BDD_CLASSES, build_native, normalize_legacy_bdd_frame

PAINTED_LANES = {"single white", "single yellow", "double white", "double yellow", "single other", "double other"}


def candidate(path):
    payload = path.read_bytes()
    document = json.loads(payload)
    frame = normalize_legacy_bdd_frame(document, path.name)
    image = path.with_suffix(".jpg")
    if not image.is_file():
        raise FileNotFoundError(image)
    if frame["labels"] is None:
        raise ValueError(f"Unannotated BDD frame needs explicit exclusion review: {path}")
    classes = Counter(BDD_CLASSES[obj["category"]] for obj in frame["labels"]
                      if obj.get("box2d") and obj.get("category") in BDD_CLASSES
                      and not (obj.get("attributes") or {}).get("crowd", False)
                      and not (obj.get("attributes") or {}).get("ignored", False))
    attrs = frame.get("attributes") or {}
    return {"id": image.name, "source": "bdd", "sequence": frame["videoName"], "split": path.parent.name,
            "image": f"{path.parent.name}/{image.name}", "annotation": f"{path.parent.name}/{path.name}",
            "annotation_sha256": hashlib.sha256(payload).hexdigest(),
            "attributes": {name: attrs.get(name, "undefined") for name in ("weather", "timeofday", "scene")},
            "instances": dict(classes),
            "road_polygon_count": sum(obj["category"] == "drivable area" and obj["attributes"].get("areaType") in ("direct", "alternative") for obj in frame["labels"]),
            "unknown_road_polygon_count": sum(obj["category"] == "drivable area" and obj["attributes"].get("areaType") == "unknown" for obj in frame["labels"]),
            "painted_lane_count": sum(obj["category"] == "lane" and obj["attributes"].get("laneType") in PAINTED_LANES for obj in frame["labels"]),
            "group_kind": "published_videoName_or_keyframe_stem_not_verified_ride"}


def select_groups(rows, target_images, seed, stratum_key=None):
    """Proportional weather/time/scene strata; never split an observed group."""
    if target_images < 1:
        raise ValueError("Target images must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence"]].append(row)
    strata = defaultdict(list)
    for name, group in sorted(groups.items()):
        # Majority source metadata is deterministic; full group remains intact.
        values = Counter(tuple(row.get("attributes", {}).get(key, "undefined") for key in ("weather", "timeofday", "scene"))
                         if stratum_key is None else str(row.get(stratum_key, "undefined")) for row in group)
        stratum = sorted(values, key=lambda item: (-values[item], item))[0]
        strata[stratum].append((name, group))
    wanted = min(target_images, len(rows))
    quota = {key: wanted * sum(len(group) for _, group in value) / len(rows) for key, value in strata.items()}
    integer = {key: math.floor(value) for key, value in quota.items()}
    for key in sorted(quota, key=lambda key: (-(quota[key] - integer[key]), key))[:wanted - sum(integer.values())]:
        integer[key] += 1
    rng = random.Random(seed)
    selected = []
    for key, values in sorted(strata.items()):
        rng.shuffle(values)
        kept = 0
        for _, group in values:
            if kept >= integer[key]:
                break
            selected.extend(group)
            kept += len(group)
    return sorted(selected, key=lambda row: row["id"])


def distribution(rows):
    classes = Counter(name for row in rows for name in row["instances"])
    boxes = Counter()
    for row in rows:
        boxes.update(row["instances"])
    return {"images": len(rows), "groups": len({row["sequence"] for row in rows}),
            "class_positive_images": dict(classes), "instances": dict(boxes),
            "road_positive_images": sum(row["road_polygon_count"] > 0 for row in rows),
            "painted_lane_positive_images": sum(row["painted_lane_count"] > 0 for row in rows),
            "attributes": {key: dict(Counter(row["attributes"].get(key, "undefined") for row in rows)) for key in ("weather", "timeofday", "scene")}}


def run(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh BDD subset directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Frozen annotated-image subset for review; no training started", "training_approved": False,
              "source_release": "author_legacy_2018", "seed": args.seed,
              "selection": "Weather/timeofday/scene proportional largest-remainder quotas; seeded shuffle within strata; complete observed groups; no model predictions",
              "group_limit": "Keyframe/video identities do not establish disjoint riders or locations",
              "splits": {}}
    seen_groups = set()
    for split, expected, target in (("train", 70000, args.train_images), ("val", 10000, args.val_images)):
        paths = sorted((root / split).glob("*.json"))
        if len(paths) != expected:
            raise ValueError(f"Expected {expected} legacy {split} annotations, found {len(paths)}")
        rows = []
        path = output / f"candidates_{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(candidate, paths):
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                if len(rows) % 5000 == 0:
                    print(json.dumps({"stage": "metadata", "split": split, "complete": len(rows), "total": expected}), flush=True)
        groups = {row["sequence"] for row in rows}
        if groups & seen_groups:
            raise ValueError("BDD observed groups overlap published train/val splits")
        seen_groups.update(groups)
        selected = select_groups(rows, target, args.seed + (split == "val"))
        for name in sorted(set(BDD_CLASSES.values())):
            if not any(name in row["instances"] for row in selected):
                raise ValueError(f"Selected {split} lacks {name}; review target size before preparing labels")
        selected_path = output / f"selected_{split}.jsonl"
        selected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
        # Canonical frame-list is generated only for retained labels; raw JSONs remain immutable.
        frames = [normalize_legacy_bdd_frame(json.loads((root / row["annotation"]).read_bytes())) for row in selected]
        normalized = output / f"normalized_{split}.json"
        normalized.write_text(json.dumps(frames), encoding="utf-8")
        built = build_native("bdd", root / split, output / f"index_{split}", labels=normalized,
                             split=split, tasks=["road", "lane", "detection"], lane_width=args.lane_width)
        source_path = Path(built["spec"])
        sources = json.loads(source_path.read_text(encoding="utf-8"))
        sources["sources"][0].update(source_release="author_legacy_2018",
                                    selection_protocol="bdd_weather_time_scene_v1",
                                    selection_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest())
        sources["sources"][0]["annotation_contract"].update(
            source_release="author_legacy_2018", selected_ids_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            selection_seed=args.seed + (split == "val"), original_label_layout="one legacy JSON per keyframe; publisher path semantics")
        source_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
        report["splits"][split] = {"effective_seed": args.seed + (split == "val"), "full": distribution(rows), "selected": distribution(selected),
                                   "candidate_index_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "selected_index_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(), "spec": built["spec"]}
        (output / "SELECTION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"stage": "prepared", "split": split, "samples": len(selected), "spec": built["spec"]}), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/phase2/raw/bdd_download/100k")
    parser.add_argument("--output", default="data/phase2/partitions/bdd_subset_v1")
    parser.add_argument("--train-images", type=int, default=6000)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lane-width", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)
