"""Prepare an honest image-level RDD validation proposal without training.

Grouping uses 63-bit pHash <=6 AND 64-bit horizontal dHash <=8, joined by
connected components. These are perceptual groups, never inferred ride identities.
The original all-training manifest and all raw images/annotations are preserved.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def image_hashes(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].reshape(-1)[1:]
    phash = sum(int(bit) << i for i, bit in enumerate(low > np.median(low)))
    small = cv2.resize(gray, (9, 8))
    dhash = sum(int(bit) << i for i, bit in enumerate((small[:, 1:] > small[:, :-1]).reshape(-1)))
    pixels = hashlib.sha256(image.shape[0].to_bytes(4, "little") + image.shape[1].to_bytes(4, "little") + image.tobytes()).hexdigest()
    return phash, dhash, pixels


def components(features, phash_limit=6, dhash_limit=8):
    parent = list(range(len(features)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    pvalues = np.asarray([row[0] for row in features], dtype=np.uint64)
    lookup = np.asarray([number.bit_count() for number in range(256)], dtype=np.uint8)
    edges = []
    exact = {}
    for i, (_, _, pixels) in enumerate(features):
        if pixels in exact:
            union(i, exact[pixels])
        else:
            exact[pixels] = i
        difference = np.bitwise_xor(pvalues[i + 1:], pvalues[i]).view(np.uint8).reshape(-1, 8)
        distances = lookup[difference].sum(axis=1)
        for relative in np.flatnonzero(distances <= phash_limit):
            j = i + 1 + int(relative)
            d_distance = (features[i][1] ^ features[j][1]).bit_count()
            if d_distance <= dhash_limit:
                union(i, j)
                edges.append((i, j, int(distances[relative]), d_distance))
    groups = defaultdict(list)
    for i in range(len(features)):
        groups[find(i)].append(i)
    return list(groups.values()), edges


def assign_groups(groups, records, val_fraction, seed):
    rng = np.random.default_rng(seed)
    validation = set()
    for positive in (False, True):
        selected = [group for group in groups if any(records[i]["boxes"] for i in group) == positive]
        if len(selected) < 2:
            raise ValueError("Too few perceptual groups for both split strata; inspect grouping.")
        order = rng.permutation(len(selected))
        count = max(1, min(len(selected) - 1, math.ceil(len(selected) * val_fraction)))
        for pos in order[:count]:
            validation.update(selected[int(pos)])
    return validation


def run(args):
    source = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh partition output directory.")
    if not 0 < args.val_fraction < .5:
        raise ValueError("Validation fraction must be between zero and one half.")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records or any(row["source"] != "rdd_india" or row["split"] != "train" for row in records):
        raise ValueError("Input must be the untouched all-training RDD India manifest.")
    records.sort(key=lambda row: row["id"])
    paths = [(source.parent / row["image"]).resolve() for row in records]
    with ThreadPoolExecutor(max_workers=4) as pool:
        features = list(pool.map(image_hashes, paths))
    groups, edges = components(features)
    validation = assign_groups(groups, records, args.val_fraction, args.seed)
    membership = {}
    for group in groups:
        identifier = hashlib.sha256("\n".join(records[i]["id"] for i in group).encode()).hexdigest()[:16]
        for i in group:
            membership[i] = "perceptual_" + identifier
    if any((a in validation) != (b in validation) for a, b, _, _ in edges):
        raise AssertionError("A detected similarity edge crosses partitions.")
    output.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, row in enumerate(records):
        item = dict(row)
        item["image"] = Path(os.path.relpath(paths[i], output)).as_posix()
        item["original_sequence"] = row["sequence"]
        item["sequence"] = membership[i]
        item["split"] = "val" if i in validation else "train"
        item["partition_group_kind"] = "perceptual_hash_component_not_a_ride"
        item["partition_protocol"] = "rdd_image_holdout_v1"
        lines.append(item)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    with (output / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "sequence", "split"])
        writer.writeheader()
        for i, row in enumerate(lines):
            writer.writerow({"image": "train/images/" + paths[i].name, "sequence": row["sequence"], "split": row["split"]})
    (output / "hashes.jsonl").write_text("".join(json.dumps({"id": row["id"], "phash63": f"{p:016x}", "dhash64": f"{d:016x}", "decoded_sha256": pixels}) + "\n"
                                              for row, (p, d, pixels) in zip(records, features)), encoding="utf-8")
    summaries = {}
    for split in ("train", "val"):
        rows = [row for row in lines if row["split"] == split]
        summaries[split] = {"images": len(rows), "positive_images": sum(bool(row["boxes"]) for row in rows),
                            "negative_images": sum(not row["boxes"] for row in rows),
                            "pothole_boxes": sum(len(row["boxes"]) for row in rows),
                            "perceptual_groups": len({row["sequence"] for row in rows})}
    report = {"protocol": "rdd_image_holdout_v1", "status": "Prepared proposal for researcher review; no training started",
              "scope": "Image-level photographed-data validation. NOT ride/location-disjoint or official challenge-test results.",
              "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "seed": args.seed,
              "requested_validation_group_fraction": args.val_fraction, "phash63_hamming_threshold": 6,
              "dhash64_hamming_threshold": 8, "near_duplicate_rule": "Both thresholds; transitive connected components",
              "group_count": len(groups), "largest_group_images": max(map(len, groups)),
              "group_size_histogram": dict(Counter(map(len, groups))), "similarity_edges": len(edges),
              "detected_similarity_edges_crossing_splits": 0,
              "splits": summaries,
              "limitations": ["Perceptual hashes miss some viewpoint/crop/temporal similarities.",
                              "Distinct groups can still belong to the same unverified recording or road.",
                              "Split stratification uses presence of pothole labels, never model predictions.",
                              "Do not tune the partition after seeing validation performance.",
                              "The published unlabeled RDD test set remains untouched."]}
    (output / "PARTITION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/phase2/prepared/rdd_india_all_train/manifest.jsonl")
    parser.add_argument("--output", default="data/phase2/partitions/rdd_image_holdout_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.2)
    run(parser.parse_args())
