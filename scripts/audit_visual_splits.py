"""Find perceptually similar photographs across prepared splits for human review.

The pHash/dHash rule is a candidate detector, not proof of a shared recording.
It can also screen same-split photographs across sources for reencoded copies.
It never changes dataset membership or annotations and does not train a model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scripts.prepare_rdd_holdout import image_hashes


def cross_split_pairs(records, features, phash_limit=6, dhash_limit=8):
    lookup = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)
    phashes = np.asarray([f[0] for f in features], dtype=np.uint64)
    splits = sorted({r["split"] for r in records})
    for split_index, left in enumerate(splits):
        left_indices = [i for i, row in enumerate(records) if row["split"] == left]
        for right in splits[split_index + 1:]:
            right_indices = np.asarray([i for i, row in enumerate(records) if row["split"] == right], dtype=np.int64)
            right_hashes = phashes[right_indices]
            for a in left_indices:
                distances = lookup[np.bitwise_xor(right_hashes, phashes[a]).view(np.uint8).reshape(-1, 8)].sum(axis=1)
                for relative in np.flatnonzero(distances <= phash_limit):
                    b = int(right_indices[relative])
                    d_distance = (features[a][1] ^ features[b][1]).bit_count()
                    if d_distance <= dhash_limit:
                        yield {"a": a, "b": b, "phash_distance": int(distances[relative]),
                               "dhash_distance": d_distance, "identical_decoded_pixels": features[a][2] == features[b][2]}


def same_split_cross_source_pairs(records, features, phash_limit=6, dhash_limit=8):
    """Screen cross-release copies while leaving within-source sequence frames alone."""
    for split in sorted({row["split"] for row in records}):
        indices = [i for i, row in enumerate(records) if row["split"] == split]
        # Reuse the identical two-hash rule, treating source names as partitions.
        proxies = [{"split": records[i]["source"]} for i in indices]
        for pair in cross_split_pairs(proxies, [features[i] for i in indices], phash_limit, dhash_limit):
            pair["a"], pair["b"] = indices[pair["a"]], indices[pair["b"]]
            yield pair


def review_pairs(records, features, include_cross_source=False):
    yield from cross_split_pairs(records, features)
    if include_cross_source:
        yield from same_split_cross_source_pairs(records, features)


def run(args):
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    raw = manifest.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    features = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(image_hashes, [(manifest.parent / r["image"]).resolve() for r in records]), 1):
            features.append(result)
            if i % 1000 == 0:
                print(json.dumps({"images_hashed": i, "total": len(records)}), flush=True)
    pairs = []
    for pair in review_pairs(records, features, args.cross_source):
        left, right = records[pair.pop("a")], records[pair.pop("b")]
        pair["relation"] = "cross_split" if left["split"] != right["split"] else "same_split_cross_source"
        pair.update(left={k: left[k] for k in ("id", "source", "split", "sequence", "image")},
                    right={k: right[k] for k in ("id", "source", "split", "sequence", "image")})
        pairs.append(pair)
    feature_path = output.with_suffix(".features.jsonl")
    with feature_path.open("w", encoding="utf-8") as stream:
        for row, (p, d, pixels) in zip(records, features):
            stream.write(json.dumps({"source": row["source"], "id": row["id"], "split": row["split"],
                                     "phash63": f"{p:016x}", "dhash64": f"{d:016x}", "decoded_sha256": pixels}) + "\n")
    report = {"manifest": str(manifest), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "images": len(records), "phash63_limit": 6, "dhash64_limit": 8,
              "candidate_pairs": pairs, "candidate_count": len(pairs),
              "same_split_cross_source_screening": bool(args.cross_source),
              "cross_split_candidates": sum(p["relation"] == "cross_split" for p in pairs),
              "same_split_cross_source_candidates": sum(p["relation"] == "same_split_cross_source" for p in pairs),
              "status": "review_required" if pairs else "no_candidates_at_declared_thresholds",
              "limitations": "Perceptual hashes miss related views and can match unrelated scenes. This is not a ride/location-independence guarantee.",
              "seconds": time.perf_counter() - started}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "candidate_pairs"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cross-source", action="store_true", help="Also screen same-split photos across source releases")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    run(parser.parse_args())
