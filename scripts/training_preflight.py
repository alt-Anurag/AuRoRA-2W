"""Review manifest metadata and RTX 4050 budgeting without model/optimizer work.

This never launches training, profiling, inference, or checkpoint creation. The
full image/annotation/hash audit and researcher review are separate requirements.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from aurora2w.config import CLASS_NAMES, load_config


def review_records(records, config, profiles, missing_files=()):
    """Summarize declared supervision; mask references do not prove valid pixels."""
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "val"]
    counts = Counter(row["source"] for row in train)
    size, source_count = len(train), len(counts)
    batches = math.ceil(size / config["batch_size"])
    accumulation = config.get("accumulate_steps", 1)
    source_sampling = {}
    for source, count in sorted(counts.items()):
        draws = size / source_count if config["source_balance"] else count
        probability = 1 / (source_count * count)
        unique = count * (-math.expm1(size * math.log1p(-probability))) if probability < 1 else count
        source_sampling[source] = {
            "manifest_training_rows": count,
            "expected_draws_per_epoch": draws,
            "expected_draws_per_row_per_epoch": draws / count,
            "expected_distinct_rows_per_epoch": unique if config["source_balance"] else count,
        }
    coverage = {}
    blockers = []
    for split in ("train", "val", "test"):
        selected = [row for row in records if row["split"] == split]
        masks = {task: sum(bool(row.get(task + "_mask")) for row in selected) for task in ("road", "lane")}
        positives = Counter(obj["class"] for row in selected for obj in row.get("boxes", []))
        exhaustive = Counter(name for row in selected for name in row.get("detection_classes", []))
        negatives = Counter(name for row in selected for name in row.get("detection_classes", [])
                            if name not in {obj["class"] for obj in row.get("boxes", [])})
        coverage[split] = {
            "images": len(selected), "mask_references_not_pixel_audit": masks,
            "positive_instances": dict(positives), "exhaustive_images": dict(exhaustive),
            "declared_exhaustive_negative_images": dict(negatives),
            "sources": dict(Counter(row["source"] for row in selected)),
        }
        if split in ("train", "val"):
            if not selected:
                blockers.append(f"No {split} records")
            for task in ("road", "lane"):
                if not masks[task]:
                    blockers.append(f"No {split} {task} mask references")
            for name in CLASS_NAMES:
                if not positives[name]:
                    blockers.append(f"No {split} positive instances for {name}")
                if not exhaustive[name]:
                    blockers.append(f"No {split} exhaustive detection coverage for {name}")
    unknown_provenance = sum(row.get("data_kind") != "real" for row in train + validation)
    if unknown_provenance:
        blockers.append(f"{unknown_provenance} train/val records are not declared real photographed data")
    trusted = sum(row.get("imu", {}).get("confidence", 0) > 0 for row in train)
    upright = sum(bool(row.get("upright_reference")) for row in train)
    conditioned = sum(row.get("imu", {}).get("confidence", 0) > 0 or bool(row.get("upright_reference")) for row in train)
    if config["alignment"] == "idfa" and not conditioned:
        blockers.append("IDFA has no calibrated IMU or explicitly curated upright references")
    ids = Counter((row["source"], row["id"]) for row in records)
    duplicate_ids = sum(count - 1 for count in ids.values() if count > 1)
    if duplicate_ids:
        blockers.append(f"{duplicate_ids} repeated source/id pairs")
    groups = defaultdict(set)
    for row in records:
        groups[(row["source"], row["sequence"])].add(row["split"])
    crossing = sum(len(splits) > 1 for splits in groups.values())
    if crossing:
        blockers.append(f"{crossing} source/sequence groups cross splits")
    if missing_files:
        blockers.append(f"{len(missing_files)} referenced image/mask files are missing")
    matched = (config["batch_size"] == 8 and config["image_size"] == [384, 640]
               and config.get("amp") is True and accumulation == 1 and config["workers"] == 2
               and config.get("channels") == 64 and config.get("neck_repeats") == 2
               and config.get("idfa_levels") == [3])
    measured = profiles.get("profiles", {}).get(config["alignment"] + "_batch8") if matched else None
    timing = {"available": bool(measured), "validation_timing_measured": False}
    if measured:
        seconds = batches * measured["median_training_step_s"]
        timing.update(
            measured_step_seconds=measured["median_training_step_s"],
            measured_images_per_second=measured["images_per_second"],
            train_only_epoch_hours=seconds / 3600,
            train_only_planned_hours=seconds * config["epochs"] / 3600,
            assumptions="Historical six post-warmup steps on repeated RDD fixture; no road/lane masks. Not sustained mixed-data speed.",
            generic_planning_scenarios="Removed: earlier 10/2 validation-images-per-second assumptions are superseded by recorded real-image timings and docs/TRAINING_REVIEW.md.",
        )
    return {
        "scope": "Metadata preflight and conditional budget only; no model created and no training performed",
        "status": "blocked" if blockers else "awaiting_full_label_audit_and_researcher_review",
        "training_approved": False,
        "blockers": blockers,
        "coverage": coverage,
        "sampling": {"training_rows": size, "draws_per_epoch": size, "batches_per_epoch": batches,
                     "optimizer_updates_per_epoch_before_amp_skips": math.ceil(batches / accumulation),
                     "planned_epochs": config["epochs"], "replacement": bool(config["source_balance"]),
                     "source_balance": bool(config["source_balance"]), "sources": source_sampling},
        "geometry": {"calibrated_imu_declared": trusted, "upright_reference_declared": upright,
                     "conditioned_training_rows": conditioned,
                     "note": "Declarations require curation evidence. Roll-only augmentation does not establish portrait, pitch/yaw or vibration robustness."},
        "missing_file_examples": list(missing_files)[:20],
        "timing": timing,
        "pending_checks": ["Full decoded-pixel and label audit at configured resolution",
                           "Near-duplicate/recording groups and cross-release IDD overlaps",
                           "Visual mask/box alignment including missing-label regions",
                           "Per-source task/class distribution and immutable held-out split review",
                           "Actual mixed-data loader and full validation timing after review"],
    }


def timing_evidence(paths):
    """Pin inference/input-only timing reports; do not mistake them for training."""
    evidence = []
    for original in paths:
        path = Path(original).resolve()
        raw = path.read_bytes()
        report = json.loads(raw)
        purpose = report.get("purpose")
        if purpose not in ("real_training_input_delivery_only", "random_untrained_model_real_image_validation_timing_only"):
            raise ValueError(f"Not a recognized read-only timing report: {path}")
        if report.get("optimizer_steps") != 0 or report.get("backward_calls") != 0 or report.get("accuracy_reported") is not False:
            raise ValueError(f"Timing report includes training or accuracy operations: {path}")
        entry = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "purpose": purpose,
                 "image_size": report["image_size"], "batch_size": report["batch_size"]}
        if purpose == "real_training_input_delivery_only":
            if report.get("model_instances") != 0:
                raise ValueError("Input-only timing report created a model")
            entry.update(training_rows=report["train_records"], source_counts=report["train_records_by_source"],
                         steady_images_per_second=report["steady_images_per_second"],
                         steady_seconds_per_batch=report["steady_mean_wall_seconds_per_batch"],
                         startup_to_first_batch_seconds=report["startup_to_first_batch_seconds"],
                         scope="Input delivery only, not joint training throughput")
        else:
            if report.get("weights_and_buffers_unchanged") is not True or report.get("checkpoint_written") is not False:
                raise ValueError("Validation timing did not establish unchanged state and no checkpoint")
            entry.update(images_per_pass=report["measured_images_per_pass"], source_split_counts=report["selected_source_split_counts"],
                         full_pass_seconds=[item["wall_seconds"] for item in report["passes"]],
                         scope="Real held-out inference/loss/metrics workload with random unchanged weights; not accuracy")
        evidence.append(entry)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/rtx4050_baseline.json")
    parser.add_argument("--profiles", default="artifacts/rtx4050-batch-profiles.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-reports", nargs="*", default=[], help="Explicit real input/inference-only evidence to pin; no optimizer profiling")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records:
        raise ValueError("Manifest is empty")
    files = set()
    for row in records:
        for key in ("id", "source", "sequence", "split", "image"):
            if not row.get(key):
                raise ValueError(f"Manifest record is missing {key}")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {row['split']}")
        for key in ("image", "road_mask", "lane_mask"):
            if row.get(key):
                files.add((manifest.parent / row[key]).resolve())
    missing = sorted(str(path) for path in files if not path.is_file())
    result = review_records(records, load_config(args.config),
                            json.loads(Path(args.profiles).read_text(encoding="utf-8-sig")), missing)
    result["manifest"] = str(manifest)
    result["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result["config"] = str(Path(args.config).resolve())
    result["config_sha256"] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    result["resolved_config_sha256"] = hashlib.sha256(json.dumps(load_config(args.config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result["historical_profiles_sha256"] = hashlib.sha256(Path(args.profiles).read_bytes()).hexdigest()
    evidence = timing_evidence(args.timing_reports)
    result["timing"]["actual_measurement_evidence"] = evidence
    result["timing"]["validation_timing_measured"] = any(item["purpose"] == "random_untrained_model_real_image_validation_timing_only" for item in evidence)
    result["timing"]["joint_training_timing_measured_for_this_manifest"] = False
    result["timing"]["budget_review"] = "docs/TRAINING_REVIEW.md; source mix, validation workload and thermal/I/O assumptions must match the frozen experiment"
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"], "training_approved": False, "blockers": result["blockers"],
                      "sampling": result["sampling"], "timing": result["timing"], "report": str(path)}, indent=2))
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
