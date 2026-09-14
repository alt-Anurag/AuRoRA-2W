"""Select whole IDD segmentation sequences from verified complete archives.

Preserves a full native index and per-image archive lineage. This is data-only
preparation; no model, optimizer, predictions or training checkpoints are used.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from aurora2w.native import build_native, relative
from scripts.prepare_bdd_subset import select_groups


def run(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    marker = root / "labelIds_SOURCE.json"
    if not marker.is_file():
        raise ValueError("Complete combined labelIds_SOURCE.json is required before selecting IDD")
    marker_bytes = marker.read_bytes()
    masks = json.loads(marker_bytes)
    if hashlib.sha256((root / "labelIds_SOURCE.jsonl").read_bytes()).hexdigest() != masks["manifest_sha256"]:
        raise ValueError("IDD mask-manifest fingerprint differs from completion marker")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh IDD subset proposal directory")
    archive_rows = {}
    archive_reports = {}
    for report_path in sorted((root / ".extraction").glob("*/report.json")):
        raw = report_path.read_bytes()
        report = json.loads(raw)
        if not report.get("usable") or not report.get("gzip_eof_verified"):
            raise ValueError(f"Incomplete archive extraction: {report_path}")
        digest = hashlib.sha256(raw).hexdigest()
        if masks["verified_extraction_report_sha256"].get(report_path.parent.name) != digest:
            raise ValueError("Mask conversion did not verify this exact archive extraction report")
        archive_reports[report_path.parent.name] = digest
        for line in (report_path.parent / "files.jsonl").read_text(encoding="utf-8-sig").splitlines():
            row = json.loads(line)
            if row["path"].startswith("leftImg8bit/"):
                if row["path"] in archive_rows:
                    raise ValueError(f"Multiple archive origins need explicit review: {row['path']}")
                archive_rows[row["path"]] = {"source_archive_sha256": report["archive_sha256"],
                                             "source_archive_member": row["archive_member"],
                                             "source_archive_name": Path(report["archive"]).name}
    if len(archive_reports) != 2:
        raise ValueError("This proposal requires both verified IDD segmentation parts")
    output.mkdir(parents=True, exist_ok=True)
    built = build_native("idd-seg", root, output / "full_index")
    full_index = output / "full_index" / "index.jsonl"
    rows = [json.loads(line) for line in full_index.read_text(encoding="utf-8-sig").splitlines()]
    groups = defaultdict(set)
    for row in rows:
        if row["image"] not in archive_rows:
            raise ValueError(f"Image lacks verified archive lineage: {row['image']}")
        row.update(archive_rows[row["image"]])
        groups[row["sequence"]].add(row["split"])
    crossing = [key for key, splits in groups.items() if len(splits) > 1]
    if crossing:
        raise ValueError(f"Observed IDD sequence directories cross official splits: {crossing[:20]}")
    actual = dict(Counter(row["split"] for row in rows))
    if actual != masks["counts"]:
        raise ValueError(f"Native index counts differ from verified mask counts: {actual}")
    full_index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    mask_rows = {row["image"]: row for row in (json.loads(line) for line in (root / "labelIds_SOURCE.jsonl").read_text(encoding="utf-8").splitlines())}
    supervised_ids = {0, 2} | set(range(3, 35))
    exclusions = []
    for row in rows:
        mask_row = mask_rows[row["image"]]
        if mask_row["mask"] != row["annotation"]:
            raise ValueError("Native annotation does not match verified mask manifest")
        row["road_has_supervision"] = bool(supervised_ids & set(mask_row["labels_present"]))
        if not row["road_has_supervision"]:
            exclusions.append(dict(row, exclusion_reason="All pixels map to ignored road labels; no other task annotated",
                                   original_ids_present=mask_row["labels_present"]))
    (output / "zero_supervision_exclusions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in exclusions), encoding="utf-8")
    selected = []
    summary = {}
    for split, target in (("train", args.train_images), ("val", args.val_images)):
        available = [row for row in rows if row["split"] == split]
        chosen = select_groups(available, target, args.seed + (split == "val"), stratum_key="source_archive_name")
        selected_group_images = len(chosen)
        chosen = [row for row in chosen if row["road_has_supervision"]]
        selected.extend(chosen)
        summary[split] = {"available_images": len(available), "target_images": target, "selected_images": len(chosen),
                          "selected_group_images_before_exclusions": selected_group_images,
                          "excluded_selected_zero_supervision": selected_group_images - len(chosen),
                          "available_usable_images": sum(row["road_has_supervision"] for row in available),
                          "available_sequences": len({row["sequence"] for row in available}),
                          "selected_sequences": len({row["sequence"] for row in chosen}),
                          "selected_by_archive": dict(Counter(row["source_archive_name"] for row in chosen)),
                          "effective_seed": args.seed + (split == "val")}
    index = output / "index.jsonl"
    index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    source = json.loads(Path(built["spec"]).read_text())["sources"][0]
    source.update(root=relative(root, output), index="index.jsonl", source_release="IDD segmentation Part I and Part II original IDs",
                  selection_protocol="idd_archive_stratified_whole_sequences_v2", selection_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
    (output / "sources.json").write_text(json.dumps({"sources": [source]}, indent=2), encoding="utf-8")
    for split in ("train", "val"):
        directory = output / ("index_" + split)
        directory.mkdir()
        (directory / "index.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected if row["split"] == split), encoding="utf-8")
        split_source = dict(source, root=relative(root, directory))
        (directory / "sources.json").write_text(json.dumps({"sources": [split_source]}, indent=2), encoding="utf-8")
    if marker.read_bytes() != marker_bytes:
        raise ValueError("Mask completion state changed during selection; do not use this proposal")
    report = {"scope": "Frozen whole-sequence selection proposal; no training started", "training_approved": False,
              "selection": "Proportional archive strata, seeded group shuffle; all supervised images of selected sequence directories retained; ignored-only rows explicitly excluded",
              "group_scope": "Published observed sequence directories; no additional claim of rider/location independence",
              "split_scope": "Official train/val membership preserved", "source_counts": actual,
              "excluded_zero_supervision": dict(Counter(row["split"] for row in exclusions)),
              "exclusions_sha256": hashlib.sha256((output / "zero_supervision_exclusions.jsonl").read_bytes()).hexdigest(),
              "mask_completion_sha256": hashlib.sha256(marker_bytes).hexdigest(), "extraction_reports": archive_reports,
              "full_index_sha256": hashlib.sha256(full_index.read_bytes()).hexdigest(),
              "selected_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "splits": summary}
    (output / "SELECTION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/phase2/raw/idd_20k")
    parser.add_argument("--output", default="data/phase2/partitions/idd_seg_subset_v1")
    parser.add_argument("--train-images", type=int, default=4000)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
