"""Rasterize official IDD polygon JSON to original label IDs, preserving object order.

Uses the AutoNUE public-code json2labelImg.py semantics (Pillow polygon rasterization,
background label 35, deleted/degenerate objects skipped, group suffix fallback).
Unknown labels fail closed instead of silently leaving misleading background.
This is annotation preparation; no model is created or trained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

LABEL_NAMES = (
    "road", "parking", "drivable fallback", "sidewalk", "rail track",
    "non-drivable fallback", "person", "animal", "rider", "motorcycle",
    "bicycle", "autorickshaw", "car", "truck", "bus", "caravan", "trailer",
    "train", "vehicle fallback", "curb", "wall", "fence", "guard rail",
    "billboard", "traffic sign", "traffic light", "pole", "polegroup",
    "obs-str-bar-fallback", "building", "bridge", "tunnel", "vegetation",
    "sky", "fallback background", "unlabeled", "ego vehicle",
    "rectification border", "out of roi", "license plate",
)
LABEL_IDS = {name: index for index, name in enumerate(LABEL_NAMES)}
CONVERTER_VERSION = "idd_original_ids_pillow_v1"
REFERENCE = "https://github.com/AutoNUE/public-code/blob/master/preperation/json2labelImg.py"


def rasterize(document):
    width, height = document["imgWidth"], document["imgHeight"]
    if not isinstance(width, int) or not isinstance(height, int) or not 0 < width <= 20000 or not 0 < height <= 20000:
        raise ValueError("Invalid annotation dimensions")
    result = Image.new("L", (width, height), LABEL_IDS["unlabeled"])
    draw = ImageDraw.Draw(result)
    for obj in document["objects"]:
        if obj.get("deleted", 0):
            continue
        polygon = obj["polygon"]
        if len(polygon) < 3:
            continue
        label = obj["label"]
        if label not in LABEL_IDS and label.endswith("group"):
            label = label[:-5]
        if label not in LABEL_IDS:
            raise ValueError(f"Unknown IDD label: {label!r}")
        vertices = []
        for point in polygon:
            if len(point) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                raise ValueError("Invalid polygon coordinate")
            vertices.append(tuple(point))
        # The official converter does not filter the annotation's 'draw' flag.
        draw.polygon(vertices, fill=LABEL_IDS[label])
    return result


def convert_one(root, annotation):
    raw = annotation.read_bytes()
    document = json.loads(raw)
    relative = annotation.relative_to(root / "gtFine")
    stem = relative.name.removesuffix("_gtFine_polygons.json")
    image_base = root / "leftImg8bit" / relative.parent / (stem + "_leftImg8bit")
    candidates = [image_base.with_suffix(ext) for ext in (".png", ".jpg", ".jpeg")]
    matched = [path for path in candidates if path.is_file()]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one image for {relative}, found {len(matched)}")
    mask = rasterize(document)
    with Image.open(matched[0]) as picture:
        if picture.size != mask.size:
            raise ValueError(f"Image/annotation dimension mismatch: {relative}")
    destination = annotation.with_name(stem + "_gtFine_labelIds.png")
    if destination.exists():
        with Image.open(destination) as existing:
            if existing.mode != "L" or existing.size != mask.size or existing.tobytes() != mask.tobytes():
                raise ValueError(f"Conflicting existing mask: {destination}")
        status = "verified_existing"
    else:
        temporary = destination.with_suffix(".png.partial")
        mask.save(temporary, format="PNG")
        temporary.replace(destination)
        status = "created"
    with destination.open("rb") as handle:
        mask_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "annotation": annotation.relative_to(root).as_posix(),
        "image": matched[0].relative_to(root).as_posix(),
        "mask": destination.relative_to(root).as_posix(),
        "annotation_sha256": hashlib.sha256(raw).hexdigest(),
        "mask_sha256": mask_sha256,
        "width": mask.width, "height": mask.height,
        "labels_present": [label for label, count in enumerate(mask.histogram()) if count], "status": status,
    }


def extraction_snapshot(root):
    """When our extractor metadata exists, require every selected archive complete."""
    directory = root / ".extraction"
    if not directory.exists():
        return {}  # Also supports manually supplied publisher layouts.
    reports = sorted(directory.glob("*/report.json"))
    if not reports:
        raise ValueError("Extraction metadata exists without a completion report")
    snapshot = {}
    for path in reports:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not value.get("usable") or not value.get("gzip_eof_verified"):
            raise ValueError(f"IDD archive extraction is incomplete: {path.parent.name}")
        snapshot[path.parent.name] = hashlib.sha256(raw).hexdigest()
    return snapshot


def run(args):
    root = Path(args.root).resolve()
    extractions = extraction_snapshot(root)
    annotations = sorted(path for split in args.splits
                         for path in (root / "gtFine" / split).rglob("*_gtFine_polygons.json"))
    if not annotations:
        raise ValueError("No extracted IDD train/validation polygons found")
    report = root / "labelIds_SOURCE.json"
    manifest = root / "labelIds_SOURCE.jsonl"
    report.unlink(missing_ok=True)  # Remove only this generated completion marker.
    temporary = manifest.with_suffix(".jsonl.partial")
    counts = {}
    created = 0
    with temporary.open("w", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(lambda path: convert_one(root, path), annotations), 1):
            output.write(json.dumps(record) + "\n")
            split = Path(record["annotation"]).parts[1]
            counts[split] = counts.get(split, 0) + 1
            created += record["status"] == "created"
            if index % 250 == 0:
                output.flush()
                print(json.dumps({"masks_processed": index, "total": len(annotations)}), flush=True)
    if extraction_snapshot(root) != extractions:
        raise ValueError("Extraction completion state changed during mask generation")
    temporary.replace(manifest)
    result = {"converter": CONVERTER_VERSION, "reference": REFERENCE,
              "label_encoding": "original IDD ID, not trainId or level3Id", "labels": LABEL_IDS,
              "background": 35, "counts": counts, "created": created,
              "verified_extraction_report_sha256": extractions,
              "verified_existing": len(annotations) - created,
              "unknown_labels": "error", "object_order": "original; later polygons overwrite earlier",
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "counts": counts, "report": str(report)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/phase2/raw/idd_20k")
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=4)
    run(parser.parse_args())
