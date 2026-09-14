"""Explicit, loss-aware conversion of local IDD, VOC/RDD, and YOLO/Kaggle data."""

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .config import CLASS_NAMES


def write_manifest(records, destination):
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    portable = []
    for original in records:
        record = {k: v for k, v in original.items() if not k.startswith("_")}
        for key in ("image", "road_mask", "lane_mask"):
            if record.get(key):
                try:
                    record[key] = Path(os.path.relpath(Path(record[key]).resolve(), path.parent)).as_posix()
                except ValueError as exc:
                    raise ValueError("Manifest and data must share a drive for portable paths") from exc
        portable.append(record)
    path.write_text("".join(json.dumps(r) + "\n" for r in portable), encoding="utf-8")


def prepare_spec(spec_path, output):
    """Spec sources require sequence lists; no random adjacent-frame splitting."""
    spec_path = Path(spec_path).resolve()
    if Path(output).exists():
        raise FileExistsError(f"Output manifest already exists: {output}")
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    records = []
    for source in spec["sources"]:
        root = (spec_path.parent / source["root"]).resolve()
        mapping = source.get("class_map", {})
        if any(v not in CLASS_NAMES for v in mapping.values()):
            raise ValueError("class_map values must belong to the canonical taxonomy")
        known = source.get("exhaustive_classes", [])
        if source["format"] != "masks" and not set(known).issubset(set(mapping.values())):
            raise ValueError("Exhaustive classes must be present in class_map")
        index_path = (spec_path.parent / source["index"]).resolve()
        # Each JSONL index row explicitly associates an image/annotation with a sequence and split.
        entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for item in entries:
            image_path = (root / item["image"]).resolve()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable image: {image_path}")
            h, w = image.shape[:2]
            record = {"id": item["id"], "source": source["name"], "sequence": item["sequence"],
                      "split": item["split"], "image": str(image_path), "boxes": [],
                      "ignore_boxes": [], "detection_classes": list(item.get("detection_classes", known)),
                      "data_kind": item.get("data_kind", source.get("data_kind", "unspecified")),
                      "upright_reference": item.get("upright_reference", source.get("upright_reference", False))}
            for provenance_key in ("source_release", "selection_protocol", "selection_sha256",
                                   "partition_protocol", "partition_group_kind", "partition_artifact_sha256",
                                   "source_archive_sha256", "source_archive_member", "source_image_sha256", "source_annotation_sha256"):
                if provenance_key in item or provenance_key in source:
                    record[provenance_key] = item.get(provenance_key, source.get(provenance_key))
            if item.get("conversion_warnings"):
                record["conversion_warnings"] = list(item["conversion_warnings"])
            if "imu" in item:
                record["imu"] = item["imu"]
            if not set(record["detection_classes"]).issubset(CLASS_NAMES):
                raise ValueError(f"Unknown detection coverage for {item['id']}")
            fmt = source["format"]
            if fmt in ("yolo", "voc"):
                annotation = (root / item["annotation"]).resolve()
                if fmt == "yolo":
                    # A missing text file is not proof that an image is negative.
                    for line in annotation.read_text(encoding="utf-8-sig").splitlines():
                        if not line.strip():
                            continue
                        values = line.split()
                        if len(values) != 5:
                            raise ValueError(f"Expected YOLO detection labels (5 columns): {annotation}")
                        label, xc, yc, bw, bh = values
                        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                        box = [(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h]
                        box = [max(0., min(v, w if i % 2 == 0 else h)) for i, v in enumerate(box)]
                        if label in mapping:
                            record["boxes"].append({"class": mapping[label], "xyxy": box})
                        else:
                            record["ignore_boxes"].append(box)
                else:
                    tree = ET.parse(annotation).getroot()
                    size = tree.find("size")
                    if size is not None and (int(size.findtext("width")), int(size.findtext("height"))) != (w, h):
                        raise ValueError(f"VOC image dimensions disagree: {annotation}")
                    # Convert explicit source pixel coordinates to pixel edges.
                    coordinates = source.get("voc_coordinates", "one_based_inclusive" if source.get("voc_one_based", True) else "zero_based_half_open")
                    if coordinates not in ("one_based_inclusive", "zero_based_inclusive", "zero_based_half_open"):
                        raise ValueError(f"Unsupported VOC coordinate convention: {coordinates}")
                    offset = 1 if coordinates == "one_based_inclusive" else 0
                    maximum_offset = 1 if coordinates == "zero_based_inclusive" else 0
                    for object_index, obj in enumerate(tree.findall("object")):
                        label = obj.findtext("name")
                        bounds = obj.find("bndbox")
                        box = [float(bounds.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                        source_box = list(box)
                        if not np.isfinite(box).all():
                            raise ValueError(f"Nonfinite VOC box: {annotation}")
                        if box[2] < box[0] or box[3] < box[1]:
                            raise ValueError(f"Inverted VOC box: {annotation}")
                        box[0] -= offset
                        box[1] -= offset
                        box[2] += maximum_offset
                        box[3] += maximum_offset
                        if source.get("clip_boxes_to_image", False):
                            clipped = [max(0., min(v, w if i % 2 == 0 else h)) for i, v in enumerate(box)]
                            if clipped != box:
                                record.setdefault("conversion_warnings", []).append({"operation": "clip_voc_box_to_image", "object_index": object_index, "category": label,
                                                                                   "coordinate_convention": coordinates, "source_xyxy": source_box,
                                                                                   "original_xyxy": box, "clipped_xyxy": clipped})
                            box = clipped
                        if box[2] <= box[0] or box[3] <= box[1]:
                            if not source.get("skip_degenerate_voc_boxes", False):
                                raise ValueError(f"Empty/inverted VOC box: {annotation}")
                            category = mapping.get(label)
                            if category in record["detection_classes"]:
                                record["detection_classes"].remove(category)
                            record.setdefault("conversion_warnings", []).append({
                                "operation": "skip_degenerate_voc_box", "category": label, "object_index": object_index,
                                "source_xyxy": source_box, "clipped_xyxy": box, "removed_exhaustive_class": category})
                            continue
                        difficult = int(obj.findtext("difficult", "0")) != 0
                        if label in mapping and not difficult:
                            record["boxes"].append({"class": mapping[label], "xyxy": box})
                        else:
                            record["ignore_boxes"].append(box)
            elif fmt == "idd_label_ids":
                # ONLY original AutoNUE `id`, never csTrainId or level3Id.
                # Official labels: road=0, parking=1, drivable fallback=2, void=35..39.
                raw = cv2.imread(str(root / item["annotation"]), cv2.IMREAD_UNCHANGED)
                if raw is None or raw.shape != (h, w):
                    raise ValueError("IDD annotation must be a single-channel original labelIds mask")
                if not np.all(((raw >= 0) & (raw < 40)) | (raw == 255)):
                    raise ValueError("Unknown original IDD label ID")
                road_ids = source.get("road_positive_ids", [0, 2])
                negative_ids = source.get("road_negative_ids", list(range(3, 35)))
                if set(road_ids) & set(negative_ids):
                    raise ValueError("Road positive/negative ID sets overlap")
                mask = np.full((h, w), 255, np.uint8)
                mask[np.isin(raw, negative_ids)] = 0
                mask[np.isin(raw, road_ids)] = 1
                # Stable name independent of potentially unsafe source/id path strings.
                import hashlib
                key = hashlib.sha256((str(image_path) + json.dumps(source, sort_keys=True)).encode()).hexdigest()[:20]
                mask_path = Path(output).resolve().parent / "masks" / f"idd_road_{key}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(mask_path), mask):
                    raise OSError(f"Cannot write {mask_path}")
                record["road_mask"] = str(mask_path)
                record["detection_classes"] = []
            elif fmt != "masks":
                raise ValueError(f"Unknown source format: {fmt}")
            for task in ("road", "lane"):
                if item.get(task + "_mask"):
                    mask_path = (root / item[task + "_mask"]).resolve()
                    value_map = source.get("mask_value_map", {}).get(task)
                    if value_map is not None:
                        raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                        if raw is None or raw.shape != (h, w):
                            raise ValueError(f"Invalid {task} source mask: {mask_path}")
                        values = {int(k): int(v) for k, v in value_map.items()}
                        if not set(values.values()).issubset({0, 1, 255}) or not set(np.unique(raw)).issubset(values):
                            raise ValueError(f"mask_value_map must account for every value in {mask_path}")
                        mask = np.full(raw.shape, 255, np.uint8)
                        for old, new in values.items():
                            mask[raw == old] = new
                        import hashlib
                        key = hashlib.sha256((str(mask_path) + json.dumps(values, sort_keys=True)).encode()).hexdigest()[:20]
                        mask_path = Path(output).resolve().parent / "masks" / f"{task}_{key}.png"
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(mask_path), mask):
                            raise OSError(mask_path)
                    record[task + "_mask"] = str(mask_path)
            if fmt == "masks":
                # Optional already-canonical boxes allow jointly labeled own/BDD frames.
                record["boxes"] = item.get("boxes", [])
                record["ignore_boxes"] = item.get("ignore_boxes", [])
            records.append(record)
    from .data import audit_records
    report = audit_records(records, check_hashes=True)
    write_manifest(records, output)
    return report
