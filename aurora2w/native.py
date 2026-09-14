"""Build explicit source indexes from original IDD, BDD100K and RDD layouts.

Sources: AutoNUE public-code/helpers/anue_labels.py; BDD100K poly2d uses
Scalabel L (line) and triples of C (cubic Bezier control/control/end) vertices.
Our binary lane stroke masks are training targets, not official BDD benchmark
rasters. No road boundary is relabeled as a painted lane.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .config import CLASS_NAMES


BDD_CLASSES = {name: name for name in CLASS_NAMES if name not in ("autorickshaw", "pothole")}
BDD_CLASSES.update({"bike": "bicycle", "motor": "motorcycle", "pedestrian": "person"})
IDD_CLASSES = {name: name for name in CLASS_NAMES if name != "pothole"}
IDD_CLASSES.update({"motorbike": "motorcycle", "auto rickshaw": "autorickshaw"})


def relative(path, base):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()
    except ValueError as exc:
        raise ValueError("Keep indexes, data and outputs on the same drive") from exc


def read_groups(path):
    """CSV image,sequence,split. Entire recording groups must share one split."""
    result, splits = {}, {}
    if path is None:
        return result
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = Path(row["image"]).as_posix()
            group, split = row["sequence"], row["split"]
            if not group or split not in ("train", "val", "test"):
                raise ValueError("groups.csv requires image, nonempty sequence, train/val/test split")
            if key in result or group in splits and splits[group] != split:
                raise ValueError("Duplicate group entry or sequence crossing splits")
            result[key] = (group, split)
            splits[group] = split
    return result


def _group(groups, image, root, default=None):
    key = relative(image, root)
    if groups:
        if key not in groups:
            raise ValueError(f"Missing group entry for {key}")
        return groups[key]
    if default is None:
        raise ValueError("This release has no verified held-out split. Supply --groups CSV with "
                         "image,sequence,split; group whole recording runs, never neighboring frames.")
    return default


def sample_curve(poly):
    """Return a polyline within roughly one source pixel of a cubic Bezier path."""
    vertices = np.asarray(poly["vertices"], dtype=np.float64)
    types = poly.get("types", "L" * len(vertices))
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2 or not np.isfinite(vertices).all():
        raise ValueError("Invalid poly2d vertices")
    if len(types) != len(vertices) or set(types) - {"L", "C"}:
        raise ValueError("Unsupported poly2d types; expected L and cubic C triples")
    points, index = [vertices[0]], 1
    while index < len(vertices):
        if types[index] == "L":
            points.append(vertices[index])
            index += 1
        else:
            if types[index:index + 3] != "CCC":
                raise ValueError("Cubic poly2d requires control/control/end C triples")
            p0, p1, p2, p3 = points[-1], *vertices[index:index + 3]
            length = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
            steps = max(8, int(np.ceil(length * 2)))
            if steps > 100000:
                raise ValueError("Unreasonable polygon extent")
            t = np.linspace(0, 1, steps + 1)[1:, None]
            points.extend((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3)
            index += 3
    return np.rint(points).astype(np.int32)


def bdd_masks(frame, shape, tasks, lane_width=8):
    """Explicit tasks declare exhaustive frame coverage; absent labels are unknown.

    Direct and alternative drivable polygons both map to road. Lane boundaries,
    curbs and crosswalks remain unknown along their strokes. This function uses
    the native 2018 categories/attributes or Scalabel task-specific categories.
    """
    labels = frame.get("labels")
    if labels is None:
        return {}
    result = {task: np.zeros(shape, np.uint8) for task in tasks if task in ("road", "lane")}
    ignored_lanes = np.zeros(shape, np.uint8)
    ignored_road = np.zeros(shape, np.uint8)
    for label in labels:
        category = label.get("category", "")
        attrs = label.get("attributes") or {}
        polys = label.get("poly2d") or []
        if category in ("drivable area", "direct", "alternative", "unknown") and "road" in result:
            area_type = attrs.get("areaType", category)
            if category == "drivable area" and area_type not in ("direct", "alternative", "unknown"):
                raise ValueError(f"Unrecognized BDD areaType: {area_type}")
            for poly in polys:
                if not poly.get("closed", True):
                    raise ValueError("Drivable polygon must be closed")
                destination = ignored_road if area_type == "unknown" else result["road"]
                cv2.fillPoly(destination, [sample_curve(poly)], 1)
        lane_types = ("single white", "single yellow", "double white", "double yellow", "single other", "double other")
        if "lane" in result and (category == "lane" or category in lane_types or category in ("road curb", "crosswalk")):
            lane_type = attrs.get("laneType", category)
            accepted = lane_type in lane_types
            destination = result["lane"] if accepted else ignored_lanes
            for poly in polys:
                cv2.polylines(destination, [sample_curve(poly)], bool(poly.get("closed", False)), 1, lane_width)
    if "road" in result:
        result["road"][ignored_road > 0] = 255
    if "lane" in result:
        result["lane"][(ignored_lanes > 0) & (result["lane"] == 0)] = 255
    return result



def normalize_legacy_bdd_poly(poly, closed=False):
    """Preserve the pre-August-2018 publisher Matplotlib Path semantics.

    First vertex is MOVETO regardless of its marker. A later C consumes three
    vertices even when its endpoint carries L (observed CCCL). Closed areas
    append the first vertex; it may be consumed as a final cubic endpoint.
    Output uses strict modern L/CCC triples, without changing coordinates.
    """
    if not isinstance(poly, list) or len(poly) < 2:
        raise ValueError("Legacy BDD poly2d requires at least two vertices")
    if any(not isinstance(point, (list, tuple)) or len(point) != 3 or point[2] not in ("L", "C") for point in poly):
        raise ValueError("Legacy BDD vertices must be [x,y,L|C]")
    vertices = [point[:2] for point in poly]
    markers = [point[2] for point in poly]
    if closed:
        vertices.append(vertices[0][:])
        markers.append("L")  # Publisher appends first point with CLOSEPOLY.
    types = ["L"] * len(vertices)
    index = 1
    while index < len(vertices):
        if markers[index] == "C":
            if index + 2 >= len(vertices):
                raise ValueError("Legacy BDD cubic path lacks control/control/end vertices")
            types[index:index + 3] = ["C", "C", "C"]
            index += 3
        else:
            index += 1
    return {"vertices": vertices, "types": "".join(types), "closed": closed}


def normalize_legacy_bdd_frame(document, annotation_name=None):
    """Convert one original keyframe JSON, preserving unknown annotation state."""
    name = document.get("name")
    frames = document.get("frames")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Legacy BDD name must be a single image stem")
    if annotation_name and Path(name).stem != Path(annotation_name).stem:
        raise ValueError("Legacy BDD document name differs from annotation stem")
    if not isinstance(frames, list) or len(frames) != 1:
        raise ValueError("Legacy BDD keyframe JSON requires exactly one frame")
    objects = frames[0].get("objects")
    result = {"name": name if Path(name).suffix else name + ".jpg",
              "videoName": document.get("videoName") or Path(name).stem,
              "attributes": document.get("attributes", {}),
              "timestamp": frames[0].get("timestamp"), "labels": None, "annotation_format": "bdd_legacy_2018"}
    if objects is None:
        return result
    if not isinstance(objects, list):
        raise ValueError("Legacy BDD objects must be a list or null")
    result["labels"] = []
    for obj in objects:
        if obj.get("deleted", False):
            continue
        label = dict(obj)
        category = label.get("category", "")
        attributes = dict(label.get("attributes") or {})
        if category.startswith("area/"):
            area = {"area/drivable": "direct", "area/alternative": "alternative", "area/unknown": "unknown"}.get(category)
            if area is None:
                raise ValueError(f"Unknown legacy BDD area category: {category}")
            label["category"] = "drivable area"
            attributes["areaType"] = area
        elif category.startswith("lane/"):
            label["category"] = "lane"
            attributes["laneType"] = category.partition("/")[2]
        label["attributes"] = attributes
        if label.get("poly2d"):
            label["poly2d"] = [normalize_legacy_bdd_poly(label["poly2d"], closed=category.startswith("area/"))]
        result["labels"].append(label)
    return result


def iter_bdd_frames(labels):
    """Stream legacy per-image directories; retain modern frame-list support."""
    path = Path(labels)
    if path.is_dir():
        paths = sorted(path.glob("*.json"))
        if not paths:
            raise ValueError("BDD annotation directory contains no JSON files")
        for annotation in paths:
            document = json.loads(annotation.read_text(encoding="utf-8-sig"))
            yield normalize_legacy_bdd_frame(document, annotation.name)
        return
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict) and raw.get("name") and "frames" in raw:
        yield normalize_legacy_bdd_frame(raw, path.name)
        return
    frames = raw if isinstance(raw, list) else raw.get("frames") if isinstance(raw, dict) else None
    if not isinstance(frames, list):
        raise ValueError("BDD requires a frame list, Scalabel frames, or legacy per-image JSON directory")
    yield from frames

def _index_nested_idd_detection(root, groups, coordinates):
    """Use publisher relative IDs, including repeated frame basenames.

    A groups CSV is also the exact selected-image allowlist. Unselected XML
    annotations may be retained locally without requiring their JPEGs.
    Coordinate origin must come from the separate release review.
    """
    if coordinates not in ("one_based_inclusive", "zero_based_inclusive", "zero_based_half_open"):
        raise ValueError("Nested IDD detection requires explicitly reviewed --voc-coordinates")
    official = {}
    for split in ("train", "val", "test"):
        path = root / (split + ".txt")
        if split in ("train", "val") and not path.is_file():
            raise FileNotFoundError(f"Missing publisher IDD split list: {path}")
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            identifier = line.strip()
            if not identifier:
                continue
            parts = Path(identifier).parts
            if (len(parts) != 3 or Path(identifier).is_absolute() or any(part in (".", "..") for part in parts)
                    or ":" in identifier or "\\" in identifier or Path(identifier).suffix):
                raise ValueError(f"Invalid nested IDD split identity: {identifier}")
            identifier = Path(identifier).as_posix()
            if identifier in official:
                raise ValueError(f"Duplicate/overlapping publisher IDD identity: {identifier}")
            official[identifier] = split
    selected = sorted(groups) if groups else ["JPEGImages/" + key + ".jpg" for key in sorted(official) if official[key] != "test"]
    rows = []
    for relative_image in selected:
        parts = Path(relative_image).parts
        if (len(parts) != 4 or parts[0] != "JPEGImages" or Path(relative_image).suffix.lower() not in (".jpg", ".jpeg", ".png")
                or any(part in (".", "..") for part in parts) or ":" in relative_image):
            raise ValueError(f"Invalid selected IDD image path: {relative_image}")
        identifier = Path(*parts[1:]).with_suffix("").as_posix()
        split = official.get(identifier)
        if split not in ("train", "val"):
            raise ValueError(f"Selected IDD image lacks labeled publisher train/val membership: {identifier}")
        image = root / relative_image
        annotation = root / "Annotations" / (identifier + ".xml")
        if not image.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Selected IDD image/XML is missing: {identifier}")
        # If the publisher divides a directory into frame splits, the default
        # group names preserve that fact without asserting recording separation.
        sequence = "official/" + split + "/" + "/".join(parts[1:-1])
        group, assigned = _group(groups, image, root, (sequence, split))
        if assigned != split:
            raise ValueError("Do not change publisher IDD split membership")
        rows.append({"id": identifier, "image": relative(image, root), "annotation": relative(annotation, root),
                     "sequence": group, "split": split, "publisher_sequence_directory": "/".join(parts[1:-1])})
    return rows


def build_native(kind, root, output, groups_path=None, labels=None, split=None,
                 tasks=None, lane_width=8, exhaustive=False, voc_coordinates=None):
    """Write a source spec/index ready for cli prepare; never overwrite an index."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new index output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    groups = read_groups(groups_path)
    rows = []
    source = {"name": kind.replace("-", "_"), "root": relative(root, output),
              "index": "index.jsonl", "upright_reference": False, "data_kind": "real"}
    if kind == "idd-seg":
        source.update(format="idd_label_ids", road_positive_ids=[0, 2])
        image_directories = {}
        for annotation in sorted(root.rglob("*_gtFine_labelIds.png")):
            rel = annotation.relative_to(root)
            parts = list(rel.parts)
            if "gtFine" not in parts:
                continue
            pos = parts.index("gtFine")
            official = parts[pos + 1]
            if official not in ("train", "val", "test") or official == "test":
                continue
            parts[pos] = "leftImg8bit"
            parts[-1] = parts[-1].replace("_gtFine_labelIds", "_leftImg8bit")
            expected = root.joinpath(*parts)
            if expected.parent not in image_directories:
                directory_images = {}
                if expected.parent.is_dir():
                    for candidate in expected.parent.iterdir():
                        if candidate.suffix.lower() in (".png", ".jpg", ".jpeg") and candidate.is_file():
                            directory_images.setdefault(candidate.stem, []).append(candidate)
                image_directories[expected.parent] = directory_images
            matches = sorted(image_directories[expected.parent].get(expected.stem, []))
            if not matches:
                raise FileNotFoundError(f"Missing IDD image stem {expected.stem}; root must contain gtFine and leftImg8bit")
            if len(matches) != 1:
                raise ValueError(f"Ambiguous IDD image stem {expected.stem}: {matches}")
            image = matches[0]
            sequence = "/".join(rel.parts[pos + 2:-1])
            if not sequence:
                raise ValueError("IDD sequence directory is missing")
            group, part = _group(groups, image, root, (sequence, official))
            if groups and part != official:
                raise ValueError("Do not change official IDD split membership")
            rows.append({"id": relative(image, root), "image": relative(image, root),
                         "annotation": relative(annotation, root), "sequence": group, "split": part})
    elif kind == "idd-det" and any((root / (name + ".txt")).exists() for name in ("train", "val", "test")):
        rows = _index_nested_idd_detection(root, groups, voc_coordinates)
        source.update(format="voc", voc_coordinates=voc_coordinates,
                      clip_boxes_to_image=True, skip_degenerate_voc_boxes=True,
                      class_map=IDD_CLASSES,
                      exhaustive_classes=sorted(set(IDD_CLASSES.values())) if exhaustive else [],
                      grouping_note="Exact publisher train/val IDs preserved; group CSV is an explicit selected-image allowlist. Default official/split/sequence names do not assert ride independence.")
    elif kind in ("rdd-india", "idd-det"):
        source.update(format="voc", voc_one_based=True,
                      clip_boxes_to_image=kind == "rdd-india",
                      class_map={"D40": "pothole"} if kind == "rdd-india" else IDD_CLASSES,
                      exhaustive_classes=["pothole"] if kind == "rdd-india" else sorted(set(IDD_CLASSES.values())) if exhaustive else [])
        images = {}
        for suffix in ("*.jpg", "*.png", "*.jpeg", "*.JPG"):
            for image in root.rglob(suffix):
                if image.stem in images and image != images[image.stem]:
                    raise ValueError(f"Ambiguous image stem {image.stem}; restrict --root to one release/country")
                images[image.stem] = image
        official_lists = {}
        for name in ("train", "val", "test"):
            paths = list(root.glob(f"**/ImageSets/Main/{name}.txt"))
            for path in paths:
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    if line.strip():
                        key = line.split()[0]
                        if key in official_lists and official_lists[key] != name:
                            raise ValueError("VOC official split lists overlap")
                        official_lists[key] = name
        for annotation in sorted(root.rglob("*.xml")):
            tree = ET.parse(annotation).getroot()
            if tree.tag != "annotation":
                continue
            image = images.get(annotation.stem)
            if image is None:
                raise FileNotFoundError(f"No matching image for {annotation}")
            if kind == "rdd-india" and not image.stem.lower().startswith("india"):
                raise ValueError("RDD India root contains non-India images")
            official = official_lists.get(image.stem)
            default = ("official/" + image.stem, official) if official else (
                ("unverified_collection_all_train", "train") if split == "train" else None)
            group, part = _group(groups, image, root, default)
            if official and part != official:
                raise ValueError("Do not change official VOC split membership")
            rows.append({"id": image.stem, "image": relative(image, root),
                         "annotation": relative(annotation, root), "sequence": group, "split": part})
        if split == "train" and not groups and not official_lists:
            source["grouping_note"] = "All images retained in training as one unverified collection; no held-out result is available for this source"
    elif kind == "bdd":
        if not labels or split not in ("train", "val", "test") or not tasks:
            raise ValueError("BDD requires --labels native.json --split train|val --tasks road lane detection (only verified annotated tasks)")
        if lane_width < 1:
            raise ValueError("lane_width must be positive source-image pixels")
        source.update(format="masks")
        for frame in iter_bdd_frames(labels):

            if frame.get("labels") is None:
                continue
            image = root / frame["name"]
            if not image.exists():
                raise FileNotFoundError(f"Missing BDD image {image}; --root is the selected split image directory")
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f"Unreadable image: {image}")
            group = frame.get("videoName") or image.stem
            group, part = _group(groups, image, root, (group, split))
            if part != split:
                raise ValueError("Do not change official BDD split membership")
            row = {"id": relative(image, root), "image": relative(image, root), "sequence": group, "split": part,
                   "boxes": [], "ignore_boxes": [], "detection_classes": sorted(set(BDD_CLASSES.values())) if "detection" in tasks else []}
            for label in frame["labels"]:
                box = label.get("box2d")
                if box is None or "detection" not in tasks:
                    continue
                # BDD coordinates are continuous; retain them, clamp only to image extent.
                h, w = pixels.shape[:2]
                xyxy = [float(np.clip(box[k], 0, w if k.startswith("x") else h)) for k in ("x1", "y1", "x2", "y2")]
                attrs = label.get("attributes") or {}
                category = BDD_CLASSES.get(label.get("category"))
                if not np.isfinite(xyxy).all():
                    raise ValueError(f"Nonfinite BDD box in {image}")
                if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                    if frame.get("annotation_format") != "bdd_legacy_2018":
                        raise ValueError(f"Empty BDD box in {image}")
                    # No geometric region can be recovered from a degenerate box.
                    # For a mapped class, all negatives/AP for that frame/class
                    # become unknown while other actual positives remain usable.
                    if category in row["detection_classes"]:
                        row["detection_classes"].remove(category)
                    row.setdefault("conversion_warnings", []).append({
                        "operation": "skip_degenerate_legacy_box", "category": label.get("category"),
                        "object_id": label.get("id"), "original_box2d": box, "clipped_xyxy": xyxy,
                        "removed_exhaustive_class": category})
                    continue
                if category and not attrs.get("crowd", False) and not attrs.get("ignored", False):
                    row["boxes"].append({"class": category, "xyxy": xyxy})
                else:
                    row["ignore_boxes"].append(xyxy)
            for task, mask in bdd_masks(frame, pixels.shape[:2], tasks, lane_width).items():
                key = hashlib.sha256(frame["name"].encode()).hexdigest()[:20]
                path = output / "masks" / f"{task}_{key}.png"
                path.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(path), mask):
                    raise OSError(path)
                row[task + "_mask"] = relative(path, root)
            rows.append(row)
        source["annotation_contract"] = {"tasks": tasks, "lane_width_source_pixels": lane_width,
                                         "lane_raster": "binary sampled cubic curves; not official benchmark raster",
                                         "labels_path": relative(labels, output),
                                         "legacy_path_semantics": "publisher MOVETO then LINETO/CURVE4 consuming 3 vertices; areas closed"}
    else:
        raise ValueError(f"Unknown native dataset: {kind}")
    if not rows:
        raise ValueError("No labeled samples found; verify the release layout and selected root")
    (output / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (output / "sources.json").write_text(json.dumps({"sources": [source]}, indent=2), encoding="utf-8")
    return {"samples": len(rows), "spec": str(output / "sources.json"),
            "note": "Inspect representative masks and coverage; then prepare and audit --hashes."}


def merge_specs(specs, output):
    """Combine native specs without changing their relative roots/indexes."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    sources = []
    for path in specs:
        path = Path(path).resolve()
        for original in json.loads(path.read_text(encoding="utf-8-sig"))["sources"]:
            source = dict(original)
            for key in ("root", "index"):
                source[key] = relative(path.parent / source[key], output.parent)
            sources.append(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"sources": sources}, indent=2), encoding="utf-8")
    return {"sources": len(sources), "spec": str(output)}
