"""BDD 2018 data conversion only; no model or optimizer execution."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from aurora2w.native import (normalize_legacy_bdd_poly, normalize_legacy_bdd_frame,
                             iter_bdd_frames, sample_curve, bdd_masks, build_native)


def publisher_segments(poly, closed):
    points = [row[:2] for row in poly]
    codes = [MplPath.LINETO if row[2] == "L" else MplPath.CURVE4 for row in poly]
    codes[0] = MplPath.MOVETO
    if closed:
        points.append(points[0])
        codes.append(MplPath.CLOSEPOLY)
    return list(MplPath(points, codes).iter_segments(curves=True, simplify=False, remove_nans=False))


@pytest.mark.parametrize("poly,closed", [
    ([[4, 28, "C"], [4, 2, "C"], [60, 2, "C"], [60, 28, "L"]], False),
    ([[0, 0, "L"], [20, 0, "L"], [20, 20, "L"], [0, 20, "L"]], True),
    ([[0, 0, "L"], [20, 0, "C"], [20, 20, "C"]], True),
])
def test_legacy_cubic_controls_match_independent_publisher_path(poly, closed):
    modern = normalize_legacy_bdd_poly(poly, closed)
    original = [(points, code) for points, code in publisher_segments(poly, closed) if code != MplPath.CLOSEPOLY]
    converted_poly = [list(point) + [code] for point, code in zip(modern["vertices"], modern["types"])]
    converted = publisher_segments(converted_poly, False)
    # An explicit line back to start is equivalent to publisher CLOSEPOLY.
    if closed and len(converted) == len(original) + 1:
        assert converted[-1][1] == MplPath.LINETO
        np.testing.assert_array_equal(converted[-1][0], poly[0][:2])
        converted = converted[:-1]
    assert [code for _, code in original] == [code for _, code in converted]
    for (expected, _), (actual, _) in zip(original, converted):
        np.testing.assert_array_equal(actual, expected)
    if not closed:
        points = sample_curve(modern)
        assert tuple(points[-1]) == (60, 28)
        assert points[:, 1].min() < 10


def test_legacy_lane_curbs_are_unknown_and_areas_are_closed():
    document = {"name": "clip", "frames": [{"timestamp": 10000, "objects": [
        {"category": "area/drivable", "poly2d": [[0, 16, "L"], [63, 16, "L"], [63, 31, "L"], [0, 31, "L"]]},
        {"category": "lane/single white", "poly2d": [[4, 28, "C"], [4, 2, "C"], [60, 2, "C"], [60, 28, "L"]]},
        {"category": "lane/road curb", "poly2d": [[2, 1, "L"], [2, 28, "L"]]},
        {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}},
    ]}]}
    frame = normalize_legacy_bdd_frame(document)
    assert frame["name"] == "clip.jpg"
    masks = bdd_masks(frame, (32, 64), ["road", "lane"], 2)
    assert masks["road"][25, 32] == 1
    assert masks["lane"][8, 32] == 1
    assert masks["lane"][12, 2] == 255
    assert masks["lane"][30, 32] == 0


def test_per_image_directory_adapter_preserves_missing_vs_empty_and_split(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    for name, objects in (("annotated", []), ("unknown", None)):
        document = {"name": name, "frames": [{"objects": objects}]}
        (root / (name + ".json")).write_text(json.dumps(document))
        cv2.imwrite(str(root / (name + ".jpg")), np.full((32, 64, 3), 128, np.uint8))
    assert len(list(iter_bdd_frames(root))) == 2
    result = build_native("bdd", root, tmp_path / "index", labels=root, split="train", tasks=["road", "lane", "detection"])
    assert result["samples"] == 1
    rows = [json.loads(line) for line in (tmp_path / "index" / "index.jsonl").read_text().splitlines()]
    assert rows[0]["split"] == "train" and rows[0]["sequence"] == "annotated"
    assert "car" in rows[0]["detection_classes"] and "pothole" not in rows[0]["detection_classes"]
    assert cv2.imread(str(root / rows[0]["road_mask"]), 0).max() == 0


def test_malformed_legacy_curve_and_mismatched_document_fail():
    with pytest.raises(ValueError, match="lacks"):
        normalize_legacy_bdd_poly([[0, 0, "L"], [1, 2, "C"]])
    with pytest.raises(ValueError, match="differs"):
        normalize_legacy_bdd_frame({"name": "other", "frames": [{"objects": []}]}, "clip.json")

def test_legacy_unknown_road_is_ignored_even_when_overlapping_positive():
    polygon = [[0, 0, "L"], [30, 0, "L"], [30, 30, "L"], [0, 30, "L"]]
    unknown = [[8, 8, "L"], [16, 8, "L"], [16, 16, "L"], [8, 16, "L"]]
    frame = normalize_legacy_bdd_frame({"name": "unknown", "frames": [{"objects": [
        {"category": "area/unknown", "poly2d": unknown},
        {"category": "area/drivable", "poly2d": polygon},
    ]}]})
    mask = bdd_masks(frame, (32, 64), ["road"])["road"]
    assert mask[12, 12] == 255
    assert mask[4, 4] == 1
    assert mask[0, 50] == 0

def test_degenerate_legacy_box_removes_only_its_class_negative_supervision(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    cv2.imwrite(str(root / "clip.jpg"), np.full((32, 64, 3), 128, np.uint8))
    document = {"name": "clip", "frames": [{"objects": [
        {"id": 1, "category": "car", "box2d": {"x1": 5, "y1": 8, "x2": 15, "y2": 8}},
        {"id": 2, "category": "traffic sign", "box2d": {"x1": 20, "y1": 8, "x2": 25, "y2": 8}},
        {"id": 3, "category": "car", "box2d": {"x1": 5, "y1": 12, "x2": 25, "y2": 28}},
    ]}]}
    (root / "clip.json").write_text(json.dumps(document))
    built = build_native("bdd", root, tmp_path / "idx", labels=root, split="train", tasks=["road", "lane", "detection"])
    row = json.loads((tmp_path / "idx" / "index.jsonl").read_text())
    assert "car" not in row["detection_classes"]
    assert "person" in row["detection_classes"]
    assert len(row["boxes"]) == 1 and len(row["conversion_warnings"]) == 2
    from aurora2w.prepare import prepare_spec
    from aurora2w.data import read_manifest
    prepare_spec(built["spec"], tmp_path / "prepared" / "manifest.jsonl")
    prepared = read_manifest(tmp_path / "prepared" / "manifest.jsonl")[0]
    assert prepared["conversion_warnings"] == row["conversion_warnings"]
    assert "car" not in prepared["detection_classes"]
