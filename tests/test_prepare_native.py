import json
import shutil

import cv2
import numpy as np
import pytest

from aurora2w.data import read_manifest, audit_records
from aurora2w.native import build_native, sample_curve, bdd_masks, merge_specs
from aurora2w.prepare import prepare_spec


def image(path, seed=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.random.default_rng(seed).integers(0, 255, (32, 64, 3), dtype=np.uint8))


def test_per_image_coverage_and_portability(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    image(root / "one.png")
    image(root / "two.png", 2)
    rows = [{"id": str(i), "image": name, "sequence": str(i), "split": split,
             "detection_classes": coverage, "boxes": [{"class": "pothole", "xyxy": [8, 8, 30, 20]}]}
            for i, (name, split, coverage) in enumerate((("one.png", "train", ["pothole"]), ("two.png", "val", [])))]
    (root / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (root / "spec.json").write_text(json.dumps({"sources": [{"name": "mask", "root": ".", "index": "index.jsonl", "format": "masks", "exhaustive_classes": ["pothole"], "data_kind": "real"}]}))
    prepare_spec(root / "spec.json", root / "prepared" / "manifest.jsonl")
    raw = (root / "prepared" / "manifest.jsonl").read_text()
    assert "../one.png" in raw
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    records = read_manifest(moved / "prepared" / "manifest.jsonl")
    assert records[0]["detection_classes"] == ["pothole"] and records[1]["detection_classes"] == []
    assert all(r["data_kind"] == "real" for r in records)
    report = audit_records(records, check_hashes=True, image_size=[64, 64])
    assert report["positive_only_images"]["pothole"] == 1


def test_idd_native_original_labels(tmp_path):
    root = tmp_path / "idd"
    image(root / "leftImg8bit" / "train" / "ride1" / "frame_leftImg8bit.png")
    path = root / "gtFine" / "train" / "ride1" / "frame_gtFine_labelIds.png"
    path.parent.mkdir(parents=True)
    mask = np.zeros((32, 64), np.uint8)
    mask[:8] = 35
    mask[8:12] = 1  # parking remains unknown
    mask[12:16] = 3
    cv2.imwrite(str(path), mask)
    built = build_native("idd-seg", root, tmp_path / "idx")
    prepare_spec(built["spec"], tmp_path / "prepared" / "manifest.jsonl")
    record = read_manifest(tmp_path / "prepared" / "manifest.jsonl")[0]
    road = cv2.imread(record["road_mask"], 0)
    assert (road[:12] == 255).all() and (road[12:16] == 0).all() and (road[16:] == 1).all()
    assert record["sequence"] == "ride1" and record["detection_classes"] == []


def test_bdd_cubic_curves_and_annotation_scope(tmp_path):
    curve = {"vertices": [[4, 28], [4, 2], [60, 2], [60, 28]], "types": "LCCC", "closed": False}
    points = sample_curve(curve)
    assert points[:, 1].min() < 10 and tuple(points[-1]) == (60, 28)
    with pytest.raises(ValueError, match="triples"):
        sample_curve({"vertices": [[0, 0], [1, 2]], "types": "LC"})
    frame = {"name": "clip.jpg", "labels": [
        {"category": "lane", "attributes": {"laneType": "single white"}, "poly2d": [curve]},
        {"category": "drivable area", "attributes": {"areaType": "direct"}, "poly2d": [
            {"vertices": [[0, 16], [63, 16], [63, 31], [0, 31]], "types": "LLLL", "closed": True}]},
        {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}}]}
    masks = bdd_masks(frame, (32, 64), ["road", "lane"], 2)
    assert masks["road"][25, 32] == 1 and masks["road"][2, 32] == 0
    assert masks["lane"][8, 32] == 1  # cubic arc, not a line joining endpoints
    assert bdd_masks({"labels": None}, (32, 64), ["road", "lane"]) == {}
    root = tmp_path / "images"
    image(root / "clip.jpg")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([frame]))
    built = build_native("bdd", root, tmp_path / "bdd_idx", labels=labels, split="train", tasks=["road", "lane", "detection"])
    prepare_spec(built["spec"], tmp_path / "prepared" / "manifest.jsonl")
    record = read_manifest(tmp_path / "prepared" / "manifest.jsonl")[0]
    assert "car" in record["detection_classes"] and "pothole" not in record["detection_classes"]
    assert len(record["boxes"]) == 1


def test_rdd_requires_groups_and_preserves_negative_images(tmp_path):
    root = tmp_path / "rdd"
    image(root / "JPEGImages" / "India_0001.jpg")
    annotations = root / "Annotations"
    annotations.mkdir()
    (annotations / "India_0001.xml").write_text("<annotation><size><width>64</width><height>32</height></size></annotation>")
    with pytest.raises(ValueError, match="--groups"):
        build_native("rdd-india", root, tmp_path / "bad")
    groups = tmp_path / "groups.csv"
    groups.write_text("image,sequence,split\nJPEGImages/India_0001.jpg,collection_a,val\n")
    built = build_native("rdd-india", root, tmp_path / "good", groups_path=groups)
    merged = merge_specs([built["spec"]], tmp_path / "combined" / "sources.json")
    prepare_spec(merged["spec"], tmp_path / "prepared" / "manifest.jsonl")
    record = read_manifest(tmp_path / "prepared" / "manifest.jsonl")[0]
    assert record["boxes"] == [] and record["detection_classes"] == ["pothole"]


def test_voc_clipping_retains_source_coordinate_warning(tmp_path):
    image(tmp_path / "image.png")
    (tmp_path / "annotation.xml").write_text("<annotation><object><name>D40</name><bndbox><xmin>0</xmin><ymin>9</ymin><xmax>20</xmax><ymax>24</ymax></bndbox></object></annotation>")
    (tmp_path / "index.jsonl").write_text(json.dumps({"id": "border", "image": "image.png", "annotation": "annotation.xml", "sequence": "seq", "split": "train"}))
    (tmp_path / "source.json").write_text(json.dumps({"sources": [{"name": "rdd", "root": ".", "index": "index.jsonl", "format": "voc", "voc_one_based": True, "clip_boxes_to_image": True, "class_map": {"D40": "pothole"}, "exhaustive_classes": ["pothole"]}]}))
    prepare_spec(tmp_path / "source.json", tmp_path / "prepared" / "manifest.jsonl")
    record = read_manifest(tmp_path / "prepared" / "manifest.jsonl")[0]
    assert record["boxes"][0]["xyxy"] == [0, 8, 20, 24]
    assert record["conversion_warnings"][0]["original_xyxy"] == [-1, 8, 20, 24]


def test_audit_reports_unassigned_tiny_pothole(tmp_path):
    image(tmp_path / "one.png")
    record = {"id": "one", "source": "fixture", "sequence": "one", "split": "train", "image": str(tmp_path / "one.png"),
              "boxes": [{"class": "pothole", "xyxy": [6, 6, 10, 10]}], "detection_classes": ["pothole"]}
    # 32x64 -> 64x64 letterbox adds 16px vertical offset, still between stride8 centers.
    report = audit_records([record], image_size=[64, 64])
    assert report["small_box_assignment"]["unassigned_instances"] == {"pothole": 1}

@pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".JPG"])
def test_idd_original_jpeg_stem_is_kept_without_reencoding(tmp_path, extension):
    root = tmp_path / "idd"
    original = root / "leftImg8bit" / "train" / "245" / ("frame0839_leftImg8bit" + extension)
    image(original)
    before = original.read_bytes()
    mask = root / "gtFine" / "train" / "245" / "frame0839_gtFine_labelIds.png"
    mask.parent.mkdir(parents=True)
    cv2.imwrite(str(mask), np.zeros((32, 64), np.uint8))
    built = build_native("idd-seg", root, tmp_path / "idx")
    rows = [json.loads(line) for line in (tmp_path / "idx" / "index.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert (root / rows[0]["image"]).resolve() == original.resolve()
    assert original.read_bytes() == before
    assert rows[0]["sequence"] == "245"


def test_idd_ambiguous_image_extensions_fail_instead_of_picking_one(tmp_path):
    root = tmp_path / "idd"
    image(root / "leftImg8bit" / "train" / "245" / "frame_leftImg8bit.png")
    image(root / "leftImg8bit" / "train" / "245" / "frame_leftImg8bit.jpg")
    mask = root / "gtFine" / "train" / "245" / "frame_gtFine_labelIds.png"
    mask.parent.mkdir(parents=True)
    cv2.imwrite(str(mask), np.zeros((32, 64), np.uint8))
    with pytest.raises(ValueError, match="Ambiguous IDD image stem"):
        build_native("idd-seg", root, tmp_path / "idx")


def test_prepared_manifest_keeps_explicit_release_and_partition_identity(tmp_path):
    image(tmp_path / "frame.jpg")
    (tmp_path / "index.jsonl").write_text(json.dumps({"id": "one", "image": "frame.jpg", "sequence": "perceptual_abc", "split": "val", "detection_classes": ["pothole"], "boxes": []}))
    source = {"name": "rdd_india", "root": ".", "index": "index.jsonl", "format": "masks", "data_kind": "real",
              "source_release": "RDD2022_India", "partition_protocol": "image_holdout", "partition_group_kind": "perceptual_not_ride", "partition_artifact_sha256": "a" * 64}
    (tmp_path / "spec.json").write_text(json.dumps({"sources": [source]}))
    prepare_spec(tmp_path / "spec.json", tmp_path / "prepared" / "manifest.jsonl")
    record = read_manifest(tmp_path / "prepared" / "manifest.jsonl")[0]
    assert record["source_release"] == "RDD2022_India"
    assert record["partition_group_kind"] == "perceptual_not_ride"
    assert record["partition_artifact_sha256"] == "a" * 64


def test_binary_mask_audit_rejects_an_unexpected_pixel_value(tmp_path):
    image(tmp_path / "one.png")
    mask = np.zeros((32, 64), np.uint8)
    mask[12, 24] = 17
    cv2.imwrite(str(tmp_path / "road.png"), mask)
    row = {"id": "one", "source": "fixture", "sequence": "one", "split": "train",
           "image": str(tmp_path / "one.png"), "road_mask": str(tmp_path / "road.png"),
           "detection_classes": ["pothole"], "boxes": []}
    with pytest.raises(ValueError, match="values 0,1,255"):
        audit_records([row])
    mask[12, 24] = 255
    mask[0, 0] = 1
    cv2.imwrite(str(tmp_path / "road.png"), mask)
    assert audit_records([row])["tasks"]["road"] == 1


def test_nested_idd_ids_preserve_repeated_basenames_and_selected_allowlist(tmp_path):
    root = tmp_path / 'idd'
    ids = ['frontFar/recording_A/frame', 'highquality_16k/recording_B_part_2/frame']
    for index, identifier in enumerate(ids):
        image(root / 'JPEGImages' / (identifier + '.jpg'), index + 1)
        annotation = root / 'Annotations' / (identifier + '.xml')
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text('<annotation><size><width>64</width><height>32</height></size><object><name>car</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>64</xmax><ymax>32</ymax></bndbox></object></annotation>')
    (root / 'train.txt').write_text(ids[0] + '\nfrontFar/unselected/frame\n')
    (root / 'val.txt').write_text(ids[1] + '\n')
    groups = tmp_path / 'groups.csv'
    groups.write_text('image,sequence,split\nJPEGImages/' + ids[0] + '.jpg,recording_A,train\nJPEGImages/' + ids[1] + '.jpg,recording_B,val\n')
    built = build_native('idd-det', root, tmp_path / 'index', groups_path=groups,
                         exhaustive=True, voc_coordinates='one_based_inclusive')
    prepare_spec(built['spec'], tmp_path / 'prepared' / 'manifest.jsonl')
    rows = read_manifest(tmp_path / 'prepared' / 'manifest.jsonl')
    assert {row['id'] for row in rows} == set(ids)
    assert all(row['boxes'][0]['xyxy'] == [0, 0, 64, 32] for row in rows)
    assert all('autorickshaw' in row['detection_classes'] for row in rows)
    # Publisher val membership must not be changed by an external selection CSV.
    groups.write_text(groups.read_text().replace('recording_B,val', 'recording_B,train'))
    with pytest.raises(ValueError, match='publisher IDD split membership'):
        build_native('idd-det', root, tmp_path / 'badindex', groups_path=groups,
                     voc_coordinates='one_based_inclusive')


def test_nested_idd_requires_reviewed_coordinate_origin(tmp_path):
    root = tmp_path / 'idd'
    root.mkdir()
    (root / 'train.txt').write_text('category/recording/frame\n')
    with pytest.raises(ValueError, match='explicitly reviewed'):
        build_native('idd-det', root, tmp_path / 'index')


@pytest.mark.parametrize('convention,bounds,expected', [
    ('one_based_inclusive', [1, 1, 64, 32], [0, 0, 64, 32]),
    ('zero_based_inclusive', [0, 0, 63, 31], [0, 0, 64, 32]),
    ('zero_based_half_open', [0, 0, 64, 32], [0, 0, 64, 32]),
])
def test_voc_coordinate_conventions_map_to_same_pixel_edges(tmp_path, convention, bounds, expected):
    image(tmp_path / 'one.png')
    box = ''.join(f'<{name}>{value}</{name}>' for name, value in zip(['xmin','ymin','xmax','ymax'], bounds))
    (tmp_path / 'one.xml').write_text('<annotation><object><name>car</name><bndbox>' + box + '</bndbox></object></annotation>')
    (tmp_path / 'index.jsonl').write_text(json.dumps({'id': 'one', 'image': 'one.png', 'annotation': 'one.xml', 'sequence': 'one', 'split': 'train'}))
    (tmp_path / 'source.json').write_text(json.dumps({'sources': [{'name': 'idd_det', 'root': '.', 'index': 'index.jsonl', 'format': 'voc', 'voc_coordinates': convention, 'class_map': {'car': 'car'}}]}))
    prepare_spec(tmp_path / 'source.json', tmp_path / 'prepared' / 'manifest.jsonl')
    assert read_manifest(tmp_path / 'prepared' / 'manifest.jsonl')[0]['boxes'][0]['xyxy'] == expected


def test_idd_ignored_only_mask_still_rejected_as_no_supervision(tmp_path):
    root = tmp_path / 'idd'
    image(root / 'leftImg8bit' / 'val' / '18' / 'frame_leftImg8bit.png')
    path = root / 'gtFine' / 'val' / '18' / 'frame_gtFine_labelIds.png'
    path.parent.mkdir(parents=True)
    cv2.imwrite(str(path), np.full((32, 64), 35, np.uint8))
    built = build_native('idd-seg', root, tmp_path / 'index')
    with pytest.raises(ValueError, match='No supervised task'):
        prepare_spec(built['spec'], tmp_path / 'prepared' / 'manifest.jsonl')


def test_degenerate_reviewed_voc_object_removes_only_its_class_coverage(tmp_path):
    image(tmp_path / 'one.png')
    xml = '<annotation><object><name>car</name><bndbox><xmin>5</xmin><ymin>1</ymin><xmax>5</xmax><ymax>10</ymax></bndbox></object><object><name>car</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>40</xmax><ymax>20</ymax></bndbox></object></annotation>'
    (tmp_path / 'one.xml').write_text(xml)
    (tmp_path / 'index.jsonl').write_text(json.dumps({'id': 'one', 'image': 'one.png', 'annotation': 'one.xml', 'sequence': 'one', 'split': 'val'}))
    source = {'name': 'idd_det', 'root': '.', 'index': 'index.jsonl', 'format': 'voc',
              'voc_coordinates': 'zero_based_half_open', 'skip_degenerate_voc_boxes': True,
              'class_map': {'car': 'car', 'truck': 'truck'}, 'exhaustive_classes': ['car', 'truck']}
    (tmp_path / 'source.json').write_text(json.dumps({'sources': [source]}))
    prepare_spec(tmp_path / 'source.json', tmp_path / 'prepared' / 'manifest.jsonl')
    record = read_manifest(tmp_path / 'prepared' / 'manifest.jsonl')[0]
    assert record['detection_classes'] == ['truck']
    assert record['boxes'] == [{'class': 'car', 'xyxy': [20, 10, 40, 20]}]
    assert record['conversion_warnings'][0]['removed_exhaustive_class'] == 'car'
