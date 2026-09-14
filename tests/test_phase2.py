import copy
import math

import numpy as np
import pytest
import torch

from aurora2w.config import CLASS_NAMES, load_config
from aurora2w.data import RoadDataset, read_manifest, audit_records, collate
from aurora2w.geometry import transform_boxes, roll_matrix
from aurora2w.idfa import RollConditionedConv
from aurora2w.losses import multitask_loss, assign_targets
from aurora2w.model import MultiTaskNet, ExportModel, flatten_detection
from aurora2w.sensors import AttitudeTimeline, slerp, camera_roll
from aurora2w.smoke import make_fixture


@pytest.fixture
def config():
    torch.set_num_threads(2)
    c = load_config("configs/phase2.json")
    c.update(image_size=[128, 192], channels=32, neck_repeats=1, pretrained_backbone=False)
    return c


def test_rotated_box_encloses_all_corners():
    box, keep = transform_boxes([[20, 30, 40, 50]], roll_matrix(100, 100, math.pi/2), 100, 100)
    np.testing.assert_allclose(box[0], [50, 20, 70, 40], atol=1e-5)
    assert keep[0]


def test_idfa_phase1_prior_and_confidence_fallback():
    torch.manual_seed(0)
    block = RollConditionedConv(8).eval()
    x = torch.randn(2, 8, 8, 10)
    offsets = block.offsets(x, torch.tensor([[math.pi/2, 1], [1., 0]]))
    # Top-left tap (-1,-1) -> (+1,-1): dy=0, dx=2.
    torch.testing.assert_close(offsets[0, :2, 4, 4], torch.tensor([0., 2.]), atol=1e-6, rtol=0)
    assert offsets[1].count_nonzero() == 0
    ordinary = RollConditionedConv(8, deformable=False).eval()
    ordinary.conv.load_state_dict(block.conv.state_dict())
    ordinary.norm.load_state_dict(block.norm.state_dict())
    torch.testing.assert_close(block(x, torch.zeros(2, 2)), ordinary(x, torch.zeros(2, 2)), atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        block.refine[-1].bias.fill_(100)
    offset = block.offsets(x, torch.tensor([[0., 1], [0., 1]]))
    assert offset.abs().max() <= .5


def test_quaternion_wrap_and_missing_samples():
    q = np.array([0., 0., 0., 1.])
    np.testing.assert_allclose(slerp(q, -q, .5), q)
    timeline = AttitudeTimeline([0, .01, 1.], [[1, 0, 0, 0]] * 3)
    assert timeline.at(-1) is None and timeline.at(.5) is None
    np.testing.assert_allclose(timeline.at(.005), [1, 0, 0, 0])
    rdc = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    roll, confidence = camera_roll([1, 0, 0, 0], rdc)
    assert roll == 0 and confidence == 1
    with pytest.raises(ValueError):
        camera_roll([1, 0, 0, 0], np.ones((3, 3)))


def test_roll_csv_wrap_and_out_of_range(tmp_path):
    from aurora2w.inference import RollTimeline
    path = tmp_path / "roll.csv"
    path.write_text(f"timestamp_s,roll_rad,confidence\n0,{math.radians(179)},1\n0.04,{math.radians(-179)},1\n")
    timeline = RollTimeline(path)
    roll, confidence = timeline.at(.02)
    assert abs(roll - math.pi) < 1e-6 and confidence == 1
    assert timeline.at(.05) == (0., 0.)


def test_idfa_training_rejects_missing_roll(config, tmp_path):
    from aurora2w.engine import train
    records = read_manifest(make_fixture(tmp_path / "data"))
    for r in records:
        r["upright_reference"] = False
    config["alignment"] = "idfa"
    with pytest.raises(ValueError, match="no trusted roll"):
        train(config, records, tmp_path / "run", max_steps=1)


@pytest.mark.parametrize("alignment", ["none", "idfa"])
def test_full_model_backward_and_initialization(config, tmp_path, alignment):
    records = read_manifest(make_fixture(tmp_path))
    config["alignment"] = alignment
    images, imu, targets = collate([RoadDataset(records, config)[i] for i in (0, 1)])
    model = MultiTaskNet(config)
    if alignment == "idfa":
        assert model.alignment["3"].refine[-1].weight.count_nonzero() == 0
    outputs = model(images, imu)
    assert outputs["road_logits"].shape == (2, 1, 32, 48)
    assert outputs["lane_logits"].shape == (2, 1, 32, 48)
    losses = multitask_loss(outputs, targets, config["loss_weights"])
    assert all(torch.isfinite(v) for v in losses.values())
    losses["total"].backward()
    for head in (model.road_head, model.lane_head, model.detector.regressor, model.detector.classifier):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    if alignment == "idfa":
        assert model.alignment["3"].refine[-1].weight.grad.abs().sum() > 0
        with pytest.raises(ValueError, match="custom backend"):
            ExportModel(model)


def test_missing_labels_have_zero_gradient(config, tmp_path):
    records = read_manifest(make_fixture(tmp_path))
    record = records[0]
    record.pop("road_mask")
    record.pop("lane_mask")
    record["detection_classes"] = ["pothole"]
    record["boxes"] = [o for o in record["boxes"] if o["class"] == "pothole"]
    image, imu, target = RoadDataset([record], config)[0]
    model = MultiTaskNet(config).eval()
    out = model(image[None], imu[None])
    for level in out["detection"]:
        level["class_logits"].retain_grad()
    out["road_logits"].retain_grad()
    loss = multitask_loss(out, [target], config["loss_weights"])
    loss["total"].backward()
    assert out["road_logits"].grad.count_nonzero() == 0
    for level in out["detection"]:
        assert level["class_logits"].grad[:, :8].count_nonzero() == 0
    assert any(level["class_logits"].grad[:, 8].abs().sum() > 0 for level in out["detection"])


def test_dataset_validation_determinism_and_split_leakage(config, tmp_path):
    records = read_manifest(make_fixture(tmp_path))
    audit_records(records, check_hashes=True)
    dataset = RoadDataset(records, config, training=False)
    first, second = dataset[0], dataset[0]
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    leaked = copy.deepcopy(records)
    leaked[4]["sequence"] = leaked[0]["sequence"]
    with pytest.raises(ValueError, match="Sequence crosses"):
        audit_records(leaked)


def test_unlabeled_classes_accept_positives_but_no_negatives(config, tmp_path):
    records = read_manifest(make_fixture(tmp_path))
    records[0]["detection_classes"] = []
    image, imu, target = RoadDataset(records, config)[0]
    out = MultiTaskNet(config).eval()(image[None], imu[None])
    cls, _, _, points, strides = flatten_detection(out["detection"])
    truth, valid, matched = assign_targets(points, strides, target, len(CLASS_NAMES))
    assert valid.any() and (matched >= 0).any()
    assert torch.equal(valid, truth.bool())


def test_metrics_perfect_boxes_and_unknown_classes(config, tmp_path):
    from aurora2w.metrics import Metrics
    records = read_manifest(make_fixture(tmp_path))
    _, _, target = RoadDataset(records, config)[0]
    target["known_classes"][:] = False
    target["known_classes"][8] = True
    pothole = target["boxes"][target["labels"] == 8]
    prediction = {"boxes": torch.cat((pothole, pothole)), "labels": torch.tensor([8, 5]), "scores": torch.tensor([.9, .95])}
    output = {k + "_logits": (target[k][None, None].float() * 2 - 1) * 20 for k in ("road", "lane")}
    metric = Metrics()
    metric.update(output, [prediction], [target])
    result = metric.compute()
    assert result["detection"]["pothole"]["ap50_95"] == 1
    assert result["detection"]["pothole"]["recall_at_score_0_3_iou_0_5"] == 1
    assert result["detection"]["car"]["labeled_images"] == 0
    assert result["segmentation"]["lane"]["foreground_iou"] == 1


@pytest.mark.parametrize("format_name", ["yolo", "voc", "idd_label_ids", "masks"])
def test_source_converters_respect_annotation_scope(tmp_path, format_name):
    import cv2
    import json
    from aurora2w.prepare import prepare_spec
    image = np.full((32, 64, 3), 100, np.uint8)
    cv2.imwrite(str(tmp_path / "image.png"), image)
    item = {"id": "test", "sequence": "seq1", "split": "train", "image": "image.png"}
    source = {"name": "test_source", "root": ".", "index": "index.jsonl", "format": format_name}
    if format_name == "yolo":
        (tmp_path / "labels.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        item["annotation"] = "labels.txt"
        source.update(class_map={"0": "pothole"}, exhaustive_classes=["pothole"])
    elif format_name == "voc":
        (tmp_path / "labels.xml").write_text("<annotation><size><width>64</width><height>32</height></size><object><name>D40</name><bndbox><xmin>17</xmin><ymin>9</ymin><xmax>48</xmax><ymax>24</ymax></bndbox></object></annotation>")
        item["annotation"] = "labels.xml"
        source.update(class_map={"D40": "pothole"}, exhaustive_classes=["pothole"])
    else:
        mask = np.zeros((32, 64), np.uint8)
        if format_name == "idd_label_ids":
            mask[:8] = 35
            item["annotation"] = "mask.png"
        else:
            mask[:, 30:34] = 255
            item["lane_mask"] = "mask.png"
            source["mask_value_map"] = {"lane": {"0": 0, "255": 1}}
        cv2.imwrite(str(tmp_path / "mask.png"), mask)
    (tmp_path / "index.jsonl").write_text(json.dumps(item) + "\n")
    (tmp_path / "sources.json").write_text(json.dumps({"sources": [source]}))
    report = prepare_spec(tmp_path / "sources.json", tmp_path / "out" / "manifest.jsonl")
    assert report["samples"] == 1
    record = read_manifest(tmp_path / "out" / "manifest.jsonl")[0]
    if format_name in ("yolo", "voc"):
        np.testing.assert_allclose(record["boxes"][0]["xyxy"], [16, 8, 48, 24])
        assert record["detection_classes"] == ["pothole"]
    elif format_name == "masks":
        assert set(np.unique(cv2.imread(record["lane_mask"], 0))) == {0, 1}
    else:
        assert set(np.unique(cv2.imread(record["road_mask"], 0))) == {1, 255}
