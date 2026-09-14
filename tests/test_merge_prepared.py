import json
import cv2
import numpy as np
import pytest
from aurora2w.data import read_manifest
from aurora2w.prepare import write_manifest
from scripts.merge_prepared import merge_prepared


def fixture(tmp_path, name, split, seed):
    image = tmp_path / (name + '.png')
    cv2.imwrite(str(image), np.random.default_rng(seed).integers(0, 255, (16, 32, 3), dtype=np.uint8))
    row = {'source': 'rdd_india', 'id': name, 'image': str(image), 'sequence': name, 'split': split,
           'data_kind': 'real', 'detection_classes': ['pothole'], 'boxes': []}
    manifest = tmp_path / name / 'manifest.jsonl'
    write_manifest([row], manifest)
    return manifest


def test_merge_rejects_cross_split_same_pixels_and_does_not_write(tmp_path):
    train = fixture(tmp_path, 'a', 'train', 1)
    val = fixture(tmp_path, 'b', 'val', 1)
    output = tmp_path / 'out' / 'manifest.jsonl'
    with pytest.raises(ValueError, match='Duplicate image pixels'):
        merge_prepared([train, val], output)
    assert not output.exists()


def test_merge_provenance_checks_identity_and_keeps_unknown_annotations(tmp_path):
    train = fixture(tmp_path, 'a', 'train', 1)
    val = fixture(tmp_path, 'b', 'val', 2)
    item = {'id': 'a', 'image': 'a.png', 'sequence': 'a', 'split': 'train'}
    index = tmp_path / 'index.jsonl'
    index.write_text(json.dumps(item))
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'sources': [{'name': 'rdd_india', 'root': '.', 'index': 'index.jsonl',
                                           'source_release': 'explicit', 'detection_classes': ['car']}]}))
    output = tmp_path / 'out' / 'manifest.jsonl'
    merge_prepared([train, val], output, [spec])
    rows = read_manifest(output)
    assert rows[0]['source_release'] == 'explicit'
    assert rows[0]['detection_classes'] == ['pothole']
    assert 'road_mask' not in rows[0] and rows[0]['boxes'] == []
    item['split'] = 'val'
    index.write_text(json.dumps(item))
    with pytest.raises(ValueError, match='Provenance identity mismatch'):
        merge_prepared([train, val], tmp_path / 'bad' / 'manifest.jsonl', [spec])


def test_merge_audits_tiny_target_assignment_at_requested_input_resolution(tmp_path):
    original = fixture(tmp_path, 'small', 'train', 3)
    record = read_manifest(original)[0]
    # A16x32image becomes32x64with16pixelstop padding; x6..10falls
    # between stride8centers4and12, so no positive target is invented.
    record['boxes'] = [{'class': 'pothole', 'xyxy': [3, 3, 5, 5]}]
    input_path = tmp_path / 'labeled' / 'manifest.jsonl'
    write_manifest([record], input_path)
    report = merge_prepared([input_path], tmp_path / 'merged' / 'manifest.jsonl', image_size=[64, 64])
    assert report['audit']['small_box_assignment']['unassigned_instances'] == {'pothole': 1}
    assert read_manifest(tmp_path / 'merged' / 'manifest.jsonl')[0]['boxes'][0]['xyxy'] == [3, 3, 5, 5]
