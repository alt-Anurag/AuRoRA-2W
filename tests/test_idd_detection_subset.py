from aurora2w.native import IDD_CLASSES
from scripts.prepare_idd_detection_subset import select_detection_rows
import pytest


def row(identity, group, split):
    return {'id': identity, 'conservative_group': group, 'official_split': split,
            'capture_category': 'frontFar', 'positive_instances': {name: 1 for name in set(IDD_CLASSES.values())},
            'coordinate_counts': {'boxes': 8}}


def test_detection_selection_excludes_shared_recording_and_retains_whole_clean_groups():
    rows = [row('correlated_train1', 'shared', 'train'), row('correlated_train2', 'shared', 'train'),
            row('publisher_val', 'shared', 'val')] + [row(str(i), 'clean', 'train') for i in range(3)]
    chosen, excluded, stats = select_detection_rows(rows, 1, 1, 42)
    assert {r['id'] for r in excluded} == {'correlated_train1', 'correlated_train2'}
    assert {r['id'] for r in chosen if r['split'] == 'train'} == {'0', '1', '2'}
    assert stats['train']['selected_images'] == 3
    assert next(r for r in chosen if r['id'] == 'publisher_val')['split'] == 'val'


def test_detection_selection_does_not_fabricate_missing_class_coverage():
    rows = [row('train', 'train_group', 'train'), row('val', 'val_group', 'val')]
    rows[1]['positive_instances'].pop('bicycle')
    with pytest.raises(ValueError, match='lack bicycle'):
        select_detection_rows(rows, 1, 1, 42)
