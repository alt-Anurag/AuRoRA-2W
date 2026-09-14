"""Freeze a metadata-selected IDD detection proposal before extracting JPEGs.

Default policy excludes training groups sharing the conservative timestamp key
with any publisher validation image. Publisher membership is never changed.
No model predictions, optimizer operations or generated labels are involved.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from aurora2w.native import IDD_CLASSES
from scripts.prepare_bdd_subset import select_groups


def select_detection_rows(rows, train_images, val_images, seed):
    val_groups = {row['conservative_group'] for row in rows if row['official_split'] == 'val'}
    excluded = [row for row in rows if row['official_split'] == 'train' and row['conservative_group'] in val_groups]
    available = [dict(row, sequence=row['conservative_group'], split=row['official_split']) for row in rows
                 if row['official_split'] == 'val' or row['conservative_group'] not in val_groups]
    chosen, statistics = [], {}
    for split, target in (('train', train_images), ('val', val_images)):
        candidates = [row for row in available if row['split'] == split]
        if not candidates:
            raise ValueError(f'No {split} images remain under the no-shared-recording policy; review actual publisher overlap')
        selection = select_groups(candidates, target, seed + (split == 'val'), stratum_key='capture_category')
        groups = defaultdict(list)
        for row in candidates:
            groups[row['sequence']].append(row)
        selected_groups = {row['sequence'] for row in selection}
        # Add complete, smallest available groups for any initially missing class.
        # This uses true annotations, never model predictions or holdout scores.
        additions = []
        for name in sorted(set(IDD_CLASSES.values())):
            if any(row['positive_instances'].get(name, 0) for row in selection):
                continue
            candidates_with_class = [(key, group) for key, group in groups.items() if key not in selected_groups
                                     and any(row['positive_instances'].get(name, 0) for row in group)]
            if not candidates_with_class:
                raise ValueError(f'Available {split} groups lack {name}; do not fabricate coverage')
            key, group = min(candidates_with_class, key=lambda item: (len(item[1]), item[0]))
            selection.extend(group)
            selected_groups.add(key)
            additions.append({'class': name, 'group': key, 'images_added': len(group)})
        selection.sort(key=lambda row: row['id'])
        chosen.extend(selection)
        instances, coordinates = Counter(), Counter()
        for row in selection:
            instances.update(row['positive_instances'])
            coordinates.update(row['coordinate_counts'])
        statistics[split] = {'target_images': target, 'selected_images': len(selection), 'available_images': len(candidates),
                             'selected_groups': len(selected_groups), 'available_groups': len(groups),
                             'selected_capture_categories': dict(Counter(row['capture_category'] for row in selection)),
                             'positive_instances': dict(instances),
                             'class_positive_images': dict(Counter(name for row in selection for name, count in row['positive_instances'].items() if count)),
                             'coordinate_counts': dict(coordinates), 'class_coverage_group_additions': additions,
                             'effective_seed': seed + (split == 'val')}
    selected_train = {row['sequence'] for row in chosen if row['split'] == 'train'}
    selected_val = {row['sequence'] for row in chosen if row['split'] == 'val'}
    if selected_train & selected_val:
        raise AssertionError('Selection groups unexpectedly cross publisher splits')
    return chosen, excluded, statistics


def run(args):
    directory, output = Path(args.audit).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh immutable detection proposal directory')
    audit_path = directory / 'AUDIT.json'
    audit = json.loads(audit_path.read_bytes())
    index = directory / 'candidates.jsonl'
    if hashlib.sha256(index.read_bytes()).hexdigest() != audit['candidates_sha256']:
        raise ValueError('Candidate metadata changed since its completed XML audit')
    rows = [json.loads(line) for line in index.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    chosen, excluded, statistics = select_detection_rows(rows, args.train_images, args.val_images, args.seed)
    byte_count = sum(row['bytes'] for row in chosen)
    if not 0 < byte_count <= 10 * 1024**3:
        raise ValueError(f'Selected JPEGs require {byte_count} bytes, exceeding the reviewed 10 GiB budget')
    output.mkdir(parents=True, exist_ok=True)
    selected = output / 'selected.jsonl'
    selected.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in chosen), encoding='utf-8')
    (output / 'excluded_training_overlap.jsonl').write_text(''.join(json.dumps({'id': row['id'], 'image': row['image'],
        'group': row['conservative_group'], 'reason': 'Shares conservative timestamp with publisher validation'}, sort_keys=True) + '\n' for row in excluded), encoding='utf-8')
    groups = output / 'groups.csv'
    with groups.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'sequence', 'split'])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in chosen)
    plan = {'schema_version': 1, 'reviewed_layout': 'idd_detection_nested_voc_v1',
            'archive_sha256': audit['archive_sha256'], 'strip_root': 'IDD_Detection',
            'maximum_image_bytes': byte_count, 'groups_csv': 'groups.csv', 'groups_sha256': hashlib.sha256(groups.read_bytes()).hexdigest(),
            'group_policy': 'Exclude all training images sharing conservative city/date/time groups with any official validation image',
            'group_limit': 'Observed timestamp/path relation, not verified independent riders or locations', 'training_approved': False}
    (output / 'extraction_PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    report = {'scope': 'Frozen data-only selected-JPEG proposal for review; no training', 'training_approved': False,
              'selection': 'Whole conservative groups, capture-category strata, proportional quotas and seeded shuffle; missing classes add the smallest complete available group',
              'split_scope': 'Publisher membership preserved; train groups sharing any publisher validation timestamp excluded',
              'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(), 'archive_sha256': audit['archive_sha256'],
              'full_labeled_source_counts': audit['split_counts'], 'excluded_training_images_due_to_overlap': len(excluded),
              'excluded_training_groups': len({row['conservative_group'] for row in excluded}),
              'selected_index_sha256': hashlib.sha256(selected.read_bytes()).hexdigest(), 'groups_sha256': plan['groups_sha256'],
              'selected_image_bytes': byte_count, 'splits': statistics, 'coordinates': 'Must be explicitly reviewed before native conversion; no origin chosen by selection'}
    (output / 'SELECTION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-images', type=int, default=4000)
    parser.add_argument('--val-images', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
