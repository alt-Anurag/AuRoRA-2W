"""Audit the verified IDD detection metadata catalog before freezing a subset.

Reads publisher XMLs and inventory only. No image extraction, predictions,
optimizer steps or modification of official split membership is performed.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from aurora2w.native import IDD_CLASSES


def recording_timestamp(directory):
    """Conservative shared timestamp grouping; not proof of rider identity."""
    match = re.match(r'^([A-Za-z]+)-(\d{4})-(\d{2})-(\d{2})[-_](\d{2})[-_:](\d{2})[-_:](\d{2})(?:_|$)', directory)
    return '-'.join(match.groups()) if match else None


def inspect_xml(task):
    root, original = task
    row = dict(original)
    path = root / row['annotation']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['annotation_sha256']:
        raise ValueError(f'XML changed since verified catalog: {path}')
    tree = ET.fromstring(raw)
    if tree.tag != 'annotation':
        raise ValueError(f'Unexpected XML root: {path}')
    size = tree.find('size')
    width, height = int(size.findtext('width')), int(size.findtext('height'))
    if min(width, height) <= 0:
        raise ValueError(f'Invalid XML image dimensions: {path}')
    classes, positives, flags, coordinates = Counter(), Counter(), Counter(), Counter()
    examples = []
    for obj in tree.findall('object'):
        label = obj.findtext('name')
        classes[label] += 1
        difficult = int(obj.findtext('difficult', '0')) != 0
        flags['difficult_present'] += obj.find('difficult') is not None
        flags['difficult_nonzero'] += difficult
        flags['truncated_present'] += obj.find('truncated') is not None
        flags['truncated_nonzero'] += int(obj.findtext('truncated', '0')) != 0
        if label in IDD_CLASSES and not difficult:
            positives[IDD_CLASSES[label]] += 1
        bndbox = obj.find('bndbox')
        box = [float(bndbox.findtext(key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not all(math.isfinite(value) for value in box):
            raise ValueError(f'Nonfinite box: {path}')
        x1, y1, x2, y2 = box
        checks = {'boxes': True, 'any_noninteger': any(value != int(value) for value in box),
                  'xmin_zero': x1 == 0, 'ymin_zero': y1 == 0, 'xmin_one': x1 == 1, 'ymin_one': y1 == 1,
                  'min_negative': x1 < 0 or y1 < 0, 'xmax_width': x2 == width, 'ymax_height': y2 == height,
                  'xmax_width_minus_one': x2 == width-1, 'ymax_height_minus_one': y2 == height-1,
                  'max_beyond_image': x2 > width or y2 > height, 'inverted': x2 < x1 or y2 < y1,
                  'zero_continuous_area': x2 == x1 or y2 == y1,
                  'one_based_inclusive_incompatible': not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height),
                  'zero_based_inclusive_incompatible': not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height),
                  'zero_based_half_open_incompatible': not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)}
        coordinates.update({key: int(value) for key, value in checks.items()})
        if any(checks[key] for key in ('xmin_zero', 'ymin_zero', 'min_negative', 'max_beyond_image', 'inverted', 'zero_continuous_area')) and len(examples) < 5:
            examples.append({'class': label, 'xyxy': box, 'size': [width, height]})
    timestamp = recording_timestamp(row['sequence_directory'])
    row.update(width=width, height=height, raw_class_instances=dict(classes), positive_instances=dict(positives),
               raw_object_flags=dict(flags), coordinate_counts=dict(coordinates), anomalous_box_examples=examples,
               conservative_timestamp=timestamp, conservative_group=timestamp or row['recording_prefix'],
               observed_sequence=row['capture_category'] + '/' + row['sequence_directory'])
    return row


def audit_catalog(report_path, output, workers=8):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    raw_report = report_path.read_bytes()
    report = json.loads(raw_report)
    if not report.get('catalog_ready') or not report.get('gzip_eof_verified') or report.get('images_ready'):
        raise ValueError('A complete verified metadata-only catalog is required')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh catalog audit directory')
    root = Path(report['destination'])
    inventory = Path(report['jpeg_inventory'])
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != report['jpeg_inventory_sha256']:
        raise ValueError('Catalog inventory hash differs from the verified report')
    originals = [json.loads(line) for line in inventory.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    missing = [row for row in originals if row['official_split'] in ('train', 'val') and not row['annotation_present']]
    if missing:
        raise ValueError(f'{len(missing)} published labeled images lack XML annotations')
    selected = [row for row in originals if row['official_split'] in ('train', 'val')]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(inspect_xml, ((root, row) for row in selected)))
    totals, categories, split_classes = Counter(), defaultdict(Counter), defaultdict(Counter)
    group_maps = {key: defaultdict(lambda: defaultdict(list)) for key in ('observed_sequence', 'recording_group', 'conservative_group')}
    for row in rows:
        totals.update(row['coordinate_counts'])
        categories[row['capture_category']].update(row['coordinate_counts'])
        split_classes[row['official_split']].update(row['positive_instances'])
        for key, groups in group_maps.items():
            groups[row[key]][row['official_split']].append(row['id'])
    overlaps = {}
    for key, groups in group_maps.items():
        shared = {group: {split: len(ids) for split, ids in split_rows.items()} for group, split_rows in groups.items() if len(split_rows) > 1}
        overlaps[key] = {'groups': len(groups), 'cross_split_groups': len(shared), 'shared_group_counts': shared,
                         'train_images_in_shared_groups': sum(value.get('train', 0) for value in shared.values()),
                         'val_images_in_shared_groups': sum(value.get('val', 0) for value in shared.values())}
    shared = set(overlaps['conservative_group']['shared_group_counts'])
    clean_train = [row for row in rows if row['official_split'] == 'train' and row['conservative_group'] not in shared]
    clean_classes = Counter()
    for row in clean_train:
        clean_classes.update(row['positive_instances'])
    output.mkdir(parents=True, exist_ok=True)
    index = output / 'candidates.jsonl'
    index.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    result = {'scope': 'Actual XML metadata and official split audit; no images or models used', 'training_approved': False,
              'catalog_report': str(report_path), 'catalog_report_sha256': hashlib.sha256(raw_report).hexdigest(),
              'archive_sha256': report['archive_sha256'], 'inventory_sha256': report['jpeg_inventory_sha256'],
              'candidates_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
              'split_counts': dict(Counter(row['official_split'] for row in rows)),
              'capture_categories': dict(Counter(row['capture_category'] for row in rows)),
              'class_positive_instances_by_split': {key: dict(value) for key, value in split_classes.items()},
              'coordinate_counts': dict(totals), 'coordinate_counts_by_category': {key: dict(value) for key, value in categories.items()},
              'raw_object_flags': dict(sum((Counter(row['raw_object_flags']) for row in rows), Counter())),
              'timestamp_parsed_images': sum(row['conservative_timestamp'] is not None for row in rows),
              'group_scope': 'Observed paths plus conservative shared city/date/time prefixes; not verified rider/location identities',
              'group_overlap': overlaps,
              'train_without_shared_conservative_groups': {'images': len(clean_train), 'groups': len({row['conservative_group'] for row in clean_train}),
                                                          'positive_instances': dict(clean_classes)}}
    (output / 'AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit_catalog(args.catalog_report, args.output, args.workers), indent=2))
