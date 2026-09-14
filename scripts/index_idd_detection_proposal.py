"""Index a frozen IDD detection proposal only after verified JPEG extraction.

The reviewed internal convention preserves raw coordinates as continuous pixel
edges, clips actual image-border overflows with warnings, and removes class
exhaustiveness for any unrecoverable zero-area annotation. This is not a claim
that the publisher specified a zero-based origin or official AP equivalence.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from aurora2w.native import build_native, relative


def run(args):
    root, selection, output = Path(args.root).resolve(), Path(args.selection).resolve(), Path(args.output).resolve()
    selected_path = selection / 'selected.jsonl'
    report = json.loads((selection / 'SELECTION.json').read_bytes())
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != report['selected_index_sha256']:
        raise ValueError('Frozen selected metadata changed')
    groups = selection / 'groups.csv'
    if hashlib.sha256(groups.read_bytes()).hexdigest() != report['groups_sha256']:
        raise ValueError('Frozen selected group membership changed')
    plan = selection / 'extraction_PLAN.json'
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    extraction_path = root / '.extraction' / report['archive_sha256'] / ('selection-' + plan_hash) / 'report.json'
    extraction = json.loads(extraction_path.read_bytes())
    if (not extraction.get('usable') or not extraction.get('images_ready') or not extraction.get('gzip_eof_verified')
            or extraction.get('selection_plan_sha256') != plan_hash):
        raise ValueError('Frozen detector JPEG extraction is not verified complete')
    selected = {row['id']: row for row in (json.loads(line) for line in selected_path.read_text(encoding='utf-8-sig').splitlines())}
    files = {row['path']: row for row in (json.loads(line) for line in Path(extraction['manifest']).read_text(encoding='utf-8-sig').splitlines())}
    for row in selected.values():
        for path, digest in ((row['image'], row['sha256']), (row['annotation'], row['annotation_sha256'])):
            if path not in files or files[path]['sha256'] != digest:
                raise ValueError(f'Extracted file differs from frozen catalog: {path}')
    built = build_native('idd-det', root, output, groups_path=groups, exhaustive=True,
                         voc_coordinates='zero_based_half_open')
    index = output / 'index.jsonl'
    rows = [json.loads(line) for line in index.read_text(encoding='utf-8-sig').splitlines()]
    if {row['id'] for row in rows} != set(selected):
        raise ValueError('Native index changed the exact selected identity set')
    for row in rows:
        original = selected[row['id']]
        row.update(source_archive_sha256=report['archive_sha256'], source_archive_member=original['archive_member'],
                   source_image_sha256=original['sha256'], source_annotation_sha256=original['annotation_sha256'])
        if (row['image'], row['split'], row['sequence']) != (original['image'], original['split'], original['sequence']):
            raise ValueError('Native index changed frozen image/group/split membership')
    index.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    source = json.loads(Path(built['spec']).read_bytes())['sources'][0]
    source.update(source_release='IDD Detection author 2019 archive',
                  selection_protocol='idd_detection_timestamp_disjoint_whole_groups_v1',
                  selection_sha256=report['selected_index_sha256'],
                  annotation_contract={'coordinate_policy': 'Raw continuous pixel edges; actual border overflows clipped and logged; unrecoverable zero-area class becomes positive-only',
                                       'publisher_origin': 'Not explicitly established; this is a reviewed internal convention',
                                       'publisher_evaluator_difference': 'Official evaluator uses inclusive +1 IoU; internal pixel-edge AP is not official IDD AP parity',
                                       'detection_scope': 'Eight mapped traffic classes exhaustive; animal/vehicle fallback/other unmapped boxes are class-agnostic ignore regions',
                                       'difficult_policy': 'Ignored if present; no difficult/truncated tags were found in the audited release',
                                       'extraction_report_sha256': hashlib.sha256(extraction_path.read_bytes()).hexdigest()})
    (output / 'sources.json').write_text(json.dumps({'sources': [source]}, indent=2), encoding='utf-8')
    for split in ('train', 'val'):
        directory = output / ('index_' + split)
        directory.mkdir()
        (directory / 'index.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows if row['split'] == split), encoding='utf-8')
        split_source = dict(source, root=relative(root, directory), index='index.jsonl')
        (directory / 'sources.json').write_text(json.dumps({'sources': [split_source]}, indent=2), encoding='utf-8')
    result = {'scope': 'Verified frozen detector index only; no training', 'training_approved': False,
              'splits': dict(Counter(row['split'] for row in rows)), 'spec': str(output / 'sources.json'),
              'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
              'selection_sha256': report['selected_index_sha256'], 'extraction_report': str(extraction_path)}
    (output / 'INDEX_REVIEW.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/phase2/raw/idd_detection_catalog')
    parser.add_argument('--selection', default='data/phase2/partitions/idd_det_subset_v1')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
