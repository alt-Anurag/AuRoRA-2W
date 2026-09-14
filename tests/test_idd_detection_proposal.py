import hashlib
import json
from types import SimpleNamespace
import pytest
from scripts.index_idd_detection_proposal import run


def test_detector_index_refuses_a_partial_extraction_even_when_plan_is_pinned(tmp_path):
    root, selection, output = tmp_path / 'raw', tmp_path / 'selection', tmp_path / 'index'
    selection.mkdir()
    index = selection / 'selected.jsonl'
    index.write_text('')
    groups = selection / 'groups.csv'
    groups.write_text('image,sequence,split\n')
    plan = selection / 'extraction_PLAN.json'
    plan.write_text(json.dumps({'scope': 'software fixture'}))
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    (selection / 'SELECTION.json').write_text(json.dumps({'archive_sha256': 'a' * 64,
        'selected_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
        'groups_sha256': hashlib.sha256(groups.read_bytes()).hexdigest()}))
    report = root / '.extraction' / ('a' * 64) / ('selection-' + plan_hash) / 'report.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'usable': False, 'images_ready': False, 'gzip_eof_verified': False,
                                 'selection_plan_sha256': plan_hash}))
    with pytest.raises(ValueError, match='not verified complete'):
        run(SimpleNamespace(root=root, selection=selection, output=output))
    assert not output.exists()
