import hashlib
from pathlib import Path
import pytest
from scripts.audit_idd_detection_catalog import inspect_xml, recording_timestamp


def test_timestamp_group_unifies_observed_camera_and_part_suffixes():
    assert recording_timestamp('BLR-2018-03-22_17-39-26_2_frontFar') == recording_timestamp('BLR-2018-03-22-17-39-26_part_4')
    assert recording_timestamp('unknown_recording') is None


def test_xml_audit_separates_one_based_and_zero_based_evidence(tmp_path):
    raw = b'<annotation><size><width>1280</width><height>720</height></size><object><name>car</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>1280</xmax><ymax>720</ymax></bndbox></object><object><name>animal</name><difficult>1</difficult><bndbox><xmin>0</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>'
    (tmp_path / 'one.xml').write_bytes(raw)
    row = {'annotation': 'one.xml', 'annotation_sha256': hashlib.sha256(raw).hexdigest(),
           'sequence_directory': 'BLR-2018-03-22_17-39-26_2_frontFar', 'recording_prefix': 'fallback',
           'capture_category': 'frontFar'}
    found = inspect_xml((tmp_path, row))
    assert found['positive_instances'] == {'car': 1}
    assert found['coordinate_counts']['boxes'] == 2
    assert found['coordinate_counts']['one_based_inclusive_incompatible'] == 1
    assert found['coordinate_counts']['zero_based_inclusive_incompatible'] == 1
    assert found['raw_object_flags']['difficult_nonzero'] == 1
    (tmp_path / 'one.xml').write_bytes(raw + b'\n')
    with pytest.raises(ValueError, match='XML changed'):
        inspect_xml((tmp_path, row))
