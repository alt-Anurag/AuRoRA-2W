import json
import numpy as np
from PIL import Image
import pytest

from scripts.prepare_idd_masks import rasterize, convert_one


def annotation(objects):
    return {"imgWidth": 5, "imgHeight": 4, "objects": objects}


def polygon(label, points=None, **kwargs):
    return {"label": label, "polygon": points or [[0, 0], [4, 0], [4, 3], [0, 3]], **kwargs}


def test_original_ids_occlusion_deleted_and_draw_flag():
    document = annotation([
        polygon("road"),
        polygon("ridergroup", [[1, 1], [2, 1], [2, 2], [1, 2]], draw=False),
        polygon("sky", deleted=1),
        polygon("polegroup", [[4, 0], [4, 3], [3, 3]]),
    ])
    mask = np.asarray(rasterize(document))
    assert mask[0, 0] == 0
    assert mask[1, 1] == 8
    assert mask[2, 2] == 8
    assert mask[3, 4] == 27  # Must not strip the recognized 'polegroup' label.


def test_unlabeled_background_degenerate_polygon_and_unknown():
    assert np.all(np.asarray(rasterize(annotation([polygon("road", [[0, 0], [1, 1]])]))) == 35)
    with pytest.raises(ValueError, match="Unknown IDD label"):
        rasterize(annotation([polygon("unexpected road category")]))
    with pytest.raises(ValueError, match="coordinate"):
        rasterize(annotation([polygon("road", [[0, 0], [1, float("nan")], [3, 3]])]))


def test_image_dimensions_and_existing_mask_must_agree(tmp_path):
    directory = tmp_path / "gtFine" / "val" / "001"
    directory.mkdir(parents=True)
    image_dir = tmp_path / "leftImg8bit" / "val" / "001"
    image_dir.mkdir(parents=True)
    path = directory / "frame0001_gtFine_polygons.json"
    path.write_text(json.dumps(annotation([polygon("road")])), encoding="utf-8")
    image = image_dir / "frame0001_leftImg8bit.png"
    Image.new("RGB", (5, 4)).save(image)
    assert convert_one(tmp_path, path)["status"] == "created"
    assert convert_one(tmp_path, path)["status"] == "verified_existing"
    Image.new("L", (5, 4), 33).save(directory / "frame0001_gtFine_labelIds.png")
    with pytest.raises(ValueError, match="Conflicting existing mask"):
        convert_one(tmp_path, path)
    Image.new("RGB", (4, 5)).save(image)
    with pytest.raises(ValueError, match="dimension mismatch"):
        convert_one(tmp_path, path)



def test_managed_extraction_must_finish_before_mask_preparation(tmp_path):
    from scripts.prepare_idd_masks import extraction_snapshot
    assert extraction_snapshot(tmp_path) == {}
    directory = tmp_path / ".extraction" / ("a" * 64)
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="without a completion"):
        extraction_snapshot(tmp_path)
    report = directory / "report.json"
    report.write_text(json.dumps({"usable": False, "gzip_eof_verified": False}))
    with pytest.raises(ValueError, match="incomplete"):
        extraction_snapshot(tmp_path)
    report.write_text(json.dumps({"usable": True, "gzip_eof_verified": True}))
    assert list(extraction_snapshot(tmp_path)) == ["a" * 64]
