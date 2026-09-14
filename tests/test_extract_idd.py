import gzip
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import extract_idd


ROOT = "idd20kII"
IMAGE = f"{ROOT}/leftImg8bit/train/17/frame_leftImg8bit.png"
POLYGON = f"{ROOT}/gtFine/val/18/frame_gtFine_polygons.json"


def archive_fixture(tmp_path, entries, tail=b""):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            if isinstance(payload, bytes):
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            else:
                info.type = payload[0]
                info.linkname = "../../outside"
                tar.addfile(info)
    path = tmp_path / "part2.tar.gz"
    path.write_bytes(gzip.compress(buffer.getvalue()) + tail)
    source = tmp_path / "SOURCE.json"
    source.write_text(json.dumps({"archive": path.name, "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    return path, source


def run(tmp_path, entries):
    archive, source = archive_fixture(tmp_path, entries)
    return extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT)


def test_selective_files_hashes_and_identical_resume(tmp_path):
    archive, source = archive_fixture(tmp_path, [(IMAGE, b"png bytes"), (POLYGON, b"{}"),
        (f"{ROOT}/leftImg8bit/test/17/test.jpg", b"skip"),
        (f"{ROOT}/gtFine/train/17/frame_gtFine_labelIds.png", b"skip"),
        (f"{ROOT}/README.txt", b"preserved in archive")])
    before = archive.read_bytes()
    first = extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT)
    second = extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT)
    assert first["usable"] and first["gzip_eof_verified"] and first["files"] == 2
    assert first["written"] == 2 and second["written"] == 0 and second["existing_verified"] == 2
    assert first["counts"] == {"leftImg8bit/train": 1, "gtFine/val": 1}
    records = [json.loads(line) for line in Path(first["manifest"]).read_text().splitlines()]
    for record in records:
        path = tmp_path / "raw" / record["path"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (tmp_path / "raw" / "leftImg8bit" / "test").exists()
    assert archive.read_bytes() == before


@pytest.mark.parametrize("name", ["/absolute.png", "../escape.png", f"{ROOT}/leftImg8bit/train/../evil.png",
    "C:/outside.png", r"idd20kII\leftImg8bit\train\evil.png", f"{ROOT}/gtFine/train/x/evil:stream.json",
    f"{ROOT}/leftImg8bit/train/x/NUL.png", f"{ROOT}/leftImg8bit/train/x/evil. /image.png"])
def test_rejects_unsafe_paths_even_if_not_selected(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe|Reserved"):
        run(tmp_path, [(name, b"untrusted")])
    assert not (tmp_path / "escape.png").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_rejects_archive_links_and_special_members(tmp_path, kind):
    with pytest.raises(ValueError, match="forbidden"):
        run(tmp_path, [(IMAGE, (kind,))])


def test_canonical_duplicates_require_matching_bytes(tmp_path):
    result = run(tmp_path, [(IMAGE, b"same"), (IMAGE, b"same")])
    assert result["files"] == 1 and result["duplicate_members_verified"] == 1
    with pytest.raises(ValueError, match="conflicts"):
        run(tmp_path, [(IMAGE, b"same"), (IMAGE, b"evil")])
    assert (tmp_path / "raw" / IMAGE.removeprefix(ROOT + "/")).read_bytes() == b"same"


def test_existing_raw_conflict_is_not_overwritten(tmp_path):
    target = tmp_path / "raw" / IMAGE.removeprefix(ROOT + "/")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user")
    with pytest.raises(ValueError, match="hash conflicts"):
        run(tmp_path, [(IMAGE, b"data")])
    assert target.read_bytes() == b"user"


def test_source_hash_failure_precedes_raw_extraction(tmp_path):
    archive, source = archive_fixture(tmp_path, [(IMAGE, b"png")])
    metadata = json.loads(source.read_text())
    metadata["sha256"] = "0" * 64
    source.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SHA-256"):
        extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT)
    assert not (tmp_path / "raw" / "leftImg8bit").exists()


def test_drains_truncated_gzip_member_after_tar_eof(tmp_path):
    archive, source = archive_fixture(tmp_path, [(IMAGE, b"png")], gzip.compress(b"trailing data")[:-4])
    with pytest.raises(EOFError):
        extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT)
    report = json.loads(next((tmp_path / "raw" / ".extraction").glob("*/report.json")).read_text())
    assert not report["usable"] and not report["gzip_eof_verified"]
    assert not Path(report["manifest"]).exists()


def test_resumes_after_disk_reserve_interruption(tmp_path, monkeypatch):
    archive, source = archive_fixture(tmp_path, [(IMAGE, b"png"), (POLYGON, b"json")])
    actual_check = extract_idd.require_space
    calls = 0
    def bounded_check(root, incoming=0):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated disk reserve interruption")
        actual_check(root, incoming)
    monkeypatch.setattr(extract_idd, "require_space", bounded_check)
    with pytest.raises(OSError, match="reserve"):
        extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT)
    assert (tmp_path / "raw" / IMAGE.removeprefix(ROOT + "/")).read_bytes() == b"png"
    monkeypatch.setattr(extract_idd, "require_space", actual_check)
    result = extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT)
    assert result["usable"] and result["existing_verified"] == 1 and result["written"] == 1


def test_free_space_reserve_accounts_for_next_payload(tmp_path, monkeypatch):
    usage = type("Usage", (), {"free": extract_idd.RESERVE_BYTES + 3})()
    monkeypatch.setattr(extract_idd.shutil, "disk_usage", lambda path: usage)
    extract_idd.require_space(tmp_path, 3)
    with pytest.raises(OSError, match="10 GiB"):
        extract_idd.require_space(tmp_path, 4)


def detection_fixture(tmp_path, extra=(), omit_annotation=False):
    # Identical basenames in different sequences are distinct classic IDD IDs.
    a, b, c = "front/ride1/frame0001", "front/ride2/frame0001", "rear/ride3/frame0099"
    entries = [(f"{ROOT}/JPEGImages/{identifier}.jpg", identifier.encode()) for identifier in (a, b, c)]
    entries += [(f"{ROOT}/Annotations/{identifier}.xml", b"<annotation/>") for identifier in (a, b, c) if not (omit_annotation and identifier == a)]
    entries += [(f"{ROOT}/train.txt", (a + "\n" + c + "\n").encode()),
                (f"{ROOT}/val.txt", (b + "\n").encode()), (f"{ROOT}/test.txt", b"rear/ride4/frame0100\n")]
    archive, source = archive_fixture(tmp_path, entries + list(extra))
    rows = [{"image": f"JPEGImages/{a}.jpg", "sequence": "front/ride1", "split": "train"},
            {"image": f"JPEGImages/{b}.jpg", "sequence": "front/ride2", "split": "val"}]
    groups = tmp_path / "frozen.csv"
    with groups.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "sequence", "split"))
        writer.writeheader()
        writer.writerows(rows)
    plan = {"schema_version": 1, "reviewed_layout": "idd_detection_nested_voc_v1",
            "strip_root": ROOT, "archive_sha256": extract_idd.sha256_file(archive),
            "groups_csv": groups.name, "groups_sha256": extract_idd.sha256_file(groups),
            "maximum_image_bytes": 1024}
    plan_path = tmp_path / "selection.json"
    plan_path.write_text(json.dumps(plan))
    return archive, source, plan_path


def run_detection(tmp_path, archive, source, plan):
    return extract_idd.extract_archive(archive, source, tmp_path / "raw", ROOT,
                                       mode="detection", detection_plan=plan)


def test_detection_selects_frozen_images_keeps_all_xml_and_splits(tmp_path):
    result = run_detection(tmp_path, *detection_fixture(tmp_path))
    assert result["usable"] and result["files"] == 8
    assert result["selected_official_split_counts"] == {"train": 1, "val": 1}
    assert result["requested_images"] == 2
    assert (tmp_path / "raw/JPEGImages/front/ride1/frame0001.jpg").exists()
    assert (tmp_path / "raw/JPEGImages/front/ride2/frame0001.jpg").exists()
    assert not (tmp_path / "raw/JPEGImages/rear/ride3/frame0099.jpg").exists()
    assert (tmp_path / "raw/Annotations/rear/ride3/frame0099.xml").exists()
    assert (tmp_path / "raw/test.txt").read_text() == "rear/ride4/frame0100\n"


def test_detection_requires_reviewed_plan(tmp_path):
    archive, source, _ = detection_fixture(tmp_path)
    with pytest.raises(ValueError, match="reviewed frozen"):
        run_detection(tmp_path, archive, source, None)


@pytest.mark.parametrize("field,value,error", [("archive_sha256", "0" * 64, "SHA/root"),
    ("reviewed_layout", "unknown", "reviewed"), ("groups_sha256", "0" * 64, "CSV SHA"),
    ("maximum_image_bytes", 11 * 1024**3, "at most 10 GiB")])
def test_detection_plan_binding_and_budget(tmp_path, field, value, error):
    archive, source, plan_path = detection_fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan[field] = value
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=error):
        run_detection(tmp_path, archive, source, plan_path)


def test_detection_refuses_unrecognized_member_schema(tmp_path):
    files = detection_fixture(tmp_path, extra=[(f"{ROOT}/JPEGImages/flattened.jpg", b"unknown")])
    with pytest.raises(ValueError, match="Unrecognized nested VOC"):
        run_detection(tmp_path, *files)


def test_detection_requires_selected_counterpart(tmp_path):
    with pytest.raises(ValueError, match="counterpart absent"):
        run_detection(tmp_path, *detection_fixture(tmp_path, omit_annotation=True))


def test_detection_validates_official_split_membership(tmp_path):
    archive, source, plan_path = detection_fixture(tmp_path)
    groups = tmp_path / "frozen.csv"
    groups.write_text(groups.read_text().replace("front/ride1,train", "front/ride1,val"))
    plan = json.loads(plan_path.read_text())
    plan["groups_sha256"] = extract_idd.sha256_file(groups)
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="official split membership"):
        run_detection(tmp_path, archive, source, plan_path)


def test_detection_enforces_frozen_image_byte_budget(tmp_path):
    archive, source, plan_path = detection_fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["maximum_image_bytes"] = 1
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(OSError, match="image byte budget"):
        run_detection(tmp_path, archive, source, plan_path)


def test_detection_rejects_relabeling_one_physical_sequence_as_two_groups(tmp_path):
    archive, source, plan_path = detection_fixture(tmp_path)
    groups = tmp_path / "frozen.csv"
    with groups.open("a", newline="") as stream:
        stream.write("JPEGImages/front/ride1/frame0002.jpg,artificial_other_group,val\n")
    plan = json.loads(plan_path.read_text())
    plan["groups_sha256"] = extract_idd.sha256_file(groups)
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="physical detection sequence"):
        run_detection(tmp_path, archive, source, plan_path)


def run_catalog(tmp_path, archive, source):
    return extract_idd.extract_archive(archive, source, tmp_path / "catalog", ROOT,
                                       mode="detection-catalog")


def test_catalog_validates_metadata_and_hashes_jpegs_without_extracting_them(tmp_path):
    archive, source, _ = detection_fixture(tmp_path)
    result = run_catalog(tmp_path, archive, source)
    assert result["catalog_ready"] and result["gzip_eof_verified"]
    assert not result["usable"] and not result["images_ready"]
    assert result["files"] == 6 and result["catalog_images"] == 3
    assert not (tmp_path / "catalog/JPEGImages").exists()
    assert (tmp_path / "catalog/Annotations/front/ride1/frame0001.xml").exists()
    rows = [json.loads(line) for line in Path(result["jpeg_inventory"]).read_text().splitlines()]
    assert len(rows) == 3
    assert result["jpeg_inventory_sha256"] == extract_idd.sha256_file(Path(result["jpeg_inventory"]))
    assert result["catalog_image_split_counts"] == {"train": 2, "val": 1, "test": 0, "unlisted": 0}
    assert result["official_ids_without_jpeg"] == 1
    for row in rows:
        payload = row["id"].encode()
        assert row["bytes"] == len(payload) and row["sha256"] == hashlib.sha256(payload).hexdigest()
        assert row["archive_sha256"] == result["archive_sha256"]
        assert row["archive_member"] == ROOT + "/" + row["image"]
        assert row["annotation_present"] and row["official_split"] in ("train", "val")
    resumed = run_catalog(tmp_path, archive, source)
    assert resumed["written"] == 0 and resumed["existing_verified"] == 6
    assert resumed["jpeg_inventory_sha256"] == result["jpeg_inventory_sha256"]


def test_selected_images_reuse_completed_catalog_without_rewriting_metadata(tmp_path):
    archive, source, plan = detection_fixture(tmp_path)
    catalog = run_catalog(tmp_path, archive, source)
    destination = tmp_path / "catalog"
    metadata = [destination / json.loads(line)["path"]
                for line in Path(catalog["manifest"]).read_text().splitlines()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in metadata}
    inventory = Path(catalog["jpeg_inventory"])
    inventory_before = inventory.read_bytes()
    selected = extract_idd.extract_archive(archive, source, destination, ROOT,
                                           mode="detection", detection_plan=plan)
    assert selected["usable"] and selected["images_ready"] and selected["gzip_eof_verified"]
    assert selected["existing_verified"] == 6 and selected["written"] == 2
    assert selected["selected_official_split_counts"] == {"train": 1, "val": 1}
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in metadata} == before
    assert inventory.read_bytes() == inventory_before
    assert not (destination / "JPEGImages/rear/ride3/frame0099.jpg").exists()


def test_catalog_exposes_recording_prefix_across_capture_categories(tmp_path):
    identifier = "highquality_16k/BLR-2018-06-19-06-21-44_part_26/0000899"
    archive, source, _ = detection_fixture(tmp_path, extra=[
        (f"{ROOT}/JPEGImages/{identifier}.jpg", b"jpeg payload"),
        (f"{ROOT}/Annotations/{identifier}.xml", b"<annotation/>")])
    result = run_catalog(tmp_path, archive, source)
    rows = [json.loads(line) for line in Path(result["jpeg_inventory"]).read_text().splitlines()]
    row = next(item for item in rows if item["id"] == identifier)
    assert row["recording_prefix"] == "BLR-2018-06-19-06-21-44"
    assert row["recording_group"] == "highquality_16k/BLR-2018-06-19-06-21-44"
    assert row["sequence_directory"].endswith("_part_26")
    assert row["official_split"] is None and result["catalog_image_split_counts"]["unlisted"] == 1


def test_catalog_duplicate_jpeg_conflicts_are_not_hidden_by_skipping_images(tmp_path):
    identifier = "front/ride1/frame0001"
    archive, source, _ = detection_fixture(tmp_path, extra=[(f"{ROOT}/JPEGImages/{identifier}.jpg", identifier.encode())])
    result = run_catalog(tmp_path, archive, source)
    assert result["catalog_images"] == 3 and result["catalog_duplicate_images_verified"] == 1
    archive, source, _ = detection_fixture(tmp_path, extra=[(f"{ROOT}/JPEGImages/{identifier}.jpg", b"conflict")])
    with pytest.raises(ValueError, match="catalog JPEG conflicts"):
        run_catalog(tmp_path, archive, source)


def test_catalog_requires_complete_gzip_after_tar_eof(tmp_path):
    archive, source, _ = detection_fixture(tmp_path)
    archive.write_bytes(archive.read_bytes() + gzip.compress(b"later gzip member")[:-4])
    metadata = json.loads(source.read_text())
    metadata.update(bytes=archive.stat().st_size, sha256=extract_idd.sha256_file(archive))
    source.write_text(json.dumps(metadata))
    with pytest.raises(EOFError):
        run_catalog(tmp_path, archive, source)
    work = tmp_path / "catalog/.extraction" / metadata["sha256"] / "catalog"
    report = json.loads((work / "report.json").read_text())
    assert not report["catalog_ready"] and not report["images_ready"] and not report["usable"]
    assert not (work / "jpeg_inventory.jsonl").exists()


def test_catalog_rejects_unknown_image_layout(tmp_path):
    archive, source, _ = detection_fixture(tmp_path, extra=[(f"{ROOT}/JPEGImages/flat.jpg", b"payload")])
    with pytest.raises(ValueError, match="Unrecognized nested VOC"):
        run_catalog(tmp_path, archive, source)


def test_catalog_requires_official_split_metadata(tmp_path):
    identifier = "front/ride1/frame0001"
    archive, source = archive_fixture(tmp_path, [
        (f"{ROOT}/JPEGImages/{identifier}.jpg", b"payload"),
        (f"{ROOT}/Annotations/{identifier}.xml", b"<annotation/>"),
        (f"{ROOT}/train.txt", (identifier + "\n").encode())])
    with pytest.raises(ValueError, match="official detection split list is absent"):
        run_catalog(tmp_path, archive, source)


@pytest.mark.parametrize("artifact", sorted(extract_idd.IGNORED_DETECTION_ARTIFACTS) + [
    "Annotations/.DS_Store", "Annotations/sideLeft/.DS_Store", "JPEGImages/sideRight/ride/.DS_Store"])
def test_catalog_ignores_only_reviewed_editor_residue(tmp_path, artifact):
    archive, source, _ = detection_fixture(tmp_path, extra=[(ROOT + "/" + artifact, b"editor swap")])
    result = run_catalog(tmp_path, archive, source)
    assert result["catalog_ready"] and result["ignored_detection_artifacts"] == [ROOT + "/" + artifact]
    assert not (tmp_path / "catalog" / artifact).exists()
    unknown = "Annotations/frontFar/BLR-2018-03-22_17-39-26_2_frontFar/.unknown.xml.swp"
    archive, source, _ = detection_fixture(tmp_path, extra=[(ROOT + "/" + unknown, b"unknown")])
    with pytest.raises(ValueError, match="Unrecognized nested VOC"):
        run_catalog(tmp_path, archive, source)


@pytest.mark.parametrize("artifact", [".DS_Store", "unknown/.DS_Store", "Annotations/sideLeft/.DS_Store.xml"])
def test_catalog_does_not_generalize_finder_artifact_to_unknown_paths(tmp_path, artifact):
    archive, source, _ = detection_fixture(tmp_path, extra=[(ROOT + "/" + artifact, b"not reviewed")])
    with pytest.raises(ValueError, match="Unrecognized"):
        run_catalog(tmp_path, archive, source)
