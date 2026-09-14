"""Selectively extract verified IDD segmentation/VOC archives without extractall.

Completed files survive interrupted runs. Re-running streams the archive again
and checks every existing file against its archived SHA-256; gzip cannot resume
at an arbitrary compressed offset. A final usable=true report certifies selected
images; detection-catalog instead sets catalog_ready=true and images_ready=false.
Both require archive SHA-256 and complete gzip stream verification.
Detection mode requires a reviewed, archive-bound selection plan described in
docs/IDD_ACCESS.md; it never invents a split or selects images by archive order.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import tarfile
import time


CHUNK = 1024 * 1024
RESERVE_BYTES = 10 * 1024**3
# Observed non-dataset editor residue in the completed official classic archive.
# Keep the exception exact; other unrecognized paths still require inspection.
IGNORED_DETECTION_ARTIFACTS = {
    "Annotations/frontFar/BLR-2018-03-22_17-39-26_2_frontFar/.001542_r.xml.swp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_parts(name: str) -> tuple[str, ...]:
    """Reject portable path escapes, Windows aliases/ADS and ambiguous names."""
    if not name or "\\" in name or "\x00" in name or PureWindowsPath(name).drive or name.startswith("/"):
        raise ValueError(f"Unsafe archive path: {name!r}")
    parts = tuple(name.rstrip("/").split("/"))
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in parts:
        if part in ("", ".", "..") or part.endswith((".", " ")) or any(c in part for c in ':<>"|?*') or any(ord(c) < 32 for c in part):
            raise ValueError(f"Unsafe archive path: {name!r}")
        if part.split(".")[0].upper() in reserved:
            raise ValueError(f"Reserved Windows path: {name!r}")
    return parts


def ensure_no_links(path: Path) -> None:
    """Also reject existing destination junctions; never follow them to write."""
    for component in (path, *path.parents):
        if component.is_symlink() or getattr(component, "is_junction", lambda: False)():
            raise ValueError(f"Destination link/junction is not allowed: {component}")


def load_detection_plan(path: Path, archive_sha256: str, strip_root: str) -> dict:
    """A reviewed, archive-bound group selection is mandatory for VOC extraction."""
    path = Path(path).absolute()
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if plan.get("schema_version") != 1 or plan.get("reviewed_layout") != "idd_detection_nested_voc_v1":
        raise ValueError("Detection plan requires an explicitly reviewed nested VOC archive layout")
    if plan.get("archive_sha256") != archive_sha256 or plan.get("strip_root") != strip_root:
        raise ValueError("Detection selection plan does not match this archive SHA/root")
    budget = plan.get("maximum_image_bytes")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 0 < budget <= 10 * 1024**3:
        raise ValueError("Detection image budget must be positive and at most 10 GiB")
    groups = (path.parent / plan["groups_csv"]).resolve()
    if sha256_file(groups) != plan.get("groups_sha256"):
        raise ValueError("Frozen detection groups CSV SHA-256 mismatch")
    selected, group_splits, physical_groups = {}, {}, {}
    with groups.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not {"image", "sequence", "split"}.issubset(rows.fieldnames or []):
            raise ValueError("Frozen groups need image,sequence,split columns")
        for row in rows:
            image_parts = checked_parts(row["image"])
            if len(image_parts) != 4 or image_parts[0] != "JPEGImages" or Path(image_parts[-1]).suffix.lower() != ".jpg":
                raise ValueError("Selected detection image must use JPEGImages/category/sequence/frame.jpg")
            if row["split"] not in ("train", "val") or not row["sequence"].strip():
                raise ValueError("Working detection selection needs explicit train/val groups")
            if row["sequence"] in group_splits and group_splits[row["sequence"]] != row["split"]:
                raise ValueError("Frozen detection group crosses train/val splits")
            group_splits[row["sequence"]] = row["split"]
            physical = "/".join(image_parts[1:3]).casefold()
            assignment = (row["sequence"], row["split"])
            if physical in physical_groups and physical_groups[physical] != assignment:
                raise ValueError("A physical detection sequence is split across frozen groups")
            physical_groups[physical] = assignment
            key = "/".join(image_parts).casefold()
            if key in selected:
                raise ValueError("Frozen detection selection repeats a canonical image")
            selected[key] = row
    if not selected:
        raise ValueError("Frozen detection selection is empty")
    return {**plan, "plan_path": str(path), "plan_sha256": sha256_file(path),
            "groups_path": str(groups), "selected": selected}


def ignored_detection_artifact(parts: tuple[str, ...]) -> bool:
    """Recognize exact editor residue and Finder metadata in reviewed trees only."""
    return "/".join(parts) in IGNORED_DETECTION_ARTIFACTS or (
        len(parts) >= 2 and parts[0] in ("Annotations", "JPEGImages") and parts[-1] == ".DS_Store"
    )


def selected_path(parts: tuple[str, ...], strip_root: str, mode: str = "segmentation",
                  plan: dict | None = None, is_directory: bool = False) -> Path | None:
    if parts[0] != strip_root:
        raise ValueError(f"Unexpected archive root {parts[0]!r}; expected {strip_root!r}")
    parts = parts[1:]
    if is_directory:
        return None
    if mode in ("detection", "detection-catalog"):
        if mode == "detection" and plan is None:
            raise ValueError("Detection extraction requires a reviewed frozen selection plan")
        if len(parts) == 1 and parts[0] in ("train.txt", "val.txt", "test.txt"):
            return Path(*parts)
        if ignored_detection_artifact(parts):
            return None
        if parts and parts[0] in ("Annotations", "JPEGImages"):
            extension = ".xml" if parts[0] == "Annotations" else ".jpg"
            if len(parts) != 4 or Path(parts[-1]).suffix.lower() != extension:
                raise ValueError(f"Unrecognized nested VOC member schema: {'/'.join(parts)}")
            relative = Path(*parts)
            if mode == "detection-catalog" or parts[0] == "Annotations" or relative.as_posix().casefold() in plan["selected"]:
                return relative
            return None  # Recognized JPEG deliberately excluded by the frozen selection.
        if len(parts) == 1 and parts[0] in ("README", "README.md", "README.txt", "LICENSE", "LICENSE.txt"):
            return None
        raise ValueError(f"Unrecognized detection archive member: {'/'.join(parts)}")
    if len(parts) < 4 or parts[1] not in ("train", "val"):
        return None
    image = parts[0] == "leftImg8bit" and Path(parts[-1]).suffix.lower() in (".png", ".jpg", ".jpeg")
    polygon = parts[0] == "gtFine" and parts[-1].endswith("_gtFine_polygons.json")
    return Path(*parts) if image or polygon else None


def read_official_detection_splits(destination: Path, seen: dict) -> dict:
    official = {}
    for split in ("train", "val", "test"):
        name = split + ".txt"
        if name not in seen:
            raise ValueError(f"Required official detection split list is absent: {name}")
        for line in (destination / name).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            identifier = line.strip()
            parts = checked_parts(identifier)
            if len(parts) != 3 or any(c.isspace() for c in identifier):
                raise ValueError("Unrecognized detection split ID; expected category/sequence/frame")
            key = identifier.casefold()
            if key in official:
                raise ValueError("Official detection split IDs overlap or repeat")
            official[key] = split
    return official


def validate_detection_selection(destination: Path, plan: dict, seen: dict) -> dict:
    """Require exact selected pairs and preserve the archive's official split IDs."""
    official = read_official_detection_splits(destination, seen)
    counts = {"train": 0, "val": 0}
    for key, row in plan["selected"].items():
        image = Path(row["image"])
        identifier = Path(*image.parts[1:]).with_suffix("").as_posix()
        annotation = "Annotations/" + identifier + ".xml"
        if key not in seen or annotation.casefold() not in seen:
            raise ValueError(f"Selected image/XML counterpart absent in archive: {row['image']}")
        if official.get(identifier.casefold()) != row["split"]:
            raise ValueError(f"Frozen selection changes or lacks official split membership: {identifier}")
        counts[row["split"]] += 1
    return counts


def catalog_image_entry(relative: Path, member: tarfile.TarInfo, tar: tarfile.TarFile,
                        archive_sha256: str) -> dict:
    """Hash compressed-archive JPEG payloads without decoding or writing images."""
    payload = tar.extractfile(member)
    if payload is None:
        raise ValueError(f"Missing catalog JPEG payload: {member.name}")
    digest, count = hashlib.sha256(), 0
    with payload:
        for block in iter(lambda: payload.read(CHUNK), b""):
            digest.update(block)
            count += len(block)
    if count != member.size:
        raise ValueError(f"Truncated catalog JPEG: {member.name}")
    category, directory = relative.parts[1:3]
    recording_prefix = re.sub(r"_part_\d+$", "", directory)
    identifier = Path(*relative.parts[1:]).with_suffix("").as_posix()
    return {"schema_version": 1, "id": identifier, "image": relative.as_posix(),
            "annotation": "Annotations/" + identifier + ".xml", "archive_member": member.name,
            "archive_sha256": archive_sha256, "bytes": count, "sha256": digest.hexdigest(),
            "capture_category": category, "sequence_directory": directory,
            "recording_prefix": recording_prefix, "recording_group": category + "/" + recording_prefix}


def finish_catalog(destination: Path, work: Path, images: dict, seen: dict) -> dict:
    official = read_official_detection_splits(destination, seen)
    if not images or not any(Path(entry["path"]).parts[0] == "Annotations" for entry in seen.values()):
        raise ValueError("Catalog needs JPEG inventory and XML annotations")
    inventory = work / "jpeg_inventory.jsonl"
    partial = work / "jpeg_inventory.jsonl.partial"
    ensure_no_links(inventory)
    ensure_no_links(partial)
    split_counts = {"train": 0, "val": 0, "test": 0, "unlisted": 0}
    missing_xml = {key: 0 for key in split_counts}
    identifiers = set()
    with partial.open("w", encoding="utf-8", newline="\n") as output:
        for key in sorted(images):
            entry = dict(images[key])
            identifier = entry["id"].casefold()
            identifiers.add(identifier)
            entry["official_split"] = official.get(identifier)
            annotation = seen.get(entry["annotation"].casefold())
            entry["annotation_present"] = annotation is not None
            if annotation:
                entry["annotation"] = annotation["path"]
                entry["annotation_sha256"] = annotation["sha256"]
            split = entry["official_split"] or "unlisted"
            split_counts[split] += 1
            missing_xml[split] += int(annotation is None)
            encoded = json.dumps(entry, separators=(",", ":")) + "\n"
            require_space(destination, len(encoded.encode("utf-8")))
            output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, inventory)
    return {"jpeg_inventory": str(inventory), "jpeg_inventory_sha256": sha256_file(inventory),
            "catalog_image_split_counts": split_counts, "catalog_images_without_xml": missing_xml,
            "official_ids_without_jpeg": sum(key not in identifiers for key in official)}


def require_space(root: Path, incoming: int = 0) -> None:
    if shutil.disk_usage(root).free - incoming < RESERVE_BYTES:
        raise OSError("Extraction would reduce D-drive free space below the 10 GiB reserve")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    ensure_no_links(temporary)
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_archive(archive: Path, source_path: Path, destination: Path, strip_root: str,
                    *, mode: str = "segmentation", detection_plan: Path | None = None) -> dict:
    archive, source_path = Path(archive).absolute(), Path(source_path).absolute()
    destination = Path(destination).absolute()
    if checked_parts(strip_root) != (strip_root,):
        raise ValueError("strip_root must be one explicit top-level directory")
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    expected = source.get("sha256", "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected.lower()):
        raise ValueError("SOURCE metadata needs a SHA-256 archive digest")
    if source.get("archive") != archive.name or source.get("bytes") != archive.stat().st_size:
        raise ValueError("Archive name/size do not match SOURCE metadata")
    if mode not in ("segmentation", "detection", "detection-catalog"):
        raise ValueError("Unknown extraction mode")
    if mode == "detection" and detection_plan is None:
        raise ValueError("Detection extraction requires a reviewed frozen selection plan")
    plan = load_detection_plan(detection_plan, expected, strip_root) if mode == "detection" else None
    ensure_no_links(destination)
    destination.mkdir(parents=True, exist_ok=True)
    require_space(destination)
    print(json.dumps({"stage": "archive_sha256", "archive": str(archive)}), flush=True)
    initial_stat = archive.stat()
    actual = sha256_file(archive)
    if actual.lower() != expected.lower():
        raise ValueError("Archive SHA-256 does not match SOURCE metadata")
    work = destination / ".extraction" / actual
    if mode == "detection-catalog":
        work /= "catalog"
    if plan:
        work /= "selection-" + plan["plan_sha256"]
    ensure_no_links(work)
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "report.json"
    manifest_path = work / "files.jsonl"
    partial_manifest = work / "files.jsonl.partial"
    temporary_file = work / "current-file.partial"
    for path in (report_path, manifest_path, partial_manifest, temporary_file):
        ensure_no_links(path)
    report = {"schema_version": 1, "usable": False, "catalog_ready": False, "images_ready": False,
              "mode": mode, "archive": str(archive), "archive_sha256": actual,
              "source_metadata": str(source_path), "hash_scope": source.get("hash_scope", "SOURCE metadata"),
              "destination": str(destination), "strip_root": strip_root, "minimum_free_bytes": RESERVE_BYTES,
              "manifest": str(manifest_path), "files": 0, "selected_bytes": 0, "written": 0,
              "existing_verified": 0, "duplicate_members_verified": 0, "skipped_members": 0,
              "ignored_detection_artifacts": [],
              "counts": {}, "gzip_eof_verified": False, "started_unix": time.time()}
    if plan:
        report.update(selection_plan=plan["plan_path"], selection_plan_sha256=plan["plan_sha256"],
                      groups_sha256=plan["groups_sha256"], maximum_image_bytes=plan["maximum_image_bytes"],
                      selected_image_bytes=0, requested_images=len(plan["selected"]))
    if mode == "detection-catalog":
        report.update(catalog_images=0, catalog_image_bytes=0, catalog_duplicate_images_verified=0,
                      readiness_note="Metadata catalog only; JPEG images are not extracted or training-ready")
    write_json(report_path, report)
    seen: dict[str, dict] = {}
    catalog_images: dict[str, dict] = {}
    last_progress = time.monotonic()
    def progress() -> None:
        nonlocal last_progress
        if time.monotonic() - last_progress >= 15:
            write_json(report_path, report)
            print(json.dumps({"stage": "cataloging" if mode == "detection-catalog" else "extracting",
                              "files": report["files"], "written": report["written"],
                              "existing_verified": report["existing_verified"],
                              "catalog_images": report.get("catalog_images", 0)}), flush=True)
            last_progress = time.monotonic()
    try:
        with partial_manifest.open("w", encoding="utf-8", newline="\n") as manifest:
            with gzip.open(archive, "rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|") as tar:
                    for member in tar:
                        parts = checked_parts(member.name)
                        if not (member.isdir() or member.isfile()) or member.issparse():
                            raise ValueError(f"Links, sparse files and special members are forbidden: {member.name}")
                        if member.size < 0:
                            raise ValueError(f"Negative archive member size: {member.name}")
                        relative = selected_path(parts, strip_root, mode, plan, member.isdir())
                        if member.isdir() or relative is None:
                            report["skipped_members"] += 1
                            if mode.startswith("detection") and not member.isdir() and ignored_detection_artifact(parts[1:]):
                                report["ignored_detection_artifacts"].append(member.name)
                            progress()
                            continue
                        key = relative.as_posix().casefold()
                        if mode == "detection-catalog" and relative.parts[0] == "JPEGImages":
                            require_space(destination)
                            entry = catalog_image_entry(relative, member, tar, actual)
                            prior_image = catalog_images.get(key)
                            if prior_image:
                                if prior_image["sha256"] != entry["sha256"] or prior_image["bytes"] != entry["bytes"]:
                                    raise ValueError(f"Duplicate canonical catalog JPEG conflicts: {relative}")
                                report["catalog_duplicate_images_verified"] += 1
                            else:
                                catalog_images[key] = entry
                                report["catalog_images"] += 1
                                report["catalog_image_bytes"] += entry["bytes"]
                            progress()
                            continue
                        prior = seen.get(key)
                        if plan and relative.parts[0] == "JPEGImages" and not prior:
                            if report["selected_image_bytes"] + member.size > plan["maximum_image_bytes"]:
                                raise OSError("Frozen detection selection exceeds its image byte budget")
                        if prior:
                            relative = Path(prior["path"])
                        target = destination / relative
                        ensure_no_links(target)
                        if not target.resolve().is_relative_to(destination.resolve()):
                            raise ValueError(f"Destination escape: {relative}")
                        if target.exists() and (not target.is_file() or target.stat().st_size != member.size):
                            raise ValueError(f"Existing canonical file conflicts with archive: {relative}")
                        require_space(destination, 0 if target.exists() else member.size)
                        archived = tar.extractfile(member)
                        if archived is None:
                            raise ValueError(f"Missing member payload: {member.name}")
                        digest, size = hashlib.sha256(), 0
                        sink = None if target.exists() else temporary_file.open("wb")
                        try:
                            with archived:
                                for block in iter(lambda: archived.read(CHUNK), b""):
                                    if sink:
                                        require_space(destination, len(block))
                                        sink.write(block)
                                    digest.update(block)
                                    size += len(block)
                            if sink:
                                sink.flush()
                                os.fsync(sink.fileno())
                        finally:
                            if sink:
                                sink.close()
                        if size != member.size:
                            raise ValueError(f"Truncated member: {member.name}")
                        entry = {"path": relative.as_posix(), "archive_member": member.name,
                                 "bytes": size, "sha256": digest.hexdigest()}
                        if prior and prior["sha256"] != entry["sha256"]:
                            raise ValueError(f"Duplicate canonical archive path conflicts: {relative}")
                        if target.exists():
                            if sha256_file(target) != entry["sha256"]:
                                raise ValueError(f"Existing canonical file hash conflicts with archive: {relative}")
                            report["existing_verified"] += 1
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            ensure_no_links(target)
                            # os.rename does not overwrite on Windows; preserve any concurrent raw file.
                            os.rename(temporary_file, target)
                            report["written"] += 1
                        if prior:
                            report["duplicate_members_verified"] += 1
                            continue
                        seen[key] = entry
                        manifest.write(json.dumps(entry, separators=(",", ":")) + "\n")
                        manifest.flush()
                        report["files"] += 1
                        report["selected_bytes"] += size
                        if plan and relative.parts[0] == "JPEGImages":
                            report["selected_image_bytes"] += size
                        group = "/".join(relative.parts[:2])
                        report["counts"][group] = report["counts"].get(group, 0) + 1
                        progress()
                # Tar EOF can occur before gzip trailer/member EOF. Drain to validate
                # every CRC/ISIZE and concatenated gzip member, including skipped data.
                for _ in iter(lambda: compressed.read(CHUNK), b""):
                    pass
            os.fsync(manifest.fileno())
        final_stat = archive.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError("Archive changed during extraction")
        require_space(destination)
        if not report["files"]:
            raise ValueError("No matching train/val images or polygon labels were found")
        if plan:
            report["selected_official_split_counts"] = validate_detection_selection(destination, plan, seen)
        if mode == "detection-catalog":
            report.update(finish_catalog(destination, work, catalog_images, seen))
        os.replace(partial_manifest, manifest_path)
        report.update(usable=mode != "detection-catalog", images_ready=mode != "detection-catalog",
                      catalog_ready=mode == "detection-catalog", gzip_eof_verified=True, finished_unix=time.time(),
                      free_bytes_after=shutil.disk_usage(destination).free)
        write_json(report_path, report)
        return report
    except BaseException as error:
        report.update(usable=False, catalog_ready=False, images_ready=False,
                      error=f"{type(error).__name__}: {error}", interrupted_unix=time.time())
        write_json(report_path, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--strip-root", required=True, help="Exact top-level folder observed in this archive")
    parser.add_argument("--mode", choices=("segmentation", "detection", "detection-catalog"), default="segmentation")
    parser.add_argument("--detection-plan", type=Path, help="Archive-bound reviewed nested-VOC layout and frozen group selection")
    args = parser.parse_args()
    local_inputs = [args.archive, args.source, args.destination]
    if args.detection_plan:
        local_inputs.append(args.detection_plan)
    if os.name == "nt" and any(path.absolute().drive.upper() != "D:" for path in local_inputs):
        parser.error("This project keeps archives, metadata, and extracted datasets on D:")
    print(json.dumps(extract_archive(args.archive, args.source, args.destination, args.strip_root,
                                    mode=args.mode, detection_plan=args.detection_plan), indent=2))


if __name__ == "__main__":
    main()
