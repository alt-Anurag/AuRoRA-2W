"""Combine already prepared manifests without regenerating or changing labels.

Every image/mask is decoded and pixel-hashed again across the combined set.
Duplicate records/images are rejected, never silently merged or discarded.
Optional source specs attach only explicit provenance after exact row identity
(image, source/id, sequence and split) checks; task annotations stay unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

from aurora2w.data import audit_records, read_manifest
from aurora2w.prepare import write_manifest

PROVENANCE_KEYS = ("source_release", "selection_protocol", "selection_sha256",
                   "partition_protocol", "partition_group_kind", "partition_artifact_sha256",
                   "source_archive_sha256", "source_archive_member", "source_image_sha256", "source_annotation_sha256")


def merge_prepared(manifests, output, provenance_specs=(), image_size=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    records, origins = [], []
    for manifest in manifests:
        path = Path(manifest).resolve()
        rows = read_manifest(path)
        records.extend(rows)
        origins.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)})
    definitions, spec_evidence = {}, []
    for spec in provenance_specs:
        path = Path(spec).resolve()
        for source in json.loads(path.read_text(encoding="utf-8-sig"))["sources"]:
            root = (path.parent / source["root"]).resolve()
            index = (path.parent / source["index"]).resolve()
            spec_evidence.append({"spec": str(path), "spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "index": str(index), "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()})
            for line in index.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                key = (source["name"], item["id"])
                if key in definitions:
                    raise ValueError(f"Repeated provenance identity: {key}")
                metadata = {key: item.get(key, source.get(key)) for key in PROVENANCE_KEYS if key in item or key in source}
                definitions[key] = (str((root / item["image"]).resolve()), item["sequence"], item["split"], metadata)
    matched = set()
    for row in records:
        key = (row["source"], row["id"])
        if key in definitions:
            image, sequence, split, metadata = definitions[key]
            if (row["image"], row["sequence"], row["split"]) != (image, sequence, split):
                raise ValueError(f"Provenance identity mismatch: {key}")
            for field, value in metadata.items():
                if field in row and row[field] != value:
                    raise ValueError(f"Conflicting provenance {field}: {key}")
                row[field] = value
            matched.add(key)
    if set(definitions) != matched:
        raise ValueError("Provenance spec contains records absent from the supplied manifests")
    audit = audit_records(records, check_hashes=True, image_size=image_size)
    write_manifest(records, output)
    report = {"operation": "concatenate_prepared_rows_and_reaudit_without_label_changes",
              "training_approved": False, "inputs": origins, "provenance_specs": spec_evidence,
              "provenance_rows_matched": len(matched), "audit": audit,
              "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (output.parent / (output.stem + "_MERGE.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--provenance-specs", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="Also audit detector sampling-center assignment at the configured input resolution")
    args = parser.parse_args()
    from aurora2w.config import load_config
    image_size = load_config(args.config)["image_size"] if args.config else None
    print(json.dumps(merge_prepared(args.manifests, args.output, args.provenance_specs, image_size), indent=2))
