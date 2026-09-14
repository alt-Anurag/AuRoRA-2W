"""Pure metadata checks: these tests never instantiate or train a model."""

import pytest

from scripts.training_preflight import review_records


def configuration(**overrides):
    return {"batch_size": 8, "image_size": [384, 640], "amp": True,
            "workers": 2, "accumulate_steps": 1, "source_balance": True,
            "epochs": 40, "alignment": "none", "channels": 64, "neck_repeats": 2,
            "idfa_levels": [3], **overrides}


def row(identity, source, split="train"):
    return {"id": identity, "source": source, "sequence": f"{source}/{identity}",
            "split": split, "data_kind": "real", "boxes": [], "detection_classes": []}


def test_source_balance_keeps_epoch_length_but_repeats_small_sources():
    records = [row("small", "a")] + [row(str(i), "b") for i in range(3)]
    result = review_records(records, configuration(), {})
    assert result["sampling"]["draws_per_epoch"] == 4
    assert result["sampling"]["batches_per_epoch"] == 1
    small, large = (result["sampling"]["sources"][key] for key in ("a", "b"))
    assert small["expected_draws_per_epoch"] == large["expected_draws_per_epoch"] == 2
    assert small["expected_distinct_rows_per_epoch"] == pytest.approx(15 / 16)
    assert large["expected_distinct_rows_per_epoch"] == pytest.approx(3 * (1 - (5 / 6) ** 4))
    unbalanced = review_records(records, configuration(source_balance=False), {})
    assert unbalanced["sampling"]["sources"]["b"]["expected_distinct_rows_per_epoch"] == 3


def test_missing_files_validation_tasks_and_geometry_fail_closed():
    result = review_records([row("1", "rdd")], configuration(alignment="idfa"), {}, ["missing.jpg"])
    assert result["status"] == "blocked"
    assert result["training_approved"] is False
    assert "No val records" in result["blockers"]
    assert "No train road mask references" in result["blockers"]
    assert "1 referenced image/mask files are missing" in result["blockers"]
    assert any("IDFA has no" in reason for reason in result["blockers"])


def test_estimate_uses_rounded_batches_and_rejects_unmeasured_config():
    records = [row(str(i), "a") for i in range(9)]
    profiles = {"profiles": {"none_batch8": {"median_training_step_s": 0.2, "images_per_second": 40}}}
    result = review_records(records, configuration(), profiles)
    assert result["timing"]["train_only_planned_hours"] == pytest.approx(2 * 0.2 * 40 / 3600)
    changed = review_records(records, configuration(batch_size=4), profiles)
    assert changed["timing"]["available"] is False


def test_cross_split_sequences_and_nonreal_records_are_blocked():
    records = [row("1", "a"), row("2", "a", "val")]
    records[1].update(sequence=records[0]["sequence"], data_kind="synthetic")
    result = review_records(records, configuration(), {})
    assert "1 source/sequence groups cross splits" in result["blockers"]
    assert "1 train/val records are not declared real photographed data" in result["blockers"]



def test_read_only_timing_evidence_rejects_training_and_pins_file(tmp_path):
    import hashlib
    import json
    from scripts.training_preflight import timing_evidence
    report = {'purpose': 'real_training_input_delivery_only', 'optimizer_steps': 0, 'backward_calls': 0,
              'accuracy_reported': False, 'model_instances': 0, 'image_size': [384, 640], 'batch_size': 8,
              'train_records': 10, 'train_records_by_source': {'rdd': 10}, 'steady_images_per_second': 30,
              'steady_mean_wall_seconds_per_batch': 8/30, 'startup_to_first_batch_seconds': 1}
    path = tmp_path / 'timing.json'
    path.write_text(json.dumps(report))
    assert timing_evidence([path])[0]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    report['optimizer_steps'] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='includes training'):
        timing_evidence([path])
