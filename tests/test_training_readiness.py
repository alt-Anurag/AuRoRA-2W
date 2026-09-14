import copy

import pytest
import torch

from aurora2w.config import load_config
from aurora2w.data import read_manifest
from aurora2w.engine import train, learning_rate_factor
from aurora2w.smoke import make_fixture


def test_resume_matches_uninterrupted_cpu_and_rejects_changed_data(tmp_path):
    torch.set_num_threads(2)
    config = load_config("configs/rtx4050_fixture.json")
    config.update(channels=16, accumulate_steps=3, max_roll_degrees=15)
    records = read_manifest(make_fixture(tmp_path / "data"))
    uninterrupted = train(config, records, tmp_path / "full")
    first = train(config, records, tmp_path / "resume", epochs=1)
    resumed = train(config, records, tmp_path / "resume", resume=first, epochs=2)
    left = torch.load(uninterrupted, weights_only=True)
    right = torch.load(resumed, weights_only=True)
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], atol=0, rtol=0)
    assert right["synthetic_smoke"] and not right["trained_on_real_data"]
    assert right["supervised_tasks"] == ["lane", "road"]
    assert right["supervised_detection_classes"] == ["car", "pothole"]
    changed = copy.deepcopy(records)
    changed[0]["detection_classes"] = []
    with pytest.raises(ValueError, match="identical data"):
        train(config, changed, tmp_path / "resume", resume=resumed, epochs=3)


def test_schedule_warmup_and_cosine_floor():
    config = {"epochs": 10, "warmup_epochs": 2, "lr_schedule": "cosine", "min_lr_factor": .1}
    assert learning_rate_factor(0, config) == .5
    assert learning_rate_factor(1, config) == 1
    assert learning_rate_factor(2, config) == 1
    assert learning_rate_factor(9, config) == pytest.approx(.1)
    assert learning_rate_factor(20, config) == pytest.approx(.1)
