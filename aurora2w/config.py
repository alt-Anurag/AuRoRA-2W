"""One taxonomy and explicit configuration for training, inference, and export."""

import json
import math
from pathlib import Path

CLASS_NAMES = (
    "person", "rider", "bicycle", "motorcycle", "autorickshaw",
    "car", "bus", "truck", "pothole",
)
# RGB, stable across videos and model variants. Potholes are red.
COLORS = ((255, 160, 32), (255, 90, 180), (160, 80, 255), (40, 140, 255),
          (0, 220, 220), (70, 210, 90), (240, 210, 40), (160, 130, 70), (255, 40, 40))


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    h, w = config["image_size"]
    if min(h, w) < 64 or h % 32 or w % 32:
        raise ValueError("image_size must be [height, width], both multiples of 32 >= 64")
    if config["alignment"] not in ("none", "idfa"):
        raise ValueError("alignment must be none or idfa")
    if not set(config["idfa_levels"]).issubset({3, 4}) or (config["alignment"] == "idfa" and not config["idfa_levels"]):
        raise ValueError("IDFA supports shared P3 and P4; start with P3 only")
    if any(not math.isfinite(config[k]) or config[k] < 0 for k in ("residual_bound", "max_roll_degrees")):
        raise ValueError("Offset and augmentation bounds must be nonnegative")
    if not 0 <= config["sensor_dropout"] <= 1:
        raise ValueError("sensor_dropout must be a probability")
    for key in ("batch_size", "epochs", "channels", "neck_repeats"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("accumulate_steps", 1) < 1 or config["workers"] < 0:
        raise ValueError("accumulate_steps must be positive; workers must be nonnegative")
    if config.get("lr_schedule", "constant") not in ("constant", "cosine"):
        raise ValueError("lr_schedule must be constant or cosine")
    if not 0 <= config.get("min_lr_factor", .05) <= 1 or config.get("warmup_epochs", 0) < 0:
        raise ValueError("Invalid learning-rate schedule parameters")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer parameters")
    return config
