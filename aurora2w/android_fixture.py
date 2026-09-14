"""Generate clearly untrained assets exclusively for Android instrumentation tests.

Never package these weights in src/main/assets or enable them in live perception.
The production importer must reject their synthetic provenance. The test builds
an InferenceEngine directly to compare the real Android ORT execution with native
PyTorch tensors. Temporary checkpoints are removed after generation.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from .config import CLASS_NAMES, load_config
from .export import _native_outputs, export_onnx
from .mobile_contract import OUTPUT_NAMES, sha256_file
from .model import MultiTaskNet


@torch.no_grad()
def build_fixture(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh fixture directory: {output}")
    torch.set_num_threads(2)
    torch.manual_seed(901)
    config = load_config("configs/phase2.json")
    config.update(image_size=[64, 96], channels=8, neck_repeats=1,
                  pretrained_backbone=False, alignment="idfa", idfa_levels=[3])
    model = MultiTaskNet(config)
    # Synthetic statistics avoid vanishing untrained activations, so changing
    # the IMU has an observable effect in the test. No road labels or accuracy.
    model.alignment["3"].refine[-1].bias.uniform_(-.3, .3)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 1.0
    model.train()
    model(torch.randn(4, 3, 64, 96), torch.tensor([[.2, 1.]] * 4))
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fixture_", dir=output.parent) as temp:
        checkpoint = Path(temp) / "synthetic.pt"
        torch.save({"config": config, "model": model.state_dict(), "classes": list(CLASS_NAMES),
                    "epoch": -1, "synthetic_smoke": True, "trained_on_real_data": False,
                    "supervised_tasks": [], "supervised_detection_classes": []}, checkpoint)
        report = export_onnx(checkpoint, output / "model.onnx", allow_untrained=True)
    image = torch.randn(1, 3, 64, 96)
    cases = []
    for roll, confidence in ((.35, 1.), (-.2, 1.), (.35, 0.)):
        imu = torch.tensor([[roll, confidence]])
        results = _native_outputs(model, image, imu)
        cases.append({"input": image.flatten().tolist(), "imu": [roll, confidence],
                      "outputs": {name: result.flatten().tolist()
                                  for name, result in zip(OUTPUT_NAMES, results)}})
    delta = max(abs(a-b) for a, b in zip(cases[0]["outputs"]["road_logits"], cases[2]["outputs"]["road_logits"]))
    if delta <= 1e-6:
        raise AssertionError("Fixture must exercise an observable sensor-conditioned output change")
    expected = {"format_version": 1, "atol": 2e-4, "rtol": 2e-3,
                "generator_sha256": sha256_file(__file__), "model_sha256": report["model_sha256"],
                "purpose": "Untrained synthetic native-PyTorch versus Android-ORT numerical check only",
                "sensor_effect_road_max_abs": delta, "cases": cases}
    (output / "expected.json").write_text(json.dumps(expected, allow_nan=False), encoding="utf-8")
    return {"directory": str(output), "model_sha256": report["model_sha256"],
            "cases": len(cases), "sensor_effect_road_max_abs": delta,
            "live_inference_allowed": False}


def ensure_fixture(output):
    """Reuse verified generated assets or replace only the three known fixtures."""
    output = Path(output)
    try:
        metadata = json.loads((output / "model.json").read_text(encoding="utf-8"))
        expected = json.loads((output / "expected.json").read_text(encoding="utf-8"))
        digest = sha256_file(output / "model.onnx")
        if (metadata["model_sha256"] == expected["model_sha256"] == digest
                and expected["generator_sha256"] == sha256_file(__file__)
                and metadata["synthetic_smoke"] is True and metadata["live_inference_allowed"] is False):
            return {"directory": str(output), "reused": True, "model_sha256": digest,
                    "cases": len(expected["cases"]), "live_inference_allowed": False}
    except (OSError, ValueError, KeyError):
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fixture_build_", dir=output.parent) as temp:
        staged = Path(temp) / "bundle"
        result = build_fixture(staged)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("model.onnx", "model.json", "expected.json"):
            (staged / name).replace(output / name)
    return {**result, "directory": str(output), "reused": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="android/app/src/androidTest/assets/untrained_idfa_fixture")
    parser.add_argument("--ensure", action="store_true", help="Reuse hash-verified assets or regenerate the test-only files")
    arguments = parser.parse_args()
    print(json.dumps((ensure_fixture if arguments.ensure else build_fixture)(arguments.output), indent=2))
