"""Parity-gated static ONNX export, including real sensor-conditioned IDFA.

Use --output artifacts/model.zip to make an Android import bundle. Unknown or
synthetic checkpoints require --allow-untrained and remain marked non-live.
"""

import argparse
import json
import numbers
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from .engine import load_checkpoint
from .mobile_contract import OUTPUT_NAMES, build_metadata, training_provenance
from .model import ExportModel, distance_to_boxes, flatten_detection


def _native_outputs(model, image, imu):
    result = model(image, imu)
    classes, distances, center, points, _ = flatten_detection(result["detection"])
    return (result["road_logits"], result["lane_logits"], classes,
            distance_to_boxes(points, distances), center)


def _parity_cases(height, width, uses_imu):
    generator = torch.Generator().manual_seed(123)
    random = torch.randn(1, 3, height, width, generator=generator)
    gradient = torch.linspace(-2., 2., height * width).reshape(1, 1, height, width).expand(1, 3, -1, -1).contiguous()
    images = [random, torch.zeros_like(random), gradient, random.flip(-1).contiguous()]
    attitudes = [(0., 1.), (.43, 1.), (-1.2, .65), (3.14159265, 1.), (.9, 0.)]
    return [(images[index % len(images)], torch.tensor([attitude], dtype=torch.float32))
            for index, attitude in enumerate(attitudes if uses_imu else attitudes[:3])]


@torch.no_grad()
def export_onnx(checkpoint, destination, allow_untrained=False, image_size=None):
    """Write a ZIP or ONNX + JSON only after native/portable/ORT comparisons pass."""
    import onnx
    import onnxruntime as ort

    path = Path(destination)
    if path.suffix.lower() not in (".onnx", ".zip"):
        raise ValueError("Export destination must end in .onnx or .zip")
    if path.exists() or (path.suffix.lower() == ".onnx" and path.with_suffix(".json").exists()):
        raise FileExistsError(f"Output already exists: {path}")
    model, state = load_checkpoint(checkpoint)
    model = model.cpu().float().eval()
    provenance = training_provenance(state)
    if not provenance["live_inference_allowed"] and not allow_untrained:
        raise ValueError("Checkpoint is synthetic or has no verified real-data training provenance; "
                         "use --allow-untrained for a clearly labeled integration fixture")
    wrapper = ExportModel(model, portable_idfa=True).eval()
    chosen_size = image_size if image_size is not None else state["config"]["image_size"]
    if (len(chosen_size) != 2 or any(isinstance(value, bool) or not isinstance(value, numbers.Integral)
                                    or value < 64 or value % 32 for value in chosen_size)):
        raise ValueError("Export image_size must be [height, width], both integer multiples of 32 >= 64")
    h, w = map(int, chosen_size)
    uses_imu = state["config"]["alignment"] == "idfa"
    cases = _parity_cases(h, w, uses_imu)
    example, imu = cases[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    # A failed export never leaves an apparently usable destination model.
    with tempfile.TemporaryDirectory(prefix="export_", dir=path.parent) as temp_dir:
        onnx_path = Path(temp_dir) / "model.onnx"
        metadata_path = Path(temp_dir) / "model.json"
        inputs = (example, imu) if uses_imu else (example,)
        torch.onnx.export(wrapper, inputs, str(onnx_path),
                          input_names=["image", "imu"] if uses_imu else ["image"],
                          output_names=OUTPUT_NAMES, opset_version=17, dynamo=False)
        graph = onnx.load(str(onnx_path))
        # Legacy exporter may leave symbolic output dim labels despite a static
        # graph. Resolve them from the native reference, then verify every value
        # below against the executing graph before publishing the bundle.
        reference_shapes = [list(value.shape) for value in _native_outputs(model, example, imu)]
        for output, shape in zip(graph.graph.output, reference_shapes):
            dimensions = output.type.tensor_type.shape.dim
            for dimension, value in zip(dimensions, shape):
                dimension.ClearField("dim_param")
                dimension.dim_value = value
        onnx.checker.check_model(graph)
        if any(node.domain not in ("", "ai.onnx") for node in graph.graph.node):
            raise RuntimeError("Export contains custom operators unsupported by the standard Android runtime")
        expected_inputs = ["image", "imu"] if uses_imu else ["image"]
        if [item.name for item in graph.graph.input] != expected_inputs:
            raise RuntimeError("Export pruned or changed a required model input")
        onnx.save(graph, str(onnx_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        per_output = {name: {"native_to_portable_max_abs": 0., "native_to_onnx_max_abs": 0.}
                      for name in OUTPUT_NAMES}
        for image, attitude in cases:
            expected = _native_outputs(model, image, attitude)
            portable = wrapper(image, attitude) if uses_imu else wrapper(image)
            feed = {"image": image.numpy()}
            if uses_imu:
                feed["imu"] = attitude.numpy()
            actual = session.run(OUTPUT_NAMES, feed)
            for name, native, local, got in zip(OUTPUT_NAMES, expected, portable, actual):
                reference = native.numpy()
                if not all(np.isfinite(value).all() for value in (reference, local.numpy(), got)):
                    raise FloatingPointError(f"Nonfinite {name} during export parity; destination was not published")
                np.testing.assert_allclose(local.numpy(), reference, atol=2e-4, rtol=2e-3, err_msg=name + " portable")
                np.testing.assert_allclose(got, reference, atol=2e-4, rtol=2e-3, err_msg=name + " ONNX")
                per_output[name]["native_to_portable_max_abs"] = max(
                    per_output[name]["native_to_portable_max_abs"], float(np.abs(local.numpy() - reference).max()))
                per_output[name]["native_to_onnx_max_abs"] = max(
                    per_output[name]["native_to_onnx_max_abs"], float(np.abs(got - reference).max()))
        output_shapes = {name: list(value.shape) for name, value in zip(OUTPUT_NAMES, actual)}
        parity = {"provider": "CPUExecutionProvider", "cases": len(cases), "atol": 2e-4, "rtol": 2e-3,
                  "torch_version": str(torch.__version__), "onnxruntime_version": ort.__version__,
                  "per_output": per_output,
                  "scope": "fixed synthetic tensors and varied roll/confidence; no road accuracy or device benchmark"}
        report = build_metadata(state, onnx_path, output_shapes, parity, [h, w])
        if path.suffix.lower() == ".onnx":
            report["model_file"] = path.name
        metadata_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if path.suffix.lower() == ".zip":
            staged = Path(temp_dir) / "bundle.zip"
            with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(onnx_path, "model.onnx")
                bundle.write(metadata_path, "model.json")
            staged.replace(path)
        else:
            metadata_path.replace(path.with_suffix(".json"))
            onnx_path.replace(path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"),
                        help="Export the same weights at another static resolution; requires task reevaluation")
    args = parser.parse_args()
    print(json.dumps(export_onnx(args.checkpoint, args.output, args.allow_untrained, args.image_size), indent=2))
