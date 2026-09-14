"""Prepare the author's pretrained YOLOP graphs for Android. No training is run.

The official ONNX graph is pruned to its two segmentation branches. Each output is
the foreground minus background pre-sigmoid score, so threshold 0 is equivalent
to the author's argmax decision at the network pixels. Android applies sigmoid
and resizes cropped probabilities. The documented mobile bilinear resize differs
from the INTER_AREA resize in the author's standalone ONNX demo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "8d8f68df318c71f01d6f813c024df646c7d1978f"
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def resize(array, height, width):
    """Float32 bilinear with half-pixel coordinates, matching Postprocess.bilinear."""
    sh, sw = array.shape[:2]
    x = np.clip((np.arange(width, dtype=np.float64)+.5)*sw/width-.5, 0, sw-1)
    y = np.clip((np.arange(height, dtype=np.float64)+.5)*sh/height-.5, 0, sh-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, sw-1), np.minimum(y0+1, sh-1)
    dx, dy = (x-x0).astype(np.float32)[None, :], (y-y0).astype(np.float32)[:, None]
    if array.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    a = array[y0[:, None], x0[None, :]].astype(np.float32)
    b = array[y0[:, None], x1[None, :]].astype(np.float32)
    c = array[y1[:, None], x0[None, :]].astype(np.float32)
    d = array[y1[:, None], x1[None, :]].astype(np.float32)
    return (a*(1-dx)+b*dx)*(1-dy)+(c*(1-dx)+d*dx)*dy


def prepare(rgb, size):
    h, w = rgb.shape[:2]
    scale = min(size/w, size/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    left, top = (size-rw)//2, (size-rh)//2
    canvas = np.full((size, size, 3), 114, np.float32)
    canvas[top:top+rh, left:left+rw] = np.rint(resize(rgb, rh, rw))
    tensor = ((canvas/np.float32(255)-MEAN)/STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), (left, top, rw, rh)


def project(values, box, height, width, threshold=.5):
    left, top, rw, rh = box
    probability = 1/(1+np.exp(-np.clip(values[0, 0], -80, 80)))
    return resize(probability[top:top+rh, left:left+rw], height, width) >= threshold


def extract(size, destination):
    repo = ROOT/"third_party/YOLOP"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Review a changed upstream revision before export: {commit}")
    original_path = repo/"weights"/f"yolop-{size}-{size}.onnx"
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--",
                    str(original_path.relative_to(repo))], check=True, capture_output=True)
    original = onnx.load(original_path)
    producers = {output: node for node in original.graph.node for output in node.output}
    ends = ["drive_area_seg", "lane_line_seg"]
    raw_names = []
    for name in ends:
        node = producers[name]
        assert node.op_type == "Sigmoid" and len(node.input) == 1
        raw_names.append(node.input[0])
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT/".tmp"/f"yolop-{size}-raw.onnx"
    onnx.utils.extract_model(str(original_path), str(raw_path), ["images"], raw_names)
    model = onnx.load(raw_path)
    for value in [*model.graph.input, *model.graph.value_info]:
        if value.name == "images":
            value.name = "image"
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == "images":
                node.input[i] = "image"
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([0], np.int64), "aurora_background_index"),
        numpy_helper.from_array(np.array([1], np.int64), "aurora_foreground_index"),
    ])
    del model.graph.output[:]
    for task, raw in zip(["road", "lane"], raw_names):
        model.graph.node.extend([
            helper.make_node("Gather", [raw, "aurora_background_index"], [task+"_background"], axis=1),
            helper.make_node("Gather", [raw, "aurora_foreground_index"], [task+"_foreground"], axis=1),
            helper.make_node("Sub", [task+"_foreground", task+"_background"], [task+"_logits"]),
        ])
        model.graph.output.append(helper.make_tensor_value_info(task+"_logits", TensorProto.FLOAT, [1, 1, size, size]))
    model.doc_string = "YOLOP authors' pretrained segmentation branches. No fine-tuning. See accompanying provenance."
    onnx.checker.check_model(model)
    onnx.save(model, destination/"model.onnx")
    source_license = (repo/"LICENSE").read_text(encoding="utf-8")
    metadata = {
        "format_version": 3, "model_family": "yolop_segmentation", "display_name": f"YOLOP {size} baseline",
        "layout": "NCHW", "color": "RGB", "image_shape": [1, 3, size, size],
        "dtype": "float32", "classes": [], "alignment": "none", "mean": MEAN.tolist(), "std": STD.tolist(),
        "pixel_scale": 255, "letterbox": {"pad_value": [114, 114, 114],
        "dimension_rounding": "ties_to_even", "resize": "bilinear_half_pixel", "placement": "center_floor_left_top",
        "independent_xy_scale_after_rounding": True},
        "input_names": ["image"], "input_shapes": {"image": [1, 3, size, size]},
        "outputs": ["road_logits", "lane_logits"],
        "output_shapes": {name+"_logits": [1, 1, size, size] for name in ["road", "lane"]},
        "mask_stride": 1, "mask_projection": "sigmoid_then_crop_bilinear_threshold",
        "mask_scores": "foreground_minus_background_presigmoid",
        "supervised_tasks": ["road", "lane"], "supervised_detection_classes": [],
        "training_status": "pretrained", "trained_on_real_data": True, "live_inference_allowed": True,
        "synthetic_smoke": False, "idd_finetuned": False, "model_sha256": digest(destination/"model.onnx"),
        "source": {"repository": "https://github.com/hustvl/YOLOP", "commit": commit,
                   "onnx_path": str(original_path.relative_to(repo)).replace("\\", "/"),
                   "onnx_sha256": digest(original_path), "training_dataset": "BDD100K",
                   "license": "MIT", "license_text": source_license},
        "limitations": "Pretrained road/lane baseline. No IDD fine-tuning, pothole head, detection or IMU fusion. Phone speed and night accuracy unvalidated.",
        "preprocessing_note": "Mobile bilinear resize differs from official standalone ONNX INTER_AREA demo; static input resolution changes require evaluation."
    }
    (destination/"model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    bundle = ROOT/"artifacts"/f"yolop-road-lane-{size}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("model.onnx", "model.json"):
            archive.write(destination/name, name)
    return original_path, raw_path, metadata


def selected_idd(limit):
    manifest = ROOT/"data/phase2/prepared/research_baseline_v2/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    rows = sorted((r for r in rows if r["split"] == "val" and r["source"] == "idd_seg"), key=lambda r: r["id"])
    selected, sequences = [], set()
    for row in rows:
        if row["sequence"] not in sequences:
            selected.append(row)
            sequences.add(row["sequence"])
        if len(selected) == limit:
            break
    return manifest, selected


def run(size=320, limit=8, assets=False):
    destination = ROOT/"artifacts"/f"yolop-road-lane-{size}"
    source_path, raw_path, metadata = extract(size, destination)
    upstream, raw, mobile = session(source_path), session(raw_path), session(destination/"model.onnx")
    manifest, rows = selected_idd(limit)
    preview_dir = ROOT/"artifacts"/f"yolop-idd-preview-{size}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    test_dir = ROOT/"android/app/src/androidTest/assets/yolop_reference"
    if assets:
        test_dir.mkdir(parents=True, exist_ok=True)
    max_error, min_agreement = 0., 1.
    for index, row in enumerate(rows):
        path = (manifest.parent/row["image"]).resolve()
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Unreadable validation image: {path}")
        rgb = bgr[..., ::-1].copy()
        tensor, box = prepare(rgb, size)
        result = mobile.run(None, {"image": tensor})
        reference_scores = upstream.run(["drive_area_seg", "lane_line_seg"], {"images": tensor})
        reference_raw = raw.run(None, {"images": tensor})
        errors, agreements = [], []
        for actual, scores, logits in zip(result, reference_scores, reference_raw):
            expected = logits[:, 1:2]-logits[:, 0:1]
            error = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
            # Ignore exact ties introduced by saturating the source sigmoid.
            separated = np.abs(scores[:, 1:2]-scores[:, 0:1]) > 1e-6
            agreement = float(((actual > 0) == (scores[:, 1:2] > scores[:, 0:1]))[separated].mean())
            assert agreement > .999
            errors.append(error);agreements.append(agreement)
        max_error = max(max_error, *errors);min_agreement = min(min_agreement, *agreements)
        masks = [project(v, box, *rgb.shape[:2]) for v in result]
        overlay = rgb.copy()
        for mask, color in zip(masks, [(255, 217, 72), (255, 77, 145)]):
            overlay[mask] = np.rint(.45*overlay[mask]+.55*np.array(color)).astype(np.uint8)
        # Diagnostic comparison, preserving the original camera image and model predictions.
        preview = np.concatenate([rgb, overlay], axis=1)
        if preview.shape[1] > 1600:
            preview = cv2.resize(preview, (1600, round(preview.shape[0]*1600/preview.shape[1])))
        cv2.imwrite(str(preview_dir/f"{index+1:02d}.jpg"), preview[..., ::-1])
        case = {"id": row["id"], "sequence": row["sequence"], "image_sha256": digest(path),
                "upstream_agreement": agreements, "raw_margin_max_abs": errors,
                "road_fraction": float(masks[0].mean()), "lane_fraction": float(masks[1].mean()),
                "preview": f"{index+1:02d}.jpg"}
        cases.append(case)
        print(f"{size}: IDD sample {index+1}/{len(rows)}, upstream decision agreement {min(agreements):.6f}", flush=True)
    if assets:
        # Use a deterministic odd-sized colour pattern to exercise resize, padding,
        # network execution and final mask projection without redistributing IDD frames.
        fixtures = []
        for index, (h, w) in enumerate([(97, 161), (163, 99)]):
            yy, xx = np.indices((h, w))
            rgb = np.stack([(xx*13+yy*7)%256, (xx*3+yy*11)%256, (xx*17+yy*5)%256], axis=-1).astype(np.uint8)
            tensor, box = prepare(rgb, size)
            values = mobile.run(None, {"image": tensor})
            cv2.imwrite(str(test_dir/f"{index}.png"), rgb[..., ::-1])
            tensor.astype("<f4").tofile(test_dir/f"{index}-input.bin")
            flags = np.zeros((h, w), np.uint8)
            for bit, (name, value) in enumerate(zip(["road", "lane"], values)):
                value.astype("<f4").tofile(test_dir/f"{index}-{name}.bin")
                flags[project(value, box, h, w)] |= 1 << bit
            flags.tofile(test_dir/f"{index}-mask.bin")
            fixtures.append({"name": str(index), "width": w, "height": h})
        (test_dir/"expected.json").write_text(json.dumps({"size": size, "cases": fixtures, "model_sha256": metadata["model_sha256"]}), encoding="utf-8")
        starter = ROOT/"android/app/src/main/assets/yolop_starter"
        starter.mkdir(parents=True, exist_ok=True)
        for name in ["model.onnx", "model.json"]:
            shutil.copy2(destination/name, starter/name)
    report = {"source": metadata["source"], "model_sha256": metadata["model_sha256"], "size": size,
              "training_performed": False, "idd_finetuned": False, "validation_selection": "First image by ID from each distinct IDD validation sequence",
              "sample_count": len(cases), "raw_margin_max_abs": max_error, "upstream_decision_min_agreement": min_agreement,
              "meaning": "Export equivalence and visual diagnostic subset. Not a benchmark or proof of phone or night performance.", "cases": cases}
    (preview_dir/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"bundle": str(destination), "report": str(preview_dir/"report.json"), "export_max_abs": max_error}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=[320, 640], default=320)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--prepare-assets", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 64:
        parser.error("--limit must be between 1 and 64 for this bounded preview")
    if args.prepare_assets and args.size != 320:
        parser.error("The bundled starter and instrumentation reference use 320 only")
    run(args.size, args.limit, args.prepare_assets)
