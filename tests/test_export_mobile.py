"""Numerical portability and provenance checks; these are not accuracy benchmarks."""

import copy
import hashlib
import json
import math
import zipfile

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.ops import deform_conv2d

from aurora2w.config import CLASS_NAMES, load_config
from aurora2w.export import export_onnx
from aurora2w.idfa import RollConditionedConv, portable_deform_conv3x3
from aurora2w.mobile_contract import training_provenance
from aurora2w.model import MultiTaskNet, ExportModel


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.manual_seed(71)
    torch.set_num_threads(2)


@pytest.mark.parametrize("height,width", [(1, 1), (1, 7), (5, 1), (7, 11)])
@pytest.mark.parametrize("offset_scale", [0., .5, 3., 100.])
def test_portable_taps_match_native_at_image_bounds(height, width, offset_scale):
    x = torch.randn(2, 3, height, width)
    weight = torch.randn(5, 3, 3, 3)
    offsets = torch.randn(2, 18, height, width) * offset_scale
    actual = portable_deform_conv3x3(x, offsets, weight)
    reference = deform_conv2d(x, offsets, weight, padding=(1, 1))
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("roll,confidence", [(0., 1.), (math.pi/2, 1.), (-.63, .4), (1., 0.), (0., 0.)])
def test_portable_idfa_preserves_roll_refinement_and_confidence(roll, confidence):
    native = RollConditionedConv(4).eval()
    with torch.no_grad():
        native.refine[-1].weight.normal_(0, .2)
        native.refine[-1].bias.uniform_(-1, 1)
    portable = copy.deepcopy(native)
    portable.portable = True
    x = torch.randn(2, 4, 13, 17)
    imu = torch.tensor([[roll, confidence], [-roll, confidence]])
    torch.testing.assert_close(portable(x, imu), native(x, imu), atol=2e-5, rtol=2e-5)
    if confidence == 0:
        ordinary = RollConditionedConv(4, deformable=False).eval()
        ordinary.conv.load_state_dict(native.conv.state_dict())
        ordinary.norm.load_state_dict(native.norm.state_dict())
        torch.testing.assert_close(portable(x, imu), ordinary(x, imu), atol=2e-5, rtol=2e-5)


def test_portable_deformed_sampling_matches_onnx_cpu(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    class TapModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))

        def forward(self, image, offsets):
            return portable_deform_conv3x3(image, offsets, self.weight)

    module = TapModule().eval()
    x, offsets = torch.randn(1, 3, 9, 13), torch.randn(1, 18, 9, 13)
    path = tmp_path / "taps.onnx"
    torch.onnx.export(module, (x, offsets), str(path), input_names=["image", "offsets"],
                      output_names=["sampled"], opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for scale in (0., .5, 2., 100.):
        moved = offsets * scale
        expected = deform_conv2d(x, moved, module.weight, padding=(1, 1)).detach().numpy()
        got = session.run(None, {"image": x.numpy(), "offsets": moved.numpy()})[0]
        np.testing.assert_allclose(got, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("alignment", ["none", "idfa"])
@pytest.mark.parametrize("export_size", [None, [96, 128]])
def test_model_bundle_export_parity_and_provenance(tmp_path, alignment, export_size):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    config = load_config("configs/phase2.json")
    config.update(image_size=[64, 96], channels=8, neck_repeats=1,
                  pretrained_backbone=False, alignment=alignment)
    model = MultiTaskNet(config).eval()
    wrapper = ExportModel(model, portable_idfa=True)
    assert all(not block.portable for block in model.alignment.values())
    if alignment == "idfa":
        assert all(block.portable for block in wrapper.model.alignment.values())
        with pytest.raises(ValueError, match="requires imu"):
            wrapper(torch.zeros(1, 3, 64, 96))
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"config": config, "model": model.state_dict(), "classes": list(CLASS_NAMES),
                "epoch": 0, "synthetic_smoke": True, "trained_on_real_data": False}, checkpoint)
    destination = tmp_path / "model.zip"
    with pytest.raises(ValueError, match="provenance"):
        export_onnx(checkpoint, destination)
    assert not destination.exists()
    for invalid in ([63, 96], [64, 100], [64.0, 96], [64, 96, 128]):
        with pytest.raises(ValueError, match="integer multiples"):
            export_onnx(checkpoint, destination, allow_untrained=True, image_size=invalid)
    assert not destination.exists()
    report = export_onnx(checkpoint, destination, allow_untrained=True, image_size=export_size)
    assert report["format_version"] == 2 and not report["live_inference_allowed"]
    assert report["alignment"] == alignment
    assert report["input_names"] == (["image", "imu"] if alignment == "idfa" else ["image"])
    h, w = export_size or [64, 96]
    assert report["output_shapes"]["road_logits"] == [1, 1, h // 4, w // 4]
    locations = sum((h // stride) * (w // stride) for stride in (8, 16, 32))
    assert report["output_shapes"]["class_logits"] == [1, locations, 9]
    assert report["training_image_shape"] == [1, 3, 64, 96]
    assert report["export_image_shape"] == report["image_shape"] == [1, 3, h, w]
    assert report["resolution_changed"] == (export_size is not None)
    assert report["requires_resolution_reevaluation"] == (export_size is not None)
    assert model.config["image_size"] == [64, 96]
    reloaded = torch.load(checkpoint, weights_only=True)
    assert reloaded["config"]["image_size"] == [64, 96]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(reloaded["model"][name], value, atol=0, rtol=0)
    with zipfile.ZipFile(destination) as bundle:
        assert set(bundle.namelist()) == {"model.onnx", "model.json"}
        meta = json.loads(bundle.read("model.json"))
        assert meta == report
        assert hashlib.sha256(bundle.read("model.onnx")).hexdigest() == meta["model_sha256"]
    with pytest.raises(FileExistsError):
        export_onnx(checkpoint, destination, allow_untrained=True)


def test_provenance_never_promotes_unknown_or_smoke_weights():
    assert not training_provenance({"epoch": 2})["live_inference_allowed"]
    assert not training_provenance({"epoch": 2, "trained_on_real_data": True,
                                    "synthetic_smoke": True})["live_inference_allowed"]
    assert not training_provenance({"epoch": -1, "trained_on_real_data": True})["live_inference_allowed"]
    assert not training_provenance({"epoch": 0, "trained_on_real_data": True})["live_inference_allowed"]
    assert training_provenance({"epoch": 0, "trained_on_real_data": True,
                                "synthetic_smoke": False, "supervised_tasks": [],
                                "supervised_detection_classes": ["pothole"]})["live_inference_allowed"]


def test_partial_supervision_is_explicit_in_mobile_provenance():
    state = {"epoch": 0, "trained_on_real_data": True, "synthetic_smoke": False,
             "supervised_tasks": [], "supervised_detection_classes": ["pothole"]}
    provenance = training_provenance(state)
    assert provenance["live_inference_allowed"] and provenance["scope_status"] == "partial"
    assert provenance["supervised_tasks"] == []
    assert provenance["supervised_detection_classes"] == ["pothole"]
    full = training_provenance({**state, "supervised_tasks": ["road", "lane"],
                                "supervised_detection_classes": list(CLASS_NAMES)})
    assert full["scope_status"] == "full"
    with pytest.raises(ValueError, match="supervised_tasks"):
        training_provenance({**state, "supervised_tasks": ["pothole"]})
    with pytest.raises(ValueError, match="supervised_detection_classes"):
        training_provenance({**state, "supervised_detection_classes": ["car", "car"]})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not present")
def test_native_cuda_idfa_matches_portable_cpu_and_amp_backpropagates():
    native = RollConditionedConv(4).eval().cuda()
    with torch.no_grad():
        native.refine[-1].weight.normal_(0, .1)
        native.refine[-1].bias.uniform_(-.5, .5)
    portable = copy.deepcopy(native).cpu()
    portable.portable = True
    image = torch.randn(2, 4, 13, 17)
    imu = torch.tensor([[.6, 1.], [-1.3, .4]])
    with torch.no_grad():
        cuda_value = native(image.cuda(), imu.cuda()).cpu()
        cpu_value = portable(image, imu)
    torch.testing.assert_close(cuda_value, cpu_value, atol=3e-5, rtol=3e-5)
    half_image = image.cuda().half().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        output = native(half_image, imu.cuda())
        loss = output.float().square().mean()
    loss.backward()
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    assert torch.isfinite(half_image.grad).all() and half_image.grad.abs().sum() > 0
    assert torch.isfinite(native.refine[-1].weight.grad).all()
    assert native.refine[-1].weight.grad.abs().sum() > 0
