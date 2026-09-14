"""Versioned model bundle contract shared with the Android importer."""

import hashlib

from .config import CLASS_NAMES


OUTPUT_NAMES = ["road_logits", "lane_logits", "class_logits", "boxes_xyxy", "centerness_logits"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_provenance(state):
    """Legacy/unknown provenance never gets promoted to a road-trained model."""
    smoke = bool(state.get("synthetic_smoke", False))
    real = state.get("trained_on_real_data") is True
    tasks = state.get("supervised_tasks", [])
    detection_classes = state.get("supervised_detection_classes", [])
    if (not isinstance(tasks, list) or not all(isinstance(value, str) for value in tasks)
            or len(set(tasks)) != len(tasks) or not set(tasks).issubset({"road", "lane"})):
        raise ValueError("supervised_tasks must be unique road/lane task names")
    if (not isinstance(detection_classes, list)
            or not all(isinstance(value, str) for value in detection_classes)
            or len(set(detection_classes)) != len(detection_classes)
            or not set(detection_classes).issubset(CLASS_NAMES)):
        raise ValueError("supervised_detection_classes must contain unique Aurora class names")
    scoped = bool(tasks or detection_classes)
    live = real and not smoke and int(state.get("epoch", -1)) >= 0 and scoped
    status = "trained" if live else ("untrained_fixture" if smoke else "unknown")
    full = set(tasks) == {"road", "lane"} and set(detection_classes) == set(CLASS_NAMES)
    return {"training_status": status, "synthetic_smoke": smoke,
            "trained_on_real_data": real, "live_inference_allowed": live,
            "completed_epochs": max(0, int(state.get("epoch", -1)) + 1),
            "supervised_tasks": tasks, "supervised_detection_classes": detection_classes,
            "scope_status": "full" if full else ("partial" if scoped else "unverified")}


def build_metadata(state, model_path, output_shapes, parity, export_image_size=None):
    config = state["config"]
    training_h, training_w = config["image_size"]
    h, w = export_image_size or config["image_size"]
    uses_imu = config["alignment"] == "idfa"
    input_shapes = {"image": [1, 3, h, w]}
    if uses_imu:
        input_shapes["imu"] = [1, 2]
    return {
        "format_version": 2,
        "model_file": "model.onnx",
        "model_sha256": sha256_file(model_path),
        "architecture": "aurora2w_mobilenetv3_bipyramid",
        "alignment": config["alignment"],
        "idfa_levels": config["idfa_levels"] if uses_imu else [],
        "idfa_implementation": "nine_bilinear_gridsample_taps" if uses_imu else "ordinary_convolution",
        "opset": 17, "dtype": "float32", "classes": list(CLASS_NAMES),
        "image_shape": [1, 3, h, w], "input_names": list(input_shapes), "input_shapes": input_shapes,
        "training_image_shape": [1, 3, training_h, training_w],
        "export_image_shape": [1, 3, h, w],
        "resolution_changed": [h, w] != [training_h, training_w],
        "requires_resolution_reevaluation": [h, w] != [training_h, training_w],
        "validation_image_shape": [1, 3, training_h, training_w],
        "outputs": OUTPUT_NAMES, "output_shapes": output_shapes,
        "layout": "NCHW", "color": "RGB", "mean": [.485, .456, .406],
        "std": [.229, .224, .225], "pixel_scale": 255.0,
        "letterbox": {
            "resize": "bilinear_half_pixel", "dimension_rounding": "ties_to_even",
            "pad_value": [114, 114, 114], "placement": "center_floor_left_top",
            "independent_xy_scale_after_rounding": True,
            "training_reference": "OpenCV INTER_LINEAR uint8; mobile float bilinear is not bit-identical",
        },
        "sensor_contract": {
            "fields": ["camera_scene_roll_rad", "confidence"],
            "roll_sign": "positive_clockwise_in_model_image",
            "axes": "optical x right, y down, z forward", "confidence_range": [0.0, 1.0],
            "missing_or_uncalibrated": [0.0, 0.0],
            "timestamp": "camera exposure midpoint aligned to monotonic sensor time",
            "requires_device_to_camera_calibration": True, "default_max_gap_s": 0.05,
        },
        "postprocess": {
            "segmentation_threshold": 0.5,
            "segmentation": "bilinear logits to input size, sigmoid, crop padding, resize to source",
            "box_coordinates": "continuous XYXY in letterboxed input pixels",
            "score": "sqrt(sigmoid(class_logits)*sigmoid(centerness_logits))",
            "score_threshold": 0.3, "nms_iou_threshold": 0.5, "max_detections": 100,
            "nms": "classwise", "pothole_masks": False,
        },
        **training_provenance(state), "validation": state.get("validation"), "parity": parity,
        "training_supervision_audit": state.get("training_supervision_audit"),
        "mobile_status": "CPU ONNX numerical parity verified; physical Android speed and accuracy unmeasured",
    }
