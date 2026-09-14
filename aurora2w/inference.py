"""Class-colored overlays and a timestamp-aware desktop video reference loop."""

import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .config import CLASS_NAMES, COLORS
from .engine import load_checkpoint
from .geometry import letterbox, normalize_image, transform_boxes
from .metrics import decode
from .mobile_contract import training_provenance


class RollTimeline:
    """Already calibrated camera roll, timestamp seconds in video's time domain."""
    def __init__(self, path, max_gap_s=.1):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.times = np.array([float(r["timestamp_s"]) for r in rows])
        self.roll = np.unwrap([float(r["roll_rad"]) for r in rows])
        self.conf = np.array([float(r["confidence"]) for r in rows])
        self.max_gap_s = max_gap_s
        if (len(rows) < 2 or not np.isfinite([self.times, self.roll, self.conf]).all()
                or not (np.diff(self.times) > 0).all() or (self.conf < 0).any() or (self.conf > 1).any()):
            raise ValueError("IMU CSV needs ordered timestamp_s, finite camera roll_rad, confidence in [0,1]")

    def at(self, timestamp):
        if not np.isfinite(timestamp) or timestamp < self.times[0] or timestamp > self.times[-1]:
            return 0., 0.
        i = int(np.searchsorted(self.times, timestamp))
        if self.times[i] == timestamp:
            return float(self.roll[i]), float(self.conf[i])
        if self.times[i] - self.times[i - 1] > self.max_gap_s:
            return 0., 0.
        roll = np.interp(timestamp, self.times[i-1:i+1], self.roll[i-1:i+1])
        return float(roll), float(min(self.conf[i-1], self.conf[i]))


def render_overlay(frame, output, prediction, matrix, size, text="", supervised_tasks=("road", "lane")):
    """Model frame -> original frame, preserving box corners and mask alignment."""
    ih, iw = frame.shape[:2]
    h, w = size
    sx, sy, left, top = matrix[0, 0], matrix[1, 1], int(matrix[0, 2]), int(matrix[1, 2])
    nw, nh = round(iw * sx), round(ih * sy)
    result = frame.copy()
    for task, rgb, alpha in (("road", (40, 210, 80), .3), ("lane", (255, 215, 0), .75)):
        if task not in supervised_tasks:
            continue
        prob = F.interpolate(output[task + "_logits"], size=(h, w), mode="bilinear", align_corners=False).sigmoid()[0, 0].cpu().numpy()
        prob = cv2.resize(prob[top:top+nh, left:left+nw], (iw, ih))
        mask = prob >= .5
        result[mask] = ((1 - alpha) * result[mask] + alpha * np.array(rgb[::-1])).astype(np.uint8)
    boxes, valid = transform_boxes(prediction["boxes"].numpy(), np.linalg.inv(matrix), iw, ih)
    for box, label, score, keep in zip(boxes, prediction["labels"], prediction["scores"], valid):
        if not keep:
            continue
        label = int(label)
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[label][::-1]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, f"{CLASS_NAMES[label]} {float(score):.2f}", (x1, max(14, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    cv2.putText(result, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def scoped_detection_output(output, supervised_classes):
    """Suppress untrained classes before top-k/NMS without changing caller tensors."""
    allowed = {CLASS_NAMES.index(name) for name in supervised_classes}
    result = dict(output)
    result["detection"] = []
    for level in output["detection"]:
        scoped = dict(level)
        scoped["class_logits"] = level["class_logits"].clone()
        for index in range(len(CLASS_NAMES)):
            if index not in allowed:
                scoped["class_logits"][:, index] = torch.finfo(scoped["class_logits"].dtype).min
        result["detection"].append(scoped)
    return result


@torch.no_grad()
def run_video(checkpoint_path, video, output_path, device="cpu", imu_csv=None,
              frame_times=None, max_frames=None, threshold=.3, allow_untrained=False):
    if Path(video).resolve() == Path(output_path).resolve():
        raise ValueError("Output must differ from input video")
    if Path(output_path).exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()
    provenance = training_provenance(checkpoint)
    if not provenance["live_inference_allowed"] and not allow_untrained:
        raise ValueError("Checkpoint lacks verified real-training scope; use --allow-untrained only for labeled software diagnostics")
    config = checkpoint["config"]
    timeline = RollTimeline(imu_csv) if imu_csv else None
    timestamps = None
    if frame_times:
        with open(frame_times, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if [int(r["frame_index"]) for r in rows] != list(range(len(rows))):
            raise ValueError("frame_times requires contiguous 0-based frame_index")
        timestamps = [float(r["timestamp_s"]) for r in rows]
        if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
            raise ValueError("Frame timestamps must be finite and increasing")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video has no valid playback FPS")
    writer, count, confident, durations, previous_time = None, 0, 0, [], -float("inf")
    try:
        while max_frames is None or count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if timestamps is not None and count >= len(timestamps):
                raise ValueError("Frame timestamp file ends before video")
            timestamp = timestamps[count] if timestamps is not None else capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timeline and (not np.isfinite(timestamp) or timestamp <= previous_time):
                raise ValueError("Decoder timestamps are not usable; supply --frame-times with captured frame PTS")
            previous_time = timestamp
            roll, confidence = timeline.at(timestamp) if timeline else (0., 0.)
            confident += confidence > 0
            start = time.perf_counter()
            image, _, matrix = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), config["image_size"])
            tensor = torch.from_numpy(normalize_image(image))[None].to(device)
            out = model(tensor, torch.tensor([[roll, confidence]], dtype=torch.float32, device=device))
            scoped = scoped_detection_output(out, provenance["supervised_detection_classes"])
            pred = decode(scoped, config["image_size"], threshold=threshold)[0]
            status = "IMU unavailable" if confidence == 0 else f"roll {np.degrees(roll):+.1f} deg"
            if not provenance["live_inference_allowed"]:
                status = "UNTRAINED / INTEGRATION WEIGHTS | " + status
            elif provenance["scope_status"] == "partial":
                status = "PARTIAL TRAINING SCOPE | " + status
            overlay = render_overlay(frame, out, pred, matrix, config["image_size"], status,
                                     provenance["supervised_tasks"])
            durations.append((time.perf_counter() - start) * 1000)
            if writer is None:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise OSError(f"Cannot create video: {output_path}")
            writer.write(overlay)
            count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not count:
        raise ValueError("Video contains no decodable frames")
    report = {"training_provenance": provenance, "frames": count, "frames_with_trusted_imu": confident,
              "median_pipeline_ms": float(np.median(durations)), "p95_pipeline_ms": float(np.percentile(durations, 95)),
              "checkpoint": str(checkpoint_path), "alignment": config["alignment"],
              "note": "Offline video; no audio. Playback FPS is not measured inference FPS. Timing excludes capture/write."}
    Path(str(output_path) + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
