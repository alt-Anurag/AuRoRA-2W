"""Run from repository root: python -m aurora2w.cli --help."""

import argparse
import csv
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert indexed local sources into a validated manifest")
    prepare.add_argument("--spec", required=True)
    prepare.add_argument("--output", required=True)
    index = commands.add_parser("index", help="Index original IDD/BDD/RDD releases; keeps official and recording splits")
    index.add_argument("--kind", choices=["idd-seg", "idd-det", "rdd-india", "bdd"], required=True)
    index.add_argument("--root", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--groups", help="CSV image,sequence,split; required when no official labeled split exists")
    index.add_argument("--labels", help="BDD frame-list/Scalabel JSON, or directory of original per-image legacy JSON files")
    index.add_argument("--split", choices=["train", "val", "test"])
    index.add_argument("--tasks", nargs="+", choices=["road", "lane", "detection"], help="BDD tasks verified exhaustively annotated in this label release")
    index.add_argument("--lane-width", type=int, default=8, help="BDD binary lane stroke width in source pixels")
    index.add_argument("--exhaustive", action="store_true", help="Declare mapped IDD detection classes exhaustive after annotation audit")
    index.add_argument("--voc-coordinates", choices=["one_based_inclusive", "zero_based_inclusive", "zero_based_half_open"], help="Required reviewed coordinate convention for nested IDD detection")
    merge = commands.add_parser("merge-specs", help="Combine native source specs with portable roots")
    merge.add_argument("--specs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    audit = commands.add_parser("audit", help="Validate annotations and split leakage")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--hashes", action="store_true")
    audit.add_argument("--config", help="Also audit detector target assignment at this input resolution")
    audit.add_argument("--output")
    doctor = commands.add_parser("doctor", help="Check Python/CUDA/torchvision custom operators")
    doctor.add_argument("--device", default="cuda")
    profile = commands.add_parser("profile", help="Short random-model training throughput/VRAM check with real label loader")
    profile.add_argument("--config", default="configs/rtx4050_baseline.json")
    profile.add_argument("--manifest", required=True)
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--steps", type=int, default=10)
    profile.add_argument("--output")
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/phase2.json")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume")
    train.add_argument("--initialize", help="Initialize IDFA model from a trained none baseline")
    train.add_argument("--alignment", choices=["none", "idfa"], help="Override config alignment")
    train.add_argument("--epochs", type=int)
    train.add_argument("--max-steps", type=int, help="SMOKE TEST ONLY: limits steps per epoch; marks checkpoint")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", required=True)
    video = commands.add_parser("video")
    video.add_argument("--checkpoint", required=True)
    video.add_argument("--video", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--device", default="cpu")
    video.add_argument("--imu", help="timestamp_s,roll_rad,confidence CSV in video time domain")
    video.add_argument("--frame-times", help="frame_index,timestamp_s CSV if decoder PTS are unsuitable")
    video.add_argument("--max-frames", type=int)
    video.add_argument("--threshold", type=float, default=.3)
    video.add_argument("--allow-untrained", action="store_true", help="Explicit software diagnostic; still suppresses unsupervised tasks")
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--allow-untrained", action="store_true", help="Explicit fixture export; metadata prevents live perception claims")
    export.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), help="Explicit export resolution override; requires new accuracy/device evaluation")
    sync = commands.add_parser("sync", help="Calibrated quaternion samples -> exposure-aligned camera roll CSV")
    sync.add_argument("--quaternions", required=True, help="timestamp_s,qw,qx,qy,qz")
    sync.add_argument("--frames", required=True, help="frame_index,timestamp_s,exposure_s (optional)")
    sync.add_argument("--calibration", required=True, help="JSON R_device_camera, camera_to_imu_offset_s")
    sync.add_argument("--output", required=True)
    sync.add_argument("--max-gap-s", type=float, default=.05, help="Maximum bracketing attitude sample interval; widening does not recover vibration")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare_spec
        result = prepare_spec(args.spec, args.output)
    elif args.command == "index":
        from .native import build_native
        result = build_native(args.kind, args.root, args.output, args.groups, args.labels, args.split,
                              args.tasks, args.lane_width, args.exhaustive, args.voc_coordinates)
    elif args.command == "merge-specs":
        from .native import merge_specs
        result = merge_specs(args.specs, args.output)
    elif args.command == "doctor":
        from .engine import doctor as run_doctor
        result = run_doctor(args.device)
    elif args.command == "profile":
        from .data import read_manifest
        from .engine import profile as run_profile
        result = run_profile(load_config(args.config), read_manifest(args.manifest), args.device, args.steps)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "audit":
        from .data import read_manifest, audit_records
        result = audit_records(read_manifest(args.manifest), args.hashes,
                               load_config(args.config)["image_size"] if args.config else None)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "train":
        from .data import read_manifest
        from .engine import train as run_train
        config = load_config(args.config)
        if args.alignment:
            config["alignment"] = args.alignment
        result = str(run_train(config, read_manifest(args.manifest), args.output,
                               args.device, args.resume, args.epochs, args.max_steps, args.initialize))
    elif args.command == "evaluate":
        from .data import read_manifest, audit_records
        from .engine import load_checkpoint, evaluate as run_evaluate
        records = read_manifest(args.manifest)
        audit_records(records)
        model, state = load_checkpoint(args.checkpoint, args.device)
        result = run_evaluate(model, [r for r in records if r["split"] == args.split], state["config"], args.device)
        result["checkpoint"], result["split"] = args.checkpoint, args.split
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "video":
        from .inference import run_video
        result = run_video(args.checkpoint, args.video, args.output, args.device, args.imu,
                           args.frame_times, args.max_frames, args.threshold, args.allow_untrained)
    elif args.command == "export":
        from .export import export_onnx
        result = export_onnx(args.checkpoint, args.output, allow_untrained=args.allow_untrained, image_size=args.image_size)
    else:
        from .sensors import AttitudeTimeline, camera_roll
        import numpy as np
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8-sig"))
        camera_roll([1, 0, 0, 0], calibration["R_device_camera"])
        offset = float(calibration["camera_to_imu_offset_s"])
        if not np.isfinite(offset):
            raise ValueError("Clock offset must be finite")
        with open(args.quaternions, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not np.isfinite(args.max_gap_s) or args.max_gap_s <= 0:
            raise ValueError("max-gap-s must be positive and finite")
        timeline = AttitudeTimeline([float(r["timestamp_s"]) for r in rows],
                                    [[float(r[k]) for k in ("qw", "qx", "qy", "qz")] for r in rows], max_gap_s=args.max_gap_s)
        with open(args.frames, newline="", encoding="utf-8-sig") as f:
            frames = list(csv.DictReader(f))
        frame_times = [float(row["timestamp_s"]) for row in frames]
        if (not frames or [int(row["frame_index"]) for row in frames] != list(range(len(frames)))
                or not np.isfinite(frame_times).all() or not (np.diff(frame_times) > 0).all()):
            raise ValueError("Frames require contiguous 0-based indices and increasing finite timestamps")
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        output = []
        for row in frames:
            t = float(row["timestamp_s"])  # Capture start, in camera time domain.
            exposure = float(row.get("exposure_s") or 0)
            if not np.isfinite(exposure) or exposure < 0:
                raise ValueError("Exposure duration must be finite and nonnegative")
            q = timeline.at(t + exposure/2 + offset)
            roll, confidence = camera_roll(q, calibration["R_device_camera"]) if q is not None else (0., 0.)
            output.append({"timestamp_s": t, "roll_rad": roll, "confidence": confidence})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp_s", "roll_rad", "confidence"])
            writer.writeheader()
            writer.writerows(output)
        result = {"frames": len(output), "trusted": sum(r["confidence"] > 0 for r in output), "max_gap_s": args.max_gap_s}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
