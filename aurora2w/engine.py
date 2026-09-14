"""Reproducible training, explicit checkpoint resume, deterministic validation."""

from collections import Counter
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import CLASS_NAMES
from .data import RoadDataset, collate, audit_records
from .losses import multitask_loss
from .metrics import Metrics, decode
from .model import MultiTaskNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader supplies a deterministic per-worker seed from its generator."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv_threads = __import__("cv2")
    cv_threads.setNumThreads(1)


def float_outputs(value):
    """Keep focal/BCE/GIoU calculations in float32 under CUDA mixed precision."""
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, dict):
        return {k: float_outputs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [float_outputs(v) for v in value]
    return value


def fingerprint_records(records):
    """Content fingerprint independent of machine paths; includes split/label scope."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: (r["source"], r["id"])):
        value = {k: v for k, v in record.items() if not k.startswith("_")}
        for key in ("image", "road_mask", "lane_mask"):
            if value.get(key):
                content = hashlib.sha256()
                with open(value[key], "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        content.update(chunk)
                value[key] = content.hexdigest()
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def run_provenance(config, records):
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code.update(path.name.encode() + path.read_bytes())
    return {"data_sha256": fingerprint_records(records), "code_sha256": code.hexdigest(),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "python": platform.python_version(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "samples_by_source_split": dict(Counter(r["source"] + "/" + r["split"] for r in records))}


def learning_rate_factor(epoch, config):
    """Epoch schedule fixed by config epochs; --epochs only controls this invocation."""
    warmup = int(config.get("warmup_epochs", 0))
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    if config.get("lr_schedule", "constant") == "constant":
        return 1.
    end = max(config["epochs"] - 1 - warmup, 1)
    progress = min(max((epoch - warmup) / end, 0.), 1.)
    floor = config.get("min_lr_factor", .05)
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != list(CLASS_NAMES):
        raise ValueError("Checkpoint taxonomy differs from Phase 2")
    model = MultiTaskNet(checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


@torch.no_grad()
def evaluate(model, records, config, device, batch_size=None):
    if not records:
        raise ValueError("No records in requested evaluation split")
    dataset = RoadDataset(records, config, training=False)
    loader = DataLoader(dataset, batch_size=batch_size or config["batch_size"], collate_fn=collate, shuffle=False)
    metrics, totals, batches = Metrics(), Counter(), 0
    grouped = {}
    durations = []
    model.eval()
    for images, imu, targets in loader:
        images, imu = images.to(device), imu.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(images, imu)
        predictions = decode(out, config["image_size"], threshold=.001)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000 / len(images))
        metrics.update(out, predictions, targets)
        for i, target in enumerate(targets):
            angle = abs(np.degrees(target["roll_rad"]))
            roll_bin = "unknown" if target["imu_confidence"] == 0 else ("0_10" if angle < 10 else "10_25" if angle < 25 else "25_plus")
            for key in ("source/" + target["source"], "roll_deg/" + roll_bin):
                grouped.setdefault(key, Metrics()).update(
                    {task + "_logits": out[task + "_logits"][i:i+1] for task in ("road", "lane")},
                    [predictions[i]], [target])
        losses = multitask_loss(out, targets, config["loss_weights"])
        for k, v in losses.items():
            totals[k] += float(v)
        batches += 1
    result = metrics.compute()
    result["groups"] = {key: value.compute() for key, value in grouped.items()}
    result["loss"] = {k: v / batches for k, v in totals.items()}
    result["forward_decode_ms_per_image"] = {"median": float(np.median(durations)), "p95": float(np.percentile(durations, 95))}
    result["timing_note"] = "Desktop batch-normalized timing incl. decode, not Android or camera-to-display latency; includes cold first batch"
    return result


def train(config, records, output_dir, device="cpu", resume=None, epochs=None, max_steps=None, initialize=None):
    audit = audit_records(records)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    if not train_records or not val_records:
        raise ValueError("Training requires separate nonempty train and val splits")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive; omit it for a full run")
    if resume and initialize:
        raise ValueError("Choose either resume or initialize")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use the D-drive CUDA environment")
    if config["alignment"] == "idfa" and not any(
            r.get("imu", {}).get("confidence", 0) > 0 or r.get("upright_reference", False) for r in train_records):
        raise ValueError("IDFA has no trusted roll supervision: supply calibrated IMU or explicitly curated upright references")
    print(json.dumps({"data_audit": audit}), flush=True)
    seed_everything(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.get("deterministic", True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists() and resume is None:
        raise FileExistsError("Run already contains last.pt; choose a new output or explicitly --resume")
    checkpoint = None
    source_state = {}
    provenance = run_provenance(config, records)
    if resume:
        model, checkpoint = load_checkpoint(resume, device)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires identical config; use --epochs to extend the run")
        previous = checkpoint.get("provenance", {})
        if previous.get("data_sha256") != provenance["data_sha256"]:
            raise ValueError("Resume requires identical data, labels and splits; legacy checkpoints have no verifiable data fingerprint")
        if previous.get("code_sha256") != provenance["code_sha256"]:
            raise ValueError("Training code changed since checkpoint; preserve the run code for reproducible resume")
        if checkpoint.get("max_steps") != max_steps:
            raise ValueError("Resume must preserve the smoke/full-run step budget")
    else:
        model = MultiTaskNet(config, pretrained=False if initialize else None).to(device)
    if initialize:
        source, source_state = load_checkpoint(initialize)
        # Optimizer/augmentation settings may change; structure must agree.
        for key in ("channels", "neck_repeats", "idfa_levels"):
            if source_state["config"][key] != config[key]:
                raise ValueError(f"Initialization architecture mismatch: {key}")
        if source_state["config"]["alignment"] != "none" or config["alignment"] != "idfa":
            raise ValueError("initialize supports none baseline -> idfa only; use resume otherwise")
        missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
        if unexpected or any(not k.startswith("alignment.") or not any(s in k for s in ("refine", "tap_")) for k in missing):
            raise ValueError(f"Unexpected initialization differences: {missing}, {unexpected}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    use_amp = bool(config.get("amp", False) and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=config.get("amp_initial_scale", 4096.))
    start_epoch, best = 0, float("inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best = checkpoint["epoch"] + 1, checkpoint["best_val_loss"]
    counts = Counter(r["source"] for r in train_records)
    generator = torch.Generator()
    sampler = WeightedRandomSampler([1 / counts[r["source"]] for r in train_records], len(train_records), replacement=True,
                                    generator=generator) if config["source_balance"] else None
    dataset = RoadDataset(train_records, config, training=True)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=sampler is None, sampler=sampler,
                        num_workers=config["workers"], collate_fn=collate, drop_last=False,
                        pin_memory=device.startswith("cuda"), worker_init_fn=seed_worker,
                        generator=generator, persistent_workers=False)
    total_epochs = epochs if epochs is not None else config["epochs"]
    if total_epochs <= start_epoch:
        raise ValueError("Requested epoch limit is already completed")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    accumulation = int(config.get("accumulate_steps", 1))
    if accumulation < 1:
        raise ValueError("accumulate_steps must be positive")
    real_data = all(r.get("data_kind") == "real" for r in train_records + val_records)
    supervision = audit_records(train_records)
    inherited = source_state if initialize else checkpoint or {}
    supervised_tasks = sorted(set(inherited.get("supervised_tasks", [])) |
                              {task for task in ("road", "lane") if supervision["tasks"][task] > 0})
    supervised_classes = sorted(set(inherited.get("supervised_detection_classes", [])) |
                                {name for name in CLASS_NAMES if supervision["instances"][name] > 0
                                 and supervision["exhaustively_labeled_images"][name] > 0})
    for epoch in range(start_epoch, total_epochs):
        # Epoch-based seed makes augmentation and source sampling resumable (workers=0).
        seed_everything(config["seed"] + epoch)
        generator.manual_seed(config["seed"] + epoch)
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * learning_rate_factor(epoch, config)
        model.train()
        total, steps = 0., 0
        successful_updates, skipped_updates = 0, 0
        step_limit = min(len(loader), max_steps) if max_steps else len(loader)
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        for step, (images, imu, targets) in enumerate(loader):
            window_start = step // accumulation * accumulation
            window_size = min(accumulation, step_limit - window_start)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=use_amp):
                outputs = model(images.to(device, non_blocking=True), imu.to(device, non_blocking=True))
            losses = multitask_loss(float_outputs(outputs), targets, config["loss_weights"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite loss; run stopped without overwriting checkpoint")
            scaler.scale(losses["total"] / window_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == step_limit:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 10), error_if_nonfinite=not use_amp)
                # GradScaler detects FP16 overflow during unscale and skips that
                # update while reducing scale. Do not abort a normal scale search.
                successful_updates += int(bool(torch.isfinite(gradient_norm)))
                skipped_updates += int(not bool(torch.isfinite(gradient_norm)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(losses["total"].detach())
            steps += 1
            if steps >= step_limit:
                break
        if successful_updates == 0:
            raise FloatingPointError("No finite optimizer updates in this epoch; lower AMP initial scale or inspect losses/data")
        report = evaluate(model, val_records, config, device)
        value = report["loss"]["total"]
        improved = value < best
        best = min(best, value)
        checkpoint = {"format_version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": config, "classes": list(CLASS_NAMES), "epoch": epoch,
                      "best_val_loss": best, "synthetic_smoke": bool(max_steps) or any(
                          r.get("data_kind") == "synthetic" or r["source"].startswith("synthetic") for r in train_records),
                      "trained_on_real_data": bool(real_data and not max_steps),
                      "supervised_tasks": supervised_tasks, "supervised_detection_classes": supervised_classes,
                      "training_supervision_audit": supervision,
                      "max_steps": max_steps, "scaler": scaler.state_dict(), "provenance": provenance,
                      "initialization_checkpoint": str(initialize) if initialize else checkpoint.get("initialization_checkpoint") if checkpoint else None,
                      "validation": report}
        temp = output_dir / "last.tmp.pt"
        torch.save(checkpoint, temp)
        temp.replace(output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        summary = {"epoch": epoch + 1, "train_loss": total / steps, "validation": report,
                   "learning_rate": optimizer.param_groups[0]["lr"], "amp": use_amp, "batches": steps,
                   "successful_optimizer_updates": successful_updates, "overflow_skipped_updates": skipped_updates,
                   "seconds": time.perf_counter() - epoch_start, "accumulate_steps": accumulation}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return output_dir / "last.pt"


def doctor(device="cuda"):
    """Report the actual interpreter/runtime; never infer CUDA from a GPU name."""
    import os
    import sys
    import torchvision
    result = {"python": sys.executable, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
              "cache_locations": {key: os.environ.get(key) for key in ("TORCH_HOME", "HF_HOME", "TEMP", "TMP", "PIP_CACHE_DIR")}}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device if device.startswith("cuda") else "cuda")
        result.update(gpu=props.name, vram_gib=props.total_memory / 1024**3)
        from torchvision.ops import deform_conv2d, nms
        x = torch.ones(1, 2, 4, 4, device="cuda")
        y = deform_conv2d(x, torch.zeros(1, 18, 4, 4, device="cuda"), torch.ones(2, 2, 3, 3, device="cuda"), padding=1)
        nms(torch.tensor([[0., 0., 2., 2.]], device="cuda"), torch.ones(1, device="cuda"), .5)
        result["cuda_custom_ops_pass"] = bool(torch.isfinite(y).all())
    return result


def profile(config, records, device="cuda", steps=10, warmup_steps=1):
    """Time actual loader+forward+backward steps; no checkpoint/accuracy claim."""
    if warmup_steps < 1 or steps <= warmup_steps:
        raise ValueError("Profile steps must exceed positive warmup_steps")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run doctor in the D-drive environment")
    selected = [r for r in records if r["split"] == "train"]
    if len(selected) < config["batch_size"]:
        raise ValueError("Profile needs at least one complete training batch")
    seed_everything(config["seed"])
    model = MultiTaskNet(config, pretrained=False).to(device).train()
    loader = DataLoader(RoadDataset(selected, config, training=True), batch_size=config["batch_size"],
                        num_workers=config["workers"], collate_fn=collate, drop_last=True,
                        pin_memory=device.startswith("cuda"), worker_init_fn=seed_worker)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    amp = bool(config.get("amp", False) and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=config.get("amp_initial_scale", 4096.))
    times, iterator = [], iter(loader)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            images, imu, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, imu, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=amp):
            output = model(images.to(device), imu.to(device))
        loss = multitask_loss(float_outputs(output), targets, config["loss_weights"])["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite profile loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    median = float(np.median(times[warmup_steps:]))
    return {"device": device, "amp": amp, "batch_size": config["batch_size"], "input_size": config["image_size"],
            "median_training_step_s": median, "images_per_second": config["batch_size"] / median,
            "estimated_epoch_train_seconds": median * math.ceil(len(selected) / config["batch_size"]),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.startswith("cuda") else None,
            "warmup_steps_excluded": warmup_steps, "measured_steps": steps - warmup_steps,
            "note": "Random model throughput including loader, full optimizer step; warmup excluded. Validation/checkpoint time extra. No trained accuracy or Android FPS."}
