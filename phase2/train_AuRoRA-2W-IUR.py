"""
train_AuRoRA-2W-IUR.py
========================
Main training script for the AuRoRA-2W IUR (Indian Unstructured Roads) multi-task model.

Usage (on Colab):
    python train_AuRoRA-2W-IUR.py \
        --idd-root  /content/datasets/IDD \
        --rdd-root  /content/datasets/RDD2022 \
        --bdd-root  /content/datasets/BDD100K \
        --epochs 100 \
        --batch-size 8 \
        --output-dir /content/drive/MyDrive/AuRoRA2W_IUR_checkpoints

Features:
  ✅ Synthetic Roll Augmentation ±30° (with bbox rotation math)
  ✅ OHEM Loss for segmentation heads (Phase 1 fix)
  ✅ Fill=255 for rotated label corners  (Phase 1 fix)
  ✅ Lane mask dilation before rotation  (Phase 2 new)
  ✅ AMP (Mixed Precision) with GradScaler
  ✅ Poly LR scheduler
  ✅ Full state checkpoint resume (epoch + optimizer + scaler)
  ✅ TensorBoard logging
  ✅ Google Drive auto-save
"""

import os
import sys
import argparse
import random
import time
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── Add phase2/ to path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from models.aurora2w_iur import get_aurora2w_iur_model
from datasets.composite_dataset import build_composite_dataset, collate_fn
from utils.augmentation import apply_batch_roll_augmentation
from utils.criterion import AuRoRA2W_IUR_Loss

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[!] TensorBoard not found. Install with: pip install tensorboard")


# ═══════════════════════════════════════════════════════════════
# Argument Parser
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='Train AuRoRA-2W-IUR Multi-Task Model')

    # Dataset roots
    parser.add_argument('--idd-root',  type=str, default=None, help='Path to IDD dataset root')
    parser.add_argument('--rdd-root',  type=str, default=None, help='Path to RDD2022 dataset root')
    parser.add_argument('--bdd-root',  type=str, default=None, help='Path to BDD100K dataset root')

    # Training hyperparameters
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch-size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--weight-decay',type=float, default=5e-4)
    parser.add_argument('--momentum',    type=float, default=0.9)
    parser.add_argument('--img-h',       type=int,   default=384)
    parser.add_argument('--img-w',       type=int,   default=640)
    parser.add_argument('--workers',     type=int,   default=4)
    parser.add_argument('--seed',        type=int,   default=42)

    # Loss weights
    parser.add_argument('--w-drive', type=float, default=1.0)
    parser.add_argument('--w-lane',  type=float, default=1.5)
    parser.add_argument('--w-det',   type=float, default=1.0)

    # Model
    parser.add_argument('--fpn-ch',     type=int, default=64)
    parser.add_argument('--pretrained-backbone', type=str, default=None)

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/iur_run_01')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume',     type=str, default=None, help='Path to checkpoint to resume from')

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Learning Rate Scheduler (Poly decay — same as Phase 1)
# ═══════════════════════════════════════════════════════════════

def poly_lr(optimizer, base_lr: float, max_iters: int, cur_iter: int, power: float = 0.9):
    lr = base_lr * ((1 - cur_iter / max_iters) ** power)
    lr = max(lr, 1e-7)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ═══════════════════════════════════════════════════════════════
# Checkpoint Save / Load
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"[✓] Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer, scaler, device):
    print(f"[*] Loading checkpoint from {path} ...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"[✓] Resumed from epoch {ckpt['epoch']}. Starting at epoch {start_epoch}.")
    return start_epoch


# ═══════════════════════════════════════════════════════════════
# One Training Epoch
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(
    model, loader, optimizer, criterion, scaler,
    base_lr, max_iters, cur_iter_start,
    device, epoch, total_epochs, writer=None
):
    model.train()
    total_loss = 0.0
    drive_loss = 0.0
    lane_loss  = 0.0
    det_loss   = 0.0
    t_start = time.time()

    for i, batch in enumerate(loader):
        cur_iter = cur_iter_start + i

        # ── Apply Synthetic Roll Augmentation ──────────────────
        batch = apply_batch_roll_augmentation(batch, angle_range=30.0)

        images      = batch['image'].to(device)
        roll_angles = batch['roll_angle'].to(device)

        optimizer.zero_grad()

        # ── Forward Pass (AMP) ─────────────────────────────────
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images, roll_angles)
            losses  = criterion(outputs, batch)
            loss    = losses['total']

        # ── Backward Pass ──────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        # ── LR Update ──────────────────────────────────────────
        lr = poly_lr(optimizer, base_lr, max_iters, cur_iter)

        # ── Accumulate ─────────────────────────────────────────
        total_loss += loss.item()
        drive_loss += losses['drive'].item()
        lane_loss  += losses['lane'].item()
        det_loss   += losses['det'].item()

        if i % 20 == 0:
            elapsed = time.time() - t_start
            print(
                f"  [Ep {epoch+1:03d}/{total_epochs} | Iter {i:04d}/{len(loader)}] "
                f"LR={lr:.6f} | Total={loss.item():.4f} | "
                f"Drive={losses['drive'].item():.4f} | "
                f"Lane={losses['lane'].item():.4f} | "
                f"Det={losses['det'].item():.4f} | "
                f"Elapsed={elapsed:.0f}s"
            )

        if writer and i % 20 == 0:
            step = cur_iter
            writer.add_scalar('Train/total', loss.item(), step)
            writer.add_scalar('Train/drive', losses['drive'].item(), step)
            writer.add_scalar('Train/lane',  losses['lane'].item(),  step)
            writer.add_scalar('Train/det',   losses['det'].item(),   step)
            writer.add_scalar('Train/lr',    lr, step)

    n = len(loader)
    return {
        'total': total_loss / n,
        'drive': drive_loss / n,
        'lane':  lane_loss  / n,
        'det':   det_loss   / n,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Reproducibility ────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")
    if device.type == 'cuda':
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Output Directory ───────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Checkpoints will be saved to: {output_dir}")

    # ── TensorBoard ────────────────────────────────────────────
    writer = None
    if HAS_TB:
        tb_dir = output_dir / 'tensorboard'
        tb_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(str(tb_dir))
        print(f"[*] TensorBoard logs: {tb_dir}")

    # ── Dataset & DataLoader ───────────────────────────────────
    print("\n[*] Building composite IUR dataset...")
    target_hw = (args.img_h, args.img_w)

    train_dataset = build_composite_dataset(
        idd_root=args.idd_root,
        rdd_root=args.rdd_root,
        bdd_root=args.bdd_root,
        split='train',
        target_hw=target_hw,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"[*] Training batches per epoch: {len(train_loader)}")

    # ── Model ──────────────────────────────────────────────────
    print("\n[*] Building AuRoRA-2W-IUR model...")
    model = get_aurora2w_iur_model(
        num_det_classes=3,
        num_drive_classes=3,
        fpn_ch=args.fpn_ch,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable parameters: {total_params/1e6:.2f}M")

    # ── Loss ───────────────────────────────────────────────────
    criterion = AuRoRA2W_IUR_Loss(
        num_det_classes=3,
        w_drive=args.w_drive,
        w_lane=args.w_lane,
        w_det=args.w_det,
        img_h=args.img_h,
        img_w=args.img_w,
    ).to(device)

    # ── Optimizer ──────────────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # ── AMP Scaler ─────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── Resume ─────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler, device)
    else:
        # Auto-detect latest checkpoint in output_dir
        ckpts = sorted(glob.glob(str(output_dir / 'aurora2w_iur_epoch_*.pt')))
        if ckpts:
            latest = ckpts[-1]
            print(f"\n[*] Auto-resuming from latest checkpoint: {latest}")
            start_epoch = load_checkpoint(latest, model, optimizer, scaler, device)

    # ── Training Loop ──────────────────────────────────────────
    max_iters = args.epochs * len(train_loader)
    print(f"\n[*] Starting training: epochs {start_epoch+1} → {args.epochs}")
    print(f"    Total iterations: {max_iters}")
    print("="*70)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n─── Epoch {epoch+1}/{args.epochs} ───────────────────────────")
        cur_iter_start = epoch * len(train_loader)

        epoch_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            base_lr=args.lr,
            max_iters=max_iters,
            cur_iter_start=cur_iter_start,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            writer=writer,
        )

        print(
            f"\n  → Epoch {epoch+1} summary | "
            f"Total={epoch_losses['total']:.4f} | "
            f"Drive={epoch_losses['drive']:.4f} | "
            f"Lane={epoch_losses['lane']:.4f} | "
            f"Det={epoch_losses['det']:.4f}"
        )

        if writer:
            for k, v in epoch_losses.items():
                writer.add_scalar(f'Epoch/{k}', v, epoch)

        # ── Save Checkpoint ────────────────────────────────────
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = output_dir / f'aurora2w_iur_epoch_{epoch}.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict':    scaler.state_dict(),
                'losses': epoch_losses,
                'args': vars(args),
            }, str(ckpt_path))

    print("\n[+] Training complete!")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
