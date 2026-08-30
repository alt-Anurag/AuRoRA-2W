#!/usr/bin/env python3
"""
AuRoRA-2W Stage 5 — IMU Roll Distribution Report
==================================================
Analyses the distribution of real-world roll angles and compares
against the +/−30° synthetic augmentation range used in training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="IMU roll distribution analysis")
    parser.add_argument("--input", type=Path,
                        default=project_root / "sensor_log" / "cleaned_orientation.csv",
                        help="Cleaned orientation CSV")
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "output" / "imu_qa",
                        help="Output directory for plots")
    parser.add_argument("--aug-range", type=float, default=30.0,
                        help="Synthetic augmentation range (degrees)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    t = df["timestamp_s"].values
    roll = df["roll_deg"].values
    pitch = df["pitch_deg"].values

    dt = np.diff(t)
    roll_rate = np.diff(roll) / dt  # deg/s

    duration = t[-1] - t[0]
    abs_roll = np.abs(roll)

    # ── Figure with 4 subplots ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Histogram of roll_deg
    ax = axes[0, 0]
    ax.hist(roll, bins=100, color="steelblue", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(-args.aug_range, color="red", linestyle="--", linewidth=1.5, label=f"−{args.aug_range}° aug limit")
    ax.axvline(+args.aug_range, color="red", linestyle="--", linewidth=1.5, label=f"+{args.aug_range}° aug limit")
    ax.axvline(roll.min(), color="limegreen", linestyle="-", linewidth=1.2, alpha=0.7, label=f"min: {roll.min():.1f}°")
    ax.axvline(roll.max(), color="limegreen", linestyle="-", linewidth=1.2, alpha=0.7, label=f"max: {roll.max():.1f}°")
    ax.set_xlabel("Roll (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Roll Angle Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Time series
    ax = axes[0, 1]
    ax.plot(t, roll, linewidth=0.5, color="steelblue")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(-args.aug_range, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(+args.aug_range, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Roll (deg)")
    ax.set_title("Roll Angle Over Time")
    ax.grid(True, alpha=0.3)

    # (c) Roll rate histogram
    ax = axes[1, 0]
    # Clip extreme rates for display
    rate_clip = np.clip(roll_rate, np.percentile(roll_rate, 0.5), np.percentile(roll_rate, 99.5))
    ax.hist(rate_clip, bins=100, color="coral", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Roll rate (deg/s)")
    ax.set_ylabel("Count")
    ax.set_title("Roll Rate of Change Distribution")
    ax.grid(True, alpha=0.3)

    # (d) CDF of |roll|
    ax = axes[1, 1]
    sorted_abs = np.sort(abs_roll)
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs) * 100
    ax.plot(sorted_abs, cdf, linewidth=2, color="darkblue")
    ax.axvline(5, color="green", linestyle="--", alpha=0.6, label="|roll| = 5°")
    ax.axvline(15, color="orange", linestyle="--", alpha=0.6, label="|roll| = 15°")
    ax.axvline(args.aug_range, color="red", linestyle="--", alpha=0.6, label=f"|roll| = {args.aug_range}°")
    ax.set_xlabel("|Roll| (deg)")
    ax.set_ylabel("Cumulative % of time")
    ax.set_title("CDF of |Roll Angle|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"IMU Roll Distribution Report — {duration:.0f}s riding session",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = args.output_dir / "roll_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # ── Compute statistics ────────────────────────────────────────────
    pcts = [1, 5, 25, 50, 75, 95, 99]
    roll_percentiles = np.percentile(roll, pcts)

    pct_straight = np.mean(abs_roll < 5) * 100
    pct_significant = np.mean(abs_roll > 15) * 100
    pct_beyond_aug = np.mean(abs_roll > args.aug_range) * 100
    pct_covered = np.mean(abs_roll <= args.aug_range) * 100

    # ── Report ────────────────────────────────────────────────────────
    print("=" * 64)
    print("IMU ROLL DISTRIBUTION REPORT")
    print("=" * 64)
    print(f"\n  Total duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Samples:        {len(roll)}")

    print(f"\n  ROLL STATISTICS (degrees):")
    print(f"    min:   {roll.min():+8.2f}°")
    print(f"    max:   {roll.max():+8.2f}°")
    print(f"    mean:  {roll.mean():+8.2f}°")
    print(f"    std:   {roll.std():8.2f}°")
    print(f"\n  Percentiles:")
    for p, v in zip(pcts, roll_percentiles):
        print(f"    P{p:02d}:   {v:+8.2f}°")

    print(f"\n  TIME DISTRIBUTION:")
    print(f"    |roll| <  5° (near straight):     {pct_straight:5.1f}%")
    print(f"    |roll| > 15° (significant lean):  {pct_significant:5.1f}%")
    print(f"    |roll| > {args.aug_range:.0f}° (beyond aug range): {pct_beyond_aug:5.1f}%")

    print(f"\n  ROLL RATE (deg/s):")
    print(f"    mean |rate|:  {np.mean(np.abs(roll_rate)):.1f} deg/s")
    print(f"    max  |rate|:  {np.max(np.abs(roll_rate)):.1f} deg/s")
    print(f"    P95  |rate|:  {np.percentile(np.abs(roll_rate), 95):.1f} deg/s")

    print(f"\n  PITCH STATISTICS (degrees):")
    print(f"    min:   {pitch.min():+8.2f}°")
    print(f"    max:   {pitch.max():+8.2f}°")
    print(f"    mean:  {pitch.mean():+8.2f}°")
    print(f"    std:   {pitch.std():8.2f}°")

    print(f"\n{'─' * 64}")
    print(f"  VERDICT:")
    print(f"    Our +/−{args.aug_range:.0f}° synthetic augmentation range covers")
    print(f"    {pct_covered:.1f}% of observed real riding angles.")
    if pct_beyond_aug > 1:
        print(f"    ⚠ {pct_beyond_aug:.1f}% of readings exceed the augmentation range.")
        print(f"      Consider extending the range or treating these as edge cases.")
    else:
        print(f"    ✓ Only {pct_beyond_aug:.1f}% exceeds the range — coverage is adequate.")
    print(f"\n✓ Plot saved → {out_path}")


if __name__ == "__main__":
    main()
