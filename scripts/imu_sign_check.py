#!/usr/bin/env python3
"""
AuRoRA-2W Stage 5 — IMU Sign Convention Check
===============================================
Finds clear left-turn and right-turn segments from the riding data and
plots roll_deg to determine whether the sensor's sign matches our IDFA
convention (positive roll = counter-clockwise rotation in IDFA).

NOTE: Uses riding turns rather than a stationary hand-lean test.
      Turns involve centripetal effects — treat as reasonable but
      not definitive.
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
    parser = argparse.ArgumentParser(description="IMU roll sign convention check")
    parser.add_argument("--input", type=Path,
                        default=project_root / "sensor_log" / "cleaned_orientation.csv",
                        help="Cleaned orientation CSV")
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "output" / "imu_qa",
                        help="Output directory for plots")
    parser.add_argument("--window-sec", type=float, default=10.0,
                        help="Window size (seconds) around each peak")
    parser.add_argument("--smooth-window", type=int, default=50,
                        help="Moving-average window (samples, ~50 = 0.5s at 10ms)")
    parser.add_argument("--offset", type=float, default=6.606,
                        help="IMU-to-video offset in seconds (video_time = imu_time - offset)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    t = df["timestamp_s"].values
    roll = df["roll_deg"].values

    # ── Smooth ────────────────────────────────────────────────────────
    kernel = np.ones(args.smooth_window) / args.smooth_window
    roll_smooth = np.convolve(roll, kernel, mode="same")

    # ── Find most positive and most negative peaks ────────────────────
    idx_pos = np.argmax(roll_smooth)
    idx_neg = np.argmin(roll_smooth)

    peaks = [
        ("Most POSITIVE roll", idx_pos, roll_smooth[idx_pos]),
        ("Most NEGATIVE roll", idx_neg, roll_smooth[idx_neg]),
    ]

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (label, peak_idx, peak_val) in zip(axes, peaks):
        t_peak = t[peak_idx]
        half_w = args.window_sec / 2
        mask = (t >= t_peak - half_w) & (t <= t_peak + half_w)

        ax.plot(t[mask], roll[mask], alpha=0.4, label="raw", color="steelblue")
        ax.plot(t[mask], roll_smooth[mask], linewidth=2, label="smoothed", color="navy")
        ax.axvline(t_peak, color="red", linestyle="--", linewidth=1.5, label=f"peak @ {t_peak:.1f}s")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("IMU time (s)")
        ax.set_ylabel("Roll (deg)")
        ax.set_title(f"{label}: {peak_val:+.1f}° @ t={t_peak:.1f}s\n"
                     f"(video ≈ {t_peak - args.offset:.1f}s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("IMU Roll Sign Convention Check (from riding turns)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = args.output_dir / "sign_check.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # ── Report ────────────────────────────────────────────────────────
    print("=" * 64)
    print("IMU ROLL SIGN CONVENTION CHECK")
    print("=" * 64)
    for label, peak_idx, peak_val in peaks:
        t_peak = t[peak_idx]
        print(f"\n  {label}:")
        print(f"    Value:      {peak_val:+.2f}°  (smoothed)")
        print(f"    Raw value:  {roll[peak_idx]:+.2f}°")
        print(f"    IMU time:   {t_peak:.2f}s")
        print(f"    Video time: ≈{t_peak - args.offset:.2f}s  (using offset {args.offset}s)")

    print(f"\n{'─' * 64}")
    print("IDFA CONVENTION:")
    print("  IDFAModule treats positive roll_angles (degrees) as the angle")
    print("  to counter-rotate. Internally it computes:")
    print("    x' = x*cos(θ) - y*sin(θ)")
    print("    y' = x*sin(θ) + y*cos(θ)")
    print("  where θ = roll_angles * π/180.")
    print("  Positive θ → counter-clockwise kernel rotation.")
    print()
    print("TO DETERMINE YOUR CONVENTION:")
    print("  1. Check the video at the reported timestamps.")
    print("  2. If the POSITIVE peak corresponds to a RIGHT lean in the")
    print("     video, then: positive roll = right lean.")
    print("  3. If the POSITIVE peak corresponds to a LEFT lean, then:")
    print("     positive roll = left lean.")
    print("  4. Compare with IDFA's expectation and note if a sign flip")
    print("     is needed in the data pipeline.")
    print()
    print("⚠  NOTE: This uses riding turns rather than a stationary hand-")
    print("   lean test. Centripetal effects during turns can slightly")
    print("   affect the fused orientation reading. Treat as a reasonable")
    print("   but not definitive check.")
    print(f"\n✓ Plot saved → {out_path}")


if __name__ == "__main__":
    main()
