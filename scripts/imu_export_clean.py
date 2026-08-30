#!/usr/bin/env python3
"""
AuRoRA-2W Stage 5 — IMU Export Clean
=====================================
Loads Sensor Logger app's Orientation.csv, converts roll/pitch/yaw from
radians to degrees, and saves a standardised clean CSV.

Input:  sensor_log/extracted/Orientation.csv + Metadata.csv
Output: sensor_log/cleaned_orientation.csv
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_extracted = project_root / "sensor_log" / "extracted"

    parser = argparse.ArgumentParser(description="Clean Sensor Logger orientation export")
    parser.add_argument("--orientation", type=Path,
                        default=default_extracted / "Orientation.csv",
                        help="Path to Orientation.csv")
    parser.add_argument("--metadata", type=Path,
                        default=default_extracted / "Metadata.csv",
                        help="Path to Metadata.csv")
    parser.add_argument("--output", type=Path,
                        default=project_root / "sensor_log" / "cleaned_orientation.csv",
                        help="Output clean CSV path")
    args = parser.parse_args()

    # ── Load Metadata ─────────────────────────────────────────────────
    meta = pd.read_csv(args.metadata)
    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    for col in meta.columns:
        print(f"  {col}: {meta[col].iloc[0]}")
    recording_epoch_ms = int(meta["recording epoch time"].iloc[0])
    print(f"\n  Recording epoch (ms): {recording_epoch_ms}")
    print(f"  Recording epoch (s):  {recording_epoch_ms / 1000:.3f}")

    # ── Load Orientation ──────────────────────────────────────────────
    orient = pd.read_csv(args.orientation)
    print(f"\n{'=' * 60}")
    print("ORIENTATION DATA")
    print("=" * 60)
    print(f"  Rows:    {len(orient)}")
    print(f"  Columns: {list(orient.columns)}")
    print(f"  Duration: {orient['seconds_elapsed'].iloc[-1]:.2f}s")

    dt = np.diff(orient["seconds_elapsed"].values)
    print(f"  Sample interval: mean={np.mean(dt)*1000:.2f}ms, "
          f"std={np.std(dt)*1000:.2f}ms, "
          f"min={np.min(dt)*1000:.2f}ms, max={np.max(dt)*1000:.2f}ms")

    # ── Convert radians → degrees ─────────────────────────────────────
    RAD2DEG = 180.0 / math.pi
    clean = pd.DataFrame({
        "timestamp_s": orient["seconds_elapsed"],
        "roll_deg":    orient["roll"]  * RAD2DEG,
        "pitch_deg":   orient["pitch"] * RAD2DEG,
        "yaw_deg":     orient["yaw"]   * RAD2DEG,
    })

    # ── Report statistics ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("CLEANED DATA STATISTICS (degrees)")
    print("=" * 60)
    for col in ["roll_deg", "pitch_deg", "yaw_deg"]:
        s = clean[col]
        print(f"\n  {col}:")
        print(f"    min:  {s.min():+8.2f}°")
        print(f"    max:  {s.max():+8.2f}°")
        print(f"    mean: {s.mean():+8.2f}°")
        print(f"    std:  {s.std():8.2f}°")

    # ── Save ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\n✓ Saved {len(clean)} rows → {args.output}")


if __name__ == "__main__":
    main()
