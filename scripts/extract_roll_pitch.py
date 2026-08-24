#!/usr/bin/env python3
"""
AuRoRA-2W Phase 1 — Two-Wheeler Attitude Estimation & Visualization
=====================================================================
Fuses gyroscope roll/pitch rates with a kinematic lean model via Kalman
filtering to produce drift-free attitude estimates synced to video frames.

Key improvements (v2):
  • Dual-axis estimation (roll + pitch) with separate 1D Kalman filters
  • Continuous-time noise parameters properly discretized (Q × dt)
  • Adaptive measurement noise — increases at low speed where the
    kinematic lean model is unreliable
  • Professional HUD overlay: artificial horizon, rolling time-series
    graph, GPS mini-map, lean severity indicator, telemetry panel

Usage:
  python extract_roll_pitch.py --visualize               # all clips
  python extract_roll_pitch.py --clip 01_019 --visualize  # single clip
"""

import argparse
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import cv2

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x   # silent fallback


# ═══════════════════════════════════════════════════════════════════════
# Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════

GRAVITY = 9.80665  # m/s²

# ── Kalman filter tuning (continuous-time noise densities) ──────────
ROLL_Q_ANGLE  = 0.50     # deg²/s  — angle process noise density
ROLL_Q_BIAS   = 0.001    # (deg/s)²/s — gyro-bias random-walk density
ROLL_R_BASE   = 2.0      # deg²  — kinematic measurement noise (v > 3 m/s)

PITCH_Q_ANGLE = 0.30     # deg²/s
PITCH_Q_BIAS  = 0.001    # (deg/s)²/s
PITCH_R_BASE  = 15.0     # deg²  — weak zero-pull (no accel data)

SPEED_LO   = 0.5   # m/s — below: kinematic model disabled
SPEED_FADE = 3.0   # m/s — below: R increases smoothly (quadratic fade)

# ── Visualisation layout ───────────────────────────────────────────
GRAPH_HISTORY    = 90       # frames of rolling history (~3 s @ 30 Hz)
GRAPH_Y_RANGE    = 15.0     # ±deg on y-axis
HUD_ALPHA        = 0.65     # panel background transparency
HORIZON_R        = 90       # artificial-horizon radius (px)
PITCH_PX_PER_DEG = 4        # horizon vertical shift per degree

# Colours (BGR)
C_BG          = (20, 20, 20)
C_TEXT        = (220, 220, 220)
C_DIM         = (130, 130, 130)
C_GRID        = (55, 55, 55)
C_SKY         = (160, 120, 50)
C_GROUND      = (40, 70, 110)
C_ROLL        = (80, 80, 255)
C_PITCH       = (255, 140, 40)
C_GREEN       = (80, 200, 80)
C_YELLOW      = (60, 220, 255)
C_ORANGE      = (50, 150, 255)
C_RED         = (60, 60, 255)
C_CYAN        = (255, 255, 0)
C_HZ_LINE     = (190, 190, 190)
C_REF         = (0, 255, 255)
C_GPS_TRAIL   = (120, 200, 120)
C_GPS_DOT     = (0, 255, 255)

# Lean-severity thresholds (deg → colour)
LEAN_THRESH = [(5, C_GREEN), (15, C_YELLOW), (30, C_ORANGE)]


# ═══════════════════════════════════════════════════════════════════════
# Kalman Filter
# ═══════════════════════════════════════════════════════════════════════

class SingleAxisKF:
    """
    1D Kalman filter for attitude estimation from gyroscope integration
    with an external scalar measurement.

    State vector
    ─────────────
      x = [angle (deg),  gyro_bias (deg/s)]ᵀ

    Prediction model
    ─────────────────
      angle_k = angle_{k-1} + (ω_gyro − bias) · Δt
      bias_k  = bias_{k-1}

    Process noise Q is multiplied by Δt at every step so the filter
    behaves identically regardless of the actual sampling rate.
    """

    def __init__(self, q_angle: float, q_bias: float, r_base: float):
        self.x = np.zeros(2, dtype=np.float64)
        self.P = np.diag([1.0, 0.1]).astype(np.float64)
        self._qa = q_angle
        self._qb = q_bias
        self._rb = r_base
        self._H  = np.array([[1.0, 0.0]], dtype=np.float64)

    # ── predict ──────────────────────────────────────────────────────

    def predict(self, gyro_dps: float, dt: float):
        if dt <= 0:
            return
        rate = gyro_dps - self.x[1]          # bias-corrected rate
        self.x[0] += rate * dt

        F = np.array([[1.0, -dt],
                      [0.0,  1.0]], dtype=np.float64)
        Q = np.array([[self._qa * dt, 0.0],
                      [0.0,           self._qb * dt]], dtype=np.float64)
        self.P = F @ self.P @ F.T + Q

    # ── update ───────────────────────────────────────────────────────

    def update(self, z: float, R: float = None) -> float:
        R = R if R is not None else self._rb
        y = z - (self._H @ self.x).item()              # innovation
        S = (self._H @ self.P @ self._H.T).item() + R  # innovation cov
        K = (self.P @ self._H.T) / S                    # gain (2×1)

        self.x += K.flatten() * y
        self.P = (np.eye(2) - K @ self._H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)            # numerical symmetry
        return float(self.x[0])

    @property
    def angle(self):    return float(self.x[0])
    @property
    def bias(self):     return float(self.x[1])
    @property
    def variance(self): return float(self.P[0, 0])


class AttitudeEstimator:
    """
    Dual-axis attitude estimator for two-wheelers.

    Roll  — gyro_x integration fused with kinematic lean
            φ = arctan(v · ω_yaw / g)   (coordinated-turn model)
    Pitch — gyro_y integration fused with a weak zero-prior
            (no accelerometer available in the dataset)
    """

    def __init__(self):
        self.roll_kf  = SingleAxisKF(ROLL_Q_ANGLE,  ROLL_Q_BIAS,  ROLL_R_BASE)
        self.pitch_kf = SingleAxisKF(PITCH_Q_ANGLE, PITCH_Q_BIAS, PITCH_R_BASE)

    def step(self, gyro_roll: float, gyro_pitch: float,
             gyro_yaw: float, speed: float, dt: float):
        """Process one IMU sample.  Returns (roll_deg, pitch_deg)."""

        # ── Roll ─────────────────────────────────────────────────────
        self.roll_kf.predict(gyro_roll, dt)
        yaw_rad = np.radians(gyro_yaw)

        if speed > SPEED_FADE:
            z = np.degrees(np.arctan2(speed * yaw_rad, GRAVITY))
            r = ROLL_R_BASE
        elif speed > SPEED_LO:
            z = np.degrees(np.arctan2(speed * yaw_rad, GRAVITY))
            r = ROLL_R_BASE * (SPEED_FADE / max(speed, 0.01)) ** 2
        else:
            z, r = 0.0, ROLL_R_BASE * 100.0

        roll = self.roll_kf.update(z, r)

        # ── Pitch ────────────────────────────────────────────────────
        self.pitch_kf.predict(gyro_pitch, dt)
        pitch = self.pitch_kf.update(0.0, PITCH_R_BASE)

        return roll, pitch

    @property
    def diagnostics(self):
        return dict(
            roll_bias  = self.roll_kf.bias,
            pitch_bias = self.pitch_kf.bias,
            roll_var   = self.roll_kf.variance,
            pitch_var  = self.pitch_kf.variance,
        )


# ═══════════════════════════════════════════════════════════════════════
# HUD Renderer
# ═══════════════════════════════════════════════════════════════════════

def _severity_colour(deg: float):
    a = abs(deg)
    for thresh, col in LEAN_THRESH:
        if a < thresh:
            return col
    return C_RED


class HUDRenderer:
    """Professional heads-up display overlay for attitude visualisation."""

    def __init__(self, w: int, h: int,
                 gps_lats: np.ndarray = None,
                 gps_lons: np.ndarray = None):
        self.w, self.h = w, h

        # Panel rectangles (x1, y1, x2, y2)
        self.info  = (20, 20, 390, 270)
        self.graph = (20, h - 260, 570, h - 20)
        self.gps   = (w - 225, h - 260, w - 20, h - 20)

        # Artificial-horizon centre
        self.hz_cx = w - 140
        self.hz_cy = 140

        # Pre-build circle mask
        sz = 2 * HORIZON_R
        self.hz_mask = np.zeros((sz, sz), dtype=np.uint8)
        cv2.circle(self.hz_mask, (HORIZON_R, HORIZON_R), HORIZON_R, 255, -1)

        # GPS normalisation
        self._init_gps(gps_lats, gps_lons)

        # Rolling history for graph
        self.roll_h  = deque(maxlen=GRAPH_HISTORY)
        self.pitch_h = deque(maxlen=GRAPH_HISTORY)

    # ── GPS helpers ──────────────────────────────────────────────────

    def _init_gps(self, lats, lons):
        self.has_gps = False
        if lats is None or len(lats) == 0:
            return
        # Simple moving-average smoothing (reduce GPS noise)
        k = min(5, len(lats))
        kern = np.ones(k) / k
        self.gps_lats = np.convolve(lats, kern, mode="same")
        self.gps_lons = np.convolve(lons, kern, mode="same")
        lat_r = self.gps_lats.max() - self.gps_lats.min()
        lon_r = self.gps_lons.max() - self.gps_lons.min()
        self.gps_span = max(lat_r, lon_r, 1e-6)
        self.gps_lat_mid = (self.gps_lats.max() + self.gps_lats.min()) / 2
        self.gps_lon_mid = (self.gps_lons.max() + self.gps_lons.min()) / 2
        self.has_gps = self.gps_span > 1e-5

    def _gps_px(self, lat, lon, bx, by, bw, bh):
        pad = 18
        nx = (lon - self.gps_lon_mid) / self.gps_span + 0.5
        ny = 0.5 - (lat - self.gps_lat_mid) / self.gps_span
        px = int(bx + pad + nx * (bw - 2 * pad))
        py = int(by + pad + ny * (bh - 2 * pad))
        return np.clip(px, bx, bx + bw), np.clip(py, by, by + bh)

    # ── Main entry ───────────────────────────────────────────────────

    def render(self, frame: np.ndarray, d: dict) -> np.ndarray:
        """Render complete HUD onto *frame* (in-place).  Returns frame."""
        self.roll_h.append(d["roll_deg"])
        self.pitch_h.append(d["pitch_deg"])

        # 1. Semi-transparent backgrounds
        ov = frame.copy()
        for box in (self.info, self.graph, self.gps):
            cv2.rectangle(ov, (box[0], box[1]), (box[2], box[3]), C_BG, -1)
        self._horizon_bg(ov, d["roll_deg"], d["pitch_deg"])

        # 2. Blend
        cv2.addWeighted(ov, HUD_ALPHA, frame, 1.0 - HUD_ALPHA, 0, frame)

        # 3. Opaque content on top
        self._info_text(frame, d)
        self._horizon_ref(frame, d["roll_deg"])
        self._graph(frame)
        self._gps_map(frame, d)
        return frame

    # ── Artificial horizon background ────────────────────────────────

    def _horizon_bg(self, ov, roll, pitch):
        r  = HORIZON_R
        sz = 2 * r
        big = sz * 3  # oversize for rotation headroom

        canvas = np.zeros((big, big, 3), dtype=np.uint8)
        mid = big // 2 + int(pitch * PITCH_PX_PER_DEG)
        canvas[:mid] = C_SKY
        canvas[mid:] = C_GROUND
        cv2.line(canvas, (0, mid), (big, mid), C_HZ_LINE, 2)

        # Pitch ladder (every 5°)
        cx = big // 2
        for deg in range(-20, 25, 5):
            if deg == 0:
                continue
            y = mid - int(deg * PITCH_PX_PER_DEG)
            ll = 35 if deg % 10 == 0 else 18
            cv2.line(canvas, (cx - ll, y), (cx + ll, y), C_HZ_LINE, 1)

        # Rotate by roll
        M = cv2.getRotationMatrix2D((big // 2, big // 2), roll, 1.0)
        canvas = cv2.warpAffine(canvas, M, (big, big),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        # Crop centre
        off = (big - sz) // 2
        crop = canvas[off:off + sz, off:off + sz]

        # Paste into overlay via mask
        x1 = self.hz_cx - r
        y1 = self.hz_cy - r
        # Clamp to frame
        fx1, fy1 = max(x1, 0), max(y1, 0)
        fx2, fy2 = min(x1 + sz, self.w), min(y1 + sz, self.h)
        dx, dy = fx1 - x1, fy1 - y1
        dw, dh = fx2 - fx1, fy2 - fy1

        m = self.hz_mask[dy:dy + dh, dx:dx + dw]
        c = crop[dy:dy + dh, dx:dx + dw]
        for ch in range(3):
            ov[fy1:fy2, fx1:fx2, ch] = np.where(m > 0, c[:, :, ch],
                                                  ov[fy1:fy2, fx1:fx2, ch])

    def _horizon_ref(self, frame, roll):
        cx, cy = self.hz_cx, self.hz_cy
        # Airplane wings
        cv2.line(frame, (cx - 30, cy), (cx - 10, cy), C_REF, 2)
        cv2.line(frame, (cx + 10, cy), (cx + 30, cy), C_REF, 2)
        cv2.circle(frame, (cx, cy), 4, C_REF, 2)
        # Border
        cv2.circle(frame, (cx, cy), HORIZON_R, C_HZ_LINE, 2)
        # Label
        cv2.putText(frame, f"ROLL {roll:+.1f} deg",
                    (cx - 48, cy + HORIZON_R + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT, 1, cv2.LINE_AA)

    # ── Info panel ───────────────────────────────────────────────────

    def _info_text(self, fr, d):
        x0, y0 = self.info[0], self.info[1]
        lx = x0 + 15
        x2 = self.info[2]

        # Title
        cv2.putText(fr, "AuRoRA-2W  ATTITUDE HUD", (lx, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_CYAN, 1, cv2.LINE_AA)
        cv2.line(fr, (x0 + 5, y0 + 34), (x2 - 5, y0 + 34), C_GRID, 1)

        y = y0 + 58
        cv2.putText(fr, f"Frame  {d['frame_idx']:04d} / {d['total_frames']:04d}",
                    (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_TEXT, 1, cv2.LINE_AA)
        y += 22
        cv2.putText(fr, f"Time   {d['timestamp_s']:.3f} s",
                    (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_TEXT, 1, cv2.LINE_AA)
        cv2.line(fr, (x0 + 5, y + 10), (x2 - 5, y + 10), C_GRID, 1)

        # Roll
        y += 30
        sc = _severity_colour(d["roll_deg"])
        cv2.putText(fr, f"Roll   {d['roll_deg']:+7.2f} deg",
                    (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, sc, 1, cv2.LINE_AA)
        # Severity bar
        bx = x0 + 260
        bw, bh = 90, 12
        fill = min(int(abs(d["roll_deg"]) / 30.0 * bw), bw)
        cv2.rectangle(fr, (bx, y - 11), (bx + bw, y + 1), C_GRID, 1)
        if fill > 0:
            cv2.rectangle(fr, (bx, y - 11), (bx + fill, y + 1), sc, -1)

        # Pitch
        y += 25
        cv2.putText(fr, f"Pitch  {d['pitch_deg']:+7.2f} deg",
                    (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_PITCH, 1,
                    cv2.LINE_AA)
        cv2.line(fr, (x0 + 5, y + 12), (x2 - 5, y + 12), C_GRID, 1)

        # Speed
        y += 32
        kmh = d["speed_mps"] * 3.6
        sp_c = C_GREEN if kmh < 25 else (C_YELLOW if kmh < 50 else C_RED)
        cv2.putText(fr, f"Speed  {kmh:5.1f} km/h  ({d['speed_mps']:.1f} m/s)",
                    (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sp_c, 1,
                    cv2.LINE_AA)

        # Lean severity label
        y += 24
        a = abs(d["roll_deg"])
        if   a < 5:  lbl, lc = "GENTLE",     C_GREEN
        elif a < 15: lbl, lc = "MODERATE",    C_YELLOW
        elif a < 30: lbl, lc = "AGGRESSIVE",  C_ORANGE
        else:        lbl, lc = "EXTREME",     C_RED
        cv2.putText(fr, f"Lean   {lbl}", (lx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, lc, 1, cv2.LINE_AA)

    # ── Time-series graph ────────────────────────────────────────────

    def _graph(self, fr):
        gx1, gy1, gx2, gy2 = self.graph
        # Plot area (inset for labels)
        px1, py1 = gx1 + 50, gy1 + 28
        px2, py2 = gx2 - 12, gy2 - 22
        pw, ph = px2 - px1, py2 - py1

        cv2.putText(fr, "ATTITUDE  (3 s window)", (gx1 + 12, gy1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_DIM, 1, cv2.LINE_AA)

        # Grid
        for deg in (-10, -5, 0, 5, 10):
            frac = (deg + GRAPH_Y_RANGE) / (2 * GRAPH_Y_RANGE)
            y = int(py2 - frac * ph)
            cv2.line(fr, (px1, y), (px2, y), C_GRID, 1)
            cv2.putText(fr, f"{deg:+d}", (gx1 + 15, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_DIM, 1,
                        cv2.LINE_AA)

        # Plot helper
        def _plot(hist, colour):
            n = len(hist)
            if n < 2:
                return
            pts = []
            for i, v in enumerate(hist):
                x = int(px1 + pw * i / (GRAPH_HISTORY - 1))
                f = np.clip((v + GRAPH_Y_RANGE) / (2 * GRAPH_Y_RANGE), 0, 1)
                y = int(py2 - f * ph)
                pts.append([x, y])
            cv2.polylines(fr, [np.array(pts, np.int32)],
                          False, colour, 2, cv2.LINE_AA)

        _plot(self.roll_h, C_ROLL)
        _plot(self.pitch_h, C_PITCH)

        # Legend
        lx = px2 - 100
        ly = gy2 - 6
        cv2.circle(fr, (lx, ly - 3), 4, C_ROLL, -1)
        cv2.putText(fr, "Roll", (lx + 8, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_ROLL, 1, cv2.LINE_AA)
        cv2.circle(fr, (lx + 55, ly - 3), 4, C_PITCH, -1)
        cv2.putText(fr, "Pitch", (lx + 63, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_PITCH, 1, cv2.LINE_AA)

    # ── GPS mini-map ─────────────────────────────────────────────────

    def _gps_map(self, fr, d):
        gx1, gy1, gx2, gy2 = self.gps
        gw, gh = gx2 - gx1, gy2 - gy1

        cv2.putText(fr, "GPS TRACK", (gx1 + 12, gy1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_DIM, 1, cv2.LINE_AA)

        if not self.has_gps:
            cv2.putText(fr, "N/A", (gx1 + gw // 2 - 12, gy1 + gh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_DIM, 1)
            return

        # Trail up to current frame
        idx = d["frame_idx"]
        n = min(idx + 1, len(self.gps_lats))
        if n > 1:
            pts = []
            for i in range(n):
                px, py = self._gps_px(self.gps_lats[i], self.gps_lons[i],
                                       gx1, gy1 + 25, gw, gh - 30)
                pts.append([px, py])
            cv2.polylines(fr, [np.array(pts, np.int32)],
                          False, C_GPS_TRAIL, 1, cv2.LINE_AA)

        # Current position
        if idx < len(self.gps_lats):
            cpx, cpy = self._gps_px(d["gps_lat"], d["gps_lon"],
                                     gx1, gy1 + 25, gw, gh - 30)
            cv2.circle(fr, (cpx, cpy), 5, C_GPS_DOT, -1)
            cv2.circle(fr, (cpx, cpy), 9, C_GPS_DOT, 1)

        # North arrow
        nx = gx2 - 18
        cv2.putText(fr, "N", (nx - 4, gy1 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_DIM, 1)
        cv2.arrowedLine(fr, (nx, gy1 + 48), (nx, gy1 + 33),
                        C_DIM, 1, tipLength=0.4)


# ═══════════════════════════════════════════════════════════════════════
# Telemetry Processing
# ═══════════════════════════════════════════════════════════════════════

def process_telemetry_file(
    input_csv: Path,
    output_csv: Path,
    roll_axis:  str = "gyro_x",
    pitch_axis: str = "gyro_y",
    yaw_axis:   str = "gyro_z",
):
    """Read raw IMU CSV → run dual-axis Kalman filter → write synced CSV."""
    df = pd.read_csv(input_csv)
    cols = {c.lower(): c for c in df.columns}

    col_frame = cols.get("frame", "frame")
    col_time  = cols.get("timestamp_s", "timestamp_s")
    col_speed = cols.get("speed_2d_mps", None)

    r_col = cols.get(roll_axis.lower(),  roll_axis)
    p_col = cols.get(pitch_axis.lower(), pitch_axis)
    y_col = cols.get(yaw_axis.lower(),   yaw_axis)

    ts          = df[col_time].values
    roll_rates  = np.degrees(df[r_col].values)
    pitch_rates = np.degrees(df[p_col].values) if p_col in df.columns else np.zeros(len(df))
    yaw_rates   = np.degrees(df[y_col].values) if y_col in df.columns else np.zeros(len(df))
    speeds      = (df[col_speed].values
                   if col_speed and col_speed in df.columns
                   else np.zeros(len(df)))
    gps_lat     = df["gps_lat"].values if "gps_lat" in df.columns else np.full(len(df), np.nan)
    gps_lon     = df["gps_lon"].values if "gps_lon" in df.columns else np.full(len(df), np.nan)

    est = AttitudeEstimator()
    dt_default = 1.0 / 30.0
    rows = []

    for i in range(len(df)):
        dt = (ts[i] - ts[i - 1]) if i > 0 else dt_default
        dt = max(dt, 1e-6)

        roll, pitch = est.step(roll_rates[i], pitch_rates[i],
                               yaw_rates[i], speeds[i], dt)
        diag = est.diagnostics

        rows.append(dict(
            frame_idx      = int(df[col_frame].values[i]) if col_frame in df.columns else i,
            timestamp_s    = ts[i],
            roll_deg       = round(roll, 3),
            pitch_deg      = round(pitch, 3),
            roll_rate_dps  = round(roll_rates[i], 3),
            pitch_rate_dps = round(pitch_rates[i], 3),
            yaw_rate_dps   = round(yaw_rates[i], 3),
            speed_mps      = round(speeds[i], 3),
            speed_kmh      = round(speeds[i] * 3.6, 2),
            gps_lat        = round(gps_lat[i], 8),
            gps_lon        = round(gps_lon[i], 8),
            roll_unc_deg   = round(np.sqrt(max(diag["roll_var"], 0)), 3),
            pitch_unc_deg  = round(np.sqrt(max(diag["pitch_var"], 0)), 3),
        ))

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[+] Saved synced CSV -> {output_csv.resolve()}")
    return out_df


# ═══════════════════════════════════════════════════════════════════════
# Overlay Video Generation
# ═══════════════════════════════════════════════════════════════════════

def create_validation_overlay(
    video_path:  Path,
    synced_csv:  Path,
    output_path: Path,
):
    """Generate professional HUD overlay video from synced CSV + raw video."""
    df  = pd.read_csv(synced_csv)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[-] Cannot open video: {video_path}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not out.isOpened():
        print(f"[-] Cannot create output: {output_path}")
        cap.release()
        return

    n_csv = len(df)
    total = min(n_vid, n_csv)

    gps_lats = df["gps_lat"].values if "gps_lat" in df.columns else None
    gps_lons = df["gps_lon"].values if "gps_lon" in df.columns else None
    hud = HUDRenderer(w, h, gps_lats, gps_lons)

    print(f"[*] Rendering overlay: {total} frames @ {fps:.0f} fps ...")

    for i in tqdm(range(total), desc="  Rendering", unit="f"):
        ret, frame = cap.read()
        if not ret:
            break

        row = df.iloc[i]
        d = dict(
            frame_idx     = i,
            total_frames  = total,
            timestamp_s   = row.get("timestamp_s",   i / fps),
            roll_deg      = row.get("roll_deg",      0.0),
            pitch_deg     = row.get("pitch_deg",     0.0),
            roll_rate_dps = row.get("roll_rate_dps", 0.0),
            pitch_rate_dps= row.get("pitch_rate_dps",0.0),
            yaw_rate_dps  = row.get("yaw_rate_dps", 0.0),
            speed_mps     = row.get("speed_mps",    0.0),
            gps_lat       = row.get("gps_lat",      0.0),
            gps_lon       = row.get("gps_lon",      0.0),
        )
        hud.render(frame, d)
        out.write(frame)

    cap.release()
    out.release()
    print(f"[+] Overlay video -> {output_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="AuRoRA-2W Phase 1: Two-Wheeler Attitude Estimation")
    ap.add_argument("--clip", type=str, default=None,
                    help="Specific clip ID (e.g. 01_019). Default: all.")
    ap.add_argument("--visualize", action="store_true",
                    help="Generate HUD overlay video(s).")
    ap.add_argument("--roll-axis",  default="gyro_x")
    ap.add_argument("--pitch-axis", default="gyro_y")
    ap.add_argument("--yaw-axis",   default="gyro_z")
    args = ap.parse_args()

    raw_imu   = Path("./data/raw_imu")
    raw_video = Path("./data/raw_video")
    synced    = Path("./data/synced")

    csvs = ([raw_imu / f"{args.clip}.csv"] if args.clip
            else sorted(raw_imu.glob("*.csv")))

    if not csvs:
        print("[-] No raw IMU CSVs in ./data/raw_imu/")
        return

    for csv_file in csvs:
        clip = csv_file.stem
        synced_csv = synced / f"{clip}_synced.csv"

        process_telemetry_file(csv_file, synced_csv,
                               roll_axis=args.roll_axis,
                               pitch_axis=args.pitch_axis,
                               yaw_axis=args.yaw_axis)

        if args.visualize:
            vid = raw_video / f"{clip}.mp4"
            if vid.exists():
                create_validation_overlay(
                    vid, synced_csv,
                    synced / f"{clip}_overlay.mp4")
            else:
                print(f"[!] Video not found: {vid}")

    print("\n[OK] Phase 1 processing complete!")


if __name__ == "__main__":
    main()