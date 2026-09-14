"""Offline reference for Android's timestamped quaternion-to-camera-roll contract.

Quaternions are normalized wxyz, device -> world. Extrinsic R_device_camera
maps optical camera axes (x right, y down, z forward) into sensor device axes.
World uses z up. No raw Android Euler component is accepted as camera roll.
"""

import bisect
import math
import numpy as np


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Expected a finite, nonzero wxyz quaternion")
    return q / np.linalg.norm(q)


def slerp(q0, q1, fraction):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("SLERP fraction must be finite and in [0,1]; extrapolation is unsupported")
    q0, q1 = normalize_quaternion(q0), normalize_quaternion(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1, 1)
    if dot > .9995:
        return normalize_quaternion(q0 + fraction * (q1 - q0))
    theta = np.arccos(dot)
    return (np.sin((1 - fraction) * theta) * q0 + np.sin(fraction * theta) * q1) / np.sin(theta)


def quaternion_matrix(q):
    w, x, y, z = normalize_quaternion(q)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def camera_roll(q_world_device, r_device_camera):
    extrinsic = np.asarray(r_device_camera, dtype=np.float64)
    if (extrinsic.shape != (3, 3) or not np.isfinite(extrinsic).all()
            or not np.allclose(extrinsic.T @ extrinsic, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(extrinsic), 1, atol=1e-5)):
        raise ValueError("R_device_camera must be a calibrated proper rotation")
    gravity = (quaternion_matrix(q_world_device) @ extrinsic).T @ np.array([0., 0., -1.])
    if np.hypot(gravity[0], gravity[1]) < .1:
        return 0., 0.  # Optical axis too close to gravity: horizon roll ill-defined.
    return math.atan2(-gravity[0], gravity[1]), 1.


class AttitudeTimeline:
    """Bracketed attitude interpolation with an explicit sampling-gap policy.

    Default 50 ms is appropriate only when the supplied stream is that frequent.
    A deliberately selected 120 ms limit can interpolate a 10 Hz source, but
    cannot reconstruct motion between those measurements or validate vibration.
    Exact stored samples remain valid on either side of a gap. No extrapolation.
    """
    def __init__(self, timestamps, quaternions, max_gap_s=.05):
        self.times = np.asarray(timestamps, dtype=np.float64)
        if (self.times.ndim != 1 or len(self.times) < 2 or not np.isfinite(self.times).all()
                or not (np.diff(self.times) > 0).all()
                or not np.isfinite(max_gap_s) or max_gap_s <= 0):
            raise ValueError("Need strictly increasing finite timestamps and positive max_gap_s")
        if len(quaternions) != len(self.times):
            raise ValueError("Quaternion/timestamp length mismatch")
        self.quaternions = [normalize_quaternion(q) for q in quaternions]
        self.max_gap_s = float(max_gap_s)

    def at(self, timestamp):
        """No extrapolation; caller must buffer to the frame exposure timestamp."""
        if not np.isfinite(timestamp) or timestamp < self.times[0] or timestamp > self.times[-1]:
            return None
        i = bisect.bisect_left(self.times, timestamp)
        if self.times[i] == timestamp:
            return self.quaternions[i].copy()
        gap = self.times[i] - self.times[i - 1]
        if gap > self.max_gap_s + 1e-12:
            return None
        return slerp(self.quaternions[i - 1], self.quaternions[i], (timestamp - self.times[i - 1]) / gap)
