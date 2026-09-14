import math

import numpy as np
import pytest

from aurora2w.sensors import AttitudeTimeline, camera_roll, slerp


@pytest.mark.parametrize("gap", [0., -1., math.inf, math.nan])
def test_invalid_sensor_gap_policy_is_rejected(gap):
    with pytest.raises(ValueError):
        AttitudeTimeline([0., .1], [[1, 0, 0, 0]] * 2, max_gap_s=gap)


def test_10hz_source_requires_explicit_interpolation_policy():
    times = [0., .1, .2, .3, .8]
    quaternions = [[1, 0, 0, 0]] * len(times)
    strict = AttitudeTimeline(times, quaternions)
    intentional = AttitudeTimeline(times, quaternions, max_gap_s=.1)
    assert strict.at(1 / 30) is None
    np.testing.assert_allclose(intentional.at(1 / 30), [1, 0, 0, 0])
    np.testing.assert_allclose(intentional.at(.25), [1, 0, 0, 0])
    assert intentional.at(.5) is None  # A genuine dropped-sample gap stays rejected.
    assert intentional.at(-.001) is None and intentional.at(.801) is None
    np.testing.assert_allclose(intentional.at(.8), [1, 0, 0, 0])


@pytest.mark.parametrize("fraction", [-.01, 1.01, math.nan])
def test_slerp_does_not_extrapolate(fraction):
    with pytest.raises(ValueError):
        slerp([1, 0, 0, 0], [1, 0, 0, 0], fraction)


def test_camera_roll_sign_in_optical_axes_and_degenerate_horizon():
    # This extrinsic gives +y optical aligned with world gravity when upright.
    extrinsic = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.]])
    angle = .3
    # Device's y axis is optical z. Positive camera roll rotates the scene -angle.
    q = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]
    roll, confidence = camera_roll(q, extrinsic)
    assert confidence == 1
    assert roll == pytest.approx(-angle)
    assert camera_roll([1, 0, 0, 0], np.eye(3)) == (0., 0.)
