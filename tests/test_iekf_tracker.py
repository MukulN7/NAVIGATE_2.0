"""
test_iekf_tracker.py — Unit Tests for Error-State EKF (IEKF) Tracker in NAVIGATE 2.0.

Tests Cover:
1. Quaternion normalization and canonical sign.
2. Quaternion multiplication, inverse, and identity.
3. Small and large angle rotation vector conversions.
4. Heading to quaternion and quaternion to heading conversion.
5. Stationary IMU gravity compensation (zero net vertical acceleration).
6. State propagation under forward acceleration and gyro rates.
7. Covariance symmetry and numerical stability.
8. Non-Holonomic Constraint (NHC) measurement update.
9. Learned VelocityModel forward speed update.
10. GNSS position update and blackout suppression.
11. Multi-step continuous propagation stability (100 steps without NaN/Inf).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.iekf_tracker import (
    ErrorStateIEKFTracker,
    GNSSBlackoutSchedule,
    STANDARD_GRAVITY,
    quat_normalize,
    quat_multiply,
    quat_inverse,
    rotvec_to_quat,
    quat_to_rotvec,
    quat_to_rotmat,
    rotmat_to_quat,
    heading_deg_to_quat,
    quat_to_heading_deg,
    skew_symmetric,
)


# ================================================================== #
#  Quaternion & Math Utilities Tests
# ================================================================== #

def test_quat_normalize():
    # Normal unit quaternion
    q = np.array([2.0, 0.0, 0.0, 0.0])
    qn = quat_normalize(q)
    assert np.isclose(np.linalg.norm(qn), 1.0)
    assert np.isclose(qn[0], 1.0)

    # Negative scalar should be flipped to positive canonical
    q_neg = np.array([-1.0, 0.0, 0.0, 0.0])
    qn_neg = quat_normalize(q_neg)
    assert np.isclose(qn_neg[0], 1.0)

    # Near-zero quaternion fallback
    q_zero = np.zeros(4)
    qn_zero = quat_normalize(q_zero)
    assert np.allclose(qn_zero, [1.0, 0.0, 0.0, 0.0])


def test_quat_multiply_and_inverse():
    q1 = rotvec_to_quat(np.array([0.1, 0.2, 0.3]))
    q_inv = quat_inverse(q1)
    
    # q * q^-1 == [1, 0, 0, 0]
    q_ident = quat_multiply(q1, q_inv)
    assert np.allclose(q_ident, [1.0, 0.0, 0.0, 0.0], atol=1e-7)

    # Identity multiplication
    e = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_multiply(q1, e), q1, atol=1e-7)


def test_rotvec_to_quat_and_back():
    # Small angle
    small_rotvec = np.array([1e-7, -2e-7, 3e-7])
    q_small = rotvec_to_quat(small_rotvec)
    rec_small = quat_to_rotvec(q_small)
    assert np.allclose(rec_small, small_rotvec, atol=1e-9)

    # Large angle (e.g. 90 deg around Z)
    rotvec_90z = np.array([0.0, 0.0, math.pi / 2.0])
    q_90 = rotvec_to_quat(rotvec_90z)
    rec_90 = quat_to_rotvec(q_90)
    assert np.allclose(rec_90, rotvec_90z, atol=1e-7)


def test_rotmat_quat_roundtrip():
    rotvec = np.array([0.2, -0.4, 0.6])
    q = rotvec_to_quat(rotvec)
    R = quat_to_rotmat(q)
    
    # Rotation matrix properties: R * R^T = I, det(R) = 1
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(R), 1.0)

    # Roundtrip back to quaternion
    q_rec = rotmat_to_quat(R)
    assert np.allclose(q, q_rec, atol=1e-7) or np.allclose(q, -q_rec, atol=1e-7)


def test_heading_to_quat_and_back():
    for heading in [0.0, 45.0, 90.0, 180.0, 270.0, 359.5]:
        q = heading_deg_to_quat(heading)
        h_rec = quat_to_heading_deg(q)
        assert np.isclose(h_rec, heading, atol=1e-4), f"Failed for heading {heading} (got {h_rec})"


# ================================================================== #
#  IMU Prediction & Stationary Tests
# ================================================================== #

def test_stationary_gravity_compensation():
    """
    When the vehicle is stationary and level facing North:
    IMU accelerometer measures specific force [0, 0, +g] in body frame.
    Gravity compensation should result in net zero acceleration in ENU frame.
    """
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=0.0,
    )
    
    accel_stationary = [0.0, 0.0, STANDARD_GRAVITY]
    gyro_stationary = [0.0, 0.0, 0.0]

    # Predict for 1 second at 10 Hz
    dt = 0.1
    for _ in range(10):
        tracker.predict(dt=dt, accel_b=accel_stationary, gyro_b=gyro_stationary)

    state = tracker.get_state()
    # Position and velocity should remain practically zero
    assert np.allclose(state["pos_enu"], [0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(state["vel_enu"], [0.0, 0.0, 0.0], atol=1e-5)


def test_prediction_changes_position_and_velocity():
    """
    Under constant forward specific force on a level vehicle facing North (heading 0 deg):
    Forward body axis (X) maps to North (Y) in ENU.
    """
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=0.0,  # North
    )

    accel_fwd = [2.0, 0.0, STANDARD_GRAVITY]  # 2 m/s^2 forward
    gyro_zero = [0.0, 0.0, 0.0]
    dt = 0.5

    tracker.predict(dt=dt, accel_b=accel_fwd, gyro_b=gyro_zero)
    state = tracker.get_state()

    # Expected: v_North = 2.0 * 0.5 = 1.0 m/s, p_North = 0.5 * 2.0 * (0.5^2) = 0.25 m
    assert np.isclose(state["vel_enu"][1], 1.0, atol=1e-5)
    assert np.isclose(state["pos_enu"][1], 0.25, atol=1e-5)
    assert np.isclose(state["vel_enu"][0], 0.0, atol=1e-5)  # East should be 0


def test_covariance_remains_finite_and_symmetric():
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=45.0,
    )

    accel = [1.0, 0.2, STANDARD_GRAVITY]
    gyro = [0.01, -0.02, 0.05]

    for _ in range(20):
        tracker.predict(dt=0.1, accel_b=accel, gyro_b=gyro)

    cov = tracker.get_state()["cov"]
    assert cov.shape == (9, 9)
    assert not np.isnan(cov).any()
    assert not np.isinf(cov).any()
    # Check symmetry: cov == cov^T
    assert np.allclose(cov, cov.T, atol=1e-8)
    # Check positive diagonal variances
    assert np.all(np.diag(cov) > 0.0)


# ================================================================== #
#  Measurement Update Tests (NHC, Velocity, GNSS)
# ================================================================== #

def test_nhc_update_executes_without_nan():
    """
    Initial state has an unrealistic lateral body velocity.
    NHC update should reduce lateral velocity towards 0 without NaN/Inf.
    """
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[5.0, 0.0, 0.0],  # Moving East
        init_heading_deg=0.0,          # Facing North -> East is lateral (body Y is -East)
    )

    tracker.update_nhc(cov_lateral=0.01, cov_vertical=0.01)
    state = tracker.get_state()

    assert not np.isnan(state["vel_enu"]).any()
    # Lateral velocity should have been significantly pulled towards zero
    assert abs(state["vel_enu"][0]) < 5.0


def test_learned_velocity_update_executes():
    """
    Vehicle is facing North. State velocity is 0 m/s.
    VelocityModel provides 10.0 m/s forward speed.
    Velocity update should correct North velocity towards 10 m/s.
    """
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=0.0,  # North
        std_vel_init=1.0,
    )

    tracker.update_velocity(forward_speed_ms=10.0, cov_speed=0.1**2)
    state = tracker.get_state()

    assert not np.isnan(state["vel_enu"]).any()
    # Velocity in North direction should be substantially positive
    assert state["vel_enu"][1] > 5.0
    assert np.isclose(state["vel_enu"][0], 0.0, atol=1e-5)  # East remains 0


def test_gnss_update_and_blackout_behavior():
    schedule = GNSSBlackoutSchedule(intervals=[(10.0, 20.0)])
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        blackout_schedule=schedule,
    )

    # 1. Non-blackout GNSS update at t=5s
    gnss_pos = [10.0, 20.0, 0.0]
    applied = tracker.update_gnss_position(gnss_pos, cov_pos=0.01, is_blackout=False)
    assert applied is True
    state = tracker.get_state()
    assert np.isclose(state["pos_enu"][0], 10.0, atol=0.5)
    assert np.isclose(state["pos_enu"][1], 20.0, atol=0.5)

    # 2. Blackout GNSS update at t=15s (should be rejected/skipped)
    curr_pos = tracker.get_state()["pos_enu"].copy()
    false_gnss = [100.0, 200.0, 0.0]
    applied_blackout = tracker.update_gnss_position(false_gnss, is_blackout=True)
    assert applied_blackout is False
    assert np.allclose(tracker.get_state()["pos_enu"], curr_pos)


def test_filter_stability_100_steps():
    """
    Runs 100 continuous 10 Hz steps through the high-level step() API
    with noisy IMU inputs, learned velocity, NHC, and intermittent GNSS blackout.
    Verifies filter does not diverge or produce NaN/Inf.
    """
    np.random.seed(42)
    schedule = GNSSBlackoutSchedule(intervals=[(3.0, 7.0)])
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 5.0, 0.0],
        init_heading_deg=0.0,
        blackout_schedule=schedule,
    )

    t = 0.0
    dt = 0.1
    for step_i in range(100):
        t += dt
        accel_noisy = [
            np.random.normal(0.0, 0.05),
            np.random.normal(0.0, 0.05),
            STANDARD_GRAVITY + np.random.normal(0.0, 0.05),
        ]
        gyro_noisy = [
            np.random.normal(0.0, 0.005),
            np.random.normal(0.0, 0.005),
            np.random.normal(0.0, 0.005),
        ]
        meas_speed = 5.0 + np.random.normal(0.0, 0.1)
        gnss_pos = [0.0, 5.0 * t, 0.0]

        pt = tracker.step(
            timestamp=t,
            accel_b=accel_noisy,
            gyro_b=gyro_noisy,
            velocity_ms=meas_speed,
            gnss_pos_enu=gnss_pos,
        )

        assert not np.isnan(pt.pos_north_m)
        assert not np.isnan(pt.speed_ms)
        assert not np.isnan(pt.heading_deg)
        if 3.0 <= t <= 7.0:
            assert pt.is_gnss_blackout is True
        else:
            assert pt.is_gnss_blackout is False

    trajectory = tracker.get_trajectory()
    assert len(trajectory) == 101  # Initial point + 100 steps
