"""
test_es_ekf_cross_check.py — Cross-check deterministic ES-EKF propagation & updates against Python reference.
"""

import math
import numpy as np
import pytest

from navigate.iekf_tracker import (
    ErrorStateIEKFTracker,
    quat_normalize,
    quat_multiply,
    quat_to_heading_deg,
    heading_deg_to_quat,
    STANDARD_GRAVITY,
)

def test_python_es_ekf_stationary_propagation():
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=0.0,
        init_lat=12.9716,
        init_lon=77.5946,
        init_alt=900.0,
        init_timestamp=0.0,
    )

    stationary_accel = [0.0, 0.0, STANDARD_GRAVITY]
    stationary_gyro = [0.0, 0.0, 0.0]

    for _ in range(10):
        tracker.predict(dt=0.1, accel_b=stationary_accel, gyro_b=stationary_gyro)

    st = tracker.get_state()
    pos = st["pos_enu"]
    vel = st["vel_enu"]

    np.testing.assert_allclose(pos, [0.0, 0.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], atol=1e-2)

def test_python_es_ekf_velocity_update():
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=90.0, # Facing East
        init_lat=12.9716,
        init_lon=77.5946,
        init_alt=900.0,
        init_timestamp=0.0,
    )

    tracker.update_velocity(forward_speed_ms=5.0)
    st = tracker.get_state()
    vel = st["vel_enu"]

    assert vel[0] > 0.0 # East velocity should increase

def test_python_es_ekf_gnss_update():
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=0.0,
        init_lat=12.9716,
        init_lon=77.5946,
        init_alt=900.0,
        init_timestamp=0.0,
    )

    applied = tracker.update_gnss_position(pos_enu_meas=[10.0, 20.0, 0.0])
    assert applied is True

    st = tracker.get_state()
    pos = st["pos_enu"]
    assert pos[0] > 0.0
    assert pos[1] > 0.0
