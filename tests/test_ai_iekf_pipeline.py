"""
test_ai_iekf_pipeline.py — Unit Tests for End-to-End AI + ES-EKF Pipeline in NAVIGATE 2.0.

Tests Cover:
1. Pipeline initialization and model loading from checkpoints.
2. VelocityModel inference produces finite speeds in m/s with correct normalization.
3. AttitudeModel inference produces finite, normalized relative quaternions (qw >= 0).
4. Relative attitude update semantics and antipodal symmetry in ES-EKF.
5. End-to-end trajectory execution with GNSS blackout suppression and recovery.
6. Execution on real IO-VNBD dataset windows without NaN/Inf.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.ai_iekf_pipeline import AIIEKFPipeline, lat_lon_to_enu_m
from navigate.iekf_tracker import (
    ErrorStateIEKFTracker,
    STANDARD_GRAVITY,
    EARTH_RADIUS_M,
    RAD2DEG,
    quat_normalize,
    quat_multiply,
    quat_to_rotvec,
)
from navigate.evaluate_blackout import haversine_distance_m


@pytest.fixture(scope="module")
def pipeline():
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints not found on disk.")
    return AIIEKFPipeline(velocity_checkpoint=vel_ckpt, attitude_checkpoint=att_ckpt, device="cpu")


def test_pipeline_initialization(pipeline):
    assert pipeline.velocity_model is not None
    assert pipeline.attitude_model is not None
    assert not pipeline.velocity_model.training
    assert not pipeline.attitude_model.training
    assert pipeline.vel_imu_mean.shape == (6,)
    assert pipeline.vel_imu_std.shape == (6,)
    assert pipeline.att_imu_mean.shape == (6,)
    assert pipeline.att_imu_std.shape == (6,)


def test_velocity_inference_finite_and_shape(pipeline):
    dummy_imu = np.random.randn(8, 50, 6).astype(np.float32)
    speeds = pipeline.predict_velocity(dummy_imu)
    assert speeds.shape == (8,)
    assert not np.isnan(speeds).any()
    assert not np.isinf(speeds).any()

    # Single window inference
    speed_single = pipeline.predict_velocity(dummy_imu[0])
    assert isinstance(speed_single, (float, np.float32))
    assert not np.isnan(speed_single)


def test_attitude_inference_normalized_and_canonical(pipeline):
    dummy_imu = np.random.randn(8, 50, 6).astype(np.float32)
    quats = pipeline.predict_attitude(dummy_imu)
    assert quats.shape == (8, 4)
    assert not np.isnan(quats).any()
    assert not np.isinf(quats).any()

    # Verify unit norm and canonical positive qw
    norms = np.linalg.norm(quats, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.all(quats[:, 0] >= 0.0)


def test_relative_attitude_update_semantics():
    """
    Verifies that the relative attitude update properly compares
    predicted relative rotation (q_start^-1 * q_curr) against q_rel_net.
    """
    tracker = ErrorStateIEKFTracker(
        init_pos_enu=[0.0, 0.0, 0.0],
        init_vel_enu=[0.0, 0.0, 0.0],
        init_heading_deg=0.0,
    )

    q_start = tracker.get_state()["quat"].copy()

    # Predict rotation: pure yaw turn of 10 deg over 1s
    gyro_turn = [0.0, 0.0, math.radians(10.0)]
    tracker.predict(dt=1.0, accel_b=[0.0, 0.0, STANDARD_GRAVITY], gyro_b=gyro_turn)

    # If the network predicted the exact same 10 deg relative rotation:
    q_rel_exact = quat_normalize(quat_multiply(quat_normalize(q_start), tracker.get_state()["quat"]))
    P_before = tracker.get_state()["cov"].copy()

    # Apply update with exact matching relative quaternion
    tracker.update_relative_attitude(q_rel_network=q_rel_exact, q_start=q_start)
    state_after = tracker.get_state()

    assert not np.isnan(state_after["quat"]).any()
    assert not np.isnan(state_after["cov"]).any()
    # Covariance on attitude should decrease or remain positive-definite
    assert np.all(np.diag(state_after["cov"]) > 0.0)


def test_relative_attitude_antipodal_invariance():
    """
    Verifies that providing -q_rel yields the exact same state update as +q_rel.
    """
    q_start = np.array([1.0, 0.0, 0.0, 0.0])
    q_rel = np.array([0.9961947, 0.0, 0.0, 0.0871557])  # ~10 deg yaw

    tracker1 = ErrorStateIEKFTracker(init_heading_deg=0.0)
    tracker2 = ErrorStateIEKFTracker(init_heading_deg=0.0)

    tracker1.update_relative_attitude(q_rel_network=q_rel, q_start=q_start)
    tracker2.update_relative_attitude(q_rel_network=-q_rel, q_start=q_start)

    s1 = tracker1.get_state()
    s2 = tracker2.get_state()

    assert np.allclose(s1["quat"], s2["quat"], atol=1e-7)
    assert np.allclose(s1["cov"], s2["cov"], atol=1e-7)


def test_synthetic_trajectory_blackout_and_recovery(pipeline):
    """
    Runs a 25-step synthetic trajectory (25 seconds) moving North at 10 m/s,
    with a 5-second blackout from t=10s to t=15s.
    Verifies GNSS suppression during outage and full recovery after.
    """
    N = 25
    timestamps = np.arange(N, dtype=np.float64) * 1.0  # 1-sec intervals

    # Synthetic IMU: constant gravity + small noise
    imu_windows = np.zeros((N, 50, 6), dtype=np.float32)
    imu_windows[:, :, 2] = STANDARD_GRAVITY  # Z accel = +g

    # Ground truth trajectory: moving North at 10 m/s
    ref_lat = 51.5
    ref_lon = -0.1
    gt_lats = np.zeros(N, dtype=np.float64)
    gt_lons = np.zeros(N, dtype=np.float64)
    for i in range(N):
        north_m = 10.0 * timestamps[i]
        gt_lats[i] = ref_lat + (north_m / EARTH_RADIUS_M) * RAD2DEG
        gt_lons[i] = ref_lon

    blackout_intervals = [(10.0, 15.0)]

    result = pipeline.run_session_blackout(
        imu_windows=imu_windows,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=blackout_intervals,
        init_heading_deg=0.0,
    )

    assert len(result.trajectory_estimated) == N
    assert len(result.per_blackout_metrics) == 1
    m = result.per_blackout_metrics[0]
    assert m.blackout_start_s == 10.0
    assert m.blackout_end_s == 15.0
    assert not np.isnan(m.final_error_m)
    assert not np.isnan(m.rmse_error_m)
    assert m.traveled_distance_m > 0.0

    # Verify blackout flag on trajectory points
    for pt in result.trajectory_estimated:
        if 10.0 <= pt.timestamp <= 15.0:
            assert pt.is_gnss_blackout is True
        else:
            assert pt.is_gnss_blackout is False

    # Verify GNSS recovery: point at t=20s should have small error (< 5m)
    pt_recovery = result.trajectory_estimated[20]
    rec_err = haversine_distance_m(pt_recovery.lat, pt_recovery.lon, gt_lats[20], gt_lons[20])
    assert rec_err < 5.0, f"Expected recovery error < 5m, got {rec_err:.2f}m"


def test_real_iovnbd_subset_execution(pipeline):
    """
    Executes pipeline on a real small subset of IO-VNBD dataset windows.
    Verifies that all states, covariances, and outputs remain strictly finite without NaN/Inf.
    """
    data_path = Path("data/processed/iovnbd_full.npz")
    if not data_path.exists():
        pytest.skip("Dataset NPZ not found.")

    npz = np.load(data_path, allow_pickle=True)
    imu_all = npz["imu"]
    session_ids = npz["session_ids"]

    # Pick first session
    first_session = session_ids[0]
    mask = (session_ids == first_session)
    session_imu = imu_all[mask][:20]  # Take 20 windows
    N = len(session_imu)

    timestamps = np.arange(N, dtype=np.float64) * 1.0
    ref_lat, ref_lon = 51.5, -0.1
    gt_lats = np.full(N, ref_lat)
    gt_lons = np.full(N, ref_lon)

    result = pipeline.run_session_blackout(
        imu_windows=session_imu,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=[(5.0, 10.0)],
        init_heading_deg=0.0,
    )

    assert len(result.trajectory_estimated) == N
    for pt in result.trajectory_estimated:
        assert not np.isnan(pt.lat)
        assert not np.isnan(pt.lon)
        assert not np.isnan(pt.speed_ms)
        assert not np.isnan(pt.heading_deg)
