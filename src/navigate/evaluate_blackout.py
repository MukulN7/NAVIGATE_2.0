"""
evaluate_blackout.py — GNSS Blackout Evaluation Module for NAVIGATE 2.0.

Evaluates dead-reckoning trajectory performance during GNSS outages/blackouts.

Key Features:
- Accepts ground-truth trajectory and estimated inputs (velocity from model/sensor, gyro yaw rate).
- Simulates GNSS blackout intervals where GNSS position corrections are suppressed.
- Calculates standard localization metrics during blackouts:
    - Final position error (m)
    - Max position error (m)
    - RMSE position error (m)
    - Traveled distance during blackout (m)
    - Relative drift percentage (%) = (final_error / traveled_distance) * 100
- Works with IO-VNBD dataset, model outputs, or synthetic benchmark trajectories.
- Designed as a modular component for drop-in replacement with future IEKF/AVNet modules.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np

from navigate.dead_reckoning import (
    DeadReckoningTracker,
    GNSSBlackoutSchedule,
    HeadingUpdater,
    TrajectoryPoint,
    haversine_distance_m,
    EARTH_RADIUS_M,
    DEG2RAD,
    RAD2DEG,
)


@dataclass
class BlackoutMetrics:
    """Metrics calculated for a single blackout window."""
    blackout_start_s: float
    blackout_end_s: float
    final_error_m: float
    max_error_m: float
    rmse_error_m: float
    traveled_distance_m: float
    relative_drift_percent: float
    points_count: int


@dataclass
class BlackoutEvaluationResult:
    """Overall evaluation result across all blackout windows in a recording."""
    per_blackout_metrics: List[BlackoutMetrics]
    mean_final_error_m: float
    mean_max_error_m: float
    mean_rmse_error_m: float
    mean_relative_drift_percent: float
    total_traveled_distance_m: float
    errors_per_step_m: np.ndarray
    trajectory_estimated: List[TrajectoryPoint]


def evaluate_trajectory_blackout(
    timestamps: np.ndarray,
    velocities_ms: np.ndarray,
    gyro_z_rad_s: np.ndarray,
    gt_lats: np.ndarray,
    gt_lons: np.ndarray,
    blackout_intervals: List[Tuple[float, float]],
    init_heading_deg: float = 0.0,
    gt_headings_deg: Optional[np.ndarray] = None,
    heading_updater: Optional[HeadingUpdater] = None,
) -> BlackoutEvaluationResult:
    """
    Evaluates dead-reckoning trajectory performance over specified GNSS blackout intervals.

    Parameters
    ----------
    timestamps          : np.ndarray [N]  Time steps in seconds.
    velocities_ms       : np.ndarray [N]  Estimated or predicted forward velocity (m/s).
    gyro_z_rad_s        : np.ndarray [N]  Gyroscope yaw rate (rad/s).
    gt_lats             : np.ndarray [N]  Ground truth latitude (degrees).
    gt_lons             : np.ndarray [N]  Ground truth longitude (degrees).
    blackout_intervals  : List[(start_s, end_s)]  Blackout time windows.
    init_heading_deg    : float  Initial heading (degrees clockwise from North).
    gt_headings_deg     : np.ndarray [N], optional  Ground truth headings for correction.
    heading_updater     : HeadingUpdater, optional  Custom heading propagation strategy.

    Returns
    -------
    BlackoutEvaluationResult containing detailed per-blackout and aggregate metrics.
    """
    N = len(timestamps)
    if N == 0:
        raise ValueError("Cannot evaluate empty trajectory arrays.")

    # Setup blackout schedule
    blackout_schedule = GNSSBlackoutSchedule(intervals=blackout_intervals)

    # Initialize tracker at ground-truth start
    tracker = DeadReckoningTracker(
        init_lat=float(gt_lats[0]),
        init_lon=float(gt_lons[0]),
        init_heading_deg=init_heading_deg,
        init_timestamp=float(timestamps[0]),
        blackout_schedule=blackout_schedule,
        heading_updater=heading_updater,
    )

    # Run step-by-step propagation
    for i in range(1, N):
        t = float(timestamps[i])
        v = float(velocities_ms[i])
        gz = float(gyro_z_rad_s[i])
        gt_lat = float(gt_lats[i])
        gt_lon = float(gt_lons[i])
        gt_hdg = float(gt_headings_deg[i]) if gt_headings_deg is not None else None

        tracker.update(
            timestamp=t,
            velocity_ms=v,
            gyro_z_rad_s=gz,
            gnss_lat=gt_lat,
            gnss_lon=gt_lon,
            gnss_heading_deg=gt_hdg,
        )

    estimated_trajectory = tracker.get_trajectory()

    # Calculate step-by-step position error (in meters)
    errors_per_step = np.zeros(N, dtype=np.float64)
    for i in range(N):
        est_pt = estimated_trajectory[i]
        errors_per_step[i] = haversine_distance_m(
            est_pt.lat, est_pt.lon, float(gt_lats[i]), float(gt_lons[i])
        )

    # Process metrics per blackout interval
    per_blackout_metrics: List[BlackoutMetrics] = []

    for start_s, end_s in blackout_intervals:
        mask = (timestamps >= start_s) & (timestamps <= end_s)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        errors_in_blackout = errors_per_step[indices]
        final_err = float(errors_in_blackout[-1])
        max_err = float(errors_in_blackout.max())
        rmse_err = float(np.sqrt(np.mean(errors_in_blackout ** 2)))

        # Traveled distance during blackout (using ground-truth distance or integrated velocity)
        if len(indices) > 1:
            dt_sub = np.diff(timestamps[indices])
            v_sub = velocities_ms[indices[:-1]]
            dist_m = float(np.sum(v_sub * dt_sub))
        else:
            dist_m = 0.0

        if dist_m > 1e-3:
            relative_drift = float((final_err / dist_m) * 100.0)
        else:
            relative_drift = 0.0

        per_blackout_metrics.append(
            BlackoutMetrics(
                blackout_start_s=start_s,
                blackout_end_s=end_s,
                final_error_m=final_err,
                max_error_m=max_err,
                rmse_error_m=rmse_err,
                traveled_distance_m=dist_m,
                relative_drift_percent=relative_drift,
                points_count=len(indices),
            )
        )

    # Aggregate summaries
    if per_blackout_metrics:
        mean_final = float(np.mean([m.final_error_m for m in per_blackout_metrics]))
        mean_max = float(np.mean([m.max_error_m for m in per_blackout_metrics]))
        mean_rmse = float(np.mean([m.rmse_error_m for m in per_blackout_metrics]))
        mean_drift = float(np.mean([m.relative_drift_percent for m in per_blackout_metrics]))
        tot_dist = float(np.sum([m.traveled_distance_m for m in per_blackout_metrics]))
    else:
        mean_final = 0.0
        mean_max = 0.0
        mean_rmse = 0.0
        mean_drift = 0.0
        tot_dist = 0.0

    return BlackoutEvaluationResult(
        per_blackout_metrics=per_blackout_metrics,
        mean_final_error_m=mean_final,
        mean_max_error_m=mean_max,
        mean_rmse_error_m=mean_rmse,
        mean_relative_drift_percent=mean_drift,
        total_traveled_distance_m=tot_dist,
        errors_per_step_m=errors_per_step,
        trajectory_estimated=estimated_trajectory,
    )


def generate_synthetic_trajectory(
    duration_s: float = 60.0,
    sample_rate_hz: float = 10.0,
    speed_ms: float = 10.0,
    init_lat: float = 51.4778,
    init_lon: float = -0.0014,
    heading_deg: float = 0.0,
    velocity_noise_std: float = 0.0,
    gyro_noise_std: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a clean synthetic ground-truth and noisy estimation dataset for testing.

    Parameters
    ----------
    duration_s         : Total duration in seconds.
    sample_rate_hz     : Sampling frequency (default 10 Hz).
    speed_ms           : Constant ground truth vehicle speed (m/s).
    init_lat           : Start latitude.
    init_lon           : Start longitude.
    heading_deg        : Vehicle motion heading (degrees).
    velocity_noise_std : Standard deviation of noise added to velocity estimates.
    gyro_noise_std     : Standard deviation of noise added to gyro measurements (rad/s).
    seed               : Random seed for reproducibility.

    Returns
    -------
    timestamps, velocities_est, gyro_z, gt_lats, gt_lons
    """
    rng = np.random.RandomState(seed)
    n_samples = int(math.ceil(duration_s * sample_rate_hz))
    dt = 1.0 / sample_rate_hz

    timestamps = np.linspace(0, duration_s, n_samples)
    gyro_z = rng.normal(0.0, gyro_noise_std, size=n_samples) if gyro_noise_std > 0 else np.zeros(n_samples)
    
    velocities_est = np.full(n_samples, speed_ms, dtype=np.float64)
    if velocity_noise_std > 0:
        velocities_est += rng.normal(0.0, velocity_noise_std, size=n_samples)

    # Compute Ground Truth Lat/Lon path
    heading_rad = math.radians(heading_deg)
    d_east_step = speed_ms * dt * math.sin(heading_rad)
    d_north_step = speed_ms * dt * math.cos(heading_rad)

    gt_lats = np.zeros(n_samples, dtype=np.float64)
    gt_lons = np.zeros(n_samples, dtype=np.float64)

    curr_lat_rad = math.radians(init_lat)
    curr_lon_rad = math.radians(init_lon)

    gt_lats[0] = init_lat
    gt_lons[0] = init_lon

    for i in range(1, n_samples):
        curr_lat_rad += d_north_step / EARTH_RADIUS_M
        curr_lon_rad += d_east_step / (EARTH_RADIUS_M * math.cos(curr_lat_rad))
        gt_lats[i] = math.degrees(curr_lat_rad)
        gt_lons[i] = math.degrees(curr_lon_rad)

    return timestamps, velocities_est, gyro_z, gt_lats, gt_lons
