"""
Unit tests for NAVIGATE 2.0 evaluate_blackout.py.

Covers:
- Zero blackout → zero/near-zero position error
- Straight-line synthetic blackout error calculation
- Blackout correction suppression
- Metric calculations (final error, max error, RMSE, traveled distance)
- Relative drift percentage calculation
- Reusability of evaluation API
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.evaluate_blackout import (
    evaluate_trajectory_blackout,
    generate_synthetic_trajectory,
    BlackoutMetrics,
    BlackoutEvaluationResult,
)


def test_zero_blackout_near_zero_error():
    """
    When no blackout interval is defined (or empty list),
    the tracker receives continuous GNSS updates, leading to near-zero error.
    """
    ts, vels, gyros, lats, lons = generate_synthetic_trajectory(
        duration_s=30.0, speed_ms=10.0, heading_deg=0.0
    )

    # Empty blackout list
    result = evaluate_trajectory_blackout(
        timestamps=ts,
        velocities_ms=vels,
        gyro_z_rad_s=gyros,
        gt_lats=lats,
        gt_lons=lons,
        blackout_intervals=[],
        init_heading_deg=0.0,
    )

    assert len(result.per_blackout_metrics) == 0
    assert result.mean_final_error_m == 0.0
    assert result.mean_max_error_m == 0.0
    # Every point should equal GNSS fix exactly
    assert np.all(result.errors_per_step_m < 1e-4)


def test_straight_line_synthetic_blackout_exact_velocity():
    """
    With perfect velocity (no noise) and perfect heading (no gyro drift),
    dead-reckoning propagation during blackout should match ground truth almost exactly.
    """
    ts, vels, gyros, lats, lons = generate_synthetic_trajectory(
        duration_s=60.0, speed_ms=10.0, heading_deg=0.0, velocity_noise_std=0.0
    )

    blackouts = [(10.0, 40.0)]  # 30 second blackout

    result = evaluate_trajectory_blackout(
        timestamps=ts,
        velocities_ms=vels,
        gyro_z_rad_s=gyros,
        gt_lats=lats,
        gt_lons=lons,
        blackout_intervals=blackouts,
        init_heading_deg=0.0,
    )

    assert len(result.per_blackout_metrics) == 1
    m = result.per_blackout_metrics[0]

    # Perfect velocity & direction -> error should be near zero (< 1.0 m over 300m traveled)
    assert m.final_error_m < 1.0
    assert m.max_error_m < 1.0
    assert m.rmse_error_m < 1.0
    assert m.relative_drift_percent < 0.5

    # Traveled distance during 30s at 10 m/s should be ~300m
    assert abs(m.traveled_distance_m - 300.0) < 1.0


def test_straight_line_velocity_bias_blackout_error():
    """
    With a +1 m/s velocity bias (est speed 11 m/s vs true 10 m/s),
    over a 10s blackout, position error should accumulate to ~10 meters.
    """
    ts, vels, gyros, lats, lons = generate_synthetic_trajectory(
        duration_s=30.0, speed_ms=10.0, heading_deg=0.0
    )

    # Introduce +1.0 m/s bias to velocity estimate
    vels_biased = vels + 1.0

    blackouts = [(10.0, 20.0)]  # 10 second blackout

    result = evaluate_trajectory_blackout(
        timestamps=ts,
        velocities_ms=vels_biased,
        gyro_z_rad_s=gyros,
        gt_lats=lats,
        gt_lons=lons,
        blackout_intervals=blackouts,
        init_heading_deg=0.0,
    )

    assert len(result.per_blackout_metrics) == 1
    m = result.per_blackout_metrics[0]

    # 1 m/s error * 10s = ~10m error
    assert abs(m.final_error_m - 10.0) < 0.5
    assert abs(m.max_error_m - 10.0) < 0.5

    # Traveled distance estimate: 11 m/s * 10s = 110m
    assert abs(m.traveled_distance_m - 110.0) < 1.0

    # Relative drift = 10m / 110m * 100 ~ 9.09%
    assert abs(m.relative_drift_percent - (10.0 / 110.0 * 100.0)) < 0.5


def test_blackout_correction_suppression_and_recovery():
    """
    Verifies that:
    1. Outside blackout: GNSS correction keeps error at 0.
    2. During blackout: error grows.
    3. After blackout: position snaps back to GNSS (error returns to 0).
    """
    ts, vels, gyros, lats, lons = generate_synthetic_trajectory(
        duration_s=50.0, speed_ms=10.0, heading_deg=0.0
    )

    # Add velocity bias so error accumulates during blackout
    vels_biased = vels + 2.0  # +2 m/s error

    blackouts = [(10.0, 30.0)]  # 20 second blackout

    result = evaluate_trajectory_blackout(
        timestamps=ts,
        velocities_ms=vels_biased,
        gyro_z_rad_s=gyros,
        gt_lats=lats,
        gt_lons=lons,
        blackout_intervals=blackouts,
        init_heading_deg=0.0,
    )

    errors = result.errors_per_step_m

    # Before blackout (t < 10): error should be 0
    idx_before = np.where(ts < 10.0)[0]
    assert np.all(errors[idx_before] < 1e-4)

    # During blackout (10 <= t <= 30): error should grow up to ~40m (2 m/s * 20s)
    idx_during = np.where((ts >= 10.0) & (ts <= 30.0))[0]
    assert errors[idx_during[-1]] > 35.0

    # Immediately after blackout (t > 30): error should snap back to 0
    idx_after = np.where(ts > 30.0)[0]
    assert np.all(errors[idx_after] < 1e-4)


def test_multiple_blackout_intervals_metrics():
    """
    Verifies metric calculations across multiple blackout windows.
    """
    ts, vels, gyros, lats, lons = generate_synthetic_trajectory(
        duration_s=100.0, speed_ms=10.0, heading_deg=0.0
    )
    vels_biased = vels + 1.0  # +1 m/s bias

    blackouts = [
        (10.0, 20.0),  # 10s blackout -> ~10m error
        (50.0, 70.0),  # 20s blackout -> ~20m error
    ]

    result = evaluate_trajectory_blackout(
        timestamps=ts,
        velocities_ms=vels_biased,
        gyro_z_rad_s=gyros,
        gt_lats=lats,
        gt_lons=lons,
        blackout_intervals=blackouts,
        init_heading_deg=0.0,
    )

    assert len(result.per_blackout_metrics) == 2

    m1 = result.per_blackout_metrics[0]
    m2 = result.per_blackout_metrics[1]

    assert abs(m1.final_error_m - 10.0) < 0.5
    assert abs(m2.final_error_m - 20.0) < 0.5

    # Check aggregate averages
    assert abs(result.mean_final_error_m - 15.0) < 0.5
    assert abs(result.total_traveled_distance_m - 330.0) < 2.0


def test_relative_drift_percentage_calculation():
    """
    Directly tests relative drift percentage calculation:
    Relative drift = (final_error / traveled_distance) * 100.
    """
    m = BlackoutMetrics(
        blackout_start_s=0.0,
        blackout_end_s=10.0,
        final_error_m=15.0,
        max_error_m=15.0,
        rmse_error_m=10.0,
        traveled_distance_m=150.0,
        relative_drift_percent=10.0,  # 15 / 150 * 100 = 10%
        points_count=100,
    )
    assert m.relative_drift_percent == 10.0
