"""
Unit tests for NAVIGATE 2.0 dead_reckoning.py.

Covers:
- Straight-line motion (heading North, East)
- Stationary motion (velocity=0)
- Basic heading change via gyro yaw rate
- GNSS blackout simulation
- Position correction (GNSS update)
- Heading wrapping
- Haversine distance utility
- Batch propagation
- Trajectory API
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.dead_reckoning import (
    DeadReckoningTracker,
    GNSSBlackoutSchedule,
    HeadingUpdater,
    TrajectoryPoint,
    haversine_distance_m,
    trajectory_to_arrays,
    EARTH_RADIUS_M,
    DEG2RAD,
    RAD2DEG,
)


# ================================================================== #
#  Fixtures
# ================================================================== #

GREENWICH_LAT = 51.4778   # degrees
GREENWICH_LON = -0.0014   # degrees


def make_tracker(
    heading_deg: float = 0.0,
    blackout_schedule=None,
) -> DeadReckoningTracker:
    return DeadReckoningTracker(
        init_lat=GREENWICH_LAT,
        init_lon=GREENWICH_LON,
        init_heading_deg=heading_deg,
        init_timestamp=0.0,
        sample_rate_hz=10.0,
        blackout_schedule=blackout_schedule,
    )


# ================================================================== #
#  Straight-Line Motion Tests
# ================================================================== #

def test_straight_line_north_displacement():
    """
    Vehicle moving due North at 10 m/s for 10 seconds should travel ~100 m North.
    Longitude should not change.
    """
    tracker = make_tracker(heading_deg=0.0)  # North
    dt = 0.1  # 10 Hz
    v = 10.0  # m/s
    n_steps = 100  # 10 seconds

    for i in range(1, n_steps + 1):
        tracker.update(timestamp=i * dt, velocity_ms=v, gyro_z_rad_s=0.0)

    final = tracker.get_trajectory()[-1]
    d = haversine_distance_m(GREENWICH_LAT, GREENWICH_LON, final.lat, final.lon)

    # Should have moved ~100 m total
    assert abs(d - 100.0) < 1.0, f"Expected ~100 m displacement, got {d:.2f} m"
    # Latitude should have increased (moved North)
    assert final.lat > GREENWICH_LAT
    # Longitude should be approximately unchanged
    assert abs(final.lon - GREENWICH_LON) < 1e-5, f"Lon should not change heading North, got delta={final.lon - GREENWICH_LON}"


def test_straight_line_east_displacement():
    """
    Vehicle moving due East (heading 90°) at 10 m/s for 10 seconds should travel ~100 m East.
    Latitude should not change.
    """
    tracker = make_tracker(heading_deg=90.0)  # East
    dt = 0.1
    v = 10.0
    n_steps = 100

    for i in range(1, n_steps + 1):
        tracker.update(timestamp=i * dt, velocity_ms=v, gyro_z_rad_s=0.0)

    final = tracker.get_trajectory()[-1]
    d = haversine_distance_m(GREENWICH_LAT, GREENWICH_LON, final.lat, final.lon)

    assert abs(d - 100.0) < 1.0, f"Expected ~100 m displacement, got {d:.2f} m"
    # Longitude should have increased (moved East)
    assert final.lon > GREENWICH_LON
    # Latitude should be approximately unchanged
    assert abs(final.lat - GREENWICH_LAT) < 1e-5


def test_straight_line_displacement_proportional_to_velocity():
    """
    Displacement should scale proportionally with velocity.
    Doubling speed for the same duration should double displacement.
    """
    dt = 0.1
    n_steps = 50

    tracker_slow = make_tracker(heading_deg=0.0)
    for i in range(1, n_steps + 1):
        tracker_slow.update(timestamp=i * dt, velocity_ms=5.0, gyro_z_rad_s=0.0)

    tracker_fast = make_tracker(heading_deg=0.0)
    for i in range(1, n_steps + 1):
        tracker_fast.update(timestamp=i * dt, velocity_ms=10.0, gyro_z_rad_s=0.0)

    slow_final = tracker_slow.get_trajectory()[-1]
    fast_final = tracker_fast.get_trajectory()[-1]

    d_slow = haversine_distance_m(GREENWICH_LAT, GREENWICH_LON, slow_final.lat, slow_final.lon)
    d_fast = haversine_distance_m(GREENWICH_LAT, GREENWICH_LON, fast_final.lat, fast_final.lon)

    assert abs(d_fast / d_slow - 2.0) < 0.01, f"Expected 2x displacement ratio, got {d_fast/d_slow:.4f}"


# ================================================================== #
#  Stationary Motion Tests
# ================================================================== #

def test_stationary_no_displacement():
    """
    Vehicle with velocity=0 and gyro_z=0 must not move from initial position.
    """
    tracker = make_tracker(heading_deg=45.0)
    for i in range(1, 101):
        tracker.update(timestamp=i * 0.1, velocity_ms=0.0, gyro_z_rad_s=0.0)

    final = tracker.get_trajectory()[-1]
    d = haversine_distance_m(GREENWICH_LAT, GREENWICH_LON, final.lat, final.lon)
    assert d < 1e-6, f"Stationary vehicle must not move, got {d:.8f} m displacement"
    assert abs(final.heading_deg - 45.0) < 1e-9


def test_stationary_heading_unchanged():
    """
    With gyro_z=0 and velocity=0, heading must remain exactly as initialized.
    """
    tracker = make_tracker(heading_deg=270.0)
    for i in range(1, 21):
        tracker.update(timestamp=i * 0.1, velocity_ms=0.0, gyro_z_rad_s=0.0)
    assert abs(tracker.current_heading_deg - 270.0) < 1e-9


# ================================================================== #
#  Heading Change Tests
# ================================================================== #

def test_gyro_heading_update_90_degrees():
    """
    A yaw rate of π/2 rad/s for 1 second should rotate heading by 90°.
    Starting at heading=0° (North), after 1s of turning right → heading=90° (East).
    """
    tracker = make_tracker(heading_deg=0.0)
    gyro_z = math.pi / 2  # rad/s
    dt = 0.1
    n_steps = 10  # 1 second

    for i in range(1, n_steps + 1):
        tracker.update(timestamp=i * dt, velocity_ms=0.0, gyro_z_rad_s=gyro_z)

    # Should be close to 90°
    assert abs(tracker.current_heading_deg - 90.0) < 0.5, \
        f"Expected ~90°, got {tracker.current_heading_deg:.4f}°"


def test_heading_wraps_correctly():
    """
    Heading must wrap correctly through 360°.
    Starting at 350°, adding 20° should give heading=10° (not 370°).
    """
    tracker = make_tracker(heading_deg=350.0)
    gyro_z = math.radians(20.0) / 1.0  # 20 deg/s to complete 20° in 1 second
    dt = 0.1
    for i in range(1, 11):  # 1 second
        tracker.update(timestamp=i * dt, velocity_ms=0.0, gyro_z_rad_s=gyro_z)

    heading = tracker.current_heading_deg
    assert 0.0 <= heading < 360.0, f"Heading must be in [0, 360), got {heading:.4f}"
    assert abs(heading - 10.0) < 0.5, f"Expected ~10°, got {heading:.4f}°"


def test_u_turn_returns_to_south():
    """
    A 180° turn (π rad/s for 1 second) from North should result in South heading.
    """
    tracker = make_tracker(heading_deg=0.0)  # North
    gyro_z = math.pi  # rad/s → 180° per second
    dt = 0.1
    for i in range(1, 11):
        tracker.update(timestamp=i * dt, velocity_ms=0.0, gyro_z_rad_s=gyro_z)

    # 180° = South. Allow small floating-point error
    assert abs(tracker.current_heading_deg - 180.0) < 0.5 or abs(tracker.current_heading_deg - 180.0) > 359.5


# ================================================================== #
#  GNSS Correction Tests
# ================================================================== #

def test_gnss_position_correction():
    """
    When a GNSS fix is provided outside a blackout, the tracker state
    should snap to the provided GNSS lat/lon.
    """
    tracker = make_tracker(heading_deg=0.0)
    target_lat = 51.5000
    target_lon = 0.0100

    tracker.update(
        timestamp=1.0, velocity_ms=5.0, gyro_z_rad_s=0.0,
        gnss_lat=target_lat, gnss_lon=target_lon
    )

    assert abs(tracker.current_lat_deg - target_lat) < 1e-8
    assert abs(tracker.current_lon_deg - target_lon) < 1e-8


# ================================================================== #
#  GNSS Blackout Tests
# ================================================================== #

def test_gnss_blackout_suppresses_position_correction():
    """
    During a GNSS blackout, a provided GNSS fix must NOT update tracker position.
    """
    blackout = GNSSBlackoutSchedule(intervals=[(0.5, 2.0)])
    tracker = DeadReckoningTracker(
        init_lat=GREENWICH_LAT, init_lon=GREENWICH_LON,
        init_heading_deg=0.0, init_timestamp=0.0,
        blackout_schedule=blackout
    )

    # Update inside blackout window at t=1.0 with GNSS fix far from start
    tracker.update(
        timestamp=1.0, velocity_ms=0.0, gyro_z_rad_s=0.0,
        gnss_lat=52.0, gnss_lon=1.0  # Far away GNSS fix
    )

    # Position must NOT have jumped to GNSS fix
    assert abs(tracker.current_lat_deg - GREENWICH_LAT) < 0.001
    assert abs(tracker.current_lon_deg - GREENWICH_LON) < 0.001


def test_gnss_blackout_is_active_flag():
    """
    TrajectoryPoint.is_gnss_blackout must be True inside and False outside blackout.
    """
    blackout = GNSSBlackoutSchedule(intervals=[(1.0, 3.0)])
    tracker = DeadReckoningTracker(
        init_lat=GREENWICH_LAT, init_lon=GREENWICH_LON,
        init_timestamp=0.0, blackout_schedule=blackout
    )

    pt_before = tracker.update(timestamp=0.5, velocity_ms=0.0, gyro_z_rad_s=0.0)
    pt_during = tracker.update(timestamp=1.5, velocity_ms=0.0, gyro_z_rad_s=0.0)
    pt_after  = tracker.update(timestamp=3.5, velocity_ms=0.0, gyro_z_rad_s=0.0)

    assert pt_before.is_gnss_blackout is False
    assert pt_during.is_gnss_blackout is True
    assert pt_after.is_gnss_blackout is False


def test_gnss_blackout_schedule_invalid_interval():
    """GNSSBlackoutSchedule.add_interval must raise ValueError if end < start."""
    schedule = GNSSBlackoutSchedule()
    with pytest.raises(ValueError):
        schedule.add_interval(5.0, 2.0)


# ================================================================== #
#  Batch Propagation Tests
# ================================================================== #

def test_run_batch_matches_step_by_step():
    """
    run_batch() output must match step-by-step update() output.
    """
    timestamps = [i * 0.1 for i in range(1, 11)]
    velocities = [5.0] * 10
    gyros = [0.01] * 10

    # Step-by-step
    tracker_step = make_tracker(heading_deg=30.0)
    for t, v, g in zip(timestamps, velocities, gyros):
        tracker_step.update(t, v, g)

    # Batch
    tracker_batch = make_tracker(heading_deg=30.0)
    tracker_batch.run_batch(timestamps, velocities, gyros)

    step_traj = tracker_step.get_trajectory()[1:]  # skip initial point
    batch_traj = tracker_batch.get_trajectory()[1:]

    for ps, pb in zip(step_traj, batch_traj):
        assert abs(ps.lat - pb.lat) < 1e-12
        assert abs(ps.lon - pb.lon) < 1e-12
        assert abs(ps.heading_deg - pb.heading_deg) < 1e-12


# ================================================================== #
#  Trajectory API Tests
# ================================================================== #

def test_trajectory_point_count():
    """Trajectory must contain init_point + N update points."""
    tracker = make_tracker()
    n_steps = 20
    for i in range(1, n_steps + 1):
        tracker.update(timestamp=i * 0.1, velocity_ms=3.0, gyro_z_rad_s=0.0)

    traj = tracker.get_trajectory()
    assert len(traj) == n_steps + 1  # +1 for the initial point


def test_trajectory_to_arrays():
    """trajectory_to_arrays must return correct shapes and dtypes."""
    tracker = make_tracker()
    n_steps = 15
    for i in range(1, n_steps + 1):
        tracker.update(timestamp=i * 0.1, velocity_ms=5.0, gyro_z_rad_s=0.05)

    traj = tracker.get_trajectory()
    ts, lats, lons, vels, hdgs = trajectory_to_arrays(traj)

    assert ts.shape == (n_steps + 1,)
    assert lats.shape == (n_steps + 1,)
    assert lons.shape == (n_steps + 1,)
    assert vels.shape == (n_steps + 1,)
    assert hdgs.shape == (n_steps + 1,)
    assert ts.dtype == np.float64


def test_reset_clears_trajectory():
    """After reset(), trajectory must contain only the new initial point."""
    tracker = make_tracker()
    for i in range(1, 11):
        tracker.update(timestamp=i * 0.1, velocity_ms=5.0, gyro_z_rad_s=0.0)

    tracker.reset(init_lat=52.0, init_lon=0.0, init_heading_deg=45.0)
    traj = tracker.get_trajectory()

    assert len(traj) == 1
    assert abs(traj[0].lat - 52.0) < 1e-8
    assert abs(traj[0].heading_deg - 45.0) < 1e-8


# ================================================================== #
#  Haversine Distance Utility Tests
# ================================================================== #

def test_haversine_same_point_is_zero():
    assert haversine_distance_m(51.0, 0.0, 51.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_distance():
    """
    Distance from Greenwich to 0.01° North should be ~1111 m.
    (1° of latitude ≈ 111.32 km → 0.01° ≈ 1113 m at equator,
    slightly different at 51.4778° due to curvature.)
    """
    d = haversine_distance_m(51.4778, 0.0, 51.4878, 0.0)
    assert 1100 < d < 1120, f"Expected ~1111 m, got {d:.2f} m"


def test_haversine_symmetric():
    """Distance A→B must equal B→A."""
    d1 = haversine_distance_m(51.0, 0.0, 52.0, 1.0)
    d2 = haversine_distance_m(52.0, 1.0, 51.0, 0.0)
    assert abs(d1 - d2) < 1e-6
