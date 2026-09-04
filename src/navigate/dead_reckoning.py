"""
dead_reckoning.py — Standalone Dead-Reckoning Trajectory Module for NAVIGATE 2.0.

Propagates vehicle position at 10 Hz using estimated forward velocity and heading,
without integrating raw accelerometer magnitude.

Design Principles
-----------------
- Uses estimated forward velocity (from VelocityModel V2) not accelerometer double-integration.
- Heading is propagated by integrating gyroscope yaw rate (gyro_z) from the IMU.
- Position is maintained in lat/lon (WGS84) using local ENU (East-North-Up) displacement.
- Clean API designed for later drop-in replacement of simple gyro heading propagation
  with the AVNet-inspired attitude model and IEKF.
- GNSS blackout simulation: GNSS position corrections are suppressed during
  configurable blackout intervals.
- Fully NumPy-based: no torch dependencies in this module.

Coordinate System
-----------------
- Heading is measured clockwise from North, in radians.
  0 = North, π/2 = East, π = South, 3π/2 (=-π/2) = West.
- Displacements are computed in local East-North frame.
- Lat/lon deltas use the equirectangular approximation (accurate to ~0.1% at ≤100 km).

API Surface
-----------
    tracker = DeadReckoningTracker(
        init_lat=51.4778, init_lon=-0.0014, init_heading_deg=0.0,
        init_timestamp=0.0, sample_rate_hz=10.0
    )
    point = tracker.update(
        timestamp=0.1, velocity_ms=5.0, gyro_z_rad_s=0.02,
        gnss_lat=None, gnss_lon=None, gnss_heading_deg=None
    )
    trajectory = tracker.get_trajectory()
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ================================================================== #
#  Constants
# ================================================================== #

# WGS84 mean Earth radius in metres
EARTH_RADIUS_M = 6_371_000.0

# Degrees to radians
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ================================================================== #
#  Data Structures
# ================================================================== #

@dataclass
class TrajectoryPoint:
    """
    A single point in the propagated trajectory.

    Attributes
    ----------
    timestamp   : float   Seconds since start
    lat         : float   Latitude in decimal degrees (WGS84)
    lon         : float   Longitude in decimal degrees (WGS84)
    velocity_ms : float   Forward vehicle speed in m/s
    heading_deg : float   Heading clockwise from North, degrees [0, 360)
    is_gnss_blackout : bool  True if this point was propagated without GNSS correction
    """
    timestamp: float
    lat: float
    lon: float
    velocity_ms: float
    heading_deg: float
    is_gnss_blackout: bool = False


@dataclass
class DeadReckoningState:
    """
    Internal mutable state of the dead-reckoning tracker.

    Attributes
    ----------
    lat          : float  Latitude in radians
    lon          : float  Longitude in radians
    heading_rad  : float  Heading from North, clockwise, in radians
    timestamp    : float  Current time in seconds
    """
    lat: float        # radians
    lon: float        # radians
    heading_rad: float
    timestamp: float


# ================================================================== #
#  GNSS Blackout Interval
# ================================================================== #

class GNSSBlackoutSchedule:
    """
    Defines zero or more time intervals during which GNSS corrections
    are suppressed to simulate GNSS signal unavailability.

    Parameters
    ----------
    intervals : list of (start_s, end_s) tuples, times in seconds.
    """

    def __init__(self, intervals: Optional[Sequence[Tuple[float, float]]] = None) -> None:
        self._intervals: List[Tuple[float, float]] = list(intervals) if intervals else []

    def is_blackout(self, timestamp: float) -> bool:
        """Returns True if timestamp falls within any configured blackout interval."""
        for start, end in self._intervals:
            if start <= timestamp <= end:
                return True
        return False

    def add_interval(self, start_s: float, end_s: float) -> None:
        """Adds a blackout interval [start_s, end_s]."""
        if end_s < start_s:
            raise ValueError(f"Blackout end ({end_s}) must be >= start ({start_s})")
        self._intervals.append((start_s, end_s))


# ================================================================== #
#  Heading Update Interface
# ================================================================== #

class HeadingUpdater:
    """
    Base class for heading propagation strategies.

    The default implementation integrates gyroscope yaw rate (gyro_z).
    Designed for drop-in replacement with:
      - AVNet attitude model output
      - IEKF-fused attitude estimates

    Override `update_heading` in a subclass to swap in a better model.
    """

    def update_heading(
        self,
        current_heading_rad: float,
        gyro_z_rad_s: float,
        dt: float,
        gnss_heading_rad: Optional[float] = None,
    ) -> float:
        """
        Returns updated heading in radians.

        Parameters
        ----------
        current_heading_rad : float  Current heading [rad, clockwise from North]
        gyro_z_rad_s        : float  Yaw rate from IMU [rad/s, positive = turn right]
        dt                  : float  Time step [s]
        gnss_heading_rad    : float or None  GNSS-derived heading [rad] if available

        Returns
        -------
        float  Updated heading in radians, wrapped to [0, 2π)
        """
        if gnss_heading_rad is not None:
            # If we have a reliable GNSS heading, blend it (simple low-pass fusion placeholder)
            # A full IEKF would replace this blending with a proper Kalman update.
            alpha = 0.1  # weight for GNSS (low; gyro integration is dominant at 10 Hz)
            new_heading = (1.0 - alpha) * (current_heading_rad + gyro_z_rad_s * dt) + alpha * gnss_heading_rad
        else:
            new_heading = current_heading_rad + gyro_z_rad_s * dt

        return _wrap_angle_rad(new_heading)


def _wrap_angle_rad(angle: float) -> float:
    """Wraps angle to [0, 2π)."""
    return angle % (2.0 * math.pi)


# ================================================================== #
#  Core Dead-Reckoning Tracker
# ================================================================== #

class DeadReckoningTracker:
    """
    Propagates vehicle position and heading at 10 Hz using:
      - Forward velocity estimates (from VelocityModel V2)
      - IMU gyroscope yaw rate for heading integration
      - Optional GNSS position corrections (suppressed during blackout)

    Parameters
    ----------
    init_lat         : float  Initial latitude in decimal degrees
    init_lon         : float  Initial longitude in decimal degrees
    init_heading_deg : float  Initial heading clockwise from North in degrees
    init_timestamp   : float  Start timestamp in seconds (default 0.0)
    sample_rate_hz   : float  Expected IMU sample rate (default 10.0)
    heading_updater  : HeadingUpdater or None
                       Custom heading strategy; defaults to gyro integration.
    blackout_schedule : GNSSBlackoutSchedule or None
                        Defines GNSS blackout windows.
    """

    def __init__(
        self,
        init_lat: float,
        init_lon: float,
        init_heading_deg: float = 0.0,
        init_timestamp: float = 0.0,
        sample_rate_hz: float = 10.0,
        heading_updater: Optional[HeadingUpdater] = None,
        blackout_schedule: Optional[GNSSBlackoutSchedule] = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self._heading_updater = heading_updater or HeadingUpdater()
        self._blackout = blackout_schedule or GNSSBlackoutSchedule()

        # Internal state in radians
        self._state = DeadReckoningState(
            lat=init_lat * DEG2RAD,
            lon=init_lon * DEG2RAD,
            heading_rad=init_heading_deg * DEG2RAD,
            timestamp=init_timestamp,
        )

        self._trajectory: List[TrajectoryPoint] = []

        # Record the initial position as the first trajectory point
        self._trajectory.append(self._make_point(
            timestamp=init_timestamp,
            velocity_ms=0.0,
            is_blackout=False,
        ))

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def update(
        self,
        timestamp: float,
        velocity_ms: float,
        gyro_z_rad_s: float,
        gnss_lat: Optional[float] = None,
        gnss_lon: Optional[float] = None,
        gnss_heading_deg: Optional[float] = None,
    ) -> TrajectoryPoint:
        """
        Propagates state by one time step.

        Parameters
        ----------
        timestamp       : float  Current time in seconds
        velocity_ms     : float  Estimated forward vehicle speed in m/s
        gyro_z_rad_s    : float  Yaw rate from IMU (rad/s, positive = right turn)
        gnss_lat        : float or None  GNSS latitude in decimal degrees
        gnss_lon        : float or None  GNSS longitude in decimal degrees
        gnss_heading_deg: float or None  GNSS-derived heading in degrees (0=North)

        Returns
        -------
        TrajectoryPoint  Current estimated position
        """
        dt = timestamp - self._state.timestamp
        if dt <= 0:
            dt = 1.0 / self.sample_rate_hz  # fallback if timestamps are not provided

        in_blackout = self._blackout.is_blackout(timestamp)

        # Determine GNSS heading for updater (suppressed during blackout)
        gnss_heading_rad: Optional[float] = None
        if gnss_heading_deg is not None and not in_blackout:
            gnss_heading_rad = gnss_heading_deg * DEG2RAD

        # 1. Update heading
        new_heading_rad = self._heading_updater.update_heading(
            current_heading_rad=self._state.heading_rad,
            gyro_z_rad_s=gyro_z_rad_s,
            dt=dt,
            gnss_heading_rad=gnss_heading_rad,
        )

        # 2. Compute displacement in local ENU frame
        #    Heading is clockwise from North:
        #      East  displacement = v * sin(heading) * dt
        #      North displacement = v * cos(heading) * dt
        d = velocity_ms * dt  # distance travelled [m]
        d_east_m  = d * math.sin(new_heading_rad)
        d_north_m = d * math.cos(new_heading_rad)

        # 3. Update lat/lon using equirectangular approximation
        new_lat = self._state.lat + (d_north_m / EARTH_RADIUS_M)
        new_lon = self._state.lon + (d_east_m / (EARTH_RADIUS_M * math.cos(self._state.lat)))

        # 4. Apply GNSS position correction if available and not in blackout
        if gnss_lat is not None and gnss_lon is not None and not in_blackout:
            new_lat = gnss_lat * DEG2RAD
            new_lon = gnss_lon * DEG2RAD

        # 5. Commit state
        self._state = DeadReckoningState(
            lat=new_lat,
            lon=new_lon,
            heading_rad=new_heading_rad,
            timestamp=timestamp,
        )

        # 6. Record trajectory point
        point = self._make_point(
            timestamp=timestamp,
            velocity_ms=velocity_ms,
            is_blackout=in_blackout,
        )
        self._trajectory.append(point)
        return point

    def run_batch(
        self,
        timestamps: Sequence[float],
        velocities_ms: Sequence[float],
        gyro_z_rad_s: Sequence[float],
        gnss_lats: Optional[Sequence[Optional[float]]] = None,
        gnss_lons: Optional[Sequence[Optional[float]]] = None,
        gnss_headings_deg: Optional[Sequence[Optional[float]]] = None,
    ) -> List[TrajectoryPoint]:
        """
        Propagates state over a batch of timestamped measurements.

        Parameters
        ----------
        timestamps        : sequence of floats [N]  Timestamps in seconds
        velocities_ms     : sequence of floats [N]  Estimated speeds in m/s
        gyro_z_rad_s      : sequence of floats [N]  Yaw rates in rad/s
        gnss_lats         : sequence of float or None [N], optional
        gnss_lons         : sequence of float or None [N], optional
        gnss_headings_deg : sequence of float or None [N], optional

        Returns
        -------
        List[TrajectoryPoint]  All propagated trajectory points for this batch
        """
        N = len(timestamps)
        _gnss_lats = gnss_lats if gnss_lats is not None else [None] * N
        _gnss_lons = gnss_lons if gnss_lons is not None else [None] * N
        _gnss_hdgs = gnss_headings_deg if gnss_headings_deg is not None else [None] * N

        batch_points: List[TrajectoryPoint] = []
        for i in range(N):
            pt = self.update(
                timestamp=timestamps[i],
                velocity_ms=velocities_ms[i],
                gyro_z_rad_s=gyro_z_rad_s[i],
                gnss_lat=_gnss_lats[i],
                gnss_lon=_gnss_lons[i],
                gnss_heading_deg=_gnss_hdgs[i],
            )
            batch_points.append(pt)
        return batch_points

    def get_trajectory(self) -> List[TrajectoryPoint]:
        """Returns a copy of the complete accumulated trajectory."""
        return list(self._trajectory)

    def reset(
        self,
        init_lat: float,
        init_lon: float,
        init_heading_deg: float = 0.0,
        init_timestamp: float = 0.0,
    ) -> None:
        """Resets tracker state to a new initial position."""
        self._state = DeadReckoningState(
            lat=init_lat * DEG2RAD,
            lon=init_lon * DEG2RAD,
            heading_rad=init_heading_deg * DEG2RAD,
            timestamp=init_timestamp,
        )
        self._trajectory = [self._make_point(init_timestamp, 0.0, False)]

    @property
    def current_lat_deg(self) -> float:
        """Current latitude in decimal degrees."""
        return self._state.lat * RAD2DEG

    @property
    def current_lon_deg(self) -> float:
        """Current longitude in decimal degrees."""
        return self._state.lon * RAD2DEG

    @property
    def current_heading_deg(self) -> float:
        """Current heading in degrees, clockwise from North, [0, 360)."""
        return self._state.heading_rad * RAD2DEG

    @property
    def current_timestamp(self) -> float:
        return self._state.timestamp

    # ------------------------------------------------------------------ #
    #  Internal Helpers
    # ------------------------------------------------------------------ #

    def _make_point(
        self,
        timestamp: float,
        velocity_ms: float,
        is_blackout: bool,
    ) -> TrajectoryPoint:
        """Creates a TrajectoryPoint from current state."""
        return TrajectoryPoint(
            timestamp=timestamp,
            lat=self._state.lat * RAD2DEG,
            lon=self._state.lon * RAD2DEG,
            velocity_ms=velocity_ms,
            heading_deg=self._state.heading_rad * RAD2DEG,
            is_gnss_blackout=is_blackout,
        )


# ================================================================== #
#  Utility: Distance Between Two TrajectoryPoints
# ================================================================== #

def haversine_distance_m(lat1_deg: float, lon1_deg: float,
                         lat2_deg: float, lon2_deg: float) -> float:
    """
    Computes the great-circle distance between two lat/lon points in metres.
    Uses the Haversine formula.
    """
    lat1 = lat1_deg * DEG2RAD
    lat2 = lat2_deg * DEG2RAD
    dlat = lat2 - lat1
    dlon = (lon2_deg - lon1_deg) * DEG2RAD
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def trajectory_to_arrays(trajectory: List[TrajectoryPoint]):
    """
    Converts a list of TrajectoryPoints to NumPy arrays for analysis.

    Returns
    -------
    timestamps   : np.ndarray [N]  seconds
    lats         : np.ndarray [N]  decimal degrees
    lons         : np.ndarray [N]  decimal degrees
    velocities   : np.ndarray [N]  m/s
    headings_deg : np.ndarray [N]  degrees
    """
    timestamps   = np.array([p.timestamp for p in trajectory], dtype=np.float64)
    lats         = np.array([p.lat for p in trajectory], dtype=np.float64)
    lons         = np.array([p.lon for p in trajectory], dtype=np.float64)
    velocities   = np.array([p.velocity_ms for p in trajectory], dtype=np.float64)
    headings_deg = np.array([p.heading_deg for p in trajectory], dtype=np.float64)
    return timestamps, lats, lons, velocities, headings_deg
