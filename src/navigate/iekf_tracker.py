"""
iekf_tracker.py — Error-State Extended Kalman Filter (ES-EKF) for NAVIGATE 2.0.

Fuses 10 Hz 6-axis smartphone IMU (accelerometer + gyroscope), learned vehicle
forward velocity (from VelocityModel V2), Non-Holonomic Constraints (NHC),
and optional GNSS position measurements into an integrated navigation state.

State Representation
--------------------
- Nominal State (10D):
    p^n : Position in local ENU frame [p_east, p_north, p_up] (m)
    v^n : Velocity in local ENU frame [v_east, v_north, v_up] (m/s)
    q_b^n : Unit quaternion from body frame to navigation frame [qw, qx, qy, qz]
- Error State (9D):
    delta_x = [delta_p^n (3), delta_v^n (3), delta_theta^n (3)]^T
- Covariance Matrix: 9x9 positive semi-definite error covariance P.

Coordinate Conventions
----------------------
- Navigation Frame (n): Local East-North-Up (ENU) tangent plane.
- Body Frame (b): Vehicle Forward-Left-Up (FLU) right-handed system.
- Heading: Clockwise angle from true North in degrees [0, 360).
- Gravity vector: g^n = [0, 0, -9.80665]^T m/s^2.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np

# WGS84 mean Earth radius in metres (matching dead_reckoning.py)
EARTH_RADIUS_M = 6_371_000.0
STANDARD_GRAVITY = 9.80665
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ================================================================== #
#  Quaternion & Rotation Utilities
# ================================================================== #

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalizes a quaternion to unit length with canonical non-negative scalar (qw >= 0).
    Input q shape: [4] (qw, qx, qy, qz).
    """
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_norm = q / norm
    if q_norm[0] < 0.0:
        q_norm = -q_norm
    return q_norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions q1 and q2.
    Format: [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """
    Returns the conjugate / inverse of a unit quaternion.
    """
    q_norm = quat_normalize(q)
    return np.array([q_norm[0], -q_norm[1], -q_norm[2], -q_norm[3]], dtype=np.float64)


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """
    Converts a 3D rotation vector (angle * axis) to a unit quaternion [qw, qx, qy, qz].
    Accurate for small and large angles.
    """
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        # Taylor expansion around 0
        qw = 1.0 - (angle ** 2) / 8.0
        scale = 0.5 - (angle ** 2) / 48.0
        q = np.array([qw, rotvec[0] * scale, rotvec[1] * scale, rotvec[2] * scale], dtype=np.float64)
        return quat_normalize(q)
    
    half_angle = angle * 0.5
    scale = math.sin(half_angle) / angle
    q = np.array([
        math.cos(half_angle),
        rotvec[0] * scale,
        rotvec[1] * scale,
        rotvec[2] * scale,
    ], dtype=np.float64)
    return quat_normalize(q)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """
    Converts a unit quaternion [qw, qx, qy, qz] to a 3D rotation vector.
    """
    q_norm = quat_normalize(q)
    qw = np.clip(q_norm[0], -1.0, 1.0)
    vec = q_norm[1:4]
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-8:
        return 2.0 * vec
    angle = 2.0 * math.atan2(vec_norm, qw)
    return (angle / vec_norm) * vec


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [qw, qx, qy, qz] to a 3x3 direction cosine matrix R_b^n.
    Transforms vectors from body frame to navigation frame: v^n = R_b^n * v^b.
    """
    q_n = quat_normalize(q)
    w, x, y, z = q_n[0], q_n[1], q_n[2], q_n[3]
    
    return np.array([
        [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x**2 + y**2)],
    ], dtype=np.float64)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a unit quaternion [qw, qx, qy, qz].
    """
    tr = np.trace(R)
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))


def heading_deg_to_quat(heading_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """
    Constructs orientation quaternion from vehicle heading (clockwise from North in ENU).
    For level vehicle (pitch=0, roll=0):
      Forward vector (body X) in ENU is [sin(psi), cos(psi), 0].
      Left vector (body Y) in ENU is [-cos(psi), sin(psi), 0].
      Up vector (body Z) in ENU is [0, 0, 1].
    """
    psi = heading_deg * DEG2RAD
    theta = pitch_deg * DEG2RAD
    phi = roll_deg * DEG2RAD

    # Rotation matrix R_b^n for level vehicle heading psi
    R_yaw = np.array([
        [math.sin(psi), -math.cos(psi), 0.0],
        [math.cos(psi),  math.sin(psi), 0.0],
        [0.0,            0.0,           1.0],
    ], dtype=np.float64)

    if abs(pitch_deg) < 1e-6 and abs(roll_deg) < 1e-6:
        return rotmat_to_quat(R_yaw)

    # Pitch around body Y (left/right) and Roll around body X (forward)
    R_pitch = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0,             1.0, 0.0],
        [-math.sin(theta),0.0, math.cos(theta)],
    ], dtype=np.float64)

    R_roll = np.array([
        [1.0, 0.0,            0.0],
        [0.0, math.cos(phi), -math.sin(phi)],
        [0.0, math.sin(phi),  math.cos(phi)],
    ], dtype=np.float64)

    R_total = R_yaw @ R_pitch @ R_roll
    return rotmat_to_quat(R_total)


def quat_to_heading_deg(q: np.ndarray) -> float:
    """
    Extracts heading (clockwise from North, [0, 360) deg) from quaternion R_b^n.
    Extracts horizontal projection of the vehicle forward axis (body X transformed to nav).
    """
    R = quat_to_rotmat(q)
    fwd_east = R[0, 0]
    fwd_north = R[1, 0]
    heading_rad = math.atan2(fwd_east, fwd_north)
    if heading_rad < 0.0:
        heading_rad += 2.0 * math.pi
    return (heading_rad * RAD2DEG) % 360.0


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Returns 3x3 skew-symmetric cross-product matrix [v]x.
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


# ================================================================== #
#  GNSS Blackout Helper
# ================================================================== #

class GNSSBlackoutSchedule:
    """
    Maintains time intervals during which GNSS measurements are suppressed.
    """

    def __init__(self, intervals: Optional[Sequence[Tuple[float, float]]] = None) -> None:
        self._intervals: List[Tuple[float, float]] = list(intervals) if intervals else []

    def is_blackout(self, timestamp: float) -> bool:
        """Returns True if timestamp falls inside any configured blackout interval."""
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
#  Filter State Dataclasses
# ================================================================== #

@dataclass
class ESEKFState:
    """
    Nominal navigation state and error covariance.
    """
    pos_enu: np.ndarray       # [3] East, North, Up (m)
    vel_enu: np.ndarray       # [3] v_East, v_North, v_Up (m/s)
    quat: np.ndarray          # [4] Unit quaternion R_b^n [qw, qx, qy, qz]
    cov: np.ndarray           # [9, 9] Error state covariance matrix
    timestamp: float = 0.0
    lat_ref: float = 0.0      # Reference latitude (rad)
    lon_ref: float = 0.0      # Reference longitude (rad)
    alt_ref: float = 0.0      # Reference altitude (m)


@dataclass
class IEKFTrajectoryPoint:
    """
    Output trajectory snapshot from the filter.
    """
    timestamp: float
    lat: float
    lon: float
    alt: float
    pos_east_m: float
    pos_north_m: float
    pos_up_m: float
    vel_east_ms: float
    vel_north_ms: float
    vel_up_ms: float
    speed_ms: float
    heading_deg: float
    is_gnss_blackout: bool = False


# ================================================================== #
#  Error-State EKF Tracker Implementation
# ================================================================== #

class ErrorStateIEKFTracker:
    """
    Error-State Extended Kalman Filter for GPS/INS/NHC/Learned Velocity Fusion.

    Parameters
    ----------
    init_pos_enu      : Array-like [3], Initial ENU position in meters.
    init_vel_enu      : Array-like [3], Initial ENU velocity in m/s.
    init_heading_deg  : Float, Initial heading clockwise from North.
    init_quat         : Optional Array-like [4], Initial orientation quaternion.
    init_lat          : Float, Reference latitude in decimal degrees.
    init_lon          : Float, Reference longitude in decimal degrees.
    init_alt          : Float, Reference altitude in meters.
    init_timestamp    : Float, Start time in seconds.
    std_pos_init      : Float, Initial position error standard deviation (m).
    std_vel_init      : Float, Initial velocity error standard deviation (m/s).
    std_att_init_deg  : Float, Initial attitude error standard deviation (deg).
    accel_noise_std   : Float, IMU accelerometer noise spectral density (m/s^2 / sqrt(Hz)).
    gyro_noise_std    : Float, IMU gyroscope noise spectral density (rad/s / sqrt(Hz)).
    blackout_schedule : Optional GNSSBlackoutSchedule.
    """

    def __init__(
        self,
        init_pos_enu: Sequence[float] = (0.0, 0.0, 0.0),
        init_vel_enu: Sequence[float] = (0.0, 0.0, 0.0),
        init_heading_deg: float = 0.0,
        init_quat: Optional[Sequence[float]] = None,
        init_lat: float = 0.0,
        init_lon: float = 0.0,
        init_alt: float = 0.0,
        init_timestamp: float = 0.0,
        std_pos_init: float = 1.0,
        std_vel_init: float = 0.5,
        std_att_init_deg: float = 2.0,
        accel_noise_std: float = 0.2,
        gyro_noise_std: float = 0.02,
        blackout_schedule: Optional[GNSSBlackoutSchedule] = None,
    ) -> None:
        self.accel_noise_std = float(accel_noise_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.blackout = blackout_schedule or GNSSBlackoutSchedule()

        pos_enu = np.array(init_pos_enu, dtype=np.float64).reshape(3)
        vel_enu = np.array(init_vel_enu, dtype=np.float64).reshape(3)

        if init_quat is not None:
            quat = quat_normalize(np.array(init_quat, dtype=np.float64).reshape(4))
        else:
            quat = heading_deg_to_quat(init_heading_deg)

        # 9x9 Error state covariance matrix: [delta_p (3), delta_v (3), delta_theta (3)]
        cov = np.zeros((9, 9), dtype=np.float64)
        cov[0:3, 0:3] = (std_pos_init ** 2) * np.eye(3)
        cov[3:6, 3:6] = (std_vel_init ** 2) * np.eye(3)
        std_att_rad = std_att_init_deg * DEG2RAD
        cov[6:9, 6:9] = (std_att_rad ** 2) * np.eye(3)

        self._state = ESEKFState(
            pos_enu=pos_enu,
            vel_enu=vel_enu,
            quat=quat,
            cov=cov,
            timestamp=init_timestamp,
            lat_ref=init_lat * DEG2RAD,
            lon_ref=init_lon * DEG2RAD,
            alt_ref=init_alt,
        )

        self._trajectory: List[IEKFTrajectoryPoint] = []
        self._record_trajectory_point(init_timestamp, is_blackout=False)

    # ------------------------------------------------------------------ #
    #  Filter Propagation (Prediction Step at 10 Hz)
    # ------------------------------------------------------------------ #

    def predict(
        self,
        dt: float,
        accel_b: Sequence[float],
        gyro_b: Sequence[float],
    ) -> None:
        """
        Propagates the nominal state and error covariance using IMU measurements.

        Parameters
        ----------
        dt      : float, Time interval in seconds.
        accel_b : Array-like [3], Accelerometer specific force in body frame (m/s^2).
        gyro_b  : Array-like [3], Gyroscope angular velocity in body frame (rad/s).
        """
        if dt <= 0.0:
            return

        accel_b_arr = np.array(accel_b, dtype=np.float64).reshape(3)
        gyro_b_arr = np.array(gyro_b, dtype=np.float64).reshape(3)

        # 1. Propagate nominal quaternion attitude: q = q (x) delta_q(gyro * dt)
        delta_q = rotvec_to_quat(gyro_b_arr * dt)
        new_quat = quat_normalize(quat_multiply(self._state.quat, delta_q))

        # 2. Rotate specific force to navigation frame & compensate gravity
        R_curr = quat_to_rotmat(self._state.quat)
        g_n = np.array([0.0, 0.0, -STANDARD_GRAVITY], dtype=np.float64)
        accel_n = R_curr @ accel_b_arr + g_n

        # 3. Propagate nominal position and velocity
        new_pos = self._state.pos_enu + self._state.vel_enu * dt + 0.5 * accel_n * (dt ** 2)
        new_vel = self._state.vel_enu + accel_n * dt

        # 4. Error State Transition Matrix F (9x9)
        # delta_p_dot = delta_v
        # delta_v_dot = - [R * a_b]x * delta_theta + R * w_a
        # delta_theta_dot = - [omega]x * delta_theta (in nav frame: - R * w_g)
        F = np.eye(9, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        a_b_rot = R_curr @ accel_b_arr
        F[0:3, 6:9] = -0.5 * skew_symmetric(a_b_rot) * (dt ** 2)
        F[3:6, 6:9] = -skew_symmetric(a_b_rot) * dt

        # 5. Discrete Process Noise Matrix Q (9x9)
        Q = np.zeros((9, 9), dtype=np.float64)
        var_a = (self.accel_noise_std ** 2) * dt
        var_g = (self.gyro_noise_std ** 2) * dt
        Q[0:3, 0:3] = (1.0 / 3.0) * var_a * (dt ** 2) * np.eye(3)
        Q[0:3, 3:6] = 0.5 * var_a * dt * np.eye(3)
        Q[3:6, 0:3] = 0.5 * var_a * dt * np.eye(3)
        Q[3:6, 3:6] = var_a * np.eye(3)
        Q[6:9, 6:9] = var_g * np.eye(3)

        # 6. Propagate Covariance
        new_cov = F @ self._state.cov @ F.T + Q
        new_cov = 0.5 * (new_cov + new_cov.T)

        # Commit nominal state
        self._state.pos_enu = new_pos
        self._state.vel_enu = new_vel
        self._state.quat = new_quat
        self._state.cov = new_cov

    # ------------------------------------------------------------------ #
    #  Generic Measurement Update & Error State Injection
    # ------------------------------------------------------------------ #

    def _apply_error_state_update(
        self,
        residual: np.ndarray,
        H: np.ndarray,
        R_cov: np.ndarray,
    ) -> None:
        """
        Executes Kalman measurement update on the error state, injects the
        estimated error into the nominal state, and resets the error state to zero.
        Uses Joseph stabilized covariance update for numerical robustness.
        """
        P = self._state.cov
        # Innovation covariance S = H * P * H^T + R
        S = H @ P @ H.T + R_cov
        # Kalman Gain K = P * H^T * inv(S)
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P @ H.T @ np.linalg.pinv(S)

        # Error state correction delta_x in R^9
        delta_x = K @ residual

        # Joseph form covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(9, dtype=np.float64) - K @ H
        P_updated = I_KH @ P @ I_KH.T + K @ R_cov @ K.T
        self._state.cov = 0.5 * (P_updated + P_updated.T)

        # Inject error into nominal state:
        # 1. Position
        self._state.pos_enu += delta_x[0:3]
        # 2. Velocity
        self._state.vel_enu += delta_x[3:6]
        # 3. Attitude quaternion: q = delta_q(delta_theta) (x) q
        delta_q = rotvec_to_quat(delta_x[6:9])
        self._state.quat = quat_normalize(quat_multiply(delta_q, self._state.quat))

    # ------------------------------------------------------------------ #
    #  Non-Holonomic Constraints (NHC) Update
    # ------------------------------------------------------------------ #

    def update_nhc(
        self,
        cov_lateral: float = 0.05 ** 2,
        cov_vertical: float = 0.05 ** 2,
    ) -> None:
        """
        Enforces 2D non-holonomic vehicle motion constraints:
        Lateral body velocity v_y^b = 0 and vertical body velocity v_z^b = 0.
        """
        R_b_n = quat_to_rotmat(self._state.quat)
        R_n_b = R_b_n.T

        # Velocity in body frame: v^b = R_n_b * v^n
        v_b = R_n_b @ self._state.vel_enu

        # Measurement residual: z - h(x) = [0 - v_y^b, 0 - v_z^b]^T
        residual = -v_b[1:3]

        # Measurement Jacobian H_nhc (2x9)
        # d(v^b)/d(delta_v) = R_n_b
        # d(v^b)/d(delta_theta) = R_n_b * [v^n]x
        H = np.zeros((2, 9), dtype=np.float64)
        H[0:2, 3:6] = R_n_b[1:3, 0:3]
        v_skew = skew_symmetric(self._state.vel_enu)
        H[0:2, 6:9] = (R_n_b @ v_skew)[1:3, 0:3]

        R_cov = np.diag([cov_lateral, cov_vertical])
        self._apply_error_state_update(residual, H, R_cov)

    # ------------------------------------------------------------------ #
    #  Learned Velocity Update (VelocityModel V2)
    # ------------------------------------------------------------------ #

    def update_velocity(
        self,
        forward_speed_ms: float,
        cov_speed: float = 0.25 ** 2,
    ) -> None:
        """
        Fuses scalar forward speed estimate from VelocityModel V2 into body X axis.
        """
        R_b_n = quat_to_rotmat(self._state.quat)
        R_n_b = R_b_n.T

        v_b = R_n_b @ self._state.vel_enu
        residual = np.array([forward_speed_ms - v_b[0]], dtype=np.float64)

        # Measurement Jacobian H_vel (1x9)
        H = np.zeros((1, 9), dtype=np.float64)
        H[0, 3:6] = R_n_b[0, 0:3]
        v_skew = skew_symmetric(self._state.vel_enu)
        H[0, 6:9] = (R_n_b @ v_skew)[0, 0:3]

        R_cov = np.array([[cov_speed]], dtype=np.float64)
        self._apply_error_state_update(residual, H, R_cov)

    # ------------------------------------------------------------------ #
    #  Learned Relative Attitude Update (AttitudeModel)
    # ------------------------------------------------------------------ #

    def update_relative_attitude(
        self,
        q_rel_network: Sequence[float],
        q_start: Sequence[float],
        cov_att_rad: float = (5.0 * DEG2RAD) ** 2,
    ) -> None:
        """
        Relative attitude measurement update from AttitudeModel over a window.

        Compares the predicted relative rotation from the EKF attitude at window start
        (q_start) to the current EKF attitude at window end (self._state.quat):
            q_rel_ekf = inverse(q_start) (x) q_curr

        against the network's predicted relative quaternion:
            q_rel_network (order: [qw, qx, qy, qz]).

        Computes the quaternion error with antipodal equivalence:
            q_error = inverse(q_rel_network) (x) q_rel_ekf

        Converts the small-angle rotation vector to a 3-vector residual in navigation
        frame, and applies the ES-EKF measurement update.

        Note / Limitation:
        If the residual rotation is not small (e.g. > 30 deg), this update uses a
        first-order error-state approximation of the SO(3) geodesic error.
        """
        q_start_n = quat_normalize(np.array(q_start, dtype=np.float64).reshape(4))
        q_curr = self._state.quat
        q_rel_ekf = quat_multiply(quat_inverse(q_start_n), q_curr)

        q_net_n = quat_normalize(np.array(q_rel_network, dtype=np.float64).reshape(4))
        q_err = quat_multiply(quat_inverse(q_net_n), q_rel_ekf)

        # Antipodal equivalence: q and -q represent identical 3D rotations
        if q_err[0] < 0.0:
            q_err = -q_err

        # Small-angle rotation vector in body frame
        rotvec_b = quat_to_rotvec(q_err)

        # Transform rotation vector from current body frame to navigation frame:
        R_curr = quat_to_rotmat(q_curr)
        rotvec_n = R_curr @ rotvec_b

        # Measurement residual: y = -rotvec_n
        residual = -rotvec_n

        # Measurement Jacobian H_att in navigation frame (3x9):
        H = np.zeros((3, 9), dtype=np.float64)
        H[0:3, 6:9] = np.eye(3)

        R_cov = cov_att_rad * np.eye(3) if not isinstance(cov_att_rad, np.ndarray) else cov_att_rad
        self._apply_error_state_update(residual, H, R_cov)

    # ------------------------------------------------------------------ #
    #  GNSS Position Update (Skipped during Blackouts)
    # ------------------------------------------------------------------ #

    def update_gnss_position(
        self,
        pos_enu_meas: Sequence[float],
        cov_pos: float = 1.0 ** 2,
        is_blackout: bool = False,
    ) -> bool:
        """
        Fuses ENU GNSS position measurement. Skipped if in blackout.
        Returns True if update was applied, False if skipped due to blackout.
        """
        if is_blackout:
            return False

        pos_meas = np.array(pos_enu_meas, dtype=np.float64).reshape(3)
        residual = pos_meas - self._state.pos_enu

        # Measurement Jacobian H_gnss (3x9)
        H = np.zeros((3, 9), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        R_cov = (cov_pos if isinstance(cov_pos, np.ndarray) else (cov_pos * np.eye(3)))
        self._apply_error_state_update(residual, H, R_cov)
        return True

    # ------------------------------------------------------------------ #
    #  Step & Trajectory Helpers
    # ------------------------------------------------------------------ #

    def step(
        self,
        timestamp: float,
        accel_b: Sequence[float],
        gyro_b: Sequence[float],
        velocity_ms: Optional[float] = None,
        gnss_pos_enu: Optional[Sequence[float]] = None,
        in_blackout: Optional[bool] = None,
        apply_nhc: bool = True,
        q_rel_network: Optional[Sequence[float]] = None,
        q_start: Optional[Sequence[float]] = None,
    ) -> IEKFTrajectoryPoint:
        """
        Advances the filter by one time step with optional measurement updates.
        """
        dt = timestamp - self._state.timestamp
        if dt <= 0.0:
            dt = 0.1  # Fallback to 10 Hz

        # 1. Determine blackout status
        is_bo = self.blackout.is_blackout(timestamp) if in_blackout is None else bool(in_blackout)

        # 2. IMU Prediction step
        self.predict(dt=dt, accel_b=accel_b, gyro_b=gyro_b)
        self._state.timestamp = timestamp

        # 3. Learned Forward Velocity update (active in both normal & blackout modes)
        if velocity_ms is not None:
            self.update_velocity(forward_speed_ms=velocity_ms)

        # 4. Non-Holonomic Constraints update (active in both normal & blackout modes)
        if apply_nhc:
            self.update_nhc()

        # 4b. Learned Relative Attitude update (from AttitudeModel)
        if q_rel_network is not None and q_start is not None:
            self.update_relative_attitude(q_rel_network=q_rel_network, q_start=q_start)

        # 5. GNSS position update (suppressed during blackout)
        if gnss_pos_enu is not None:
            self.update_gnss_position(pos_enu_meas=gnss_pos_enu, is_blackout=is_bo)

        # 6. Record and return trajectory point
        return self._record_trajectory_point(timestamp, is_blackout=is_bo)

    def _record_trajectory_point(self, timestamp: float, is_blackout: bool) -> IEKFTrajectoryPoint:
        """
        Creates a trajectory point from the current state and appends it to history.
        """
        # Convert local ENU to WGS84 Lat/Lon
        lat_deg = (self._state.lat_ref + (self._state.pos_enu[1] / EARTH_RADIUS_M)) * RAD2DEG
        lon_deg = (self._state.lon_ref + (self._state.pos_enu[0] / (EARTH_RADIUS_M * math.cos(self._state.lat_ref)))) * RAD2DEG
        alt_m = self._state.alt_ref + self._state.pos_enu[2]

        speed_ms = float(np.linalg.norm(self._state.vel_enu))
        heading_deg = quat_to_heading_deg(self._state.quat)

        pt = IEKFTrajectoryPoint(
            timestamp=timestamp,
            lat=lat_deg,
            lon=lon_deg,
            alt=alt_m,
            pos_east_m=float(self._state.pos_enu[0]),
            pos_north_m=float(self._state.pos_enu[1]),
            pos_up_m=float(self._state.pos_enu[2]),
            vel_east_ms=float(self._state.vel_enu[0]),
            vel_north_ms=float(self._state.vel_enu[1]),
            vel_up_ms=float(self._state.vel_enu[2]),
            speed_ms=speed_ms,
            heading_deg=heading_deg,
            is_gnss_blackout=is_blackout,
        )
        self._trajectory.append(pt)
        return pt

    def get_state(self) -> Dict[str, Any]:
        """
        Returns full state snapshot as a dictionary.
        """
        return {
            "timestamp": self._state.timestamp,
            "pos_enu": self._state.pos_enu.copy(),
            "vel_enu": self._state.vel_enu.copy(),
            "quat": self._state.quat.copy(),
            "cov": self._state.cov.copy(),
            "heading_deg": quat_to_heading_deg(self._state.quat),
            "speed_ms": float(np.linalg.norm(self._state.vel_enu)),
        }

    def get_trajectory(self) -> List[IEKFTrajectoryPoint]:
        """
        Returns the recorded trajectory points history.
        """
        return list(self._trajectory)
