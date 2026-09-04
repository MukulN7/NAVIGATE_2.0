"""
ai_iekf_pipeline.py — End-to-End AI + ES-EKF Navigation Pipeline for NAVIGATE 2.0.

Fuses:
1. 10 Hz 6-axis Smartphone IMU (propagates ES-EKF state and covariance at 10 Hz).
2. VelocityModel V2 (estimates vehicle forward speed from 50-sample IMU windows).
3. AttitudeModel (estimates relative quaternion q_rel over 50-sample IMU windows).
4. Non-Holonomic Constraints (NHC: v_y^b = 0, v_z^b = 0).
5. GNSS position corrections (suppressed during blackout intervals).

Critical Relative Attitude Semantics
------------------------------------
The AttitudeModel predicts a RELATIVE quaternion over each 5-second window:
    q_rel = q_start^-1 (x) q_end
with quaternion order [qw, qx, qy, qz].

The pipeline implements this as a relative attitude constraint:
The predicted relative rotation from the EKF attitude at window start to the
EKF attitude at window end is compared against the network's q_rel.
The quaternion residual uses antipodal equivalence:
    q_error = inverse(q_rel_network) (x) q_rel_EKF
and is converted to a 3-vector residual in the navigation frame.
If the residual is not small, this first-order ES-EKF measurement update
is an approximation of the SO(3) geodesic error.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

import numpy as np
import torch

from navigate.models.velocity_model import VelocityModel
from navigate.models.attitude_model import AttitudeModel
from navigate.iekf_tracker import (
    ErrorStateIEKFTracker,
    GNSSBlackoutSchedule,
    IEKFTrajectoryPoint,
    EARTH_RADIUS_M,
    DEG2RAD,
    RAD2DEG,
    quat_normalize,
    quat_multiply,
    quat_inverse,
    quat_to_rotmat,
    quat_to_heading_deg,
    heading_deg_to_quat,
)
from navigate.evaluate_blackout import (
    BlackoutMetrics,
    BlackoutEvaluationResult,
    haversine_distance_m,
)

logger = logging.getLogger("ai_iekf_pipeline")


# ================================================================== #
#  Coordinate Conversion Helpers
# ================================================================== #

def lat_lon_to_enu_m(
    lat_deg: float,
    lon_deg: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
) -> Tuple[float, float]:
    """
    Converts WGS84 (lat, lon) in degrees to local East-North (m)
    relative to reference (ref_lat, ref_lon) using spherical Earth approximation.
    """
    d_lat = (lat_deg - ref_lat_deg) * DEG2RAD
    d_lon = (lon_deg - ref_lon_deg) * DEG2RAD
    ref_lat_rad = ref_lat_deg * DEG2RAD

    north_m = d_lat * EARTH_RADIUS_M
    east_m = d_lon * EARTH_RADIUS_M * math.cos(ref_lat_rad)
    return east_m, north_m


# ================================================================== #
#  End-to-End AI + ES-EKF Pipeline
# ================================================================== #

class AIIEKFPipeline:
    """
    End-to-end integration of trained VelocityModel V2, AttitudeModel, and ES-EKF.
    """

    def __init__(
        self,
        velocity_checkpoint: Union[str, Path] = "models/velocity_model_v2.pt",
        attitude_checkpoint: Union[str, Path] = "models/attitude_model.pt",
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing AIIEKFPipeline on device: {self.device}")

        # 1. Load VelocityModel V2
        vel_path = Path(velocity_checkpoint)
        if not vel_path.exists():
            raise FileNotFoundError(f"VelocityModel checkpoint not found: {vel_path}")
        vel_ckpt = torch.load(vel_path, map_location=self.device, weights_only=False)

        vel_cfg = vel_ckpt.get("model_config", {})
        self.velocity_model = VelocityModel(
            in_channels=vel_cfg.get("in_channels", 6),
            hidden_size=vel_cfg.get("hidden_size", 128),
            window_size=vel_cfg.get("window_size", 50),
            dropout_rate=vel_cfg.get("dropout_rate", 0.2),
        ).to(self.device)
        self.velocity_model.load_state_dict(vel_ckpt["model_state_dict"])
        self.velocity_model.eval()

        self.vel_imu_mean = np.array(vel_ckpt["imu_mean"], dtype=np.float32)
        self.vel_imu_std = np.array(vel_ckpt["imu_std"], dtype=np.float32)
        self.vel_mean = float(vel_ckpt["vel_mean"])
        self.vel_std = float(vel_ckpt["vel_std"])

        # 2. Load AttitudeModel
        att_path = Path(attitude_checkpoint)
        if not att_path.exists():
            raise FileNotFoundError(f"AttitudeModel checkpoint not found: {att_path}")
        att_ckpt = torch.load(att_path, map_location=self.device, weights_only=False)

        att_cfg = att_ckpt.get("model_config", {})
        self.attitude_model = AttitudeModel(
            in_channels=att_cfg.get("in_channels", 6),
            hidden_size=att_cfg.get("hidden_size", 128),
            window_size=att_cfg.get("window_size", 50),
            dropout_rate=att_cfg.get("dropout_rate", 0.25),
        ).to(self.device)
        self.attitude_model.load_state_dict(att_ckpt["model_state_dict"])
        self.attitude_model.eval()

        self.att_imu_mean = np.array(att_ckpt["imu_mean"], dtype=np.float32)
        self.att_imu_std = np.array(att_ckpt["imu_std"], dtype=np.float32)

        logger.info(
            f"Successfully loaded VelocityModel V2 (mean={self.vel_mean:.2f}, std={self.vel_std:.2f}) "
            f"and AttitudeModel."
        )

    # ------------------------------------------------------------------ #
    #  Model Inference Methods
    # ------------------------------------------------------------------ #

    def predict_velocity(self, imu_windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Runs batched inference with VelocityModel V2.
        Input: [N, 50, 6] IMU windows.
        Output: [N] forward vehicle speeds in m/s.
        """
        if len(imu_windows) == 0:
            return np.empty((0,), dtype=np.float32)

        is_single = (imu_windows.ndim == 2)
        if is_single:
            imu_windows = imu_windows[np.newaxis, ...]

        N = len(imu_windows)
        preds_list = []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                batch_raw = imu_windows[start:start + batch_size]
                # Standardize using VelocityModel training stats
                batch_norm = (batch_raw - self.vel_imu_mean) / self.vel_imu_std
                tensor_in = torch.tensor(batch_norm, dtype=torch.float32, device=self.device)
                norm_out, _ = self.velocity_model(tensor_in)
                # Un-normalize to physical m/s
                speed_ms = (norm_out * self.vel_std + self.vel_mean).squeeze(-1).cpu().numpy()
                preds_list.append(speed_ms)

        speeds = np.concatenate(preds_list, axis=0).astype(np.float32)
        return speeds[0] if is_single else speeds

    def predict_attitude(self, imu_windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Runs batched inference with AttitudeModel.
        Input: [N, 50, 6] IMU windows.
        Output: [N, 4] normalized unit relative quaternions [qw, qx, qy, qz] (qw >= 0).
        """
        if len(imu_windows) == 0:
            return np.empty((0, 4), dtype=np.float32)

        is_single = (imu_windows.ndim == 2)
        if is_single:
            imu_windows = imu_windows[np.newaxis, ...]

        N = len(imu_windows)
        preds_list = []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                batch_raw = imu_windows[start:start + batch_size]
                # Standardize using AttitudeModel training stats
                batch_norm = (batch_raw - self.att_imu_mean) / self.att_imu_std
                tensor_in = torch.tensor(batch_norm, dtype=torch.float32, device=self.device)
                q_out, _ = self.attitude_model(tensor_in)
                preds_list.append(q_out.cpu().numpy())

        quats = np.concatenate(preds_list, axis=0).astype(np.float32)
        # Ensure canonical non-negative qw
        neg_mask = quats[:, 0] < 0.0
        quats[neg_mask] = -quats[neg_mask]
        return quats[0] if is_single else quats

    # ------------------------------------------------------------------ #
    #  Trajectory Filter & Blackout Evaluation
    # ------------------------------------------------------------------ #

    def run_session_blackout(
        self,
        imu_windows: np.ndarray,
        timestamps: np.ndarray,
        gt_lats: np.ndarray,
        gt_lons: np.ndarray,
        blackout_intervals: List[Tuple[float, float]],
        init_heading_deg: float,
        gt_headings_deg: Optional[np.ndarray] = None,
        apply_nhc: bool = True,
        apply_attitude_update: bool = True,
        apply_velocity_update: bool = True,
        cov_speed: float = 0.25 ** 2,
        cov_att_deg: float = 5.0,
        cov_gnss_pos: float = 0.1 ** 2,
    ) -> BlackoutEvaluationResult:
        """
        Executes full AI + ES-EKF fusion trajectory tracking with GNSS blackout simulation.

        Parameters
        ----------
        imu_windows            : [N, 50, 6] IMU windows.
        timestamps             : [N] Window timestamps in seconds (typically 1.0s hop).
        gt_lats                : [N] Ground truth latitude in decimal degrees.
        gt_lons                : [N] Ground truth longitude in decimal degrees.
        blackout_intervals     : List of (start_s, end_s) blackout intervals.
        init_heading_deg       : Initial vehicle heading in degrees (clockwise from North).
        gt_headings_deg        : Optional [N] Ground truth heading in degrees.
        apply_nhc              : Whether to apply Non-Holonomic Constraints.
        apply_attitude_update  : Whether to apply relative attitude updates.
        apply_velocity_update  : Whether to apply learned forward velocity updates.
        cov_speed              : Variance of learned forward speed measurement.
        cov_att_deg            : Standard deviation in degrees for relative attitude update.
        cov_gnss_pos           : Variance of GNSS position measurement.
        """
        N = len(imu_windows)
        if N == 0:
            raise ValueError("imu_windows cannot be empty.")
        if len(timestamps) != N or len(gt_lats) != N or len(gt_lons) != N:
            raise ValueError(f"Array length mismatch: N={N}, ts={len(timestamps)}, lat={len(gt_lats)}")

        # 1. Run AI model inference across all windows
        logger.info(f"Running AI model inference for {N} windows...")
        speeds_ms = self.predict_velocity(imu_windows)
        quats_rel = self.predict_attitude(imu_windows)

        # 2. Setup Blackout Schedule
        schedule = GNSSBlackoutSchedule(intervals=blackout_intervals)

        # 3. Initialize ErrorStateIEKFTracker
        ref_lat = float(gt_lats[0])
        ref_lon = float(gt_lons[0])
        t_0 = float(timestamps[0])

        tracker = ErrorStateIEKFTracker(
            init_pos_enu=[0.0, 0.0, 0.0],
            init_vel_enu=[0.0, 0.0, 0.0],
            init_heading_deg=init_heading_deg,
            init_lat=ref_lat,
            init_lon=ref_lon,
            init_timestamp=t_0,
            blackout_schedule=schedule,
        )

        cov_att_rad = float((cov_att_deg * DEG2RAD) ** 2)

        # Buffer of recorded attitudes at window ends: list of (timestamp, quat)
        attitude_history: List[Tuple[float, np.ndarray]] = [
            (t_0, tracker.get_state()["quat"].copy())
        ]

        # 4. Step through trajectory windows
        # Note: Each window i is spaced by delta_t = timestamps[i] - timestamps[i-1] (1s).
        # In window i (50 samples at 10 Hz), the last 10 samples (samples 40:50) represent
        # the 10 Hz IMU propagation for this 1-second interval.
        for i in range(1, N):
            t_curr = float(timestamps[i])
            t_prev = float(timestamps[i - 1])
            dt_step = t_curr - t_prev
            if dt_step <= 0.0:
                dt_step = 1.0

            is_bo = schedule.is_blackout(t_curr)

            # IMU propagation at 10 Hz over the 10 samples of this 1-second interval
            window_imu = imu_windows[i]  # [50, 6]
            step_samples = window_imu[40:50]  # [10, 6]
            dt_sample = dt_step / 10.0

            for s in range(10):
                accel_s = step_samples[s, :3]
                gyro_s = step_samples[s, 3:]
                tracker.predict(dt=dt_sample, accel_b=accel_s, gyro_b=gyro_s)

            # Update tracker internal timestamp
            tracker._state.timestamp = t_curr

            # A. Learned Forward Velocity Update (VelocityModel V2)
            if apply_velocity_update:
                speed_target = max(0.0, float(speeds_ms[i]))
                tracker.update_velocity(forward_speed_ms=speed_target, cov_speed=cov_speed)

            # B. Non-Holonomic Constraints Update (lateral & vertical body velocity = 0)
            if apply_nhc:
                tracker.update_nhc()

            # C. Learned Relative Attitude Update (AttitudeModel)
            # Window i is a 5-second window. The start of this 5-second window corresponds to
            # approximately 4 to 5 steps ago in history.
            if apply_attitude_update:
                target_start_t = t_curr - 5.0
                # Find the closest recorded attitude near window start
                best_q_start = attitude_history[0][1]
                min_dt = abs(attitude_history[0][0] - target_start_t)
                for hist_t, hist_q in attitude_history:
                    diff_t = abs(hist_t - target_start_t)
                    if diff_t < min_dt:
                        min_dt = diff_t
                        best_q_start = hist_q

                q_rel_net = quats_rel[i]
                tracker.update_relative_attitude(
                    q_rel_network=q_rel_net,
                    q_start=best_q_start,
                    cov_att_rad=cov_att_rad,
                )

            # Record attitude at end of this step
            attitude_history.append((t_curr, tracker.get_state()["quat"].copy()))

            # D. GNSS Position Update (suppressed during blackout)
            gt_e, gt_n = lat_lon_to_enu_m(float(gt_lats[i]), float(gt_lons[i]), ref_lat, ref_lon)
            gnss_pos_enu = [gt_e, gt_n, 0.0]
            tracker.update_gnss_position(
                pos_enu_meas=gnss_pos_enu,
                cov_pos=cov_gnss_pos,
                is_blackout=is_bo,
            )

            # Record trajectory point
            tracker._record_trajectory_point(t_curr, is_blackout=is_bo)

        # 5. Extract trajectory & compute metrics
        estimated_traj = tracker.get_trajectory()

        # Step-by-step position error (haversine distance in meters)
        errors_per_step = np.zeros(N, dtype=np.float64)
        for i in range(N):
            pt = estimated_traj[i]
            errors_per_step[i] = haversine_distance_m(
                pt.lat, pt.lon, float(gt_lats[i]), float(gt_lons[i])
            )

        # Calculate metrics for each blackout interval
        per_blackout_metrics: List[BlackoutMetrics] = []

        for start_s, end_s in blackout_intervals:
            mask = (timestamps >= start_s) & (timestamps <= end_s)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            errors_in_bo = errors_per_step[indices]
            final_err = float(errors_in_bo[-1])
            max_err = float(errors_in_bo.max())
            rmse_err = float(np.sqrt(np.mean(errors_in_bo ** 2)))

            # Traveled distance during blackout interval
            if len(indices) > 1:
                dt_sub = np.diff(timestamps[indices])
                v_sub = speeds_ms[indices[:-1]]
                dist_m = float(np.sum(v_sub * dt_sub))
            else:
                dist_m = 0.0

            relative_drift = float((final_err / dist_m) * 100.0) if dist_m > 1e-3 else 0.0

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

        # Aggregate metrics across all blackout intervals
        if per_blackout_metrics:
            mean_final = float(np.mean([m.final_error_m for m in per_blackout_metrics]))
            mean_max = float(np.mean([m.max_error_m for m in per_blackout_metrics]))
            mean_rmse = float(np.mean([m.rmse_error_m for m in per_blackout_metrics]))
            total_dist = float(np.sum([m.traveled_distance_m for m in per_blackout_metrics]))
            mean_drift = float(np.mean([m.relative_drift_percent for m in per_blackout_metrics]))
        else:
            mean_final = 0.0
            mean_max = 0.0
            mean_rmse = 0.0
            total_dist = 0.0
            mean_drift = 0.0

        return BlackoutEvaluationResult(
            per_blackout_metrics=per_blackout_metrics,
            mean_final_error_m=mean_final,
            mean_max_error_m=mean_max,
            mean_rmse_error_m=mean_rmse,
            mean_relative_drift_percent=mean_drift,
            total_traveled_distance_m=total_dist,
            errors_per_step_m=errors_per_step,
            trajectory_estimated=estimated_traj,
        )
