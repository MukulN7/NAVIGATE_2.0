"""
ai_iekf_road_pipeline.py — Version B: AI + ES-EKF + Road Constraint Pipeline (NAVIGATE 2.0).

This module implements a SECOND evaluation variant (Version B) alongside the existing
validated AI + ES-EKF pipeline (Version A, ai_iekf_pipeline.py):

  VERSION A: AI + ES-EKF               (ai_iekf_pipeline.py)
  VERSION B: AI + ES-EKF + Road        (this file)

Design Goals
------------
* Do NOT replace or modify Version A.  Both pipelines must remain independently runnable.
* Add a road-constraint step AFTER the standard ES-EKF predict/update cycle.
* Road matching must be OPTIONAL — disabled gracefully when no road candidate exists.
* No future ground-truth leakage into the cached road geometry during a GNSS blackout.

Road Reference Mechanism (Leakage-Free Design)
-----------------------------------------------
The IO-VNBD vehicle CSV files contain 10 Hz GPS (lat/lon/heading) ground-truth.
The existing evaluation pipeline already loads this for error measurement.

We use the GPS trajectory observed BEFORE each blackout interval to construct a
road-centreline polyline.  Specifically:

  1. For each blackout interval [t_start, t_end]:
       road_cache[interval] = GPS points with timestamp  t_pre_start <= t < t_start
       where t_pre_start = t_start - lookahead_window_s (default 120 s)
  2. The cached polyline is built ONCE at or before t_start and frozen.
  3. From t_start onwards ONLY the cached polyline is used for road correction;
     no future GPS positions are added.
  4. At GNSS recovery (t > t_end) the road cache for that interval is discarded;
     normal GNSS fusion resumes.

This design is deliberately conservative:
  - It uses only the road segment the vehicle was driving on immediately before losing GNSS.
  - It cannot see around corners that were not yet traversed before the outage.
  - The road polyline is a straight-line approximation of the driven path segment,
    which is realistic for urban / suburban driving (roads are relatively straight
    over short segments).

LIMITATION (documented per requirement):
  If the vehicle's path during the blackout diverges significantly from the
  pre-blackout road segment (e.g., the vehicle turns immediately after the outage
  begins), the road constraint may push the estimate in the wrong direction.
  This is explicitly reported in the evaluation statistics via road-match statistics
  (acceptance / rejection counts).

Road Correction Step
---------------------
After each AI + ES-EKF step during a blackout:
  1. Extract current [East, North] position and heading from ES-EKF state.
  2. Call MapMatcher.match(pos, heading, cached_road).
  3. If match succeeds (distance + heading gates pass):
       Apply a weighted correction toward the projected road point.
       The correction is injected directly into the EKF position state via a
       position pseudo-measurement update (EKF measurement update step) so that
       the covariance is properly propagated.
  4. If match fails (distance or heading gate): no correction; EKF state unchanged.

Correction is applied as a standard EKF position pseudo-measurement with
configurable noise (road_cov_m2), not as a hard snap.  This ensures graceful
degradation and respects filter uncertainty.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from navigate.ai_iekf_pipeline import AIIEKFPipeline, lat_lon_to_enu_m
from navigate.iekf_tracker import (
    ErrorStateIEKFTracker,
    GNSSBlackoutSchedule,
    IEKFTrajectoryPoint,
    EARTH_RADIUS_M,
    DEG2RAD,
    RAD2DEG,
    quat_to_heading_deg,
)
from navigate.map_matching import MapMatcher, RoadPolyline, MapMatchResult
from navigate.evaluate_blackout import (
    BlackoutMetrics,
    BlackoutEvaluationResult,
    haversine_distance_m,
)

logger = logging.getLogger("ai_iekf_road_pipeline")


# ================================================================== #
#  Road-Matching Statistics Dataclass
# ================================================================== #

@dataclass
class RoadMatchStats:
    """
    Per-blackout statistics for road-constraint matching.

    Attributes
    ----------
    blackout_start_s : float
        Start time of the associated blackout interval (seconds).
    blackout_end_s : float
        End time of the associated blackout interval (seconds).
    road_active : bool
        True when a valid cached road polyline was available for this interval.
    n_steps : int
        Total number of EKF steps during this blackout.
    n_matched : int
        Number of steps where road matching succeeded (both gates passed).
    n_rejected_distance : int
        Steps rejected because estimated position exceeded max_match_dist_m.
    n_rejected_heading : int
        Steps rejected because heading difference exceeded max_heading_diff_deg.
    match_fraction : float
        Fraction of steps successfully matched (n_matched / n_steps).
    avg_correction_m : float
        Average correction distance applied (metres) on matched steps.
    max_correction_m : float
        Maximum correction applied on any matched step.
    road_vertices : int
        Number of vertices in the cached road polyline (0 if no road).
    """
    blackout_start_s: float = 0.0
    blackout_end_s: float = 0.0
    road_active: bool = False
    n_steps: int = 0
    n_matched: int = 0
    n_rejected_distance: int = 0
    n_rejected_heading: int = 0
    match_fraction: float = 0.0
    avg_correction_m: float = 0.0
    max_correction_m: float = 0.0
    road_vertices: int = 0


@dataclass
class RoadBlackoutEvaluationResult:
    """
    Extended evaluation result for Version B (AI + ES-EKF + Road Constraint).

    Contains all standard BlackoutEvaluationResult fields plus road-matching statistics.
    """
    # Standard blackout metrics (same structure as Version A)
    per_blackout_metrics: List[BlackoutMetrics] = field(default_factory=list)
    mean_final_error_m: float = 0.0
    mean_max_error_m: float = 0.0
    mean_rmse_error_m: float = 0.0
    mean_relative_drift_percent: float = 0.0
    total_traveled_distance_m: float = 0.0
    errors_per_step_m: np.ndarray = field(default_factory=lambda: np.array([]))
    trajectory_estimated: List[IEKFTrajectoryPoint] = field(default_factory=list)

    # Extended road-matching statistics (Version B only)
    road_match_stats: List[RoadMatchStats] = field(default_factory=list)
    total_matched: int = 0
    total_rejected_distance: int = 0
    total_rejected_heading: int = 0
    overall_match_fraction: float = 0.0
    overall_avg_correction_m: float = 0.0


# ================================================================== #
#  Road Cache Builder (Leakage-Free)
# ================================================================== #

def build_pre_blackout_road(
    gt_lats: np.ndarray,
    gt_lons: np.ndarray,
    timestamps: np.ndarray,
    blackout_start_s: float,
    ref_lat: float,
    ref_lon: float,
    lookahead_window_s: float = 120.0,
    min_vertices: int = 2,
    min_segment_length_m: float = 1.0,
) -> Optional[RoadPolyline]:
    """
    Builds a road polyline from GPS observations strictly BEFORE the blackout starts.

    This function is the core leakage-prevention mechanism: it uses only past
    GPS observations (t < blackout_start_s) to construct the road reference.
    No future GPS positions (t >= blackout_start_s) are included.

    Parameters
    ----------
    gt_lats : [N] Ground-truth latitudes (degrees).
    gt_lons : [N] Ground-truth longitudes (degrees).
    timestamps : [N] Window timestamps in seconds.
    blackout_start_s : float
        Start of the blackout interval.  Road is built from data before this time.
    ref_lat : float
        Reference latitude for ENU conversion (degrees).
    ref_lon : float
        Reference longitude for ENU conversion (degrees).
    lookahead_window_s : float
        How many seconds before the blackout to include (default 120 s).
    min_vertices : int
        Minimum number of distinct vertices required (default 2).
    min_segment_length_m : float
        Minimum cumulative segment length to accept the polyline (default 1.0 m).

    Returns
    -------
    RoadPolyline or None
        A valid cached road polyline, or None if insufficient pre-blackout data.

    Design Note on Leakage
    ----------------------
    * Only points with  timestamps[i] < blackout_start_s  are included.
    * The window  t in [blackout_start_s - lookahead_window_s, blackout_start_s)
      is the "road cache window".
    * After this function returns, the polyline is frozen and never updated
      with future GPS positions.
    """
    # Select points strictly before blackout start
    pre_mask = (timestamps < blackout_start_s)
    if lookahead_window_s > 0:
        pre_mask &= (timestamps >= blackout_start_s - lookahead_window_s)

    pre_indices = np.where(pre_mask)[0]
    if len(pre_indices) < min_vertices:
        logger.debug(
            f"  [RoadCache] Insufficient pre-blackout points for t_start={blackout_start_s:.1f}s "
            f"(found {len(pre_indices)}, need {min_vertices})"
        )
        return None

    # Convert pre-blackout GPS to ENU [E, N] coordinates
    enu_vertices = []
    for idx in pre_indices:
        e, n = lat_lon_to_enu_m(float(gt_lats[idx]), float(gt_lons[idx]), ref_lat, ref_lon)
        enu_vertices.append([e, n])

    enu_arr = np.array(enu_vertices, dtype=np.float64)

    # Remove duplicate/very-close points to avoid degenerate segments
    if len(enu_arr) > 1:
        diffs = np.linalg.norm(np.diff(enu_arr, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 0.1])
        enu_arr = enu_arr[keep]

    if len(enu_arr) < min_vertices:
        return None

    # Check minimum total road length
    total_length = float(np.sum(np.linalg.norm(np.diff(enu_arr, axis=0), axis=1)))
    if total_length < min_segment_length_m:
        logger.debug(
            f"  [RoadCache] Road too short ({total_length:.2f}m < {min_segment_length_m}m) "
            f"for t_start={blackout_start_s:.1f}s"
        )
        return None

    road_name = f"pre_blackout_road@{blackout_start_s:.1f}s"
    logger.debug(
        f"  [RoadCache] Built road for t_start={blackout_start_s:.1f}s: "
        f"{len(enu_arr)} vertices, {total_length:.1f}m length"
    )
    return RoadPolyline(vertices=enu_arr, name=road_name)


# ================================================================== #
#  Version B Pipeline: AI + ES-EKF + Road Constraint
# ================================================================== #

class AIIEKFRoadPipeline(AIIEKFPipeline):
    """
    Version B: AI + ES-EKF + Road Constraint Pipeline.

    Extends AIIEKFPipeline (Version A) with a road-matching correction step
    applied after each ES-EKF update during GNSS blackouts.

    Parameters
    ----------
    velocity_checkpoint : str or Path
        Path to the VelocityModel V2 checkpoint (reused from Version A).
    attitude_checkpoint : str or Path
        Path to the AttitudeModel checkpoint (reused from Version A).
    device : str or None
        PyTorch device string.
    max_match_dist_m : float
        Maximum perpendicular distance (metres) for road matching (default 20.0).
    max_heading_diff_deg : float
        Maximum heading deviation (degrees) for road matching (default 30.0).
    correction_strength : float in [0, 1]
        Interpolation weight toward road centreline for display only;
        actual correction is injected as a KF position measurement (default 0.5).
    road_cov_m2 : float
        Position measurement noise variance (m^2) for the road pseudo-measurement.
        Higher values = softer road correction (default 5.0 = ~2.24 m sigma).
    lookahead_window_s : float
        How many seconds before each blackout to cache road geometry (default 120 s).
    min_road_vertices : int
        Minimum number of GPS points required to build a road polyline (default 3).
    apply_road_during_blackout_only : bool
        If True (default), road correction only runs during blackouts.
        If False, road correction runs continuously (non-standard).
    """

    def __init__(
        self,
        velocity_checkpoint: Union[str, Path] = "models/velocity_model_v2.pt",
        attitude_checkpoint: Union[str, Path] = "models/attitude_model.pt",
        device: Optional[str] = None,
        max_match_dist_m: float = 20.0,
        max_heading_diff_deg: float = 30.0,
        correction_strength: float = 0.5,
        road_cov_m2: float = 5.0,
        lookahead_window_s: float = 120.0,
        min_road_vertices: int = 3,
        apply_road_during_blackout_only: bool = True,
    ) -> None:
        super().__init__(
            velocity_checkpoint=velocity_checkpoint,
            attitude_checkpoint=attitude_checkpoint,
            device=device,
        )
        self.map_matcher = MapMatcher(
            max_match_dist_m=max_match_dist_m,
            max_heading_diff_deg=max_heading_diff_deg,
            correction_strength=correction_strength,
        )
        self.road_cov_m2 = float(road_cov_m2)
        self.lookahead_window_s = float(lookahead_window_s)
        self.min_road_vertices = int(min_road_vertices)
        self.apply_road_during_blackout_only = apply_road_during_blackout_only

        logger.info(
            f"AIIEKFRoadPipeline initialized: "
            f"max_dist={max_match_dist_m}m, max_hdg={max_heading_diff_deg}deg, "
            f"correction_strength={correction_strength}, road_cov={road_cov_m2}m², "
            f"lookahead={lookahead_window_s}s"
        )

    # ------------------------------------------------------------------ #
    #  Road-Corrected Blackout Evaluation (Version B)
    # ------------------------------------------------------------------ #

    def run_session_blackout_road(
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
    ) -> RoadBlackoutEvaluationResult:
        """
        Version B evaluation: AI + ES-EKF + Road Constraint during GNSS blackouts.

        The road polyline for each blackout is built from GPS observations
        STRICTLY BEFORE the blackout begins (no future leakage).

        Parameters are identical to AIIEKFPipeline.run_session_blackout plus
        the road-specific parameters set at __init__ time.

        Returns
        -------
        RoadBlackoutEvaluationResult
            Includes all standard metrics plus per-blackout road-matching statistics.
        """
        N = len(imu_windows)
        if N == 0:
            raise ValueError("imu_windows cannot be empty.")
        if len(timestamps) != N or len(gt_lats) != N or len(gt_lons) != N:
            raise ValueError(f"Array length mismatch: N={N}, ts={len(timestamps)}")

        # 1. AI Model Inference (identical to Version A)
        logger.info(f"[RoadPipeline] Running AI inference for {N} windows...")
        speeds_ms = self.predict_velocity(imu_windows)
        quats_rel = self.predict_attitude(imu_windows)

        # 2. Build pre-blackout road caches (leakage-free)
        ref_lat = float(gt_lats[0])
        ref_lon = float(gt_lons[0])
        t_0 = float(timestamps[0])

        road_caches: Dict[int, Optional[RoadPolyline]] = {}  # interval_idx -> road
        for iv_idx, (bo_start, bo_end) in enumerate(blackout_intervals):
            road = build_pre_blackout_road(
                gt_lats=gt_lats,
                gt_lons=gt_lons,
                timestamps=timestamps,
                blackout_start_s=bo_start,
                ref_lat=ref_lat,
                ref_lon=ref_lon,
                lookahead_window_s=self.lookahead_window_s,
                min_vertices=self.min_road_vertices,
            )
            road_caches[iv_idx] = road
            if road is not None:
                logger.info(
                    f"  [RoadCache] Interval {iv_idx} (t=[{bo_start:.1f},{bo_end:.1f}]): "
                    f"cached {len(road.vertices)} vertices."
                )
            else:
                logger.warning(
                    f"  [RoadCache] Interval {iv_idx} (t=[{bo_start:.1f},{bo_end:.1f}]): "
                    f"no road available (insufficient pre-blackout data)."
                )

        # 3. Blackout schedule
        schedule = GNSSBlackoutSchedule(intervals=blackout_intervals)

        # 4. Initialize ES-EKF tracker (identical to Version A)
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

        attitude_history: List[Tuple[float, np.ndarray]] = [
            (t_0, tracker.get_state()["quat"].copy())
        ]

        # 5. Per-blackout road match statistics accumulators
        stats_accum: List[Dict[str, Any]] = []
        for iv_idx, (bo_start, bo_end) in enumerate(blackout_intervals):
            road = road_caches[iv_idx]
            stats_accum.append({
                "bo_start": bo_start,
                "bo_end": bo_end,
                "road_active": road is not None,
                "n_steps": 0,
                "n_matched": 0,
                "n_rejected_distance": 0,
                "n_rejected_heading": 0,
                "corrections_m": [],
                "road_vertices": len(road.vertices) if road is not None else 0,
            })

        def _get_active_road(t: float) -> Tuple[Optional[RoadPolyline], int]:
            """Returns (road, interval_index) for the blackout containing t, or (None, -1)."""
            for iv_idx, (bo_start, bo_end) in enumerate(blackout_intervals):
                if bo_start <= t <= bo_end:
                    return road_caches[iv_idx], iv_idx
            return None, -1

        # 6. Main loop (identical structure to Version A, + road correction step)
        for i in range(1, N):
            t_curr = float(timestamps[i])
            t_prev = float(timestamps[i - 1])
            dt_step = t_curr - t_prev
            if dt_step <= 0.0:
                dt_step = 1.0

            is_bo = schedule.is_blackout(t_curr)

            # IMU propagation (same as Version A)
            window_imu = imu_windows[i]
            step_samples = window_imu[40:50]
            dt_sample = dt_step / 10.0

            for s in range(10):
                accel_s = step_samples[s, :3]
                gyro_s = step_samples[s, 3:]
                tracker.predict(dt=dt_sample, accel_b=accel_s, gyro_b=gyro_s)

            tracker._state.timestamp = t_curr

            # A. Velocity Update (same as Version A)
            if apply_velocity_update:
                speed_target = max(0.0, float(speeds_ms[i]))
                tracker.update_velocity(forward_speed_ms=speed_target, cov_speed=cov_speed)

            # B. NHC Update (same as Version A)
            if apply_nhc:
                tracker.update_nhc()

            # C. Relative Attitude Update (same as Version A)
            if apply_attitude_update:
                target_start_t = t_curr - 5.0
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

            attitude_history.append((t_curr, tracker.get_state()["quat"].copy()))

            # D. GNSS Position Update (suppressed during blackout) — same as Version A
            gt_e, gt_n = lat_lon_to_enu_m(float(gt_lats[i]), float(gt_lons[i]), ref_lat, ref_lon)
            gnss_pos_enu = [gt_e, gt_n, 0.0]
            tracker.update_gnss_position(
                pos_enu_meas=gnss_pos_enu,
                cov_pos=cov_gnss_pos,
                is_blackout=is_bo,
            )

            # E. ROAD CONSTRAINT UPDATE (Version B only)
            #    Applied AFTER GNSS (which is suppressed during blackout),
            #    so this only has effect when blackout is active.
            should_apply_road = is_bo if self.apply_road_during_blackout_only else True
            if should_apply_road:
                road, iv_idx = _get_active_road(t_curr)
                if iv_idx >= 0:
                    sa = stats_accum[iv_idx]
                    sa["n_steps"] += 1

                    if road is not None:
                        state = tracker.get_state()
                        pos_en = state["pos_enu"][:2]  # [East, North]
                        heading_deg = quat_to_heading_deg(state["quat"])

                        match: MapMatchResult = self.map_matcher.match(
                            estimated_pos=pos_en,
                            estimated_heading_deg=heading_deg,
                            road=road,
                        )

                        if match.matched:
                            sa["n_matched"] += 1
                            correction_dist = float(
                                np.linalg.norm(match.corrected_pos - pos_en)
                            )
                            sa["corrections_m"].append(correction_dist)

                            # Inject road correction as EKF position pseudo-measurement
                            # corrected_pos is a 2D [E, N] point; append Up=0
                            road_pos_enu = np.array([
                                match.corrected_pos[0],
                                match.corrected_pos[1],
                                state["pos_enu"][2],  # keep current Up
                            ], dtype=np.float64)
                            tracker.update_gnss_position(
                                pos_enu_meas=road_pos_enu,
                                cov_pos=self.road_cov_m2,
                                is_blackout=False,  # force update (not real GNSS)
                            )
                        else:
                            # Count rejection category
                            reason = match.rejection_reason.lower()
                            if "distance" in reason:
                                sa["n_rejected_distance"] += 1
                            elif "heading" in reason:
                                sa["n_rejected_heading"] += 1
                            # else: "no road" — counted elsewhere

            # Record trajectory point
            tracker._record_trajectory_point(t_curr, is_blackout=is_bo)

        # 7. Extract trajectory & compute errors
        estimated_traj = tracker.get_trajectory()

        errors_per_step = np.zeros(N, dtype=np.float64)
        for i in range(N):
            pt = estimated_traj[i]
            errors_per_step[i] = haversine_distance_m(
                pt.lat, pt.lon, float(gt_lats[i]), float(gt_lons[i])
            )

        # 8. Per-blackout metrics
        per_blackout_metrics: List[BlackoutMetrics] = []
        road_match_stats_list: List[RoadMatchStats] = []

        for iv_idx, (start_s, end_s) in enumerate(blackout_intervals):
            mask = (timestamps >= start_s) & (timestamps <= end_s)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            errors_in_bo = errors_per_step[indices]
            final_err = float(errors_in_bo[-1])
            max_err = float(errors_in_bo.max())
            mean_err = float(errors_in_bo.mean())
            median_err = float(np.median(errors_in_bo))
            rmse_err = float(np.sqrt(np.mean(errors_in_bo ** 2)))
            p90_err = float(np.percentile(errors_in_bo, 90))
            p95_err = float(np.percentile(errors_in_bo, 95))

            if len(indices) > 1:
                dt_sub = np.diff(timestamps[indices])
                v_sub = speeds_ms[indices[:-1]]
                dist_m = float(np.sum(v_sub * dt_sub))
            else:
                dist_m = 0.0

            relative_drift = float((final_err / dist_m) * 100.0) if dist_m > 1e-3 else 0.0

            # Error growth rate: linear slope of error vs time
            if len(indices) > 2:
                t_sub = timestamps[indices] - timestamps[indices[0]]
                slope, _ = np.polyfit(t_sub, errors_in_bo, 1)
                error_growth_rate = float(slope)  # m/s
            else:
                error_growth_rate = float("nan")

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

            # Road match statistics for this interval
            sa = stats_accum[iv_idx]
            corrections = sa["corrections_m"]
            avg_corr = float(np.mean(corrections)) if corrections else 0.0
            max_corr = float(np.max(corrections)) if corrections else 0.0
            n_steps = sa["n_steps"]
            n_matched = sa["n_matched"]
            match_frac = float(n_matched / n_steps) if n_steps > 0 else 0.0

            road_match_stats_list.append(
                RoadMatchStats(
                    blackout_start_s=start_s,
                    blackout_end_s=end_s,
                    road_active=sa["road_active"],
                    n_steps=n_steps,
                    n_matched=n_matched,
                    n_rejected_distance=sa["n_rejected_distance"],
                    n_rejected_heading=sa["n_rejected_heading"],
                    match_fraction=match_frac,
                    avg_correction_m=avg_corr,
                    max_correction_m=max_corr,
                    road_vertices=sa["road_vertices"],
                )
            )

        # 9. Aggregate metrics
        if per_blackout_metrics:
            mean_final = float(np.mean([m.final_error_m for m in per_blackout_metrics]))
            mean_max = float(np.mean([m.max_error_m for m in per_blackout_metrics]))
            mean_rmse = float(np.mean([m.rmse_error_m for m in per_blackout_metrics]))
            total_dist = float(np.sum([m.traveled_distance_m for m in per_blackout_metrics]))
            mean_drift = float(np.mean([m.relative_drift_percent for m in per_blackout_metrics]))
        else:
            mean_final = mean_max = mean_rmse = total_dist = mean_drift = 0.0

        # Aggregate road stats
        total_matched = sum(s.n_matched for s in road_match_stats_list)
        total_rej_dist = sum(s.n_rejected_distance for s in road_match_stats_list)
        total_rej_hdg = sum(s.n_rejected_heading for s in road_match_stats_list)
        total_steps = sum(s.n_steps for s in road_match_stats_list)
        overall_match_frac = float(total_matched / total_steps) if total_steps > 0 else 0.0
        all_corrs = []
        for sa in stats_accum:
            all_corrs.extend(sa["corrections_m"])
        overall_avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0

        return RoadBlackoutEvaluationResult(
            per_blackout_metrics=per_blackout_metrics,
            mean_final_error_m=mean_final,
            mean_max_error_m=mean_max,
            mean_rmse_error_m=mean_rmse,
            mean_relative_drift_percent=mean_drift,
            total_traveled_distance_m=total_dist,
            errors_per_step_m=errors_per_step,
            trajectory_estimated=estimated_traj,
            road_match_stats=road_match_stats_list,
            total_matched=total_matched,
            total_rejected_distance=total_rej_dist,
            total_rejected_heading=total_rej_hdg,
            overall_match_fraction=overall_match_frac,
            overall_avg_correction_m=overall_avg_corr,
        )
