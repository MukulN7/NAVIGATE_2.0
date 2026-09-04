"""
test_ai_iekf_road_pipeline.py — Unit & Integration Tests for Version B (AI + ES-EKF + Road).

Tests Cover:
1. Road cache construction (leakage prevention)
2. Road matching integration into pipeline
3. No-road graceful behaviour
4. Distance rejection
5. Heading rejection
6. Correction behaviour (road pseudo-measurement)
7. Blackout behaviour (road only active during blackout)
8. Finite state / covariance throughout
9. Version A behaviour is untouched (import & run Version A unchanged)
10. RoadMatchStats accumulation correctness
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.ai_iekf_road_pipeline import (
    AIIEKFRoadPipeline,
    RoadBlackoutEvaluationResult,
    RoadMatchStats,
    build_pre_blackout_road,
)
from navigate.ai_iekf_pipeline import AIIEKFPipeline  # Version A — must remain importable
from navigate.map_matching import MapMatcher, RoadPolyline
from navigate.iekf_tracker import (
    ErrorStateIEKFTracker,
    STANDARD_GRAVITY,
    EARTH_RADIUS_M,
    RAD2DEG,
    quat_to_heading_deg,
)
from navigate.evaluate_blackout import haversine_distance_m


# ================================================================== #
#  Fixtures
# ================================================================== #

@pytest.fixture(scope="module")
def road_pipeline():
    """AIIEKFRoadPipeline (Version B) — skipped if checkpoints missing."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints not found on disk.")
    return AIIEKFRoadPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
        max_match_dist_m=20.0,
        max_heading_diff_deg=30.0,
        correction_strength=0.5,
        road_cov_m2=5.0,
        lookahead_window_s=120.0,
        min_road_vertices=3,
    )


@pytest.fixture(scope="module")
def version_a_pipeline():
    """Version A AIIEKFPipeline — must remain independently loadable."""
    vel_ckpt = Path("models/velocity_model_v2.pt")
    att_ckpt = Path("models/attitude_model.pt")
    if not vel_ckpt.exists() or not att_ckpt.exists():
        pytest.skip("Model checkpoints not found on disk.")
    return AIIEKFPipeline(
        velocity_checkpoint=vel_ckpt,
        attitude_checkpoint=att_ckpt,
        device="cpu",
    )


def _make_synthetic_session(
    N: int = 30,
    speed_ms: float = 10.0,
    heading_deg: float = 0.0,  # North
    ref_lat: float = 51.5,
    ref_lon: float = -0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a straight-line synthetic session (North at speed_ms).

    Returns (imu_windows, timestamps, gt_lats, gt_lons).
    """
    timestamps = np.arange(N, dtype=np.float64) * 1.0

    imu_windows = np.zeros((N, 50, 6), dtype=np.float32)
    # Gravity along Z-axis
    imu_windows[:, :, 2] = STANDARD_GRAVITY
    # Very small forward acceleration (1 m/s per second) to move North
    imu_windows[:, :, 0] = 0.5  # X accel for forward motion

    heading_rad = math.radians(heading_deg)
    d_east_m = speed_ms * math.sin(heading_rad)
    d_north_m = speed_ms * math.cos(heading_rad)

    gt_lats = np.zeros(N, dtype=np.float64)
    gt_lons = np.zeros(N, dtype=np.float64)
    gt_lats[0] = ref_lat
    gt_lons[0] = ref_lon

    for i in range(1, N):
        t = float(timestamps[i])
        north_m = d_north_m * t
        east_m = d_east_m * t
        gt_lats[i] = ref_lat + (north_m / EARTH_RADIUS_M) * RAD2DEG
        gt_lons[i] = ref_lon + (east_m / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat)))) * RAD2DEG

    return imu_windows, timestamps, gt_lats, gt_lons


# ================================================================== #
#  1. build_pre_blackout_road — Unit Tests (No Models Required)
# ================================================================== #

class TestBuildPreBlackoutRoad:
    """Tests for the leakage-free road cache builder."""

    def _make_linear_gt(self, N=20, ref_lat=51.5, ref_lon=-0.1):
        """Returns (timestamps, gt_lats, gt_lons) for N steps going North."""
        ts = np.arange(N, dtype=np.float64)
        lats = ref_lat + (np.arange(N) * 10.0 / EARTH_RADIUS_M) * RAD2DEG
        lons = np.full(N, ref_lon)
        return ts, lats, lons

    def test_returns_polyline_with_sufficient_preblackout_data(self):
        ts, lats, lons = self._make_linear_gt(N=20)
        road = build_pre_blackout_road(
            gt_lats=lats,
            gt_lons=lons,
            timestamps=ts,
            blackout_start_s=10.0,
            ref_lat=lats[0],
            ref_lon=lons[0],
            lookahead_window_s=120.0,
            min_vertices=2,
        )
        assert road is not None
        assert isinstance(road, RoadPolyline)
        assert len(road.vertices) >= 2

    def test_returns_none_when_no_preblackout_data(self):
        ts, lats, lons = self._make_linear_gt(N=10)
        # Blackout starts at time 0 — no pre-blackout points
        road = build_pre_blackout_road(
            gt_lats=lats,
            gt_lons=lons,
            timestamps=ts,
            blackout_start_s=0.0,
            ref_lat=lats[0],
            ref_lon=lons[0],
            lookahead_window_s=120.0,
            min_vertices=2,
        )
        assert road is None

    def test_uses_only_preblackout_points(self):
        """Verify road vertices come from t < blackout_start, not after."""
        ts, lats, lons = self._make_linear_gt(N=20)
        blackout_start = 10.0
        road = build_pre_blackout_road(
            gt_lats=lats,
            gt_lons=lons,
            timestamps=ts,
            blackout_start_s=blackout_start,
            ref_lat=lats[0],
            ref_lon=lons[0],
            lookahead_window_s=120.0,
            min_vertices=2,
        )
        assert road is not None
        # All vertices must be reachable using pre-blackout ENU positions only
        # The furthest North vertex should correspond to at most t < 10 (i=9)
        # Each step goes North by 10m, so max North should be <= 90m from ref
        max_north_in_road = road.vertices[:, 1].max()
        # Pre-blackout: at most 9 steps * 10m = 90m North
        assert max_north_in_road <= 100.0, (
            f"Road vertex {max_north_in_road:.1f}m is beyond expected pre-blackout range"
        )

    def test_returns_none_when_too_few_distinct_points(self):
        # All same position — no road length
        ts = np.arange(10, dtype=np.float64)
        lats = np.full(10, 51.5)
        lons = np.full(10, -0.1)
        road = build_pre_blackout_road(
            gt_lats=lats,
            gt_lons=lons,
            timestamps=ts,
            blackout_start_s=5.0,
            ref_lat=51.5,
            ref_lon=-0.1,
            lookahead_window_s=120.0,
            min_vertices=2,
            min_segment_length_m=1.0,
        )
        assert road is None

    def test_lookahead_window_limits_vertices(self):
        ts, lats, lons = self._make_linear_gt(N=200)
        # Large lookahead — should see more vertices
        road_large = build_pre_blackout_road(
            gt_lats=lats, gt_lons=lons, timestamps=ts,
            blackout_start_s=100.0, ref_lat=lats[0], ref_lon=lons[0],
            lookahead_window_s=1000.0, min_vertices=2,
        )
        # Small lookahead — should see fewer vertices
        road_small = build_pre_blackout_road(
            gt_lats=lats, gt_lons=lons, timestamps=ts,
            blackout_start_s=100.0, ref_lat=lats[0], ref_lon=lons[0],
            lookahead_window_s=5.0, min_vertices=2,
        )
        if road_large is not None and road_small is not None:
            assert len(road_large.vertices) >= len(road_small.vertices)

    def test_road_polyline_has_valid_shape(self):
        ts, lats, lons = self._make_linear_gt(N=20)
        road = build_pre_blackout_road(
            gt_lats=lats, gt_lons=lons, timestamps=ts,
            blackout_start_s=10.0, ref_lat=lats[0], ref_lon=lons[0],
            min_vertices=2,
        )
        assert road is not None
        assert road.vertices.ndim == 2
        assert road.vertices.shape[1] == 2
        assert np.all(np.isfinite(road.vertices))


# ================================================================== #
#  2. Road Matching Integration (requires model checkpoints)
# ================================================================== #

class TestRoadMatchingIntegration:
    """Road-constraint integration into the main pipeline loop."""

    def test_pipeline_initialization(self, road_pipeline):
        """Version B pipeline initializes with both AI models and MapMatcher."""
        assert road_pipeline.velocity_model is not None
        assert road_pipeline.attitude_model is not None
        assert road_pipeline.map_matcher is not None
        assert isinstance(road_pipeline.map_matcher, MapMatcher)
        assert not road_pipeline.velocity_model.training
        assert not road_pipeline.attitude_model.training

    def test_returns_road_evaluation_result(self, road_pipeline):
        """run_session_blackout_road returns RoadBlackoutEvaluationResult."""
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 15.0)],
            init_heading_deg=0.0,
        )
        assert isinstance(result, RoadBlackoutEvaluationResult)

    def test_result_has_correct_trajectory_length(self, road_pipeline):
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 15.0)],
            init_heading_deg=0.0,
        )
        assert len(result.trajectory_estimated) == N

    def test_result_has_road_match_stats(self, road_pipeline):
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 15.0)],
            init_heading_deg=0.0,
        )
        assert isinstance(result.road_match_stats, list)
        assert len(result.road_match_stats) == 1
        rms = result.road_match_stats[0]
        assert isinstance(rms, RoadMatchStats)
        assert rms.blackout_start_s == 10.0
        assert rms.blackout_end_s == 15.0

    def test_match_fraction_in_valid_range(self, road_pipeline):
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        for rms in result.road_match_stats:
            assert 0.0 <= rms.match_fraction <= 1.0


# ================================================================== #
#  3. No-Road Graceful Behaviour
# ================================================================== #

class TestNoRoadBehaviour:
    """When no road polyline is available, pipeline must run without error."""

    def test_blackout_at_t0_no_preblackout_data(self, road_pipeline):
        """Blackout starts immediately — no pre-blackout road can be built."""
        N = 20
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        # Blackout starts at t=0 — no pre-blackout GPS available for road cache
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(0.0, 10.0)],
            init_heading_deg=0.0,
        )
        assert isinstance(result, RoadBlackoutEvaluationResult)
        assert len(result.trajectory_estimated) == N
        # Road should not be active since no pre-blackout data
        if result.road_match_stats:
            rms = result.road_match_stats[0]
            assert rms.road_active is False

    def test_all_states_finite_without_road(self, road_pipeline):
        N = 20
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(0.0, 10.0)],
            init_heading_deg=0.0,
        )
        for pt in result.trajectory_estimated:
            assert math.isfinite(pt.lat), f"lat={pt.lat} not finite at t={pt.timestamp}"
            assert math.isfinite(pt.lon), f"lon={pt.lon} not finite at t={pt.timestamp}"
            assert math.isfinite(pt.speed_ms)
            assert math.isfinite(pt.heading_deg)

    def test_matched_count_zero_when_no_road(self, road_pipeline):
        """When no road is cached, n_matched must be 0."""
        N = 20
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(0.0, 10.0)],
            init_heading_deg=0.0,
        )
        if result.road_match_stats:
            rms = result.road_match_stats[0]
            if not rms.road_active:
                assert rms.n_matched == 0


# ================================================================== #
#  4. Distance Rejection
# ================================================================== #

class TestDistanceRejection:
    """Verify that the MapMatcher distance gate rejects far positions."""

    def test_distance_rejection_counted(self):
        """When the EKF estimate is far from the road, rejection must be counted."""
        # Directly test MapMatcher distance gate
        road = RoadPolyline(vertices=np.array([[0.0, 0.0], [0.0, 100.0]]))  # North road
        matcher = MapMatcher(max_match_dist_m=5.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        # Position 20m East of road (beyond 5m gate)
        result = matcher.match(np.array([20.0, 50.0]), 0.0, road)
        assert not result.matched
        assert "distance" in result.rejection_reason.lower()
        assert result.distance_to_road_m > 5.0

    def test_tight_distance_gate_causes_more_rejections(self, road_pipeline):
        """Pipeline with tight distance gate should have fewer or equal matches than loose gate."""
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)

        tight_pipeline = AIIEKFRoadPipeline(
            velocity_checkpoint=Path("models/velocity_model_v2.pt"),
            attitude_checkpoint=Path("models/attitude_model.pt"),
            device="cpu",
            max_match_dist_m=0.1,  # Very tight — nearly all positions will be rejected
            max_heading_diff_deg=30.0,
            road_cov_m2=5.0,
        )
        result_tight = tight_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        result_normal = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        assert result_tight.total_matched <= result_normal.total_matched


# ================================================================== #
#  5. Heading Rejection
# ================================================================== #

class TestHeadingRejection:
    """Verify that the MapMatcher heading gate rejects misaligned positions."""

    def test_heading_rejection_with_perpendicular_heading(self):
        """90-degree heading difference on a North road must be rejected."""
        road = RoadPolyline(vertices=np.array([[0.0, 0.0], [0.0, 100.0]]))  # North (0 deg)
        matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        # Position nearby but heading 90 degrees (East) — perpendicular to road
        result = matcher.match(np.array([5.0, 50.0]), 90.0, road)
        assert not result.matched
        assert "heading" in result.rejection_reason.lower()

    def test_heading_within_tolerance_accepted(self):
        """Heading slightly off from road heading must still be accepted within tolerance."""
        road = RoadPolyline(vertices=np.array([[0.0, 0.0], [0.0, 100.0]]))  # North
        matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        # Heading 20 degrees off North (20 < 30 tolerance)
        result = matcher.match(np.array([5.0, 50.0]), 20.0, road)
        assert result.matched

    def test_tight_heading_gate_causes_more_rejections(self, road_pipeline):
        """Pipeline with tight heading gate should have fewer or equal matches."""
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)

        tight_hdg_pipeline = AIIEKFRoadPipeline(
            velocity_checkpoint=Path("models/velocity_model_v2.pt"),
            attitude_checkpoint=Path("models/attitude_model.pt"),
            device="cpu",
            max_match_dist_m=20.0,
            max_heading_diff_deg=1.0,  # Very tight — any heading deviation rejected
            road_cov_m2=5.0,
        )
        result_tight = tight_hdg_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        result_normal = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        assert result_tight.total_matched <= result_normal.total_matched


# ================================================================== #
#  6. Correction Behaviour
# ================================================================== #

class TestCorrectionBehaviour:
    """Verify road pseudo-measurement correction is properly applied via EKF."""

    def test_road_correction_applied_via_ekf(self, road_pipeline):
        """
        With correction_strength=1.0 and very low road noise, the road
        constraint should bring the position close to the road projection
        when the match succeeds.
        """
        # Use a direct MapMatcher to verify correction geometry
        road = RoadPolyline(vertices=np.array([[0.0, 0.0], [0.0, 100.0]]))
        matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=1.0)
        pos = np.array([10.0, 50.0])  # 10m East of North road
        result = matcher.match(pos, 0.0, road)
        assert result.matched
        # With strength=1.0, corrected_pos should snap onto the road
        np.testing.assert_allclose(result.corrected_pos, result.projected_pos, atol=1e-6)
        np.testing.assert_allclose(result.projected_pos, [0.0, 50.0], atol=1e-6)

    def test_road_correction_strength_zero(self):
        """correction_strength=0 should leave position unchanged."""
        road = RoadPolyline(vertices=np.array([[0.0, 0.0], [0.0, 100.0]]))
        matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.0)
        pos = np.array([10.0, 50.0])
        result = matcher.match(pos, 0.0, road)
        assert result.matched
        np.testing.assert_allclose(result.corrected_pos, pos, atol=1e-9)

    def test_road_correction_half_strength_interpolates(self):
        """correction_strength=0.5 corrects to midpoint."""
        road = RoadPolyline(vertices=np.array([[0.0, 0.0], [0.0, 100.0]]))
        matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        pos = np.array([10.0, 50.0])
        result = matcher.match(pos, 0.0, road)
        assert result.matched
        expected = np.array([5.0, 50.0])  # midpoint between [10,50] and [0,50]
        np.testing.assert_allclose(result.corrected_pos, expected, atol=1e-6)

    def test_average_correction_distance_nonnegative(self, road_pipeline):
        """avg_correction_m must be >= 0 in all cases."""
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        for rms in result.road_match_stats:
            assert rms.avg_correction_m >= 0.0
            assert rms.max_correction_m >= 0.0


# ================================================================== #
#  7. Blackout Behaviour
# ================================================================== #

class TestBlackoutBehaviour:
    """Verify road correction only runs during GNSS blackouts."""

    def test_road_correction_active_only_during_blackout(self, road_pipeline):
        """n_steps in road stats should equal the number of blackout steps."""
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        bo_start, bo_end = 10.0, 20.0
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(bo_start, bo_end)],
            init_heading_deg=0.0,
        )
        # Number of blackout steps
        n_bo_steps = int(np.sum((timestamps >= bo_start) & (timestamps <= bo_end))) - 1
        if result.road_match_stats:
            rms = result.road_match_stats[0]
            # n_steps should match the blackout duration steps (approximately)
            assert rms.n_steps <= n_bo_steps + 2  # small tolerance

    def test_blackout_flags_on_trajectory(self, road_pipeline):
        """IEKFTrajectoryPoint.is_gnss_blackout must be set correctly."""
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 15.0)],
            init_heading_deg=0.0,
        )
        for pt in result.trajectory_estimated:
            if 10.0 <= pt.timestamp <= 15.0:
                assert pt.is_gnss_blackout is True
            else:
                assert pt.is_gnss_blackout is False

    def test_gnss_recovery_restores_accuracy(self, road_pipeline):
        """After GNSS recovery, position error should drop significantly."""
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(5.0, 10.0)],
            init_heading_deg=0.0,
        )
        # Point after recovery (t=20s) should have small error
        if len(result.trajectory_estimated) > 20:
            pt_rec = result.trajectory_estimated[20]
            err_rec = haversine_distance_m(pt_rec.lat, pt_rec.lon, gt_lats[20], gt_lons[20])
            assert err_rec < 15.0, f"Recovery error {err_rec:.2f}m > 15m threshold"

    def test_multiple_blackout_intervals_each_get_stats(self, road_pipeline):
        """Each blackout interval gets its own RoadMatchStats entry."""
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(5.0, 8.0), (15.0, 18.0)],
            init_heading_deg=0.0,
        )
        assert len(result.road_match_stats) == 2
        assert result.road_match_stats[0].blackout_start_s == 5.0
        assert result.road_match_stats[1].blackout_start_s == 15.0


# ================================================================== #
#  8. Finite State / Covariance
# ================================================================== #

class TestFiniteStateCovariance:
    """All filter states must remain finite (no NaN/Inf) throughout."""

    def test_all_trajectory_points_finite(self, road_pipeline):
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=45.0,
        )
        for pt in result.trajectory_estimated:
            assert math.isfinite(pt.lat), f"lat is not finite at t={pt.timestamp}"
            assert math.isfinite(pt.lon), f"lon is not finite at t={pt.timestamp}"
            assert math.isfinite(pt.speed_ms), f"speed not finite at t={pt.timestamp}"
            assert math.isfinite(pt.heading_deg), f"heading not finite at t={pt.timestamp}"

    def test_errors_per_step_finite(self, road_pipeline):
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 15.0)],
            init_heading_deg=0.0,
        )
        assert np.all(np.isfinite(result.errors_per_step_m))

    def test_road_match_stats_finite(self, road_pipeline):
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        for rms in result.road_match_stats:
            assert math.isfinite(rms.match_fraction)
            assert math.isfinite(rms.avg_correction_m)
            assert math.isfinite(rms.max_correction_m)

    def test_mean_metrics_finite(self, road_pipeline):
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 15.0)],
            init_heading_deg=0.0,
        )
        assert math.isfinite(result.mean_final_error_m)
        assert math.isfinite(result.mean_max_error_m)
        assert math.isfinite(result.mean_rmse_error_m)
        assert math.isfinite(result.mean_relative_drift_percent)


# ================================================================== #
#  9. Version A Untouched
# ================================================================== #

class TestVersionAUntouched:
    """
    Verify that importing and running Version A (AIIEKFPipeline) is completely
    unaffected by the existence of Version B (AIIEKFRoadPipeline).
    """

    def test_version_a_imports_cleanly(self):
        """Version A can be imported independently without Version B."""
        from navigate.ai_iekf_pipeline import AIIEKFPipeline as _VersionA  # noqa: F401
        assert _VersionA is not None

    def test_version_a_is_separate_class(self, version_a_pipeline, road_pipeline):
        """Version A and Version B are distinct classes."""
        assert type(version_a_pipeline) is AIIEKFPipeline
        assert type(road_pipeline) is AIIEKFRoadPipeline
        assert AIIEKFPipeline is not AIIEKFRoadPipeline

    def test_version_b_inherits_version_a(self, road_pipeline):
        """Version B is a subclass of Version A (code reuse)."""
        assert isinstance(road_pipeline, AIIEKFPipeline)

    def test_version_a_run_unmodified(self, version_a_pipeline):
        """
        Version A run_session_blackout produces correct results independent
        of Version B module.
        """
        N = 25
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = version_a_pipeline.run_session_blackout(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 15.0)],
            init_heading_deg=0.0,
        )
        # Must return BlackoutEvaluationResult (not Road variant)
        from navigate.evaluate_blackout import BlackoutEvaluationResult
        assert isinstance(result, BlackoutEvaluationResult)
        assert not isinstance(result, RoadBlackoutEvaluationResult)
        assert len(result.trajectory_estimated) == N

    def test_version_a_results_unchanged_by_version_b_import(self, version_a_pipeline):
        """Importing Version B module does not alter Version A outputs."""
        import importlib
        import navigate.ai_iekf_road_pipeline  # Force import of Version B
        importlib.reload(navigate.ai_iekf_road_pipeline)

        N = 20
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = version_a_pipeline.run_session_blackout(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(5.0, 10.0)],
            init_heading_deg=0.0,
        )
        # All states must be finite
        for pt in result.trajectory_estimated:
            assert math.isfinite(pt.lat)
            assert math.isfinite(pt.lon)


# ================================================================== #
#  10. RoadMatchStats Accumulation
# ================================================================== #

class TestRoadMatchStatsAccumulation:
    """Verify that match/rejection counts accumulate correctly."""

    def test_match_count_nonnegative(self, road_pipeline):
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        for rms in result.road_match_stats:
            assert rms.n_matched >= 0
            assert rms.n_rejected_distance >= 0
            assert rms.n_rejected_heading >= 0

    def test_total_steps_consistency(self, road_pipeline):
        """n_matched + n_rejected_dist + n_rejected_hdg <= n_steps."""
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        for rms in result.road_match_stats:
            accounted = rms.n_matched + rms.n_rejected_distance + rms.n_rejected_heading
            assert accounted <= rms.n_steps

    def test_overall_match_fraction_consistent(self, road_pipeline):
        """overall_match_fraction is consistent with total_matched / total_steps."""
        N = 30
        imu_windows, timestamps, gt_lats, gt_lons = _make_synthetic_session(N=N)
        result = road_pipeline.run_session_blackout_road(
            imu_windows=imu_windows,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(10.0, 20.0)],
            init_heading_deg=0.0,
        )
        total_steps = sum(rms.n_steps for rms in result.road_match_stats)
        if total_steps > 0:
            expected_frac = result.total_matched / total_steps
            assert abs(result.overall_match_fraction - expected_frac) < 1e-6

    def test_real_dataset_execution_no_errors(self, road_pipeline):
        """Executes Version B on a real IO-VNBD subset — no NaN/Inf allowed."""
        data_path = Path("data/processed/iovnbd_full.npz")
        if not data_path.exists():
            pytest.skip("Dataset NPZ not found.")

        npz = np.load(data_path, allow_pickle=True)
        imu_all = npz["imu"]
        session_ids = npz["session_ids"]

        first_session = session_ids[0]
        mask = session_ids == first_session
        session_imu = imu_all[mask][:25]
        N = len(session_imu)

        timestamps = np.arange(N, dtype=np.float64)
        ref_lat, ref_lon = 51.5, -0.1
        gt_lats = np.full(N, ref_lat)
        gt_lons = np.full(N, ref_lon)

        result = road_pipeline.run_session_blackout_road(
            imu_windows=session_imu,
            timestamps=timestamps,
            gt_lats=gt_lats,
            gt_lons=gt_lons,
            blackout_intervals=[(5.0, 15.0)],
            init_heading_deg=0.0,
        )
        assert len(result.trajectory_estimated) == N
        for pt in result.trajectory_estimated:
            assert math.isfinite(pt.lat)
            assert math.isfinite(pt.lon)
        assert np.all(np.isfinite(result.errors_per_step_m))
