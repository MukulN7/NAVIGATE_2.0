"""
test_map_matching.py — Unit tests for src/navigate/map_matching.py (Stage 18).

Coverage:
  - nearest-point projection geometry
  - correct segment selection (multi-segment polyline)
  - heading agreement (valid match)
  - heading rejection (invalid heading)
  - distance gating (too far from road)
  - correction behaviour (strength 0 / 0.5 / 1.0)
  - no-map / no-match graceful behaviour
  - finite / non-NaN outputs
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make src importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.map_matching import (
    MapMatchResult,
    MapMatcher,
    RoadPolyline,
    _heading_diff_deg,
    _project_point_onto_segment,
    _segment_heading_deg,
    match_position,
)


# ===========================================================================
# Helper: a simple straight East-going road from (0,0) to (100,0)
# ===========================================================================

@pytest.fixture
def east_road() -> RoadPolyline:
    """Straight road heading East (90 deg): [E,N] from (0,0) to (100,0)."""
    return RoadPolyline(vertices=np.array([[0.0, 0.0], [100.0, 0.0]]))


@pytest.fixture
def north_road() -> RoadPolyline:
    """Straight road heading North (0 deg): [E,N] from (0,0) to (0,100)."""
    return RoadPolyline(vertices=np.array([[0.0, 0.0], [0.0, 100.0]]))


@pytest.fixture
def multi_seg_road() -> RoadPolyline:
    """L-shaped road: first segment East (0,0)->(50,0), then North (50,0)->(50,50)."""
    return RoadPolyline(
        vertices=np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]]),
        name="L-road",
    )


@pytest.fixture
def matcher() -> MapMatcher:
    return MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.5)


# ===========================================================================
# 1. Geometry helpers
# ===========================================================================

class TestProjectPointOntoSegment:
    def test_midpoint_projection(self):
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([5.0, 3.0])
        proj, t = _project_point_onto_segment(p, a, b)
        np.testing.assert_allclose(proj, [5.0, 0.0], atol=1e-9)
        assert abs(t - 0.5) < 1e-9

    def test_before_start_clamps_to_a(self):
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([-5.0, 0.0])
        proj, t = _project_point_onto_segment(p, a, b)
        np.testing.assert_allclose(proj, [0.0, 0.0], atol=1e-9)
        assert t == 0.0

    def test_beyond_end_clamps_to_b(self):
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        p = np.array([15.0, 0.0])
        proj, t = _project_point_onto_segment(p, a, b)
        np.testing.assert_allclose(proj, [10.0, 0.0], atol=1e-9)
        assert t == 1.0

    def test_degenerate_segment_returns_a(self):
        a = np.array([5.0, 5.0])
        b = np.array([5.0, 5.0])
        p = np.array([10.0, 10.0])
        proj, t = _project_point_onto_segment(p, a, b)
        np.testing.assert_allclose(proj, [5.0, 5.0], atol=1e-9)
        assert t == 0.0


class TestSegmentHeading:
    def test_north(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 10.0])
        assert abs(_segment_heading_deg(a, b) - 0.0) < 1e-6

    def test_east(self):
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        assert abs(_segment_heading_deg(a, b) - 90.0) < 1e-6

    def test_south(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, -10.0])
        assert abs(_segment_heading_deg(a, b) - 180.0) < 1e-6

    def test_west(self):
        a = np.array([0.0, 0.0])
        b = np.array([-10.0, 0.0])
        assert abs(_segment_heading_deg(a, b) - 270.0) < 1e-6


class TestHeadingDiff:
    def test_same_heading_zero_diff(self):
        assert _heading_diff_deg(45.0, 45.0) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_heading_bidirectional_zero(self):
        # 0 deg and 180 deg are opposite — bidirectional diff = 0
        assert _heading_diff_deg(0.0, 180.0) == pytest.approx(0.0, abs=1e-6)

    def test_90_deg_diff(self):
        assert _heading_diff_deg(0.0, 90.0) == pytest.approx(90.0, abs=1e-6)

    def test_wrap_around(self):
        assert _heading_diff_deg(350.0, 10.0) == pytest.approx(20.0, abs=1e-6)

    def test_result_always_non_negative(self):
        for h1 in range(0, 360, 15):
            for h2 in range(0, 360, 15):
                assert _heading_diff_deg(float(h1), float(h2)) >= 0.0


# ===========================================================================
# 2. MapMatcher — nearest-point projection
# ===========================================================================

class TestNearestPointProjection:
    def test_projects_onto_east_road(self, east_road, matcher):
        # Point directly above midpoint of road
        pos = np.array([50.0, 5.0])
        result = matcher.match(pos, 90.0, east_road)
        assert result.matched
        np.testing.assert_allclose(result.projected_pos, [50.0, 0.0], atol=1e-6)

    def test_projects_onto_north_road(self, north_road, matcher):
        pos = np.array([5.0, 50.0])
        result = matcher.match(pos, 0.0, north_road)
        assert result.matched
        np.testing.assert_allclose(result.projected_pos, [0.0, 50.0], atol=1e-6)


# ===========================================================================
# 3. Correct segment selection on multi-segment polyline
# ===========================================================================

class TestSegmentSelection:
    def test_selects_first_segment(self, multi_seg_road, matcher):
        # Point near midpoint of first (East) segment
        pos = np.array([25.0, 5.0])
        result = matcher.match(pos, 90.0, multi_seg_road)
        assert result.matched
        assert result.segment_idx == 0

    def test_selects_second_segment(self, multi_seg_road, matcher):
        # Point near midpoint of second (North) segment
        pos = np.array([55.0, 25.0])
        result = matcher.match(pos, 0.0, multi_seg_road)
        assert result.matched
        assert result.segment_idx == 1


# ===========================================================================
# 4. Heading agreement — valid match
# ===========================================================================

class TestHeadingAgreement:
    def test_exact_heading_match(self, east_road, matcher):
        pos = np.array([50.0, 5.0])
        result = matcher.match(pos, 90.0, east_road)
        assert result.matched
        assert result.heading_diff_deg == pytest.approx(0.0, abs=1e-5)

    def test_reverse_heading_bidirectional_accepted(self, east_road, matcher):
        # Travelling West (270 deg) on an East road — bidirectional OK
        pos = np.array([50.0, 5.0])
        result = matcher.match(pos, 270.0, east_road)
        assert result.matched

    def test_heading_within_tolerance_accepted(self, east_road, matcher):
        # 20 deg off, tolerance is 30 deg
        pos = np.array([50.0, 5.0])
        result = matcher.match(pos, 90.0 + 20.0, east_road)
        assert result.matched

    def test_heading_just_within_tolerance(self, east_road):
        m = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.0)
        pos = np.array([50.0, 5.0])
        result = m.match(pos, 90.0 + 29.9, east_road)
        assert result.matched


# ===========================================================================
# 5. Heading rejection
# ===========================================================================

class TestHeadingRejection:
    def test_perpendicular_heading_rejected(self, east_road, matcher):
        # North (0 deg) perpendicular to East road (90 deg) — diff = 90 > 30 threshold
        pos = np.array([50.0, 5.0])
        result = matcher.match(pos, 0.0, east_road)
        assert not result.matched
        assert "heading" in result.rejection_reason.lower()

    def test_heading_just_outside_tolerance_rejected(self, east_road):
        m = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        pos = np.array([50.0, 5.0])
        result = m.match(pos, 90.0 + 31.0, east_road)
        assert not result.matched
        assert "heading" in result.rejection_reason.lower()


# ===========================================================================
# 6. Distance gating
# ===========================================================================

class TestDistanceGating:
    def test_position_within_gate_matches(self, east_road):
        m = MapMatcher(max_match_dist_m=10.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        pos = np.array([50.0, 8.0])    # 8m from road
        assert m.match(pos, 90.0, east_road).matched

    def test_position_outside_gate_rejected(self, east_road):
        m = MapMatcher(max_match_dist_m=10.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        pos = np.array([50.0, 15.0])   # 15m from road > 10m gate
        result = m.match(pos, 90.0, east_road)
        assert not result.matched
        assert "distance" in result.rejection_reason.lower()

    def test_exact_gate_boundary(self, east_road):
        # Position exactly on the boundary should pass (<=)
        m = MapMatcher(max_match_dist_m=5.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        pos = np.array([50.0, 5.0])    # exactly 5m from road
        result = m.match(pos, 90.0, east_road)
        assert result.matched

    def test_report_distance_is_accurate(self, east_road, matcher):
        pos = np.array([50.0, 7.3])
        result = matcher.match(pos, 90.0, east_road)
        assert result.matched
        assert result.distance_to_road_m == pytest.approx(7.3, abs=1e-4)


# ===========================================================================
# 7. Correction behaviour
# ===========================================================================

class TestCorrectionBehaviour:
    def test_strength_zero_keeps_position(self, east_road):
        m = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.0)
        pos = np.array([50.0, 10.0])
        result = m.match(pos, 90.0, east_road)
        assert result.matched
        np.testing.assert_allclose(result.corrected_pos, pos, atol=1e-9)

    def test_strength_one_snaps_to_road(self, east_road):
        m = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=1.0)
        pos = np.array([50.0, 10.0])
        result = m.match(pos, 90.0, east_road)
        assert result.matched
        np.testing.assert_allclose(result.corrected_pos, result.projected_pos, atol=1e-9)

    def test_strength_half_interpolates(self, east_road):
        m = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.5)
        pos = np.array([50.0, 10.0])
        result = m.match(pos, 90.0, east_road)
        expected = np.array([50.0, 5.0])   # midpoint of pos and projected [50, 0]
        np.testing.assert_allclose(result.corrected_pos, expected, atol=1e-6)


# ===========================================================================
# 8. No-map / no-match graceful behaviour
# ===========================================================================

class TestNoMapBehaviour:
    def test_none_road_returns_unmatched(self, matcher):
        pos = np.array([10.0, 5.0])
        result = matcher.match(pos, 0.0, None)
        assert not result.matched
        assert "no road" in result.rejection_reason.lower()

    def test_unmatched_corrected_pos_equals_input(self, matcher):
        pos = np.array([10.0, 5.0])
        result = matcher.match(pos, 0.0, None)
        np.testing.assert_allclose(result.corrected_pos, pos, atol=1e-9)

    def test_unmatched_segment_idx_is_minus_one(self, matcher):
        result = matcher.match(np.array([0.0, 0.0]), 0.0, None)
        assert result.segment_idx == -1

    def test_unmatched_distance_is_inf(self, matcher):
        result = matcher.match(np.array([0.0, 0.0]), 0.0, None)
        assert math.isinf(result.distance_to_road_m)


# ===========================================================================
# 9. Finite / non-NaN outputs on matched result
# ===========================================================================

class TestFiniteOutputs:
    def test_matched_result_all_finite(self, east_road, matcher):
        pos = np.array([50.0, 5.0])
        result = matcher.match(pos, 90.0, east_road)
        assert result.matched
        assert np.all(np.isfinite(result.projected_pos))
        assert np.all(np.isfinite(result.corrected_pos))
        assert math.isfinite(result.road_heading_deg)
        assert math.isfinite(result.distance_to_road_m)
        assert math.isfinite(result.heading_diff_deg)

    def test_trajectory_match_returns_correct_length(self, east_road, matcher):
        T = 20
        positions = np.column_stack([
            np.linspace(0, 100, T),
            np.random.default_rng(42).uniform(-5, 5, T),
        ])
        headings = np.full(T, 90.0)
        results = matcher.match_trajectory(positions, headings, east_road)
        assert len(results) == T

    def test_match_position_convenience_function(self, east_road):
        pos = np.array([50.0, 3.0])
        result = match_position(pos, 90.0, east_road,
                                max_match_dist_m=20.0,
                                max_heading_diff_deg=30.0,
                                correction_strength=0.5)
        assert result.matched


# ===========================================================================
# 10. RoadPolyline validation
# ===========================================================================

class TestRoadPolylineValidation:
    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            RoadPolyline(vertices=np.array([1.0, 2.0, 3.0]))

    def test_single_vertex_raises(self):
        with pytest.raises(ValueError, match="2 vertices"):
            RoadPolyline(vertices=np.array([[1.0, 2.0]]))

    def test_invalid_correction_strength_raises(self):
        with pytest.raises(ValueError, match="correction_strength"):
            MapMatcher(correction_strength=1.5)

    def test_invalid_max_dist_raises(self):
        with pytest.raises(ValueError, match="max_match_dist_m"):
            MapMatcher(max_match_dist_m=-1.0)
