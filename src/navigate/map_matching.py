"""
map_matching.py — Lightweight Road / Map-Matching Layer for NAVIGATE 2.0

Accepts a road centerline (polyline) in local ENU metres and projects an
estimated position onto the closest road segment, gated by:
  1. Distance threshold  (max_match_dist_m)
  2. Heading consistency (max_heading_diff_deg)

When a valid match is found a weighted correction can be applied toward the
road centreline.  All behaviour degrades gracefully when no road polyline is
provided or when no segment passes the gates.

Design constraints:
  - Pure NumPy — no external map libraries required at runtime.
  - No internet access required.
  - Compatible with ENU coordinates produced by ai_iekf_pipeline.py.
  - Does NOT modify any trained model or EKF internals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class RoadPolyline:
    """
    A road centreline represented as an ordered sequence of [E, N] vertices
    in local ENU metres.

    Attributes
    ----------
    vertices : np.ndarray  shape (N, 2)  dtype float64
        Ordered [East, North] waypoints defining the road centreline.
    name : str
        Optional human-readable identifier used in debug output.
    """
    vertices: np.ndarray          # (N, 2) float64
    name: str = "road"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("RoadPolyline vertices must be shape (N, 2).")
        if len(self.vertices) < 2:
            raise ValueError("RoadPolyline requires at least 2 vertices.")


@dataclass
class MapMatchResult:
    """
    Result of a single map-matching query.

    Attributes
    ----------
    matched : bool
        True when a valid road match was found (distance + heading gates passed).
    projected_pos : np.ndarray  shape (2,)
        The nearest [E, N] point on the matched segment (or the original
        estimated position when matched=False).
    corrected_pos : np.ndarray  shape (2,)
        The corrected position after applying correction_strength toward the
        road centreline (or the original estimated position when matched=False).
    road_heading_deg : float
        Heading of the matched road segment [0, 360) measured clockwise from
        North in degrees (nan when no match).
    segment_idx : int
        Index of the matched segment in the polyline (-1 when no match).
    distance_to_road_m : float
        Perpendicular distance from estimated position to the projected point
        in metres (inf when no match).
    heading_diff_deg : float
        Absolute angular difference between estimated and road heading in degrees
        (nan when no match).
    rejection_reason : str
        Human-readable explanation when matched=False.
    """
    matched: bool
    projected_pos: np.ndarray          # (2,)
    corrected_pos: np.ndarray          # (2,)
    road_heading_deg: float
    segment_idx: int
    distance_to_road_m: float
    heading_diff_deg: float
    rejection_reason: str = ""

    def __repr__(self) -> str:
        if self.matched:
            return (
                f"MapMatchResult(matched=True, seg={self.segment_idx}, "
                f"dist={self.distance_to_road_m:.2f}m, "
                f"hdg_diff={self.heading_diff_deg:.1f}deg)"
            )
        return f"MapMatchResult(matched=False, reason='{self.rejection_reason}')"


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _project_point_onto_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Project point *p* onto the finite line segment [a, b].

    Returns
    -------
    proj : np.ndarray  shape (2,)
        Nearest point on [a, b] to *p*.
    t : float
        Parametric coordinate in [0, 1] along the segment.
    """
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq < 1e-12:
        # Degenerate segment — return start vertex
        return a.copy(), 0.0
    t = float(np.dot(p - a, ab)) / len_sq
    t = max(0.0, min(1.0, t))
    return a + t * ab, t


def _segment_heading_deg(a: np.ndarray, b: np.ndarray) -> float:
    """
    Bearing of road segment a->b in degrees [0, 360), measured clockwise
    from North in the ENU plane (i.e. atan2(dE, dN)).
    """
    de = float(b[0] - a[0])
    dn = float(b[1] - a[1])
    hdg = math.degrees(math.atan2(de, dn)) % 360.0
    return hdg


def _heading_diff_deg(h1: float, h2: float) -> float:
    """
    Minimum angular difference between two headings in degrees,
    accounting for both forward and reverse road alignment (bidirectional).
    Returns a value in [0, 90].
    """
    diff = abs(h1 - h2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    # For a two-way road h and h+180 are both valid
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


# ---------------------------------------------------------------------------
# MapMatcher — main public class
# ---------------------------------------------------------------------------

class MapMatcher:
    """
    Lightweight polyline-based map matcher operating in local ENU metres.

    Parameters
    ----------
    max_match_dist_m : float
        Maximum perpendicular distance (metres) allowed for a valid match.
        Positions farther than this are returned as unmatched.
    max_heading_diff_deg : float
        Maximum angular deviation (degrees) between the estimated heading
        and the road segment heading for a valid match.  Uses bidirectional
        road semantics, so the effective range is [0, 90].
    correction_strength : float in [0, 1]
        Linear interpolation weight toward the projected road point.
        0.0 -> no correction (keep estimated position).
        1.0 -> snap fully onto road centreline.
    """

    def __init__(
        self,
        max_match_dist_m: float = 20.0,
        max_heading_diff_deg: float = 30.0,
        correction_strength: float = 0.5,
    ) -> None:
        if not (0.0 <= correction_strength <= 1.0):
            raise ValueError("correction_strength must be in [0, 1].")
        if max_match_dist_m <= 0.0:
            raise ValueError("max_match_dist_m must be positive.")
        if not (0.0 < max_heading_diff_deg <= 90.0):
            raise ValueError("max_heading_diff_deg must be in (0, 90].")

        self.max_match_dist_m = float(max_match_dist_m)
        self.max_heading_diff_deg = float(max_heading_diff_deg)
        self.correction_strength = float(correction_strength)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        estimated_pos: np.ndarray,
        estimated_heading_deg: float,
        road: Optional[RoadPolyline],
    ) -> MapMatchResult:
        """
        Match *estimated_pos* against *road*.

        Parameters
        ----------
        estimated_pos : array-like  shape (2,)
            Current estimated [East, North] position in ENU metres.
        estimated_heading_deg : float
            Current estimated heading in degrees [0, 360), clockwise from North.
        road : RoadPolyline or None
            Road centreline to match against.  Pass None when no road geometry
            is available — the result will be unmatched.

        Returns
        -------
        MapMatchResult
        """
        pos = np.asarray(estimated_pos, dtype=np.float64).flatten()[:2]

        # ------ No road available ----------------------------------------
        if road is None:
            return self._no_match(pos, "no road polyline provided")

        verts = road.vertices
        n_segs = len(verts) - 1

        # ------ Find nearest segment -------------------------------------
        best_dist = math.inf
        best_proj = pos.copy()
        best_seg_idx = -1
        best_road_hdg = float("nan")
        best_hdg_diff = float("nan")

        for i in range(n_segs):
            a, b = verts[i], verts[i + 1]
            proj, _ = _project_point_onto_segment(pos, a, b)
            dist = float(np.linalg.norm(pos - proj))

            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                best_seg_idx = i
                best_road_hdg = _segment_heading_deg(a, b)
                best_hdg_diff = _heading_diff_deg(
                    estimated_heading_deg % 360.0, best_road_hdg
                )

        # ------ Distance gate --------------------------------------------
        if best_dist > self.max_match_dist_m:
            return self._no_match(
                pos,
                f"distance {best_dist:.1f}m exceeds max_match_dist_m "
                f"({self.max_match_dist_m:.1f}m)",
            )

        # ------ Heading gate ---------------------------------------------
        if best_hdg_diff > self.max_heading_diff_deg:
            return self._no_match(
                pos,
                f"heading diff {best_hdg_diff:.1f}deg exceeds max_heading_diff_deg "
                f"({self.max_heading_diff_deg:.1f}deg)",
            )

        # ------ Valid match — apply correction ---------------------------
        corrected = pos + self.correction_strength * (best_proj - pos)

        return MapMatchResult(
            matched=True,
            projected_pos=best_proj,
            corrected_pos=corrected,
            road_heading_deg=best_road_hdg,
            segment_idx=best_seg_idx,
            distance_to_road_m=best_dist,
            heading_diff_deg=best_hdg_diff,
            rejection_reason="",
        )

    def match_trajectory(
        self,
        positions: np.ndarray,
        headings_deg: np.ndarray,
        road: Optional[RoadPolyline],
    ) -> List[MapMatchResult]:
        """
        Apply map matching to a full trajectory.

        Parameters
        ----------
        positions : np.ndarray  shape (T, 2)
            Array of [E, N] estimated positions over time.
        headings_deg : np.ndarray  shape (T,)
            Estimated heading in degrees [0, 360) for each position.
        road : RoadPolyline or None

        Returns
        -------
        List[MapMatchResult]  length T
        """
        positions = np.asarray(positions, dtype=np.float64)
        headings_deg = np.asarray(headings_deg, dtype=np.float64).flatten()
        if positions.shape[0] != headings_deg.shape[0]:
            raise ValueError("positions and headings_deg must have the same length.")

        return [
            self.match(positions[i], float(headings_deg[i]), road)
            for i in range(len(positions))
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _no_match(self, pos: np.ndarray, reason: str) -> MapMatchResult:
        return MapMatchResult(
            matched=False,
            projected_pos=pos.copy(),
            corrected_pos=pos.copy(),
            road_heading_deg=float("nan"),
            segment_idx=-1,
            distance_to_road_m=math.inf,
            heading_diff_deg=float("nan"),
            rejection_reason=reason,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def match_position(
    estimated_pos: np.ndarray,
    estimated_heading_deg: float,
    road: Optional[RoadPolyline],
    max_match_dist_m: float = 20.0,
    max_heading_diff_deg: float = 30.0,
    correction_strength: float = 0.5,
) -> MapMatchResult:
    """
    One-shot convenience wrapper around MapMatcher.

    All parameters are forwarded to MapMatcher and MapMatcher.match.
    """
    matcher = MapMatcher(
        max_match_dist_m=max_match_dist_m,
        max_heading_diff_deg=max_heading_diff_deg,
        correction_strength=correction_strength,
    )
    return matcher.match(estimated_pos, estimated_heading_deg, road)
