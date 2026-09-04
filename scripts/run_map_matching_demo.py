"""
run_map_matching_demo.py — Stage 18 demonstration of map matching for NAVIGATE 2.0.

This script demonstrates the MapMatcher API on a synthetic scenario that mirrors
realistic vehicle navigation conditions.  It does NOT require any internet
connection, road database, or pretrained model.

Real-road evaluation status:
  No road geometry (GeoJSON, OSM, shapefile, or GPX) was found in the local
  IO-VNBD dataset or the NAVIGATE 2.0 results directory.  Therefore:
    "Map-matching algorithm implemented, but real-road accuracy not evaluated
     because no road geometry is available."
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.map_matching import MapMatcher, RoadPolyline, match_position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_result(label: str, result) -> None:
    status = "MATCHED  " if result.matched else "NO MATCH "
    if result.matched:
        print(
            f"  [{status}] {label:40s}  "
            f"dist={result.distance_to_road_m:6.2f}m  "
            f"hdg_diff={result.heading_diff_deg:5.1f}deg  "
            f"proj={result.projected_pos}  "
            f"corrected={result.corrected_pos}"
        )
    else:
        print(f"  [{status}] {label:40s}  reason: {result.rejection_reason}")


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def demo_basic_match() -> None:
    """Scenario 1: vehicle driving East, near a straight East-going road."""
    _print_header("Scenario 1: Straight East-going road — basic match")

    road = RoadPolyline(
        vertices=np.array([[0.0, 0.0], [500.0, 0.0]]),
        name="East Highway"
    )
    matcher = MapMatcher(max_match_dist_m=25.0, max_heading_diff_deg=30.0, correction_strength=0.5)

    cases = [
        ("On-road, correct heading",   np.array([250.0,  0.0]),  90.0),
        ("5m N of road, correct hdg",  np.array([250.0,  5.0]),  90.0),
        ("15m N of road, correct hdg", np.array([250.0, 15.0]),  90.0),
        ("30m N of road (too far)",    np.array([250.0, 30.0]),  90.0),
        ("Correct dist, wrong heading",np.array([250.0,  5.0]),   0.0),
        ("Travelling West (reverse OK)",np.array([250.0,  5.0]), 270.0),
    ]

    for label, pos, hdg in cases:
        _print_result(label, matcher.match(pos, hdg, road))


def demo_multi_segment_road() -> None:
    """Scenario 2: L-shaped road — segment selection test."""
    _print_header("Scenario 2: L-shaped road — correct segment selection")

    road = RoadPolyline(
        vertices=np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0]]),
        name="L-Road"
    )
    matcher = MapMatcher(max_match_dist_m=25.0, max_heading_diff_deg=30.0, correction_strength=0.5)

    cases = [
        ("Near seg-0 (East leg)",  np.array([100.0, 10.0]), 90.0),
        ("Near seg-1 (North leg)", np.array([190.0, 100.0]), 0.0),
        ("At corner, East heading", np.array([210.0, 10.0]), 90.0),
    ]

    for label, pos, hdg in cases:
        result = matcher.match(pos, hdg, road)
        if result.matched:
            print(
                f"  [MATCHED  ] {label:40s}  seg={result.segment_idx}  "
                f"dist={result.distance_to_road_m:.2f}m"
            )
        else:
            print(f"  [NO MATCH ] {label:40s}  reason: {result.rejection_reason}")


def demo_correction_strengths() -> None:
    """Scenario 3: Effect of different correction_strength values."""
    _print_header("Scenario 3: Correction strength comparison (0, 0.5, 1.0)")

    road = RoadPolyline(
        vertices=np.array([[0.0, 0.0], [100.0, 0.0]]),
        name="North-offset road"
    )
    pos = np.array([50.0, 12.0])  # 12m north of road
    hdg = 90.0

    print(f"\n  Estimated position : {pos}  (12m north of the road)")
    print(f"  Projected position :  [50.0, 0.0]  (nearest road point)")
    print()

    for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = match_position(
            pos, hdg, road,
            max_match_dist_m=20.0,
            max_heading_diff_deg=30.0,
            correction_strength=strength
        )
        print(
            f"  strength={strength:.2f}  matched={result.matched}  "
            f"corrected_pos={np.round(result.corrected_pos, 3)}"
        )


def demo_no_road_graceful() -> None:
    """Scenario 4: No road provided — graceful no-match."""
    _print_header("Scenario 4: No road available — graceful no-match")

    matcher = MapMatcher()
    pos = np.array([123.0, 456.0])
    result = matcher.match(pos, 45.0, road=None)

    print(f"  matched          : {result.matched}")
    print(f"  corrected_pos    : {result.corrected_pos}  (unchanged from input)")
    print(f"  rejection_reason : {result.rejection_reason}")
    assert not result.matched
    assert np.allclose(result.corrected_pos, pos)
    print("  [OK] corrected_pos equals estimated_pos when no road is provided.")


def demo_trajectory() -> None:
    """Scenario 5: Trajectory-level map matching over 50 time steps."""
    _print_header("Scenario 5: Trajectory matching (50 steps, straight road)")

    road = RoadPolyline(
        vertices=np.array([[0.0, 0.0], [500.0, 0.0]]),
        name="Straight road"
    )
    matcher = MapMatcher(max_match_dist_m=20.0, max_heading_diff_deg=30.0, correction_strength=0.5)

    rng = np.random.default_rng(seed=0)
    T = 50
    # Vehicle travels East with small lateral noise
    east_positions = np.linspace(0, 490, T)
    lateral_noise = rng.normal(0, 5, T)    # sigma=5m lateral error
    positions = np.column_stack([east_positions, lateral_noise])
    headings = np.full(T, 90.0)            # all heading East

    results = matcher.match_trajectory(positions, headings, road)

    n_matched = sum(r.matched for r in results)
    dists = [r.distance_to_road_m for r in results if r.matched]
    corrections = [
        float(np.linalg.norm(r.corrected_pos - positions[i]))
        for i, r in enumerate(results)
        if r.matched
    ]

    print(f"  Timesteps       : {T}")
    print(f"  Matched         : {n_matched} / {T}")
    print(f"  Mean dist to road (matched): {np.mean(dists):.2f}m")
    print(f"  Mean correction applied    : {np.mean(corrections):.2f}m")
    assert n_matched > 0, "Expected at least some matches in this demo."
    print("  [OK] Trajectory matching completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 70)
    print("  NAVIGATE 2.0 — Stage 18: Map Matching Demo")
    print("=" * 70)
    print()
    print("  Real-road evaluation status:")
    print()
    print("    Map-matching algorithm implemented, but real-road accuracy")
    print("    not evaluated because no road geometry is available.")
    print()
    print("  (No GeoJSON / OSM / shapefile / GPX road data was found in the")
    print("   IO-VNBD dataset or results/ directory.  All demonstrations")
    print("   below use synthetic road centrelines.)")

    demo_basic_match()
    demo_multi_segment_road()
    demo_correction_strengths()
    demo_no_road_graceful()
    demo_trajectory()

    print()
    print("=" * 70)
    print("  All Stage 18 demo scenarios completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
