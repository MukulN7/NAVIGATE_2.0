"""
run_ai_iekf_road_evaluation.py — Stage 19: Version B (AI + ES-EKF + Road) Evaluation.

Evaluates NAVIGATE 2.0 Version B against Version A and baseline dead-reckoning
across held-out IO-VNBD test sessions S-M, S-Vfa01, S-Vw2 with 5/10/30/60 s
GNSS blackout durations.

Road Reference Method (Leakage-Free):
  - GPS observations strictly BEFORE each blackout interval are cached as a
    road polyline (up to lookahead_window_s seconds prior to blackout start).
  - No future GPS positions are used during the blackout.
  - Road constraint is applied as an EKF position pseudo-measurement with
    configurable noise (road_cov_m2).

Results Written:
  results/ai_iekf_road/road_comparison_results.json
  results/ai_iekf_road/road_comparison_results.csv

Three-Way Comparison:
  1. Baseline Dead-Reckoning   (from results/baseline_blackout_results_v2.json)
  2. Version A: AI + ES-EKF   (from results/ai_iekf_blackout_results.json)
  3. Version B: AI + ES-EKF + Road  (computed here)
"""

import argparse
import csv
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root/src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from navigate.ai_iekf_road_pipeline import AIIEKFRoadPipeline, RoadBlackoutEvaluationResult, RoadMatchStats
from navigate.evaluate_blackout import BlackoutMetrics

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("road_eval")


# ================================================================== #
#  Ground Truth Loader (shared with Version A eval script)
# ================================================================== #

def load_vehicle_ground_truth(
    base_dir: Path,
    session_id: str,
    num_windows: int,
    stride: int = 10,
    window_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads row-synchronized vehicle ground truth (lat, lon, heading) for a session.
    Matches window end sample index = i * stride + window_size - 1.
    (Identical to the function in run_ai_iekf_evaluation.py — no modification.)
    """
    v_name = f"V-{session_id[2:]}.csv"
    matches = list(base_dir.rglob(v_name))
    if not matches:
        raise FileNotFoundError(
            f"Vehicle GT file not found for session {session_id} ({v_name})"
        )

    v_path = matches[0]
    lats, lons, hdgs = [], [], []
    with open(v_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = [c.strip().upper() for c in next(reader)]
        lat_col = next(i for i, h in enumerate(header) if "LATITUDE" in h)
        lon_col = next(i for i, h in enumerate(header) if "LONGITUDE" in h)
        hdg_col = next(i for i, h in enumerate(header) if "HEADING" in h)
        for row in reader:
            try:
                lats.append(float(row[lat_col]))
                lons.append(float(row[lon_col]))
                hdgs.append(float(row[hdg_col]))
            except (ValueError, IndexError):
                continue

    lats_arr = np.array(lats, dtype=np.float64)
    lons_arr = np.array(lons, dtype=np.float64)
    hdgs_arr = np.array(hdgs, dtype=np.float64)

    sample_indices = np.array(
        [i * stride + window_size - 1 for i in range(num_windows)], dtype=int
    )
    sample_indices = np.clip(sample_indices, 0, len(lats_arr) - 1)
    return lats_arr[sample_indices], lons_arr[sample_indices], hdgs_arr[sample_indices]


def generate_deterministic_blackouts(
    total_duration_s: float,
    blackout_duration_s: float,
    num_intervals: int = 4,
) -> List[Tuple[float, float]]:
    """
    Generates reproducible blackout intervals (identical to Version A evaluator).
    """
    if total_duration_s <= blackout_duration_s + 10.0:
        start = max(1.0, float(int((total_duration_s - blackout_duration_s) / 2.0)))
        return [(start, start + blackout_duration_s)]

    fractions = [0.2, 0.4, 0.6, 0.8][:num_intervals]
    intervals = []
    for frac in fractions:
        start = float(int(frac * total_duration_s))
        end = start + blackout_duration_s
        if end < total_duration_s - 2.0:
            intervals.append((start, end))
    return intervals


# ================================================================== #
#  Additional Metric Helpers
# ================================================================== #

def compute_extended_metrics(
    errors_m: np.ndarray,
    durations: np.ndarray,
    speeds_ms: np.ndarray,
) -> Dict[str, float]:
    """
    Computes the full set of per-blackout position metrics.

    Parameters
    ----------
    errors_m : [K] position errors during blackout (metres)
    durations : [K] elapsed time from blackout start for each point (seconds)
    speeds_ms : [K-1] speeds used for distance integration

    Returns
    -------
    dict with final/max/mean/median/rmse/p90/p95/drift metrics.
    """
    if len(errors_m) == 0:
        return {}

    dist_m = float(np.sum(speeds_ms * np.diff(durations))) if len(durations) > 1 else 0.0
    final_err = float(errors_m[-1])
    rel_drift = float((final_err / dist_m) * 100.0) if dist_m > 1e-3 else 0.0

    # Error growth rate via linear fit
    if len(durations) > 2:
        t_rel = durations - durations[0]
        slope, _ = np.polyfit(t_rel, errors_m, 1)
        growth_rate = float(slope)
    else:
        growth_rate = float("nan")

    return {
        "final_error_m": round(final_err, 3),
        "max_error_m": round(float(errors_m.max()), 3),
        "mean_error_m": round(float(errors_m.mean()), 3),
        "median_error_m": round(float(np.median(errors_m)), 3),
        "rmse_error_m": round(float(np.sqrt(np.mean(errors_m ** 2))), 3),
        "p90_error_m": round(float(np.percentile(errors_m, 90)), 3),
        "p95_error_m": round(float(np.percentile(errors_m, 95)), 3),
        "traveled_distance_m": round(dist_m, 3),
        "relative_drift_percent": round(rel_drift, 3),
        "error_growth_rate_ms": round(growth_rate, 4) if not np.isnan(growth_rate) else None,
    }


# ================================================================== #
#  Main Evaluation
# ================================================================== #

def run_road_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    start_time = time.time()

    # Validate inputs
    data_path = Path(args.data)
    vel_ckpt_path = Path(args.velocity_checkpoint)
    att_ckpt_path = Path(args.attitude_checkpoint)
    raw_dataset_dir = Path(args.dataset_raw)

    for p in [data_path, vel_ckpt_path, att_ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    logger.info("=" * 70)
    logger.info("NAVIGATE 2.0 — Stage 19: Version B (AI + ES-EKF + Road) Evaluation")
    logger.info("=" * 70)

    # Instantiate Version B Pipeline
    pipeline = AIIEKFRoadPipeline(
        velocity_checkpoint=vel_ckpt_path,
        attitude_checkpoint=att_ckpt_path,
        device=args.device,
        max_match_dist_m=args.max_match_dist_m,
        max_heading_diff_deg=args.max_heading_diff_deg,
        correction_strength=args.correction_strength,
        road_cov_m2=args.road_cov_m2,
        lookahead_window_s=args.lookahead_window_s,
        min_road_vertices=3,
    )

    # Load processed NPZ dataset
    logger.info(f"Loading dataset: {data_path}")
    npz = np.load(data_path, allow_pickle=True)
    imu_all = npz["imu"]
    session_ids_all = npz["session_ids"]

    all_sessions = sorted(set(session_ids_all))
    if args.smoke:
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions][:1]
        if not test_sessions:
            test_sessions = all_sessions[:1]
        logger.info(f"[SMOKE] Testing 1 session: {test_sessions}")
    else:
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions]
        if not test_sessions:
            logger.warning("Held-out sessions not found — using first 3 available.")
            test_sessions = all_sessions[:3]

    blackout_durations_s = [5, 10, 30, 60] if not args.smoke else [5, 10]
    logger.info(f"Test Sessions: {test_sessions}")
    logger.info(f"Blackout Durations: {blackout_durations_s} s")

    # Results containers
    session_results: Dict[str, Any] = {}
    durations_collector: Dict[int, List[BlackoutMetrics]] = {d: [] for d in blackout_durations_s}
    road_stats_collector: Dict[int, List[RoadMatchStats]] = {d: [] for d in blackout_durations_s}

    for session_id in test_sessions:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Session: {session_id}")
        logger.info(f"{'=' * 60}")

        mask = session_ids_all == session_id
        session_imu = imu_all[mask]
        N_win = len(session_imu)

        if args.smoke and N_win > 100:
            logger.info(f"[SMOKE] Truncating {session_id} from {N_win} to 100 windows.")
            session_imu = session_imu[:100]
            N_win = 100

        if N_win == 0:
            logger.warning(f"No windows for {session_id}. Skipping.")
            continue

        timestamps = np.arange(N_win, dtype=np.float64)  # 1-second steps

        try:
            gt_lats, gt_lons, gt_hdgs = load_vehicle_ground_truth(
                base_dir=raw_dataset_dir,
                session_id=session_id,
                num_windows=N_win,
            )
        except FileNotFoundError as e:
            logger.warning(f"GT not found: {e}. Using placeholder.")
            gt_lats = np.full(N_win, 51.5)
            gt_lons = np.full(N_win, -0.1)
            gt_hdgs = np.zeros(N_win)

        session_duration_s = float(timestamps[-1])
        logger.info(f"  Windows: {N_win} | Duration: {session_duration_s:.1f}s")

        session_durations_res: Dict[str, Any] = {}

        for duration_s in blackout_durations_s:
            blackout_intervals = generate_deterministic_blackouts(
                total_duration_s=session_duration_s,
                blackout_duration_s=duration_s,
                num_intervals=4,
            )

            eval_res: RoadBlackoutEvaluationResult = pipeline.run_session_blackout_road(
                imu_windows=session_imu,
                timestamps=timestamps,
                gt_lats=gt_lats,
                gt_lons=gt_lons,
                blackout_intervals=blackout_intervals,
                init_heading_deg=float(gt_hdgs[0]),
                gt_headings_deg=gt_hdgs,
                apply_nhc=True,
                apply_attitude_update=True,
                apply_velocity_update=True,
            )

            durations_collector[duration_s].extend(eval_res.per_blackout_metrics)
            road_stats_collector[duration_s].extend(eval_res.road_match_stats)

            # Per-session per-duration summary
            mean_dist = (
                float(np.mean([m.traveled_distance_m for m in eval_res.per_blackout_metrics]))
                if eval_res.per_blackout_metrics else 0.0
            )

            # Road matching aggregates for this duration+session
            n_total_steps = sum(r.n_steps for r in eval_res.road_match_stats)
            n_total_matched = sum(r.n_matched for r in eval_res.road_match_stats)
            n_rej_dist = sum(r.n_rejected_distance for r in eval_res.road_match_stats)
            n_rej_hdg = sum(r.n_rejected_heading for r in eval_res.road_match_stats)
            n_roads_active = sum(1 for r in eval_res.road_match_stats if r.road_active)
            match_frac = n_total_matched / n_total_steps if n_total_steps > 0 else 0.0
            all_corrs = [r.avg_correction_m for r in eval_res.road_match_stats if r.n_matched > 0]
            avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0

            session_durations_res[f"{duration_s}s"] = {
                "blackout_intervals": blackout_intervals,
                "final_position_error_m": round(eval_res.mean_final_error_m, 3),
                "max_position_error_m": round(eval_res.mean_max_error_m, 3),
                "rmse_position_error_m": round(eval_res.mean_rmse_error_m, 3),
                "traveled_distance_m": round(mean_dist, 3),
                "total_traveled_distance_m": round(eval_res.total_traveled_distance_m, 3),
                "relative_drift_percent": round(eval_res.mean_relative_drift_percent, 3),
                "road_matching": {
                    "intervals_with_road": n_roads_active,
                    "total_intervals": len(blackout_intervals),
                    "total_steps": n_total_steps,
                    "matched": n_total_matched,
                    "rejected_distance": n_rej_dist,
                    "rejected_heading": n_rej_hdg,
                    "match_fraction": round(match_frac, 4),
                    "avg_correction_m": round(avg_corr, 3),
                },
            }

            logger.info(
                f"  [{duration_s}s] Final: {eval_res.mean_final_error_m:7.2f}m | "
                f"RMSE: {eval_res.mean_rmse_error_m:7.2f}m | "
                f"Dist: {mean_dist:7.1f}m | "
                f"Drift: {eval_res.mean_relative_drift_percent:5.1f}% | "
                f"Match: {match_frac*100:.0f}% ({n_total_matched}/{n_total_steps})"
            )

        session_results[session_id] = session_durations_res

    # Aggregate across all sessions
    duration_overall_summary: Dict[str, Any] = {}
    for d in blackout_durations_s:
        mets = durations_collector[d]
        rsts = road_stats_collector[d]
        if mets:
            mean_final = float(np.mean([m.final_error_m for m in mets]))
            mean_max = float(np.mean([m.max_error_m for m in mets]))
            mean_rmse = float(np.mean([m.rmse_error_m for m in mets]))
            mean_dist = float(np.mean([m.traveled_distance_m for m in mets]))
            mean_drift = float(np.mean([m.relative_drift_percent for m in mets]))
        else:
            mean_final = mean_max = mean_rmse = mean_dist = mean_drift = 0.0

        total_steps = sum(r.n_steps for r in rsts)
        total_matched = sum(r.n_matched for r in rsts)
        total_rej_dist = sum(r.n_rejected_distance for r in rsts)
        total_rej_hdg = sum(r.n_rejected_heading for r in rsts)
        match_frac = total_matched / total_steps if total_steps > 0 else 0.0
        n_roads_active = sum(1 for r in rsts if r.road_active)
        all_corrs = [r.avg_correction_m for r in rsts if r.n_matched > 0]
        avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0

        duration_overall_summary[f"{d}s"] = {
            "final_position_error_m": round(mean_final, 3),
            "max_position_error_m": round(mean_max, 3),
            "rmse_position_error_m": round(mean_rmse, 3),
            "traveled_distance_m": round(mean_dist, 3),
            "relative_drift_percent": round(mean_drift, 3),
            "road_matching": {
                "intervals_with_road": n_roads_active,
                "total_intervals": len(rsts),
                "total_steps": total_steps,
                "matched": total_matched,
                "rejected_distance": total_rej_dist,
                "rejected_heading": total_rej_hdg,
                "match_fraction": round(match_frac, 4),
                "avg_correction_m": round(avg_corr, 3),
            },
        }

    elapsed_s = round(time.time() - start_time, 2)

    # Load Version A and Baseline results for comparison
    version_a_summary: Optional[Dict] = None
    baseline_summary: Optional[Dict] = None

    version_a_path = Path(args.version_a_results)
    if version_a_path.exists():
        try:
            with open(version_a_path) as f:
                va = json.load(f)
            version_a_summary = va.get("overall_mean_by_duration")
        except Exception as e:
            logger.warning(f"Could not load Version A results: {e}")

    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        try:
            with open(baseline_path) as f:
                bl = json.load(f)
            baseline_summary = bl.get("overall_mean_by_duration")
        except Exception as e:
            logger.warning(f"Could not load baseline results: {e}")

    # Print comparison table
    logger.info("\n" + "=" * 90)
    logger.info("NAVIGATE 2.0 — THREE-WAY COMPARISON: Baseline vs Version A vs Version B")
    logger.info("=" * 90)
    logger.info(
        f"{'Duration':<10} | {'Metric':<22} | {'Baseline':<14} | "
        f"{'V-A (AI+ESEKF)':<16} | {'V-B (AI+ESEKF+Road)':<18} | "
        f"{'VB vs VA':<14}"
    )
    logger.info("-" * 90)

    for d_str, vb_m in duration_overall_summary.items():
        va_m = version_a_summary.get(d_str, {}) if version_a_summary else {}
        bl_m = baseline_summary.get(d_str, {}) if baseline_summary else {}

        metrics_to_report = [
            ("final_position_error_m", "Final Error (m)"),
            ("max_position_error_m", "Max Error (m)"),
            ("rmse_position_error_m", "RMSE (m)"),
            ("traveled_distance_m", "Distance (m)"),
            ("relative_drift_percent", "Drift (%)"),
        ]
        for key, label in metrics_to_report:
            bl_v = bl_m.get(key, float("nan"))
            va_v = va_m.get(key, float("nan"))
            vb_v = vb_m.get(key, float("nan"))
            if not np.isnan(va_v) and not np.isnan(vb_v) and va_v > 0:
                diff = vb_v - va_v
                pct = (diff / va_v) * 100.0
                s = "+" if diff > 0 else ""
                diff_str = f"{s}{diff:.2f} ({s}{pct:.1f}%)"
            else:
                diff_str = "n/a"
            logger.info(
                f"{d_str:<10} | {label:<22} | "
                f"{bl_v if not np.isnan(bl_v) else 'n/a':>14.2f} | "
                f"{va_v if not np.isnan(va_v) else 'n/a':>16.2f} | "
                f"{vb_v if not np.isnan(vb_v) else 'n/a':>18.2f} | "
                f"{diff_str:<14}"
            )

        # Road stats line
        rd = vb_m.get("road_matching", {})
        logger.info(
            f"{'':10}   {'Road Match%':<22} | {'':14} | {'':16} | "
            f"{rd.get('match_fraction', 0)*100:>17.1f}% | "
            f"{rd.get('matched', 0):>4} matched / {rd.get('total_steps', 0):>4} steps"
        )
        logger.info("-" * 90)

    # Improvement assessment (honest)
    logger.info("\n>>> IMPROVEMENT ASSESSMENT (Version B vs Version A) <<<")
    for d_str, vb_m in duration_overall_summary.items():
        va_m = version_a_summary.get(d_str, {}) if version_a_summary else {}
        va_final = va_m.get("final_position_error_m", float("nan"))
        vb_final = vb_m.get("final_position_error_m", float("nan"))
        if not np.isnan(va_final) and not np.isnan(vb_final):
            improvement = va_final - vb_final
            improved = improvement > 0
            pct = abs(improvement) / va_final * 100.0
            direction = "BETTER" if improved else "WORSE"
            logger.info(
                f"  {d_str}: V-B final error {vb_final:.2f}m vs V-A {va_final:.2f}m "
                f"→ {direction} by {abs(improvement):.2f}m ({pct:.1f}%)"
            )
        else:
            logger.info(f"  {d_str}: Cannot compare (missing data)")

    # Save JSON results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "road_comparison_results.json"

    final_output = {
        "metadata": {
            "pipeline_version": "Version B: AI + ES-EKF + Road Constraint",
            "velocity_checkpoint": str(vel_ckpt_path),
            "attitude_checkpoint": str(att_ckpt_path),
            "dataset_npz": str(data_path),
            "evaluated_sessions": test_sessions,
            "blackout_durations_s": blackout_durations_s,
            "road_config": {
                "max_match_dist_m": args.max_match_dist_m,
                "max_heading_diff_deg": args.max_heading_diff_deg,
                "correction_strength": args.correction_strength,
                "road_cov_m2": args.road_cov_m2,
                "lookahead_window_s": args.lookahead_window_s,
            },
            "road_data_source": (
                "GPS observations strictly before each blackout interval "
                "(t < blackout_start), up to lookahead_window_s seconds prior. "
                "No future GT is used. Road geometry is frozen at blackout onset."
            ),
            "leakage_prevention": (
                "Pre-blackout GPS used only (timestamps < blackout_start_s). "
                "Cached road polyline is immutable during blackout. "
                "EKF road pseudo-measurement uses configurable noise (road_cov_m2). "
                "Post-blackout: road cache discarded, normal GNSS fusion resumes."
            ),
            "limitation": (
                "If vehicle path during blackout diverges from pre-blackout road segment "
                "(e.g., turns immediately after outage begins), road constraint may be "
                "unhelpful or harmful. Reported via match/rejection statistics."
            ),
            "runtime_seconds": elapsed_s,
        },
        "version_b_overall_by_duration": duration_overall_summary,
        "version_a_comparison": version_a_summary,
        "baseline_comparison": baseline_summary,
        "per_session_results": session_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"\nJSON results saved: {json_path}")

    # Save CSV comparison table
    csv_path = out_dir / "road_comparison_results.csv"
    csv_rows = []
    csv_rows.append([
        "duration_s", "session", "metric",
        "baseline", "version_a_AI_ESEKF", "version_b_AI_ESEKF_Road",
        "vb_vs_va_diff", "vb_vs_va_pct",
    ])

    metrics_csv = [
        ("final_position_error_m", "final_error_m"),
        ("max_position_error_m", "max_error_m"),
        ("rmse_position_error_m", "rmse_m"),
        ("traveled_distance_m", "distance_m"),
        ("relative_drift_percent", "drift_pct"),
    ]

    for session_id in test_sessions:
        s_res = session_results.get(session_id, {})
        for duration_s in blackout_durations_s:
            d_str = f"{duration_s}s"
            vb_d = s_res.get(d_str, {})
            va_m = (version_a_summary.get(d_str, {}) if version_a_summary else {})
            bl_m = (baseline_summary.get(d_str, {}) if baseline_summary else {})

            for key, label in metrics_csv:
                bl_v = bl_m.get(key, "")
                va_v = va_m.get(key, "")
                vb_v = vb_d.get(key, "")
                diff = ""
                pct = ""
                try:
                    va_f = float(va_v)
                    vb_f = float(vb_v)
                    if va_f != 0:
                        diff = round(vb_f - va_f, 3)
                        pct = round((vb_f - va_f) / va_f * 100, 2)
                except (TypeError, ValueError):
                    pass

                csv_rows.append([
                    duration_s, session_id, label,
                    bl_v, va_v, vb_v, diff, pct,
                ])

    # Aggregate rows
    for duration_s in blackout_durations_s:
        d_str = f"{duration_s}s"
        vb_d = duration_overall_summary.get(d_str, {})
        va_m = (version_a_summary.get(d_str, {}) if version_a_summary else {})
        bl_m = (baseline_summary.get(d_str, {}) if baseline_summary else {})
        for key, label in metrics_csv:
            bl_v = bl_m.get(key, "")
            va_v = va_m.get(key, "")
            vb_v = vb_d.get(key, "")
            diff = ""
            pct = ""
            try:
                va_f = float(va_v)
                vb_f = float(vb_v)
                if va_f != 0:
                    diff = round(vb_f - va_f, 3)
                    pct = round((vb_f - va_f) / va_f * 100, 2)
            except (TypeError, ValueError):
                pass
            csv_rows.append([
                duration_s, "OVERALL", label, bl_v, va_v, vb_v, diff, pct,
            ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    logger.info(f"CSV results saved: {csv_path}")
    logger.info(f"Total runtime: {elapsed_s:.1f}s")
    logger.info("=" * 90)

    return final_output


# ================================================================== #
#  CLI
# ================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NAVIGATE 2.0 Stage 19: AI + ES-EKF + Road Constraint Evaluation"
    )
    p.add_argument("--data", type=str, default="data/processed/iovnbd_full.npz",
                   help="Path to processed NPZ dataset")
    p.add_argument("--velocity-checkpoint", type=str, default="models/velocity_model_v2.pt")
    p.add_argument("--attitude-checkpoint", type=str, default="models/attitude_model.pt")
    p.add_argument(
        "--dataset-raw", type=str,
        default=r"D:\Career\Competitons\Devesh Aug-Sep Hackathons\IO-VNBD\Synchronised V abd S datasets",
        help="Path to raw IO-VNBD dataset directory (for GPS ground truth)"
    )
    p.add_argument("--version-a-results", type=str,
                   default="results/ai_iekf_blackout_results.json",
                   help="Path to Version A results JSON for comparison")
    p.add_argument("--baseline", type=str,
                   default="results/baseline_blackout_results_v2.json",
                   help="Path to baseline dead-reckoning results JSON")
    p.add_argument("--output-dir", type=str, default="results/ai_iekf_road",
                   help="Output directory for Version B results")
    p.add_argument("--device", type=str, default=None,
                   help="PyTorch device ('cpu', 'cuda', etc.)")
    p.add_argument("--smoke", action="store_true",
                   help="Run quick smoke test on 1 session with 2 durations")

    # Road matching parameters
    p.add_argument("--max-match-dist-m", type=float, default=20.0,
                   help="Maximum perpendicular distance (m) for road matching")
    p.add_argument("--max-heading-diff-deg", type=float, default=30.0,
                   help="Maximum heading difference (deg) for road matching")
    p.add_argument("--correction-strength", type=float, default=0.5,
                   help="Road correction interpolation weight [0,1]")
    p.add_argument("--road-cov-m2", type=float, default=5.0,
                   help="Road pseudo-measurement noise variance (m^2)")
    p.add_argument("--lookahead-window-s", type=float, default=120.0,
                   help="Seconds of pre-blackout GPS to cache as road polyline")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_road_evaluation(args)
