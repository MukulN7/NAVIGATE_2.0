"""
run_ai_iekf_evaluation.py — Full AI + ES-EKF Navigation Blackout Evaluation Pipeline.

Evaluates the end-to-end NAVIGATE 2.0 AI + ES-EKF system:
1. VelocityModel V2 (forward speed estimation from 50-sample IMU windows).
2. AttitudeModel (relative quaternion estimation over 50-sample windows).
3. ErrorStateIEKFTracker (10 Hz IMU propagation, NHC, velocity update, relative attitude update).
4. GNSS position update with deterministic blackout simulation (5s, 10s, 30s, 60s).
5. Direct comparison against baseline dead-reckoning results (results/baseline_blackout_results_v2.json).

Test Sessions:
- S-M
- S-Vfa01
- S-Vw2
"""

import argparse
import csv
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root and src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from navigate.ai_iekf_pipeline import AIIEKFPipeline
from navigate.evaluate_blackout import BlackoutMetrics, BlackoutEvaluationResult

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_iekf_eval")


# ================================================================== #
#  Ground Truth & Blackout Generators (Aligned with Baseline)
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
    """
    v_name = f"V-{session_id[2:]}.csv"
    matches = list(base_dir.rglob(v_name))
    if not matches:
        raise FileNotFoundError(f"Vehicle ground truth file not found for session {session_id} ({v_name})")

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

    # Map to window end sample indices
    sample_indices = np.array([i * stride + window_size - 1 for i in range(num_windows)], dtype=int)
    sample_indices = np.clip(sample_indices, 0, len(lats_arr) - 1)

    return lats_arr[sample_indices], lons_arr[sample_indices], hdgs_arr[sample_indices]


def generate_deterministic_blackouts(
    total_duration_s: float,
    blackout_duration_s: float,
    num_intervals: int = 4
) -> List[Tuple[float, float]]:
    """
    Generates reproducible blackout intervals evenly spaced along the recording timeline.
    Matches baseline evaluation generator.
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
#  Evaluation Runner
# ================================================================== #

def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    start_time = time.time()

    # 1. Check datasets and models
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {data_path}")

    vel_ckpt_path = Path(args.velocity_checkpoint)
    att_ckpt_path = Path(args.attitude_checkpoint)
    if not vel_ckpt_path.exists():
        raise FileNotFoundError(f"Velocity checkpoint not found: {vel_ckpt_path}")
    if not att_ckpt_path.exists():
        raise FileNotFoundError(f"Attitude checkpoint not found: {att_ckpt_path}")

    # 2. Instantiate Pipeline
    pipeline = AIIEKFPipeline(
        velocity_checkpoint=vel_ckpt_path,
        attitude_checkpoint=att_ckpt_path,
        device=args.device,
    )

    # 3. Load Processed NPZ Dataset
    logger.info(f"Loading processed dataset: {data_path}")
    npz = np.load(data_path, allow_pickle=True)
    imu_all = npz["imu"]               # [N, 50, 6]
    session_ids_all = npz["session_ids"] # [N]

    all_sessions = sorted(list(set(session_ids_all)))
    if args.smoke:
        logger.info("[SMOKE MODE] Evaluating small real slice of 1 test session for quick verification.")
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions][:1]
        if not test_sessions:
            test_sessions = all_sessions[:1]
    else:
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions]
        if not test_sessions:
            logger.warning("Held-out test sessions not found. Evaluating available sessions.")
            test_sessions = all_sessions[:3]

    logger.info(f"Held-Out Test Sessions to Evaluate ({len(test_sessions)}): {test_sessions}")

    raw_dataset_dir = Path(args.dataset_raw)
    blackout_durations_s = [5, 10, 30, 60] if not args.smoke else [5, 10]
    session_results: Dict[str, Any] = {}
    durations_collector: Dict[int, List[BlackoutMetrics]] = {d: [] for d in blackout_durations_s}

    for session_id in test_sessions:
        logger.info(f"==================================================")
        logger.info(f"Evaluating Session: {session_id}")
        logger.info(f"==================================================")

        mask = (session_ids_all == session_id)
        session_imu = imu_all[mask]
        N_win = len(session_imu)

        if args.smoke and N_win > 100:
            logger.info(f"[SMOKE MODE] Truncating session {session_id} from {N_win} to 100 windows.")
            session_imu = session_imu[:100]
            N_win = 100

        if N_win == 0:
            logger.warning(f"No windows found for session {session_id}. Skipping.")
            continue

        timestamps = np.arange(N_win, dtype=np.float64) * 1.0  # 1-second step

        # Load GT vehicle trajectory (lat, lon, heading)
        gt_lats, gt_lons, gt_hdgs = load_vehicle_ground_truth(
            base_dir=raw_dataset_dir,
            session_id=session_id,
            num_windows=N_win
        )

        session_duration_s = float(timestamps[-1])
        logger.info(f"  Windows: {N_win} | Duration: {session_duration_s:.1f} s")

        session_durations_res: Dict[str, Any] = {}

        for duration_s in blackout_durations_s:
            blackout_intervals = generate_deterministic_blackouts(
                total_duration_s=session_duration_s,
                blackout_duration_s=duration_s,
                num_intervals=4
            )

            eval_res: BlackoutEvaluationResult = pipeline.run_session_blackout(
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
            mean_dist_per_interval = float(np.mean([m.traveled_distance_m for m in eval_res.per_blackout_metrics])) if eval_res.per_blackout_metrics else 0.0

            session_durations_res[f"{duration_s}s"] = {
                "blackout_intervals": blackout_intervals,
                "final_position_error_m": round(eval_res.mean_final_error_m, 3),
                "max_position_error_m": round(eval_res.mean_max_error_m, 3),
                "rmse_position_error_m": round(eval_res.mean_rmse_error_m, 3),
                "traveled_distance_m": round(mean_dist_per_interval, 3),
                "total_traveled_distance_m": round(eval_res.total_traveled_distance_m, 3),
                "relative_drift_percent": round(eval_res.mean_relative_drift_percent, 3),
            }

            logger.info(
                f"  [{duration_s}s Outage] Final Err: {eval_res.mean_final_error_m:6.2f} m | "
                f"RMSE: {eval_res.mean_rmse_error_m:6.2f} m | "
                f"Dist: {mean_dist_per_interval:6.1f} m | "
                f"Drift: {eval_res.mean_relative_drift_percent:5.1f}%"
            )

        session_results[session_id] = session_durations_res

    # Aggregate overall summary by duration
    duration_overall_summary: Dict[str, Dict[str, float]] = {}
    for d in blackout_durations_s:
        mets = durations_collector[d]
        if mets:
            mean_final = float(np.mean([m.final_error_m for m in mets]))
            mean_max = float(np.mean([m.max_error_m for m in mets]))
            mean_rmse = float(np.mean([m.rmse_error_m for m in mets]))
            mean_dist = float(np.mean([m.traveled_distance_m for m in mets]))
            mean_drift = float(np.mean([m.relative_drift_percent for m in mets]))
        else:
            mean_final, mean_max, mean_rmse, mean_dist, mean_drift = 0.0, 0.0, 0.0, 0.0, 0.0

        duration_overall_summary[f"{d}s"] = {
            "final_position_error_m": round(mean_final, 3),
            "max_position_error_m": round(mean_max, 3),
            "rmse_position_error_m": round(mean_rmse, 3),
            "traveled_distance_m": round(mean_dist, 3),
            "relative_drift_percent": round(mean_drift, 3),
        }

    elapsed_time = round(time.time() - start_time, 2)

    # 4. Load Baseline Results for Side-by-Side Comparison
    baseline_summary: Optional[Dict[str, Any]] = None
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
                baseline_summary = baseline_data.get("overall_mean_by_duration")
        except Exception as e:
            logger.warning(f"Failed to parse baseline results: {e}")

    # 5. Format Comparison Table
    logger.info("\n" + "=" * 80)
    logger.info("NAVIGATE 2.0 — AI + ES-EKF vs BASELINE DEAD-RECKONING COMPARISON")
    logger.info("=" * 80)
    header_str = (
        f"{'Duration':<10} | {'Metric':<18} | {'Baseline V2':<15} | "
        f"{'AI + ES-EKF':<15} | {'Change':<12}"
    )
    logger.info(header_str)
    logger.info("-" * 80)

    for d_str, cur_m in duration_overall_summary.items():
        base_m = baseline_summary.get(d_str, {}) if baseline_summary else {}
        for metric_key, metric_name in [
            ("final_position_error_m", "Final Err (m)"),
            ("max_position_error_m", "Max Err (m)"),
            ("rmse_position_error_m", "RMSE Err (m)"),
            ("traveled_distance_m", "Distance (m)"),
            ("relative_drift_percent", "Drift (%)"),
        ]:
            b_val = base_m.get(metric_key, float("nan"))
            c_val = cur_m.get(metric_key, float("nan"))
            diff_str = ""
            if not np.isnan(b_val) and not np.isnan(c_val) and b_val > 0:
                diff = c_val - b_val
                pct = (diff / b_val) * 100.0
                sign = "+" if diff > 0 else ""
                diff_str = f"{sign}{diff:.2f} ({sign}{pct:.1f}%)"

            logger.info(f"{d_str:<10} | {metric_name:<18} | {b_val:<15.2f} | {c_val:<15.2f} | {diff_str:<12}")
        logger.info("-" * 80)

    # 6. Save JSON Output
    final_output = {
        "metadata": {
            "velocity_checkpoint": str(vel_ckpt_path),
            "attitude_checkpoint": str(att_ckpt_path),
            "dataset_npz": str(data_path),
            "evaluated_test_sessions": test_sessions,
            "blackout_durations_s": blackout_durations_s,
            "runtime_seconds": elapsed_time,
            "architecture": "AIIEKFPipeline (VelocityModel V2 + AttitudeModel q_rel + NHC + ES-EKF)",
        },
        "overall_mean_by_duration": duration_overall_summary,
        "baseline_comparison": baseline_summary,
        "per_session_results": session_results,
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Results saved to: {out_file}")
    logger.info("=" * 80)
    return final_output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run NAVIGATE 2.0 AI + ES-EKF Blackout Evaluation")
    p.add_argument("--data", type=str, default="data/processed/iovnbd_full.npz")
    p.add_argument("--velocity-checkpoint", type=str, default="models/velocity_model_v2.pt")
    p.add_argument("--attitude-checkpoint", type=str, default="models/attitude_model.pt")
    p.add_argument("--dataset-raw", type=str, default=r"D:\Career\Competitons\Devesh Aug-Sep Hackathons\IO-VNBD\Synchronised V abd S datasets")
    p.add_argument("--baseline", type=str, default="results/baseline_blackout_results_v2.json")
    p.add_argument("--output", type=str, default="results/ai_iekf_blackout_results.json")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="Run fast smoke test on 1 session")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_evaluation(args)
