"""
run_baseline_evaluation.py — Baseline NAVIGATE 2.0 Evaluation Pipeline.

Runs dead-reckoning trajectory evaluation with GNSS blackout simulation on held-out test sessions
using the trained VelocityModel V2.

Held-out Test Sessions evaluated:
- S-M
- S-Vfa01
- S-Vw2

Evaluated Blackout Durations: 5s, 10s, 30s, 60s
Outputs saved to: results/baseline_blackout_results.json
"""

import argparse
import csv
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from navigate.models.velocity_model import VelocityModel
from navigate.evaluate_blackout import (
    evaluate_trajectory_blackout,
    BlackoutEvaluationResult,
    BlackoutMetrics,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("baseline_eval")


# ================================================================== #
#  GT Vehicle CSV Loader
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


# ================================================================== #
#  Deterministic Blackout Generator
# ================================================================== #

def generate_deterministic_blackouts(
    total_duration_s: float,
    blackout_duration_s: float,
    num_intervals: int = 4
) -> List[Tuple[float, float]]:
    """
    Generates reproducible blackout intervals evenly spaced along the recording timeline.
    Aligns start times to integer seconds to guarantee exact blackout duration coverage.
    """
    if total_duration_s <= blackout_duration_s + 10.0:
        start = max(1.0, float(int((total_duration_s - blackout_duration_s) / 2.0)))
        return [(start, start + blackout_duration_s)]

    # Space intervals at fractions (e.g. 20%, 40%, 60%, 80% of recording length)
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

    # 1. Load Processed NPZ Dataset
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {data_path}")

    logger.info(f"Loading processed dataset: {data_path}")
    npz = np.load(data_path, allow_pickle=True)
    imu_all = npz["imu"]               # [N, 50, 6]
    session_ids_all = npz["session_ids"] # [N]
    timestamps_all = npz["timestamps"]   # [N, 2]

    # 2. Load VelocityModel V2 Checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    logger.info(f"Loading VelocityModel V2 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    imu_mean = np.array(ckpt["imu_mean"], dtype=np.float32)
    imu_std = np.array(ckpt["imu_std"], dtype=np.float32)
    vel_mean = float(ckpt["vel_mean"])
    vel_std = float(ckpt["vel_std"])

    model_config = ckpt.get("model_config", {})
    model = VelocityModel(
        in_channels=model_config.get("in_channels", 6),
        hidden_size=model_config.get("hidden_size", 128),
        window_size=model_config.get("window_size", 50),
        dropout_rate=model_config.get("dropout_rate", 0.2)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Determine Test Sessions to Evaluate
    all_sessions = sorted(list(set(session_ids_all)))
    if args.smoke:
        logger.info("[SMOKE MODE] Evaluating single test session for quick verification.")
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions][:1]
        if not test_sessions:
            test_sessions = all_sessions[:1]
    else:
        test_sessions = [s for s in ["S-M", "S-Vfa01", "S-Vw2"] if s in all_sessions]
        if not test_sessions:
            logger.warning("Target test sessions ['S-M', 'S-Vfa01', 'S-Vw2'] not found in NPZ. Using available sessions.")
            test_sessions = all_sessions[:3]

    logger.info(f"Held-Out Test Sessions to Evaluate ({len(test_sessions)}): {test_sessions}")

    raw_dataset_dir = Path(args.dataset_raw)

    blackout_durations_s = [5, 10, 30, 60]
    session_results: Dict[str, Any] = {}
    duration_overall_summary: Dict[int, Dict[str, float]] = {}

    # Store per-duration metrics across all sessions
    durations_collector: Dict[int, List[BlackoutMetrics]] = {d: [] for d in blackout_durations_s}

    for session_id in test_sessions:
        logger.info(f"--------------------------------------------------")
        logger.info(f"Evaluating Session: {session_id}")

        mask = (session_ids_all == session_id)
        session_imu = imu_all[mask]               # [N_win, 50, 6]
        session_ts_window = timestamps_all[mask]  # [N_win, 2]
        N_win = len(session_imu)

        if N_win == 0:
            logger.warning(f"No windows found for session {session_id}. Skipping.")
            continue

        # Inference: Predict speeds in m/s
        imu_norm = (session_imu - imu_mean) / imu_std
        with torch.no_grad():
            pred_norm, _ = model(torch.tensor(imu_norm, dtype=torch.float32))
            pred_ms = (pred_norm * vel_std + vel_mean).squeeze(-1).numpy()

        # Extract Gyro Z (yaw rate rad/s) & Timestamps (relative seconds, 1s per window hop)
        gyro_z_rad_s = session_imu[:, :, 5].mean(axis=1)
        timestamps = np.arange(N_win, dtype=np.float64) * 1.0  # 1-second step

        # Load GT vehicle trajectory (lat, lon, heading)
        gt_lats, gt_lons, gt_hdgs = load_vehicle_ground_truth(
            base_dir=raw_dataset_dir,
            session_id=session_id,
            num_windows=N_win
        )

        session_duration_s = float(timestamps[-1])
        logger.info(f"  Windows: {N_win} | Duration: {session_duration_s:.1f} s | Speed Mean: {pred_ms.mean()*3.6:.2f} km/h")

        session_durations_res: Dict[str, Any] = {}

        for duration_s in blackout_durations_s:
            blackout_intervals = generate_deterministic_blackouts(
                total_duration_s=session_duration_s,
                blackout_duration_s=duration_s,
                num_intervals=4
            )

            eval_res: BlackoutEvaluationResult = evaluate_trajectory_blackout(
                timestamps=timestamps,
                velocities_ms=pred_ms,
                gyro_z_rad_s=gyro_z_rad_s,
                gt_lats=gt_lats,
                gt_lons=gt_lons,
                blackout_intervals=blackout_intervals,
                init_heading_deg=float(gt_hdgs[0]),
                gt_headings_deg=gt_hdgs,
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
                f"  [{duration_s:>2d}s Blackout] "
                f"Final Err: {eval_res.mean_final_error_m:6.2f} m | "
                f"Max Err: {eval_res.mean_max_error_m:6.2f} m | "
                f"RMSE: {eval_res.mean_rmse_error_m:6.2f} m | "
                f"Mean Dist: {mean_dist_per_interval:6.1f} m | "
                f"Drift: {eval_res.mean_relative_drift_percent:5.2f}%"
            )

        session_results[session_id] = session_durations_res

    # Overall Summary across all Test Sessions
    logger.info("==================================================")
    logger.info("OVERALL MEAN BASELINE METRICS ACROSS TEST SESSIONS:")
    logger.info("==================================================")

    for d in blackout_durations_s:
        metrics_list = durations_collector[d]
        if metrics_list:
            mean_final = float(np.mean([m.final_error_m for m in metrics_list]))
            mean_max = float(np.mean([m.max_error_m for m in metrics_list]))
            mean_rmse = float(np.mean([m.rmse_error_m for m in metrics_list]))
            mean_dist = float(np.mean([m.traveled_distance_m for m in metrics_list]))
            mean_drift = float(np.mean([m.relative_drift_percent for m in metrics_list]))
        else:
            mean_final = mean_max = mean_rmse = mean_dist = mean_drift = 0.0

        duration_overall_summary[d] = {
            "final_position_error_m": round(mean_final, 3),
            "max_position_error_m": round(mean_max, 3),
            "rmse_position_error_m": round(mean_rmse, 3),
            "traveled_distance_m": round(mean_dist, 3),
            "relative_drift_percent": round(mean_drift, 3),
        }

        logger.info(
            f"  {d:>2d}s Duration -> "
            f"Final Err: {mean_final:6.2f} m | "
            f"Max Err: {mean_max:6.2f} m | "
            f"RMSE: {mean_rmse:6.2f} m | "
            f"Dist: {mean_dist:6.1f} m | "
            f"Drift: {mean_drift:5.2f}%"
        )

    elapsed_time = round(time.time() - start_time, 2)

    final_output = {
        "metadata": {
            "model_checkpoint": str(ckpt_path),
            "dataset_npz": str(data_path),
            "evaluated_test_sessions": test_sessions,
            "blackout_durations_s": blackout_durations_s,
            "runtime_seconds": elapsed_time,
            "limitations_and_assumptions": [
                "Baseline navigation uses VelocityModel V2 forward speed estimates + Gyroscope integration (no IEKF / map matching).",
                "GNSS blackout intervals are generated deterministically across test session timelines.",
                "Vehicle ground-truth lat/lon/heading is aligned with window end sample indices.",
            ]
        },
        "overall_mean_by_duration": {f"{d}s": duration_overall_summary[d] for d in blackout_durations_s},
        "per_session_results": session_results
    }

    # Save to JSON
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info("==================================================")
    logger.info(f"Baseline evaluation completed in {elapsed_time} seconds.")
    logger.info(f"Results saved to: {out_file}")
    logger.info("==================================================")

    return final_output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run NAVIGATE 2.0 Baseline Blackout Evaluation")
    p.add_argument("--data", type=str, default="data/processed/iovnbd_full.npz")
    p.add_argument("--checkpoint", type=str, default="models/velocity_model_v2.pt")
    p.add_argument("--dataset-raw", type=str, default=r"D:\Career\Competitons\Devesh Aug-Sep Hackathons\IO-VNBD\Synchronised V abd S datasets")
    p.add_argument("--output", type=str, default="results/baseline_blackout_results_v2.json")
    p.add_argument("--smoke", action="store_true", help="Run a fast smoke test evaluation on 1 session")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_evaluation(args)
