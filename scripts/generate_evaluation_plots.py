"""
generate_evaluation_plots.py — NAVIGATE 2.0 Evaluation Visualization & Summary Artifact Generator.

Generates publication/demo-quality plots and structured CSV/JSON summaries comparing:
1. Baseline Dead-Reckoning (VelocityModel V2 + Gyroscope)
2. Full AI + ES-EKF Navigation System (VelocityModel V2 + AttitudeModel q_rel + NHC + ES-EKF)

Generated Artifacts:
- results/figures/final_position_error_comparison.png
- results/figures/drift_percentage_comparison.png
- results/figures/error_growth_over_blackout_duration.png
- results/figures/representative_trajectory_blackout.png
- results/evaluation_summary.json
- results/evaluation_summary.csv
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt

# Import project modules to run representative trajectory simulation for Plot 4
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from navigate.ai_iekf_pipeline import AIIEKFPipeline, lat_lon_to_enu_m
from navigate.evaluate_blackout import evaluate_trajectory_blackout

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("generate_plots")


# ================================================================== #
#  Styling & Config
# ================================================================== #

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

COLOR_BASELINE = "#D9534F"     # Coral Red / Warm Rose
COLOR_AI_IEKF = "#0275D8"      # Deep Royal Blue
COLOR_GT = "#5CB85C"           # Emerald Green
COLOR_BLACKOUT = "#FFC107"     # Amber / Gold Shading


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
#  Plot Generation Functions
# ================================================================== #

def plot_final_position_error_comparison(
    durations: List[str],
    baseline_errs: List[float],
    ai_iekf_errs: List[float],
    out_path: Path,
) -> None:
    """Plot 1: Grouped bar chart comparing baseline vs AI+ES-EKF final position error."""
    x = np.arange(len(durations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width/2, baseline_errs, width, label="Baseline (DR)", color=COLOR_BASELINE, alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, ai_iekf_errs, width, label="NAVIGATE 2.0 (AI + ES-EKF)", color=COLOR_AI_IEKF, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Final Position Error (meters)", fontsize=12, fontweight="bold")
    ax.set_xlabel("GNSS Blackout Duration", fontsize=12, fontweight="bold")
    ax.set_title("Final Position Error: Baseline vs AI + ES-EKF", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(durations, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Value labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}m", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}m", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Plot 1: {out_path}")


def plot_drift_percentage_comparison(
    durations: List[str],
    baseline_drifts: List[float],
    ai_iekf_drifts: List[float],
    out_path: Path,
) -> None:
    """Plot 2: Grouped bar chart comparing baseline vs AI+ES-EKF relative drift percentage."""
    x = np.arange(len(durations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width/2, baseline_drifts, width, label="Baseline (DR)", color=COLOR_BASELINE, alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, ai_iekf_drifts, width, label="NAVIGATE 2.0 (AI + ES-EKF)", color=COLOR_AI_IEKF, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Relative Drift (% of Traveled Distance)", fontsize=12, fontweight="bold")
    ax.set_xlabel("GNSS Blackout Duration", fontsize=12, fontweight="bold")
    ax.set_title("Relative Drift Percentage: Baseline vs AI + ES-EKF", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(durations, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle="--", alpha=0.5)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Plot 2: {out_path}")


def plot_error_growth(
    durations_num: List[int],
    baseline_errs: List[float],
    ai_iekf_errs: List[float],
    baseline_rmses: List[float],
    ai_iekf_rmses: List[float],
    out_path: Path,
) -> None:
    """Plot 3: Line plot showing error growth trends over blackout duration."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    ax.plot(durations_num, baseline_errs, "o-", color=COLOR_BASELINE, linewidth=2.5, markersize=8, label="Baseline Final Error (m)")
    ax.plot(durations_num, ai_iekf_errs, "s-", color=COLOR_AI_IEKF, linewidth=2.5, markersize=8, label="AI + ES-EKF Final Error (m)")

    ax.plot(durations_num, baseline_rmses, "o--", color=COLOR_BASELINE, linewidth=1.5, alpha=0.6, label="Baseline RMSE (m)")
    ax.plot(durations_num, ai_iekf_rmses, "s--", color=COLOR_AI_IEKF, linewidth=1.5, alpha=0.6, label="AI + ES-EKF RMSE (m)")

    ax.set_ylabel("Localization Error (meters)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Blackout Outage Duration (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Localization Error Growth vs GNSS Outage Duration", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(durations_num)
    ax.set_xticklabels([f"{d}s" for d in durations_num], fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Plot 3: {out_path}")


def plot_representative_trajectory(
    pipeline: AIIEKFPipeline,
    npz_data_path: Path,
    raw_dataset_dir: Path,
    out_path: Path,
) -> None:
    """Plot 4: Spatial 2D trajectory comparison showing GNSS GT vs Baseline DR vs AI+ES-EKF during a 30s blackout."""
    logger.info("Generating representative 2D trajectory plot for 30s blackout segment...")

    npz = np.load(npz_data_path, allow_pickle=True)
    imu_all = npz["imu"]
    session_ids_all = npz["session_ids"]

    # Select representative session S-Vw2
    target_session = "S-Vw2" if "S-Vw2" in session_ids_all else session_ids_all[0]
    mask = (session_ids_all == target_session)
    session_imu = imu_all[mask][:250]  # Take 250 windows (250s slice)
    N_win = len(session_imu)

    gt_lats, gt_lons, gt_hdgs = load_vehicle_ground_truth(
        base_dir=raw_dataset_dir,
        session_id=target_session,
        num_windows=N_win
    )

    timestamps = np.arange(N_win, dtype=np.float64) * 1.0
    ref_lat, ref_lon = float(gt_lats[0]), float(gt_lons[0])

    # Convert GT to local ENU meters
    gt_e = np.zeros(N_win)
    gt_n = np.zeros(N_win)
    for i in range(N_win):
        gt_e[i], gt_n[i] = lat_lon_to_enu_m(float(gt_lats[i]), float(gt_lons[i]), ref_lat, ref_lon)

    # 30s blackout interval around middle of recording (t=100s to 130s)
    blackout_interval = [(100.0, 130.0)]

    # 1. Run Baseline DR
    vel_ckpt = torch.load("models/velocity_model_v2.pt", map_location="cpu", weights_only=False)
    vel_mean, vel_std = float(vel_ckpt["vel_mean"]), float(vel_ckpt["vel_std"])
    vel_imu_mean = np.array(vel_ckpt["imu_mean"], dtype=np.float32)
    vel_imu_std = np.array(vel_ckpt["imu_std"], dtype=np.float32)

    imu_norm = (session_imu - vel_imu_mean) / vel_imu_std
    with torch.no_grad():
        pred_norm, _ = pipeline.velocity_model(torch.tensor(imu_norm, dtype=torch.float32, device=pipeline.device))
        baseline_speeds = (pred_norm * vel_std + vel_mean).squeeze(-1).cpu().numpy()

    gyro_z = session_imu[:, :, 5].mean(axis=1)

    eval_base = evaluate_trajectory_blackout(
        timestamps=timestamps,
        velocities_ms=baseline_speeds,
        gyro_z_rad_s=gyro_z,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=blackout_interval,
        init_heading_deg=float(gt_hdgs[0]),
        gt_headings_deg=gt_hdgs,
    )

    base_e = np.zeros(N_win)
    base_n = np.zeros(N_win)
    for i, pt in enumerate(eval_base.trajectory_estimated):
        base_e[i], base_n[i] = lat_lon_to_enu_m(pt.lat, pt.lon, ref_lat, ref_lon)

    # 2. Run AI + ES-EKF
    eval_ai = pipeline.run_session_blackout(
        imu_windows=session_imu,
        timestamps=timestamps,
        gt_lats=gt_lats,
        gt_lons=gt_lons,
        blackout_intervals=blackout_interval,
        init_heading_deg=float(gt_hdgs[0]),
        gt_headings_deg=gt_hdgs,
    )

    ai_e = np.zeros(N_win)
    ai_n = np.zeros(N_win)
    for i, pt in enumerate(eval_ai.trajectory_estimated):
        ai_e[i], ai_n[i] = lat_lon_to_enu_m(pt.lat, pt.lon, ref_lat, ref_lon)

    # Plot 2D Trajectory
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # Focus slice around blackout window: t=80s to t=150s
    slice_idx = np.where((timestamps >= 80.0) & (timestamps <= 150.0))[0]
    bo_slice = np.where((timestamps >= 100.0) & (timestamps <= 130.0))[0]

    ax.plot(gt_e[slice_idx], gt_n[slice_idx], "k-", linewidth=2.5, label="GNSS Ground Truth", zorder=3)
    ax.plot(base_e[slice_idx], base_n[slice_idx], "--", color=COLOR_BASELINE, linewidth=2.0, label="Baseline DR (Speed + Gyro)", zorder=4)
    ax.plot(ai_e[slice_idx], ai_n[slice_idx], "-", color=COLOR_AI_IEKF, linewidth=2.5, label="NAVIGATE 2.0 (AI + ES-EKF)", zorder=5)

    # Highlight Blackout Segment on GT
    ax.plot(gt_e[bo_slice], gt_n[bo_slice], "-", color=COLOR_BLACKOUT, linewidth=5.0, alpha=0.8, label="30s GNSS Blackout Outage", zorder=2)

    # Start and End points
    ax.scatter(gt_e[bo_slice[0]], gt_n[bo_slice[0]], color="black", s=70, marker="o", zorder=6, label="Blackout Start")
    ax.scatter(gt_e[bo_slice[-1]], gt_n[bo_slice[-1]], color="red", s=90, marker="X", zorder=6, label="Blackout End (GT)")
    ax.scatter(ai_e[bo_slice[-1]], ai_n[bo_slice[-1]], color=COLOR_AI_IEKF, s=90, marker="P", zorder=6, label="AI + ES-EKF End")

    ax.set_xlabel("Local East (meters)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Local North (meters)", fontsize=12, fontweight="bold")
    ax.set_title("Representative 2D Trajectory during 30s GNSS Outage", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9.5, frameon=True, facecolor="white", loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Plot 4: {out_path}")


# ================================================================== #
#  Main Execution
# ================================================================== #

def main() -> None:
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_json_path = PROJECT_ROOT / "results" / "baseline_blackout_results_v2.json"
    ai_json_path = PROJECT_ROOT / "results" / "ai_iekf_blackout_results.json"

    if not base_json_path.exists() or not ai_json_path.exists():
        raise FileNotFoundError("Missing evaluation JSON files in results/")

    with open(base_json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)["overall_mean_by_duration"]

    with open(ai_json_path, "r", encoding="utf-8") as f:
        ai_data = json.load(f)["overall_mean_by_duration"]

    durations_key = ["5s", "10s", "30s", "60s"]
    durations_num = [5, 10, 30, 60]

    baseline_final_errs = [base_data[k]["final_position_error_m"] for k in durations_key]
    ai_iekf_final_errs = [ai_data[k]["final_position_error_m"] for k in durations_key]

    baseline_rmses = [base_data[k]["rmse_position_error_m"] for k in durations_key]
    ai_iekf_rmses = [ai_data[k]["rmse_position_error_m"] for k in durations_key]

    baseline_drifts = [base_data[k]["relative_drift_percent"] for k in durations_key]
    ai_iekf_drifts = [ai_data[k]["relative_drift_percent"] for k in durations_key]

    # 1. Generate Figures
    plot_final_position_error_comparison(durations_key, baseline_final_errs, ai_iekf_final_errs, figures_dir / "final_position_error_comparison.png")
    plot_drift_percentage_comparison(durations_key, baseline_drifts, ai_iekf_drifts, figures_dir / "drift_percentage_comparison.png")
    plot_error_growth(durations_num, baseline_final_errs, ai_iekf_final_errs, baseline_rmses, ai_iekf_rmses, figures_dir / "error_growth_over_blackout_duration.png")

    # Pipeline instance for Plot 4
    pipeline = AIIEKFPipeline(device="cpu")
    raw_dataset_dir = Path(r"D:\Career\Competitons\Devesh Aug-Sep Hackathons\IO-VNBD\Synchronised V abd S datasets")
    plot_representative_trajectory(pipeline, PROJECT_ROOT / "data" / "processed" / "iovnbd_full.npz", raw_dataset_dir, figures_dir / "representative_trajectory_blackout.png")

    # 2. Build Summary Data
    summary_rows = []
    for k in durations_key:
        b_err = base_data[k]["final_position_error_m"]
        a_err = ai_data[k]["final_position_error_m"]
        abs_imp = round(b_err - a_err, 3)
        pct_imp = round(((b_err - a_err) / b_err) * 100.0, 2)
        b_drift = base_data[k]["relative_drift_percent"]
        a_drift = ai_data[k]["relative_drift_percent"]

        summary_rows.append({
            "blackout_duration_s": k,
            "baseline_final_error_m": b_err,
            "ai_iekf_final_error_m": a_err,
            "absolute_improvement_m": abs_imp,
            "percentage_improvement_percent": pct_imp,
            "baseline_drift_percent": b_drift,
            "ai_iekf_drift_percent": a_drift,
        })

    # Save summary JSON
    summary_json_path = PROJECT_ROOT / "results" / "evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({"evaluation_summary": summary_rows}, f, indent=2)
    logger.info(f"Saved summary JSON: {summary_json_path}")

    # Save summary CSV
    summary_csv_path = PROJECT_ROOT / "results" / "evaluation_summary.csv"
    fieldnames = [
        "blackout_duration_s",
        "baseline_final_error_m",
        "ai_iekf_final_error_m",
        "absolute_improvement_m",
        "percentage_improvement_percent",
        "baseline_drift_percent",
        "ai_iekf_drift_percent",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Saved summary CSV: {summary_csv_path}")

    logger.info("==================================================")
    logger.info("Stage 17 visualization & artifact generation COMPLETE.")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
