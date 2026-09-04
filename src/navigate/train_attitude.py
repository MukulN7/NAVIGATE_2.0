"""
train_attitude.py — IO-VNBD Attitude Model Training Pipeline for NAVIGATE 2.0.

Trains AttitudeModel to estimate 5-second relative vehicle orientation quaternions
[qw, qx, qy, qz] from 6-axis smartphone IMU sliding windows (50 samples at 10 Hz).

Key Implementation Principles
-----------------------------
1. Strictly Disjoint Session Splitting: Guarantees zero session leakage across
   train, validation, and test splits (set(train) ∩ set(val) == ∅).
2. 4D Target Quaternion Reconstruction: Reconstructs full 4D unit quaternion
   q_target = [qw, qx, qy, qz] from stored 3D vector component [qx, qy, qz]:
       qw = sqrt(clamp(1.0 - (qx² + qy² + qz²), min=0.0))
3. Quaternion Cosine Loss: Invariant to antipodal representations (q ≡ -q):
       loss = 1.0 - (pred · target)²
4. Angular Error Metric: Calculates geodesic 3D rotation error in degrees:
       angle_deg = rad2deg(2 * acos(|pred · target|))
   Tracks mean, median, and 90th percentile angular error on validation set.
5. Early Stopping & Checkpoint: Saves best model state to models/attitude_model.pt
   monitoring validation mean angular error in degrees.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navigate.models.attitude_model import AttitudeModel

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_attitude")


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================== #
#  4D Target Quaternion Reconstruction
# ================================================================== #

def reconstruct_target_quaternions(rel_q: np.ndarray) -> np.ndarray:
    """
    Reconstructs full 4D unit quaternions [qw, qx, qy, qz] from stored 3D vector components [qx, qy, qz].
    qw = sqrt(max(0.0, 1.0 - (qx² + qy² + qz²)))
    """
    v_sq = np.sum(rel_q ** 2, axis=-1, keepdims=True)
    qw = np.sqrt(np.maximum(0.0, 1.0 - v_sq))
    q_4d = np.hstack([qw, rel_q]).astype(np.float32)
    norms = np.linalg.norm(q_4d, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (q_4d / norms).astype(np.float32)


# ================================================================== #
#  Loss & Metric Functions
# ================================================================== #

def quaternion_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Quaternion inner product / cosine loss: loss = 1.0 - (pred · target)²
    Invariant to antipodal orientation representation (q ≡ -q).
    """
    dot = torch.sum(pred * target, dim=-1)
    return torch.mean(1.0 - (dot ** 2))


def calculate_angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates geodesic 3D rotation error in degrees per sample:
    angle_rad = 2 * acos(|pred · target|)
    angle_deg = rad2deg(angle_rad)
    Normalizes inputs to unit norm to prevent float precision drift in acos.
    """
    pred_n = torch.nn.functional.normalize(pred, p=2, dim=-1)
    target_n = torch.nn.functional.normalize(target, p=2, dim=-1)
    abs_dot = torch.clamp(torch.abs(torch.sum(pred_n * target_n, dim=-1)), max=1.0)
    angle_rad = 2.0 * torch.acos(abs_dot)
    return torch.rad2deg(angle_rad)


# ================================================================== #
#  PyTorch Dataset Definition
# ================================================================== #

class IOVNBDAttitudeDataset(Dataset):
    """
    PyTorch Dataset for IO-VNBD attitude estimation.
    Normalizes IMU features using training-set statistics and reconstructs 4D unit target quaternions.
    """

    def __init__(
        self,
        imu: np.ndarray,
        rel_quaternion_3d: np.ndarray,
        imu_mean: np.ndarray,
        imu_std: np.ndarray,
    ) -> None:
        # Standardize 6-channel IMU features
        self.imu = torch.tensor((imu - imu_mean) / imu_std, dtype=torch.float32)
        # Reconstruct 4D unit target quaternion [qw, qx, qy, qz]
        q_4d = reconstruct_target_quaternions(rel_quaternion_3d)
        self.target = torch.tensor(q_4d, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.imu[idx], self.target[idx]


# ================================================================== #
#  Strictly Disjoint Session-Wise Split (Train / Val / Test)
# ================================================================== #

def session_split_train_val_test(
    session_ids: np.ndarray,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], Dict[str, List[str]]]:
    """
    Splits windows into Train, Validation, and Test sets by SESSION ID.
    Guarantees zero session leakage across splits.
    """
    unique_sessions = sorted(list(set(session_ids)))
    n_sessions = len(unique_sessions)
    
    rng = random.Random(seed)
    shuffled_sessions = list(unique_sessions)
    rng.shuffle(shuffled_sessions)
    
    if n_sessions >= 3:
        n_val = max(1, int(round(n_sessions * val_fraction)))
        n_test = max(1, int(round(n_sessions * test_fraction)))
        n_train = n_sessions - n_val - n_test
        
        if n_train <= 0:
            n_train = 1
            n_val = max(1, (n_sessions - 1) // 2)
            n_test = n_sessions - n_train - n_val

        train_sessions = sorted(shuffled_sessions[:n_train])
        val_sessions = sorted(shuffled_sessions[n_train:n_train + n_val])
        test_sessions = sorted(shuffled_sessions[n_train + n_val:])
    elif n_sessions == 2:
        logger.warning(
            f"[WARNING] Dataset has only 2 sessions ({shuffled_sessions}). "
            f"Performing safe 2-way split: 1 Train session, 1 Val session, 0 Test sessions."
        )
        train_sessions = [shuffled_sessions[0]]
        val_sessions = [shuffled_sessions[1]]
        test_sessions = []
    else:  # n_sessions == 1
        logger.warning(
            f"[WARNING] Dataset has only 1 session ({shuffled_sessions}). "
            f"Assigning 1 Train session, 0 Val sessions, 0 Test sessions."
        )
        train_sessions = [shuffled_sessions[0]]
        val_sessions = []
        test_sessions = []

    train_set, val_set, test_set = set(train_sessions), set(val_sessions), set(test_sessions)

    assert train_set.isdisjoint(val_set), f"Train and Val overlap: {train_set & val_set}"
    assert train_set.isdisjoint(test_set), f"Train and Test overlap: {train_set & test_set}"
    assert val_set.isdisjoint(test_set), f"Val and Test overlap: {val_set & test_set}"

    train_idx = [i for i, s in enumerate(session_ids) if s in train_set]
    val_idx = [i for i, s in enumerate(session_ids) if s in val_set]
    test_idx = [i for i, s in enumerate(session_ids) if s in test_set]

    partition_sessions = {
        "train": train_sessions,
        "val": val_sessions,
        "test": test_sessions
    }
    return train_idx, val_idx, test_idx, partition_sessions


# ================================================================== #
#  Evaluation Routine
# ================================================================== #

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """
    Evaluates AttitudeModel on a dataset partition.
    Returns (loss, mean_angular_error_deg, median_angular_error_deg, p90_angular_error_deg).
    """
    model.eval()
    total_loss = 0.0
    all_errors_deg = []

    with torch.no_grad():
        for imu, target in dataloader:
            imu, target = imu.to(device), target.to(device)
            pred, _ = model(imu)

            loss = quaternion_cosine_loss(pred, target)
            errors_deg = calculate_angular_error_deg(pred, target)

            total_loss += loss.item() * len(target)
            all_errors_deg.extend(errors_deg.cpu().numpy().tolist())

    n_samples = len(dataloader.dataset)
    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / n_samples
    errors_arr = np.array(all_errors_deg, dtype=np.float32)

    mean_deg = float(np.mean(errors_arr))
    median_deg = float(np.median(errors_arr))
    p90_deg = float(np.percentile(errors_arr, 90))

    return avg_loss, mean_deg, median_deg, p90_deg


# ================================================================== #
#  Training Runner
# ================================================================== #

def train_attitude_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Main training routine for AttitudeModel."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Processed NPZ Dataset
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    logger.info(f"Loading processed dataset: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    imu_all = data["imu"]                      # [N, 50, 6]
    rel_q_all = data["rel_quaternion"]         # [N, 3]
    session_ids_all = data["session_ids"]       # [N]

    logger.info(f"Total dataset windows: {len(imu_all):,} | Total sessions: {len(set(session_ids_all))}")

    # 2. Perform Session-Wise Train/Val/Test Split
    train_idx, val_idx, test_idx, partition_sessions = session_split_train_val_test(
        session_ids=session_ids_all,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=args.seed
    )

    logger.info(f"Split Summary (Session-Wise Disjoint):")
    logger.info(f"  Train: {len(train_idx):>6,} windows ({len(partition_sessions['train']):>2d} sessions)")
    logger.info(f"  Val  : {len(val_idx):>6,} windows ({len(partition_sessions['val']):>2d} sessions)")
    logger.info(f"  Test : {len(test_idx):>6,} windows ({len(partition_sessions['test']):>2d} sessions)")

    # 3. Compute Feature Normalization Stats on TRAIN SET ONLY
    train_imu_raw = imu_all[train_idx]
    imu_flat = train_imu_raw.reshape(-1, 6)
    imu_mean = np.mean(imu_flat, axis=0).astype(np.float32)
    imu_std = np.std(imu_flat, axis=0).astype(np.float32)
    imu_std[imu_std == 0] = 1.0

    # 4. Construct Datasets & DataLoaders
    train_dataset = IOVNBDAttitudeDataset(
        imu=train_imu_raw,
        rel_quaternion_3d=rel_q_all[train_idx],
        imu_mean=imu_mean,
        imu_std=imu_std
    )
    val_dataset = IOVNBDAttitudeDataset(
        imu=imu_all[val_idx],
        rel_quaternion_3d=rel_q_all[val_idx],
        imu_mean=imu_mean,
        imu_std=imu_std
    )
    test_dataset = IOVNBDAttitudeDataset(
        imu=imu_all[test_idx],
        rel_quaternion_3d=rel_q_all[test_idx],
        imu_mean=imu_mean,
        imu_std=imu_std
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Instantiate AttitudeModel & Optimizer
    model = AttitudeModel(
        in_channels=6,
        hidden_size=128,
        window_size=50,
        dropout_rate=0.25
    ).to(device)

    logger.info(f"Instantiated AttitudeModel ({model.count_parameters():,} trainable parameters)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    output_ckpt_path = Path(args.output)
    output_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Training Loop with Early Stopping
    best_val_angle_deg = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    logger.info("==================================================")
    logger.info(f"Starting AttitudeModel Training ({args.epochs} Epochs)")
    logger.info("==================================================")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_errors_deg = []

        for imu_b, target_b in train_loader:
            imu_b, target_b = imu_b.to(device), target_b.to(device)

            optimizer.zero_grad()
            pred_b, _ = model(imu_b)

            loss = quaternion_cosine_loss(pred_b, target_b)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(target_b)
            with torch.no_grad():
                errs = calculate_angular_error_deg(pred_b, target_b)
                train_errors_deg.extend(errs.cpu().numpy().tolist())

        avg_train_loss = train_loss / len(train_dataset)
        train_mean_deg = float(np.mean(train_errors_deg))

        # Validation Phase
        val_loss, val_mean_deg, val_median_deg, val_p90_deg = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch:>2d}/{args.epochs:>2d} | "
            f"Train Loss: {avg_train_loss:.5f} ({train_mean_deg:5.2f}°) | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Angle: Mean={val_mean_deg:5.2f}° Med={val_median_deg:5.2f}° P90={val_p90_deg:5.2f}°"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mean_angle_deg": train_mean_deg,
            "val_loss": val_loss,
            "val_mean_angle_deg": val_mean_deg,
            "val_median_angle_deg": val_median_deg,
            "val_p90_angle_deg": val_p90_deg,
        })

        # Save Checkpoint if Validation Angular Error improves
        if val_mean_deg < best_val_angle_deg:
            best_val_angle_deg = val_mean_deg
            best_epoch = epoch
            patience_counter = 0

            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_angle_deg": best_val_angle_deg,
                "val_metrics": {
                    "val_loss": val_loss,
                    "val_mean_angle_deg": val_mean_deg,
                    "val_median_angle_deg": val_median_deg,
                    "val_p90_angle_deg": val_p90_deg,
                },
                "model_config": {
                    "in_channels": model.in_channels,
                    "hidden_size": model.hidden_size,
                    "window_size": model.window_size,
                    "dropout_rate": model.dropout_rate,
                },
                "imu_mean": imu_mean.tolist(),
                "imu_std": imu_std.tolist(),
                "partition_sessions": partition_sessions,
            }
            torch.save(checkpoint_dict, output_ckpt_path)
            logger.info(f"  [SAVED CHECKPOINT] Best Val Angle: {best_val_angle_deg:.2f}° @ Epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    # 7. Final Evaluation on Held-Out Test Set
    if output_ckpt_path.exists() and len(test_dataset) > 0:
        logger.info("==================================================")
        logger.info("Evaluating Best Checkpoint on Held-Out Test Set...")
        ckpt = torch.load(output_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_mean_deg, test_median_deg, test_p90_deg = evaluate(model, test_loader, device)

        logger.info(
            f"Test Set Evaluation Results (Best Epoch {best_epoch}): "
            f"Loss={test_loss:.5f} | "
            f"Mean Angle={test_mean_deg:.2f}° | "
            f"Med Angle={test_median_deg:.2f}° | "
            f"P90 Angle={test_p90_deg:.2f}°"
        )

    return {
        "best_epoch": best_epoch,
        "best_val_angle_deg": best_val_angle_deg,
        "history": history,
        "partition_sessions": partition_sessions,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train AttitudeModel for NAVIGATE 2.0")
    p.add_argument("--data", type=str, default="data/processed/iovnbd_full.npz", help="Path to processed dataset NPZ")
    p.add_argument("--output", type=str, default="models/attitude_model.pt", help="Path to save best model checkpoint")
    p.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--smoke", action="store_true", help="Run fast smoke test training on 1 epoch")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch_size = 16
    train_attitude_pipeline(args)
