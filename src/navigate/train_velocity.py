"""
train_velocity.py — IO-VNBD Velocity Model Training Pipeline V2 for NAVIGATE 2.0.

Key Improvements in V2:
1. Target Normalization: Normalizes velocity targets using TRAIN-SET statistics only.
   Un-normalizes predictions back to m/s for metric evaluation.
2. Robust Loss: Uses SmoothL1Loss (Huber Loss) on normalized targets to reduce
   sensitivity to high-speed outliers.
3. Early Stopping: Tracks Validation MAE (km/h) with configurable patience and restores
   the best checkpoint automatically.
4. Comprehensive Target Inspection: Prints Train/Val/Test velocity mean, std, min, max
   in m/s and km/h before training begins.
5. Strictly Disjoint Session Splitting: Guarantees zero session leakage across splits.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navigate.models.velocity_model import VelocityModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_velocity")


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================== #
#  Dataset Definition with Dual Target Normalization
# ================================================================== #

class IOVNBDVelocityDataset(Dataset):
    """
    PyTorch Dataset for IO-VNBD velocity estimation.
    Normalizes IMU features and velocity targets using training-set statistics.
    """

    def __init__(
        self,
        imu: np.ndarray,
        velocity: np.ndarray,
        imu_mean: np.ndarray,
        imu_std: np.ndarray,
        vel_mean: float,
        vel_std: float,
    ) -> None:
        self.imu = torch.tensor((imu - imu_mean) / imu_std, dtype=torch.float32)
        # Normalized target for model training
        self.vel_norm = torch.tensor((velocity - vel_mean) / vel_std, dtype=torch.float32).unsqueeze(-1)
        # Ground-truth raw velocity in m/s for exact metric computation
        self.vel_raw = torch.tensor(velocity, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.vel_raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.imu[idx], self.vel_norm[idx], self.vel_raw[idx]


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
    
    Guarantees:
    - Zero session leakage: Every session ID belongs to EXACTLY one partition.
    - Strictly disjoint sets: set(train) ∩ set(val) == ∅, set(train) ∩ set(test) == ∅, set(val) ∩ set(test) == ∅.
    - Deterministic partitioning using the provided random seed.
    - Safe handling for small datasets: never duplicates sessions across partitions.
    """
    unique_sessions = sorted(list(set(session_ids)))
    n_sessions = len(unique_sessions)
    
    # Shuffle unique session list deterministically
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
            f"Requested 3-way split (80/10/10) is impossible. "
            f"Performing safe 2-way split: 1 Train session ({shuffled_sessions[:1]}), 1 Validation session ({shuffled_sessions[1:]}), 0 Test sessions."
        )
        train_sessions = [shuffled_sessions[0]]
        val_sessions = [shuffled_sessions[1]]
        test_sessions = []
    else:  # n_sessions == 1
        logger.warning(
            f"[WARNING] Dataset has only 1 session ({shuffled_sessions}). "
            f"Requested 3-way split (80/10/10) is impossible. "
            f"Assigning 1 Train session, 0 Validation sessions, 0 Test sessions."
        )
        train_sessions = [shuffled_sessions[0]]
        val_sessions = []
        test_sessions = []

    # Unconditional Disjoint Checks
    train_set = set(train_sessions)
    val_set = set(val_sessions)
    test_set = set(test_sessions)

    assert train_set.isdisjoint(val_set), f"Train and Val sessions overlap: {train_set & val_set}"
    assert train_set.isdisjoint(test_set), f"Train and Test sessions overlap: {train_set & test_set}"
    assert val_set.isdisjoint(test_set), f"Val and Test sessions overlap: {val_set & test_set}"

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
#  Normalization & Metrics Helpers
# ================================================================== #

def compute_train_normalization(
    imu: np.ndarray,
    velocity: np.ndarray,
    train_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Computes IMU and Velocity mean/std from training windows ONLY."""
    train_imu = imu[train_indices].reshape(-1, 6)
    imu_mean = train_imu.mean(axis=0).astype(np.float32)
    imu_std = train_imu.std(axis=0).astype(np.float32)
    imu_std[imu_std == 0] = 1.0

    train_vel = velocity[train_indices]
    vel_mean = float(train_vel.mean())
    vel_std = float(train_vel.std())
    if vel_std == 0:
        vel_std = 1.0

    return imu_mean, imu_std, vel_mean, vel_std


def print_partition_stats(name: str, velocity: np.ndarray, indices: List[int]) -> None:
    """Logs detailed target speed statistics for a partition."""
    if not indices:
        logger.info(f"  {name:10s} -> EMPTY PARTITION")
        return
    part_vel = velocity[indices]
    min_ms, max_ms = part_vel.min(), part_vel.max()
    mean_ms, std_ms = part_vel.mean(), part_vel.std()
    min_kmh, max_kmh = min_ms * 3.6, max_ms * 3.6
    mean_kmh, std_kmh = mean_ms * 3.6, std_ms * 3.6
    logger.info(
        f"  {name:10s} -> Count: {len(indices):>6d} | "
        f"Mean ± Std: {mean_ms:6.2f} ± {std_ms:5.2f} m/s ({mean_kmh:6.2f} ± {std_kmh:5.2f} km/h) | "
        f"Min..Max: [{min_ms:5.2f} .. {max_ms:5.2f}] m/s ([{min_kmh:6.2f} .. {max_kmh:6.2f}] km/h)"
    )


def train_one_epoch(
    model: VelocityModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs one training epoch on normalized targets. Returns average SmoothL1 loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for imu_batch, vel_norm_batch, _ in loader:
        imu_batch = imu_batch.to(device)
        vel_norm_batch = vel_norm_batch.to(device)

        optimizer.zero_grad()
        pred_norm, _ = model(imu_batch)  # [B, 1] in normalized space
        loss = loss_fn(pred_norm, vel_norm_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: VelocityModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    vel_mean: float,
    vel_std: float,
) -> Tuple[float, float, float]:
    """
    Evaluates model performance.
    Un-normalizes predictions back to m/s before computing physical metrics.
    
    Returns
    -------
    loss_val : float (SmoothL1 loss in normalized space)
    mae_ms   : float (Mean Absolute Error in m/s)
    mae_kmh  : float (Mean Absolute Error in km/h)
    """
    model.eval()
    total_loss = 0.0
    total_abs_err_ms = 0.0
    n_samples = 0

    with torch.no_grad():
        for imu_batch, vel_norm_batch, vel_raw_batch in loader:
            imu_batch = imu_batch.to(device)
            vel_norm_batch = vel_norm_batch.to(device)
            vel_raw_batch = vel_raw_batch.to(device)

            pred_norm, _ = model(imu_batch)
            loss = loss_fn(pred_norm, vel_norm_batch)
            total_loss += loss.item() * len(vel_norm_batch)

            # Convert predictions back to raw m/s space
            pred_ms = pred_norm * vel_std + vel_mean

            total_abs_err_ms += (pred_ms - vel_raw_batch).abs().sum().item()
            n_samples += len(vel_raw_batch)

    eval_loss = total_loss / max(n_samples, 1)
    mae_ms = total_abs_err_ms / max(n_samples, 1)
    mae_kmh = mae_ms * 3.6
    return eval_loss, mae_ms, mae_kmh


# ================================================================== #
#  Main Training Pipeline V2
# ================================================================== #

def train_pipeline(args: argparse.Namespace) -> Dict[str, Optional[float]]:
    set_seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # Load dataset archive
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {data_path}")

    logger.info(f"Loading processed dataset: {data_path}")
    npz = np.load(data_path, allow_pickle=True)
    imu_all = npz["imu"]             # [N, 50, 6]
    vel_all = npz["velocity"]        # [N] in m/s
    session_ids = npz["session_ids"]    # [N]

    if args.smoke_windows is not None and args.smoke_windows < len(vel_all):
        logger.info(f"[SMOKE TEST] Limiting to first {args.smoke_windows} windows.")
        imu_all = imu_all[: args.smoke_windows]
        vel_all = vel_all[: args.smoke_windows]
        session_ids = session_ids[: args.smoke_windows]

    # Session-wise split
    train_idx, val_idx, test_idx, partition_sessions = session_split_train_val_test(
        session_ids,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed
    )

    logger.info("==================================================")
    logger.info("SESSION SPLIT SUMMARY:")
    logger.info(f"  TRAIN sessions ({len(partition_sessions['train'])}): {partition_sessions['train']}")
    logger.info(f"  VAL   sessions ({len(partition_sessions['val'])}): {partition_sessions['val']}")
    logger.info(f"  TEST  sessions ({len(partition_sessions['test'])}): {partition_sessions['test']}")
    logger.info("==================================================")
    logger.info("TARGET VELOCITY STATISTICS BEFORE TRAINING:")
    print_partition_stats("TRAIN", vel_all, train_idx)
    print_partition_stats("VAL", vel_all, val_idx)
    print_partition_stats("TEST", vel_all, test_idx)
    logger.info("==================================================")

    # Compute normalisation statistics ONLY on training split
    imu_mean, imu_std, vel_mean, vel_std = compute_train_normalization(imu_all, vel_all, train_idx)
    logger.info(f"Train IMU Normalisation Mean: {imu_mean}")
    logger.info(f"Train IMU Normalisation Std:  {imu_std}")
    logger.info(f"Train Target Velocity Mean:   {vel_mean:.4f} m/s ({vel_mean*3.6:.2f} km/h)")
    logger.info(f"Train Target Velocity Std:    {vel_std:.4f} m/s ({vel_std*3.6:.2f} km/h)")

    # Build PyTorch Datasets & DataLoaders
    full_dataset = IOVNBDVelocityDataset(imu_all, vel_all, imu_mean, imu_std, vel_mean, vel_std)
    train_dataset = Subset(full_dataset, train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = None
    if len(val_idx) > 0:
        val_dataset = Subset(full_dataset, val_idx)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda")
        )

    test_loader = None
    if len(test_idx) > 0:
        test_dataset = Subset(full_dataset, test_idx)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda")
        )

    # Instantiate VelocityModel V2
    model = VelocityModel(in_channels=6, hidden_size=128, window_size=50, dropout_rate=0.2)
    model.to(device)
    logger.info("\n" + model.architecture_summary())
    logger.info(f"Trainable Parameters Count: {model.count_parameters():,}")

    # Robust Loss: Huber / SmoothL1 Loss
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_mae_kmh = float("inf")
    best_epoch = -1
    patience_counter = 0

    logger.info("==================================================")
    logger.info(f"Starting training V2: {args.epochs} epoch(s), batch={args.batch_size}, lr={args.learning_rate}, patience={args.patience}")
    logger.info("==================================================")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        
        if val_loader is not None:
            val_loss, val_mae_ms, val_mae_kmh = evaluate(model, val_loader, loss_fn, device, vel_mean, vel_std)
        else:
            val_loss = train_loss
            val_mae_ms = 0.0
            val_mae_kmh = 0.0

        improving_str = ""
        if val_mae_kmh < best_val_mae_kmh or epoch == 1:
            improving_str = " (IMPROVING ✓)"

        logger.info(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae_ms:.4f} m/s ({val_mae_kmh:.4f} km/h){improving_str}"
        )

        # Early stopping & checkpoint saving based on Validation MAE (km/h)
        if val_loader is not None:
            if val_mae_kmh < best_val_mae_kmh:
                best_val_mae_kmh = val_mae_kmh
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_mae_ms": val_mae_ms,
                        "val_mae_kmh": val_mae_kmh,
                        "imu_mean": imu_mean.tolist(),
                        "imu_std": imu_std.tolist(),
                        "vel_mean": vel_mean,
                        "vel_std": vel_std,
                        "partition_sessions": partition_sessions,
                        "model_config": {
                            "in_channels": 6,
                            "hidden_size": 128,
                            "window_size": 50,
                            "dropout_rate": 0.2
                        }
                    },
                    ckpt_path
                )
                logger.info(f"  ✓ Saved best checkpoint to {ckpt_path} (Val MAE: {best_val_mae_kmh:.4f} km/h)")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"[EARLY STOPPING] Validation MAE did not improve for {args.patience} epochs. Stopping early.")
                    break

    # ============================================================== #
    #  Final Evaluation on Held-Out Test Set (Using Best Checkpoint)
    # ============================================================== #
    logger.info("==================================================")
    if test_loader is not None and ckpt_path.exists():
        logger.info(f"Restoring best checkpoint for TEST evaluation: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_mae_ms, test_mae_kmh = evaluate(model, test_loader, loss_fn, device, vel_mean, vel_std)

        logger.info("FINAL EVALUATION METRICS:")
        logger.info(f"  Best Epoch          : {best_epoch}")
        logger.info(f"  Best Validation MAE : {best_val_mae_kmh:.4f} km/h")
        logger.info(f"  Held-Out Test Loss  : {test_loss:.6f}")
        logger.info(f"  Held-Out Test MAE   : {test_mae_ms:.4f} m/s")
        logger.info(f"  Held-Out Test MAE   : {test_mae_kmh:.4f} km/h")
        logger.info("==================================================")

        return {
            "val_mae_kmh": best_val_mae_kmh,
            "test_loss": test_loss,
            "test_mae_ms": test_mae_ms,
            "test_mae_kmh": test_mae_kmh
        }
    else:
        logger.info("[INFO] Test evaluation skipped (test set is empty).")
        logger.info("==================================================")
        return {
            "val_mae_kmh": best_val_mae_kmh,
            "test_loss": None,
            "test_mae_ms": None,
            "test_mae_kmh": None
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NAVIGATE 2.0 VelocityModel V2 on IO-VNBD")
    p.add_argument("--data", type=str, default="data/processed/iovnbd_smoke_test.npz")
    p.add_argument("--checkpoint", type=str, default="models/velocity_model.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--smoke-windows", type=int, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_pipeline(args)
