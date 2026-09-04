"""
Unit tests for NAVIGATE 2.0 train_attitude.py pipeline.

Covers:
- Target 4D quaternion reconstruction from 3D vector components
- Quaternion cosine loss calculation & antipodal sign invariance (q ≡ -q)
- Angular error calculation in degrees
- Session-wise disjoint split safety (zero overlap across train/val/test)
- One tiny training step execution (smoke run)
- Checkpoint structure and mandatory metadata keys
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.train_attitude import (
    reconstruct_target_quaternions,
    quaternion_cosine_loss,
    calculate_angular_error_deg,
    session_split_train_val_test,
    train_attitude_pipeline,
    build_parser,
    IOVNBDAttitudeDataset,
)


# ------------------------------------------------------------------ #
#  1. Target Reconstruction Tests
# ------------------------------------------------------------------ #

def test_target_reconstruction_shape_and_norm():
    """Verifies 3D -> 4D quaternion reconstruction and unit norm property."""
    # 5 random 3D vector components with norm <= 0.8
    rel_q = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [0.5, 0.0, 0.0],
        [0.0, 0.6, 0.0],
        [-0.2, 0.3, -0.4],
    ], dtype=np.float32)

    q_4d = reconstruct_target_quaternions(rel_q)

    assert q_4d.shape == (5, 4), f"Expected shape (5, 4), got {q_4d.shape}"
    norms = np.linalg.norm(q_4d, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Reconstructed quaternions not unit norm: {norms}"
    # First row is identity rotation -> [1, 0, 0, 0]
    assert np.allclose(q_4d[0], [1.0, 0.0, 0.0, 0.0], atol=1e-5)


def test_target_reconstruction_clamping_for_large_vectors():
    """Verifies safe handling when input vector norm approaches or exceeds 1.0."""
    rel_q = np.array([
        [1.0, 0.0, 0.0],
        [0.999999, 0.0, 0.0],
        [1.05, 0.0, 0.0],  # slightly > 1 due to float precision
    ], dtype=np.float32)

    q_4d = reconstruct_target_quaternions(rel_q)
    assert not np.any(np.isnan(q_4d))
    assert not np.any(np.isinf(q_4d))
    norms = np.linalg.norm(q_4d, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ------------------------------------------------------------------ #
#  2. Loss Function Tests
# ------------------------------------------------------------------ #

def test_quaternion_loss_identical_quaternions():
    """Identical quaternions must yield zero loss."""
    c45 = float(np.cos(np.pi / 4))
    s45 = float(np.sin(np.pi / 4))
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [c45, 0.0, 0.0, s45]], dtype=torch.float32)
    q2 = q1.clone()
    loss = quaternion_cosine_loss(q1, q2)
    assert pytest.approx(loss.item(), abs=1e-5) == 0.0


def test_quaternion_loss_antipodal_invariance():
    """Loss must be zero for antipodal quaternions q and -q (q ≡ -q)."""
    c45 = float(np.cos(np.pi / 4))
    s45 = float(np.sin(np.pi / 4))
    q1 = torch.tensor([[c45, 0.0, s45, 0.0]], dtype=torch.float32)
    q2 = -q1
    loss = quaternion_cosine_loss(q1, q2)
    assert pytest.approx(loss.item(), abs=1e-5) == 0.0


def test_quaternion_loss_orthogonal_quaternions():
    """Orthogonal quaternions must yield maximum loss of 1.0."""
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    loss = quaternion_cosine_loss(q1, q2)
    assert pytest.approx(loss.item(), abs=1e-5) == 1.0


# ------------------------------------------------------------------ #
#  3. Angular Error Calculation Tests
# ------------------------------------------------------------------ #

def test_angular_error_identical_quaternions():
    """Identical quaternions must yield 0.0° angular error."""
    c45 = float(np.cos(np.pi / 4))
    s45 = float(np.sin(np.pi / 4))
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [c45, 0.0, 0.0, s45]], dtype=torch.float32)
    q2 = q1.clone()
    err_deg = calculate_angular_error_deg(q1, q2)
    assert torch.allclose(err_deg, torch.tensor([0.0, 0.0]), atol=1e-4)


def test_angular_error_known_90_degree_rotation():
    """90° rotation quaternion [cos(45°), 0, 0, sin(45°)] vs identity must yield ~90.0° error."""
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    # 90-degree yaw rotation around Z axis: cos(45°)=0.7071068, sin(45°)=0.7071068
    c45 = float(np.cos(np.pi / 4))
    s45 = float(np.sin(np.pi / 4))
    q_90_yaw = torch.tensor([[c45, 0.0, 0.0, s45]], dtype=torch.float32)

    err_deg = calculate_angular_error_deg(q_identity, q_90_yaw)
    assert pytest.approx(err_deg.item(), abs=1e-3) == 90.0


def test_angular_error_antipodal_is_zero():
    """q and -q must have 0.0° angular error."""
    q1 = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    q2 = -q1
    err_deg = calculate_angular_error_deg(q1, q2)
    assert pytest.approx(err_deg.item(), abs=1e-4) == 0.0


# ------------------------------------------------------------------ #
#  4. Session-Wise Disjoint Split Safety Tests
# ------------------------------------------------------------------ #

def test_session_split_disjoint_full_dataset_simulation():
    """Verifies that train/val/test partitions have zero session overlap for 10 sessions."""
    sessions = [f"S-S{i}" for i in range(1, 11)]
    # Repeat sessions to simulate 1,000 dataset windows
    session_ids = np.random.choice(sessions, size=1000)

    train_idx, val_idx, test_idx, partition_sessions = session_split_train_val_test(
        session_ids=session_ids, val_fraction=0.1, test_fraction=0.1, seed=42
    )

    train_set = set(partition_sessions["train"])
    val_set = set(partition_sessions["val"])
    test_set = set(partition_sessions["test"])

    # Disjoint checks
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(sessions)

    # Window count checks
    assert len(train_idx) + len(val_idx) + len(test_idx) == 1000


def test_session_split_smoke_2_session_safety():
    """Verifies safe 2-way split when dataset has only 2 sessions."""
    session_ids = np.array(["S-M"] * 100 + ["S-Vfa01"] * 50)
    train_idx, val_idx, test_idx, partition_sessions = session_split_train_val_test(
        session_ids=session_ids, seed=42
    )

    assert len(partition_sessions["train"]) == 1
    assert len(partition_sessions["val"]) == 1
    assert len(partition_sessions["test"]) == 0
    assert set(partition_sessions["train"]).isdisjoint(set(partition_sessions["val"]))


# ------------------------------------------------------------------ #
#  5. One Tiny Training Step (Smoke Run)
# ------------------------------------------------------------------ #

def test_one_tiny_training_step(tmp_path):
    """Runs 1 epoch of training on a small synthetic NPZ dataset."""
    N = 100
    imu_synth = np.random.randn(N, 50, 6).astype(np.float32)
    rel_q_synth = (np.random.randn(N, 3) * 0.1).astype(np.float32)
    session_synth = np.array(["S-S1"] * 50 + ["S-S2"] * 30 + ["S-S3"] * 20)

    synth_npz_path = tmp_path / "iovnbd_synth.npz"
    ckpt_path = tmp_path / "attitude_model_synth.pt"

    np.savez_compressed(
        synth_npz_path,
        imu=imu_synth,
        rel_quaternion=rel_q_synth,
        session_ids=session_synth,
    )

    parser = build_parser()
    args = parser.parse_args([
        "--data", str(synth_npz_path),
        "--output", str(ckpt_path),
        "--epochs", "1",
        "--batch-size", "16",
        "--lr", "1e-3",
        "--seed", "42"
    ])

    res = train_attitude_pipeline(args)

    assert res["best_epoch"] == 1
    assert res["best_val_angle_deg"] < 180.0
    assert ckpt_path.exists()


# ------------------------------------------------------------------ #
#  6. Checkpoint Structure & Metadata Keys
# ------------------------------------------------------------------ #

def test_checkpoint_structure_and_keys(tmp_path):
    """Verifies that the saved checkpoint contains all mandatory keys."""
    N = 80
    imu_synth = np.random.randn(N, 50, 6).astype(np.float32)
    rel_q_synth = (np.random.randn(N, 3) * 0.1).astype(np.float32)
    session_synth = np.array(["S-A"] * 40 + ["S-B"] * 20 + ["S-C"] * 20)

    synth_npz = tmp_path / "test_synth.npz"
    ckpt_file = tmp_path / "attitude_model.pt"

    np.savez_compressed(
        synth_npz,
        imu=imu_synth,
        rel_quaternion=rel_q_synth,
        session_ids=session_synth
    )

    parser = build_parser()
    args = parser.parse_args([
        "--data", str(synth_npz),
        "--output", str(ckpt_file),
        "--epochs", "1",
        "--batch-size", "16",
        "--seed", "42"
    ])

    train_attitude_pipeline(args)

    assert ckpt_file.exists()
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # Mandatory Checkpoint Keys Check
    required_keys = [
        "model_state_dict",
        "epoch",
        "best_val_angle_deg",
        "val_metrics",
        "model_config",
        "imu_mean",
        "imu_std",
        "partition_sessions"
    ]
    for k in required_keys:
        assert k in ckpt, f"Missing key in checkpoint: {k}"

    assert ckpt["model_config"]["in_channels"] == 6
    assert ckpt["model_config"]["hidden_size"] == 128
    assert len(ckpt["imu_mean"]) == 6
    assert len(ckpt["imu_std"]) == 6
