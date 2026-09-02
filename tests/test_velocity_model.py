"""
Unit tests for the NAVIGATE 2.0 VelocityModel V2 and training pipeline logic.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.models.velocity_model import VelocityModel
from navigate.train_velocity import session_split_train_val_test, evaluate, IOVNBDVelocityDataset


# ================================================================== #
#  Fixtures & Helper Functions
# ================================================================== #

@pytest.fixture
def model() -> VelocityModel:
    """Default model instance, CPU, eval mode."""
    m = VelocityModel(in_channels=6, hidden_size=128, window_size=50, dropout_rate=0.2)
    m.eval()
    return m


def make_batch(B: int = 4, T: int = 50, C: int = 6) -> torch.Tensor:
    """Returns a random IMU tensor [B, T, C]."""
    return torch.randn(B, T, C, dtype=torch.float32)


# ================================================================== #
#  Model Architecture V2 & Shape Tests
# ================================================================== #

def test_output_shape_single_item(model):
    """Model must accept [1, 50, 6] and return [1, 1]."""
    x = make_batch(B=1)
    speed, hx = model(x)
    assert speed.shape == (1, 1), f"Expected (1,1), got {speed.shape}"
    assert hx.shape == (2, 1, 128)  # 2-layer GRU


def test_output_shape_batch(model):
    """Model must accept [B, 50, 6] and return [B, 1]."""
    for B in [1, 4, 32, 64]:
        x = make_batch(B=B)
        speed, hx = model(x)
        assert speed.shape == (B, 1), f"B={B}: Expected ({B},1), got {speed.shape}"
        assert hx.shape == (2, B, 128)


def test_output_no_nan(model):
    """Forward pass must produce no NaN or Inf."""
    x = make_batch(B=8)
    speed, hx = model(x)
    assert not torch.isnan(speed).any(), "NaN in speed output"
    assert not torch.isinf(speed).any(), "Inf in speed output"
    assert not torch.isnan(hx).any(), "NaN in hidden state"


def test_parameter_count(model):
    """Model V2 parameter count should be ~359k (reduced for generalization)."""
    n_params = model.count_parameters()
    assert n_params > 0
    assert 250_000 < n_params < 500_000, f"Expected ~359k params in V2, got {n_params:,}"


def test_does_not_assume_50hz_or_200hz():
    """Model config must explicitly reflect 10 Hz / 50-sample windows."""
    m = VelocityModel()
    assert m.window_size == 50, f"window_size should be 50 (10 Hz × 5s), got {m.window_size}"
    assert m.in_channels == 6, f"in_channels should be 6, got {m.in_channels}"
    assert m._seq_len == 10, f"Expected seq_len=10 for 50-sample input, got {m._seq_len}"


def test_seq_len_computed_correctly():
    """_compute_seq_len must match analytical formula for window_size=50."""
    import math
    m = VelocityModel(window_size=50)
    L = 50
    L = math.floor((L - m.CONV1_KERNEL + 1) / m.POOL_SIZE)  # 46 -> 23
    L = math.floor((L - m.CONV2_KERNEL + 1) / m.POOL_SIZE)  # 21 -> 10
    assert m._seq_len == L


def test_architecture_summary(model):
    """architecture_summary() must return a formatted summary string."""
    s = model.architecture_summary()
    assert isinstance(s, str)
    assert len(s) > 50
    assert "VelocityModel V2" in s


def test_backward_pass_works():
    """Backpropagation must compute non-None gradients for all trainable parameters."""
    model = VelocityModel()
    model.train()
    x = make_batch(B=4)
    target = torch.rand(4, 1)
    pred, _ = model(x)
    loss = torch.nn.functional.smooth_l1_loss(pred, target)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_hidden_state_propagation():
    """GRU hidden state must carry information across sequential calls."""
    model = VelocityModel()
    model.eval()
    x = make_batch(B=2)
    with torch.no_grad():
        _, hx1 = model(x)
        _, hx2 = model(x, hx=hx1)
    assert not torch.allclose(hx1, hx2), "Hidden state did not update"


# ================================================================== #
#  Session Split & Pipeline Tests
# ================================================================== #

def test_full_144_session_split_simulation():
    """
    Verifies 80/10/10 split on a synthetic 144-session dataset (matching full IO-VNBD).
    Checks: exactly 144 sessions total, 80/10/10 ratio (116/14/14), zero overlap, determinism.
    """
    session_ids = np.array([f"S-{i:03d}" for i in range(144)] * 10)
    
    train_idx, val_idx, test_idx, partitions = session_split_train_val_test(
        session_ids, val_fraction=0.1, test_fraction=0.1, seed=42
    )
    
    train_set = set(partitions["train"])
    val_set = set(partitions["val"])
    test_set = set(partitions["test"])
    
    assert len(train_set) + len(val_set) + len(test_set) == 144
    assert len(train_set) == 116
    assert len(val_set) == 14
    assert len(test_set) == 14
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_2_session_smoke_split_safety():
    """
    Verifies that for a 2-session smoke dataset, sessions are NEVER duplicated
    across val and test partitions, and set intersection remains strictly empty.
    """
    session_ids = np.array(["S-M", "S-S1"] * 50)
    
    train_idx, val_idx, test_idx, partitions = session_split_train_val_test(
        session_ids, val_fraction=0.1, test_fraction=0.1, seed=42
    )
    
    train_set = set(partitions["train"])
    val_set = set(partitions["val"])
    test_set = set(partitions["test"])
    
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(test_set) == 0
    assert len(test_idx) == 0


def test_test_metric_calculation_unnormalization():
    """Verifies un-normalization metric calculation logic in evaluate function."""
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn

    # Dummy model predicting normalized speed 0.0
    class DummyModel(nn.Module):
        def forward(self, x, hx=None):
            return torch.full((len(x), 1), 0.0), None

    dummy_model = DummyModel()
    imu_dummy = torch.zeros(10, 50, 6)
    vel_norm_dummy = torch.full((10, 1), 0.0)
    vel_raw_dummy = torch.full((10, 1), 10.0)  # Ground truth 10.0 m/s
    
    dataset = TensorDataset(imu_dummy, vel_norm_dummy, vel_raw_dummy)
    loader = DataLoader(dataset, batch_size=5)

    loss_fn = nn.SmoothL1Loss()
    vel_mean = 5.0
    vel_std = 2.0
    # Model predicts 0.0 norm -> pred_ms = 0.0 * 2.0 + 5.0 = 5.0 m/s
    # Ground truth = 10.0 m/s -> Error = |5.0 - 10.0| = 5.0 m/s -> MAE km/h = 18.0

    eval_loss, mae_ms, mae_kmh = evaluate(dummy_model, loader, loss_fn, torch.device("cpu"), vel_mean, vel_std)

    assert pytest.approx(mae_ms, abs=1e-4) == 5.0
    assert pytest.approx(mae_kmh, abs=1e-4) == 18.0  # 5.0 * 3.6
