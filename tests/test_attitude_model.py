"""
Unit tests for NAVIGATE 2.0 AttitudeModel (attitude_model.py).

Covers:
- Correct output shape: [B, 4]
- Quaternion unit normalization: ||q||_2 = 1 for all outputs
- Forward pass with various batch sizes (1, 4, 16, 32)
- Hidden state output shape: [num_layers=2, B, hidden_size]
- No NaN/Inf in outputs
- Backward pass (gradient flow)
- Architecture parameter count is in expected range (~150k-500k)
- No hard-coded 50Hz / 200Hz assumptions (custom window_size works)
- Sequence length computed correctly: seq_len = window_size // 4
- Hidden state continuity: can pass hx from one call to the next
- Model eval mode: output deterministic (no dropout randomness on inference)
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.models.attitude_model import AttitudeModel


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def model():
    """Default AttitudeModel in eval mode."""
    m = AttitudeModel()
    m.eval()
    return m


@pytest.fixture
def model_train():
    """Default AttitudeModel in train mode."""
    m = AttitudeModel()
    m.train()
    return m


# ------------------------------------------------------------------ #
#  1. Output Shape Tests
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_output_quaternion_shape(model, batch_size):
    """Quaternion output must be [B, 4] for all batch sizes."""
    x = torch.randn(batch_size, 50, 6)
    with torch.no_grad():
        q, _ = model(x)
    assert q.shape == (batch_size, 4), (
        f"Expected shape ({batch_size}, 4), got {q.shape}"
    )


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_output_hidden_state_shape(model, batch_size):
    """GRU hidden state must be [2, B, hidden_size]."""
    x = torch.randn(batch_size, 50, 6)
    with torch.no_grad():
        _, hx = model(x)
    assert hx.shape == (2, batch_size, model.hidden_size), (
        f"Expected hidden shape (2, {batch_size}, {model.hidden_size}), got {hx.shape}"
    )


# ------------------------------------------------------------------ #
#  2. Quaternion Normalization Tests
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_quaternion_unit_norm(model, batch_size):
    """All output quaternions must have L2 norm exactly 1.0."""
    x = torch.randn(batch_size, 50, 6)
    with torch.no_grad():
        q, _ = model(x)
    norms = torch.norm(q, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), (
        f"Quaternion norms deviate from 1.0: {norms}"
    )


def test_quaternion_no_nan(model):
    """Quaternion output must not contain NaN or Inf."""
    x = torch.randn(8, 50, 6)
    with torch.no_grad():
        q, hx = model(x)
    assert not torch.any(torch.isnan(q)), "NaN in quaternion output"
    assert not torch.any(torch.isinf(q)), "Inf in quaternion output"
    assert not torch.any(torch.isnan(hx)), "NaN in hidden state"
    assert not torch.any(torch.isinf(hx)), "Inf in hidden state"


def test_quaternion_norm_with_zero_input(model):
    """Even all-zeros IMU input must produce a valid unit quaternion."""
    x = torch.zeros(2, 50, 6)
    with torch.no_grad():
        q, _ = model(x)
    norms = torch.norm(q, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
    assert not torch.any(torch.isnan(q))


# ------------------------------------------------------------------ #
#  3. Forward Pass Tests
# ------------------------------------------------------------------ #

def test_forward_pass_runs(model):
    """Basic sanity: model forward pass does not raise."""
    x = torch.randn(4, 50, 6)
    with torch.no_grad():
        q, hx = model(x)
    assert q is not None
    assert hx is not None


def test_forward_pass_with_explicit_hx(model):
    """Model accepts pre-supplied GRU hidden state without error."""
    B = 4
    x = torch.randn(B, 50, 6)
    hx_init = torch.zeros(2, B, model.hidden_size)
    with torch.no_grad():
        q, hx_out = model(x, hx=hx_init)
    assert q.shape == (B, 4)
    assert hx_out.shape == (2, B, model.hidden_size)


def test_hidden_state_continuity(model):
    """Hidden state from one call can be passed to the next (streaming)."""
    B = 2
    x1 = torch.randn(B, 50, 6)
    x2 = torch.randn(B, 50, 6)
    with torch.no_grad():
        q1, hx1 = model(x1)
        q2, hx2 = model(x2, hx=hx1)
    assert q1.shape == (B, 4)
    assert q2.shape == (B, 4)
    # Hidden state should change between calls with different input
    assert not torch.allclose(hx1, hx2, atol=1e-4)


# ------------------------------------------------------------------ #
#  4. Backward Pass (Gradient Flow)
# ------------------------------------------------------------------ #

def test_backward_pass_works(model_train):
    """Gradient must flow through the full model."""
    x = torch.randn(4, 50, 6, requires_grad=True)
    q, _ = model_train(x)
    # Geodesic-style loss: minimize distance from identity quaternion [1,0,0,0]
    target = torch.zeros_like(q)
    target[:, 0] = 1.0  # identity quaternion
    loss = F.mse_loss(q, target)
    loss.backward()
    assert x.grad is not None, "No gradient w.r.t. input"
    assert not torch.any(torch.isnan(x.grad)), "NaN in input gradient"


def test_no_dead_gradients(model_train):
    """Parameters must receive gradients."""
    x = torch.randn(4, 50, 6)
    q, _ = model_train(x)
    loss = q.sum()
    loss.backward()
    for name, param in model_train.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for param: {name}"


# ------------------------------------------------------------------ #
#  5. Architecture / Parameter Count
# ------------------------------------------------------------------ #

def test_parameter_count_in_range():
    """Parameter count must be in the lightweight range: 100k–500k."""
    m = AttitudeModel()
    n = m.count_parameters()
    assert 100_000 <= n <= 500_000, (
        f"Unexpected parameter count: {n:,} (expected 100k–500k)"
    )


def test_architecture_summary_string():
    """architecture_summary() must return a non-empty string."""
    m = AttitudeModel()
    s = m.architecture_summary()
    assert isinstance(s, str) and len(s) > 50
    assert "AttitudeModel" in s
    assert "qw" in s
    assert f"{m.count_parameters():,}" in s


# ------------------------------------------------------------------ #
#  6. Frequency / Window Size Independence
# ------------------------------------------------------------------ #

def test_no_hardcoded_50hz_or_200hz_assumption():
    """Model must work correctly for any window_size, not just 50."""
    # 10 Hz x 10 s = 100 samples
    m100 = AttitudeModel(window_size=100)
    x = torch.randn(2, 100, 6)
    with torch.no_grad():
        q, _ = m100(x)
    assert q.shape == (2, 4)
    norms = torch.norm(q, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_seq_len_computed_correctly():
    """Internal seq_len must equal window_size // 4 for default model."""
    m = AttitudeModel(window_size=50)
    assert m._seq_len == 50 // 4, f"Expected 12, got {m._seq_len}"

    m80 = AttitudeModel(window_size=80)
    assert m80._seq_len == 80 // 4, f"Expected 20, got {m80._seq_len}"


# ------------------------------------------------------------------ #
#  7. Eval Mode Determinism
# ------------------------------------------------------------------ #

def test_eval_mode_deterministic():
    """In eval mode, two forward passes with identical input produce identical output."""
    m = AttitudeModel()
    m.eval()
    x = torch.randn(4, 50, 6)
    with torch.no_grad():
        q1, _ = m(x)
        q2, _ = m(x)
    assert torch.allclose(q1, q2), "Non-deterministic output in eval mode"
