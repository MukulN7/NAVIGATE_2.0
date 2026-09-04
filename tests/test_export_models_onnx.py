"""
test_export_models_onnx.py — Unit tests for ONNX model export and verification.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from navigate.models.velocity_model import VelocityModel
from navigate.models.attitude_model import AttitudeModel


@pytest.mark.skipif(not HAS_ONNX, reason="ONNX or ONNX Runtime not installed")
def test_velocity_model_v2_onnx_export_and_inference():
    onnx_path = Path("models/velocity_model_v2.onnx")
    pt_path = Path("models/velocity_model_v2.pt")

    assert pt_path.exists(), f"PyTorch checkpoint {pt_path} missing"
    assert onnx_path.exists(), f"ONNX model file {onnx_path} missing"

    # 1. Load ONNX model and check validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. Load PyTorch model
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    pt_model = VelocityModel(
        in_channels=cfg.get("in_channels", 6),
        hidden_size=cfg.get("hidden_size", 128),
        window_size=cfg.get("window_size", 50),
        dropout_rate=cfg.get("dropout_rate", 0.2),
    )
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()

    # 3. Test ORT Session vs PyTorch
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    torch.manual_seed(123)
    test_x = torch.randn(2, 50, 6, dtype=torch.float32)

    with torch.no_grad():
        pt_speed, pt_hx = pt_model(test_x)

    ort_inputs = {"imu_input": test_x.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_speed = ort_outputs[0]

    max_diff = np.max(np.abs(pt_speed.numpy() - ort_speed))
    assert max_diff < 1e-4, f"VelocityModel ONNX vs PyTorch diff too large: {max_diff}"


@pytest.mark.skipif(not HAS_ONNX, reason="ONNX or ONNX Runtime not installed")
def test_attitude_model_onnx_export_and_inference():
    onnx_path = Path("models/attitude_model.onnx")
    pt_path = Path("models/attitude_model.pt")

    assert pt_path.exists(), f"PyTorch checkpoint {pt_path} missing"
    assert onnx_path.exists(), f"ONNX model file {onnx_path} missing"

    # 1. Load ONNX model and check validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. Load PyTorch model
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    pt_model = AttitudeModel(
        in_channels=cfg.get("in_channels", 6),
        hidden_size=cfg.get("hidden_size", 128),
        window_size=cfg.get("window_size", 50),
        dropout_rate=cfg.get("dropout_rate", 0.25),
    )
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()

    # 3. Test ORT Session vs PyTorch
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    torch.manual_seed(123)
    test_x = torch.randn(2, 50, 6, dtype=torch.float32)

    with torch.no_grad():
        pt_quat, pt_hx = pt_model(test_x)

    ort_inputs = {"imu_input": test_x.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_quat = ort_outputs[0]

    max_diff = np.max(np.abs(pt_quat.numpy() - ort_quat))
    assert max_diff < 1e-4, f"AttitudeModel ONNX vs PyTorch diff too large: {max_diff}"

    # 4. Verify Quaternion L2 Normalization (||q||_2 == 1.0)
    ort_norms = np.linalg.norm(ort_quat, axis=-1)
    assert np.allclose(ort_norms, 1.0, atol=1e-4), f"ONNX Quaternion norm not 1.0: {ort_norms}"
