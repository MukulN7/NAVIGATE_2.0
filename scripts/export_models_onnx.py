#!/usr/bin/env python3
"""
export_models_onnx.py — Export NAVIGATE 2.0 PyTorch models to ONNX format.

Exports:
  1. models/velocity_model_v2.pt -> models/velocity_model_v2.onnx
  2. models/attitude_model.pt   -> models/attitude_model.onnx

Verifies:
  - ONNX model graph validity using `onnx.checker`
  - Numerical precision & outputs between PyTorch and ONNX Runtime (ORT)
  - Dynamic batch dimension support ([B, 50, 6])
  - Quaternion L2 normalization (norm == 1.0)
"""

import sys
import io
import os
from pathlib import Path

# Force UTF-8 output encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import numpy as np
import torch
import onnx
import onnxruntime as ort

# Add src to Python path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navigate.models.velocity_model import VelocityModel
from navigate.models.attitude_model import AttitudeModel


def export_velocity_model(
    pt_path: Path = Path("models/velocity_model_v2.pt"),
    onnx_path: Path = Path("models/velocity_model_v2.onnx"),
    opset_version: int = 14,
) -> dict:
    """Exports VelocityModel V2 to ONNX and returns verification metrics."""
    print(f"\n--- Exporting VelocityModel V2 ---")
    if not pt_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {pt_path}")

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = VelocityModel(
        in_channels=cfg.get("in_channels", 6),
        hidden_size=cfg.get("hidden_size", 128),
        window_size=cfg.get("window_size", 50),
        dropout_rate=cfg.get("dropout_rate", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 50, 6, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["imu_input"],
        output_names=["speed", "hx_out"],
        dynamic_axes={
            "imu_input": {0: "batch_size"},
            "speed": {0: "batch_size"},
            "hx_out": {1: "batch_size"},
        },
        dynamo=False,
    )
    print(f"Saved ONNX model to: {onnx_path} (Size: {onnx_path.stat().st_size / 1024:.2f} KB)")

    # Verify ONNX structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure check passed.")

    # Numerical verification via ONNX Runtime
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test with batch size 4
    torch.manual_seed(42)
    test_input_pt = torch.randn(4, 50, 6, dtype=torch.float32)

    with torch.no_grad():
        pt_speed, pt_hx = model(test_input_pt)

    ort_inputs = {"imu_input": test_input_pt.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_speed, ort_hx = ort_outputs[0], ort_outputs[1]

    speed_max_diff = float(np.max(np.abs(pt_speed.numpy() - ort_speed)))
    speed_mean_diff = float(np.mean(np.abs(pt_speed.numpy() - ort_speed)))
    hx_max_diff = float(np.max(np.abs(pt_hx.numpy() - ort_hx)))

    print(f"Velocity Speed Max Abs Diff  : {speed_max_diff:.8e}")
    print(f"Velocity Speed Mean Abs Diff : {speed_mean_diff:.8e}")
    print(f"Velocity Hidden Max Abs Diff : {hx_max_diff:.8e}")

    return {
        "onnx_path": str(onnx_path),
        "size_kb": onnx_path.stat().st_size / 1024.0,
        "input_shape": "[B, 50, 6]",
        "output_shape": "[B, 1]",
        "speed_max_diff": speed_max_diff,
        "speed_mean_diff": speed_mean_diff,
    }


def export_attitude_model(
    pt_path: Path = Path("models/attitude_model.pt"),
    onnx_path: Path = Path("models/attitude_model.onnx"),
    opset_version: int = 14,
) -> dict:
    """Exports AttitudeModel to ONNX and returns verification metrics."""
    print(f"\n--- Exporting AttitudeModel ---")
    if not pt_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {pt_path}")

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = AttitudeModel(
        in_channels=cfg.get("in_channels", 6),
        hidden_size=cfg.get("hidden_size", 128),
        window_size=cfg.get("window_size", 50),
        dropout_rate=cfg.get("dropout_rate", 0.25),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 50, 6, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["imu_input"],
        output_names=["quaternion", "hx_out"],
        dynamic_axes={
            "imu_input": {0: "batch_size"},
            "quaternion": {0: "batch_size"},
            "hx_out": {1: "batch_size"},
        },
        dynamo=False,
    )
    print(f"Saved ONNX model to: {onnx_path} (Size: {onnx_path.stat().st_size / 1024:.2f} KB)")

    # Verify ONNX structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure check passed.")

    # Numerical verification via ONNX Runtime
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test with batch size 4
    torch.manual_seed(42)
    test_input_pt = torch.randn(4, 50, 6, dtype=torch.float32)

    with torch.no_grad():
        pt_quat, pt_hx = model(test_input_pt)

    ort_inputs = {"imu_input": test_input_pt.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_quat, ort_hx = ort_outputs[0], ort_outputs[1]

    quat_max_diff = float(np.max(np.abs(pt_quat.numpy() - ort_quat)))
    quat_mean_diff = float(np.mean(np.abs(pt_quat.numpy() - ort_quat)))

    # Verify Quaternion L2 Normalization (||q|| == 1.0)
    pt_norms = np.linalg.norm(pt_quat.numpy(), axis=-1)
    ort_norms = np.linalg.norm(ort_quat, axis=-1)

    pt_norm_valid = bool(np.allclose(pt_norms, 1.0, atol=1e-5))
    ort_norm_valid = bool(np.allclose(ort_norms, 1.0, atol=1e-5))

    print(f"Attitude Quat Max Abs Diff   : {quat_max_diff:.8e}")
    print(f"Attitude Quat Mean Abs Diff  : {quat_mean_diff:.8e}")
    print(f"PyTorch Quat Norm Valid (==1.0): {pt_norm_valid} (Norms: {pt_norms})")
    print(f"ORT Quat Norm Valid (==1.0)    : {ort_norm_valid} (Norms: {ort_norms})")

    return {
        "onnx_path": str(onnx_path),
        "size_kb": onnx_path.stat().st_size / 1024.0,
        "input_shape": "[B, 50, 6]",
        "output_shape": "[B, 4]",
        "quat_max_diff": quat_max_diff,
        "quat_mean_diff": quat_mean_diff,
        "quat_norm_valid": ort_norm_valid,
    }


def main():
    print("==================================================")
    print("NAVIGATE 2.0 — PyTorch to ONNX Model Exporter")
    print("==================================================")

    vel_res = export_velocity_model()
    att_res = export_attitude_model()

    print("\n==================================================")
    print("EXPORT & VERIFICATION SUMMARY:")
    print("==================================================")
    print(f"VelocityModel V2 ONNX : {vel_res['onnx_path']} ({vel_res['size_kb']:.2f} KB)")
    print(f"  Input Shape         : {vel_res['input_shape']}")
    print(f"  Output Shape        : {vel_res['output_shape']}")
    print(f"  Max Abs Diff        : {vel_res['speed_max_diff']:.8e}")
    print(f"  Mean Abs Diff       : {vel_res['speed_mean_diff']:.8e}")
    print()
    print(f"AttitudeModel ONNX    : {att_res['onnx_path']} ({att_res['size_kb']:.2f} KB)")
    print(f"  Input Shape         : {att_res['input_shape']}")
    print(f"  Output Shape        : {att_res['output_shape']}")
    print(f"  Max Abs Diff        : {att_res['quat_max_diff']:.8e}")
    print(f"  Mean Abs Diff       : {att_res['quat_mean_diff']:.8e}")
    print(f"  Quaternion Norm == 1: {att_res['quat_norm_valid']}")
    print("==================================================")


if __name__ == "__main__":
    main()
