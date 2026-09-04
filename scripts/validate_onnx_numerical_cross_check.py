#!/usr/bin/env python3
"""
validate_onnx_numerical_cross_check.py — Deterministic numerical cross-validation
between Python ONNX Runtime and Kotlin OnnxInferenceHelper.
"""

import sys
import io
import numpy as np
import torch
import onnxruntime as ort

# Force UTF-8 output encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    print("==================================================")
    print("NAVIGATE 2.0 — Deterministic ONNX Numerical Validation")
    print("==================================================")

    # 1. Load ONNX models
    vel_session = ort.InferenceSession("models/velocity_model_v2.onnx", providers=["CPUExecutionProvider"])
    att_session = ort.InferenceSession("models/attitude_model.onnx", providers=["CPUExecutionProvider"])

    # 2. Checkpoint statistics
    vel_ckpt = torch.load("models/velocity_model_v2.pt", map_location="cpu", weights_only=False)
    att_ckpt = torch.load("models/attitude_model.pt", map_location="cpu", weights_only=False)

    vel_imu_mean = np.array(vel_ckpt["imu_mean"], dtype=np.float32)
    vel_imu_std = np.array(vel_ckpt["imu_std"], dtype=np.float32)
    vel_mean = float(vel_ckpt["vel_mean"])
    vel_std = float(vel_ckpt["vel_std"])

    att_imu_mean = np.array(att_ckpt["imu_mean"], dtype=np.float32)
    att_imu_std = np.array(att_ckpt["imu_std"], dtype=np.float32)

    # 3. Create exact deterministic fixed 50-sample window [50, 6]
    window = np.zeros((50, 6), dtype=np.float32)
    for i in range(50):
        window[i, 0] = 0.01 * (i % 5)
        window[i, 1] = -0.02 * (i % 3)
        window[i, 2] = 9.81 + 0.05 * (i % 2)
        window[i, 3] = 0.001 * i
        window[i, 4] = -0.002 * i
        window[i, 5] = 0.0005 * i

    window_batch = window[np.newaxis, ...] # [1, 50, 6]

    # 4. Velocity Model Inference
    vel_norm_input = (window_batch - vel_imu_mean) / vel_imu_std
    vel_ort_out = vel_session.run(None, {"imu_input": vel_norm_input})
    norm_speed = vel_ort_out[0][0, 0]
    physical_speed_ms = float(np.maximum(0.0, norm_speed * vel_std + vel_mean))

    # 5. Attitude Model Inference
    att_norm_input = (window_batch - att_imu_mean) / att_imu_std
    att_ort_out = att_session.run(None, {"imu_input": att_norm_input})
    quat = att_ort_out[0][0] # [qw, qx, qy, qz]

    if quat[0] < 0.0:
        quat = -quat

    quat_norm = float(np.linalg.norm(quat))

    print(f"Velocity Output (m/s)  : {physical_speed_ms:.6f}")
    print(f"Attitude Quaternion    : [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    print(f"Quaternion L2 Norm     : {quat_norm:.8f}")
    print("==================================================")

if __name__ == "__main__":
    main()
