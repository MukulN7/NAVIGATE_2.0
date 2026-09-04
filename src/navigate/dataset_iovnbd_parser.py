"""
IO-VNBD Dataset Preprocessing and Windowing Pipeline for NAVIGATE 2.0.

Standardizes 10 Hz smartphone sensor recordings and paired vehicle CAN-bus logs,
creating 5-second (50-sample at 10 Hz) sliding windows for odometry and orientation model training.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("iovnbd_parser")

# Target IMU Channel Specification (6 channels)
IMU_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
DEFAULT_SAMPLE_RATE_HZ = 10.0
DEFAULT_WINDOW_SIZE = 50  # 5 seconds at 10 Hz
DEFAULT_STRIDE = 10       # 1 second hop at 10 Hz


def normalize_smartphone_header(header: List[str]) -> Dict[str, int]:
    """
    Normalizes smartphone CSV header variants into standardized channel indices.
    
    Handles schema variants:
    - GYROSCOPE Yaw/Pitch/Roll vs GYROSCOPE X/Y/Z
    - ORIENTATION Yaw/Pitch/Roll vs ORIENTATION Azimuth/Pitch/Roll
    - Extra whitespace and degree/unit symbols.
    """
    clean_header = [col.strip() for col in header]
    mapping: Dict[str, int] = {}
    
    for idx, col in enumerate(clean_header):
        col_upper = col.upper()
        
        # Timestamp
        if "TIME SINCE START" in col_upper:
            mapping["time_ms"] = idx
        elif col_upper.startswith("DATE"):
            mapping["date_str"] = idx
            
        # Accelerometer
        elif "ACCELEROMETER X" in col_upper:
            mapping["accel_x"] = idx
        elif "ACCELEROMETER Y" in col_upper:
            mapping["accel_y"] = idx
        elif "ACCELEROMETER Z" in col_upper:
            mapping["accel_z"] = idx
            
        # Gyroscope variants
        elif "GYROSCOPE YAW" in col_upper or col_upper == "GYROSCOPE X (RAD/S)":
            mapping["gyro_x"] = idx
        elif "GYROSCOPE PITCH" in col_upper or col_upper == "GYROSCOPE Y (RAD/S)":
            mapping["gyro_y"] = idx
        elif "GYROSCOPE ROLL" in col_upper or col_upper == "GYROSCOPE Z (RAD/S)":
            mapping["gyro_z"] = idx
            
        # Gravity
        elif "GRAVITY X" in col_upper:
            mapping["gravity_x"] = idx
        elif "GRAVITY Y" in col_upper:
            mapping["gravity_y"] = idx
        elif "GRAVITY Z" in col_upper:
            mapping["gravity_z"] = idx
            
        # Magnetometer
        elif "MAGNETIC FIELD X" in col_upper:
            mapping["mag_x"] = idx
        elif "MAGNETIC FIELD Y" in col_upper:
            mapping["mag_y"] = idx
        elif "MAGNETIC FIELD Z" in col_upper:
            mapping["mag_z"] = idx
            
        # Orientation variants
        elif "ORIENTATION (YAW)" in col_upper or "ORIENTATION (AZIMUTH)" in col_upper:
            mapping["orient_yaw"] = idx
        elif "ORIENTATION (PITCH)" in col_upper:
            mapping["orient_pitch"] = idx
        elif "ORIENTATION (ROLL" in col_upper:
            mapping["orient_roll"] = idx

    # Validate mandatory 6 IMU channels
    missing = [ch for ch in IMU_CHANNELS if ch not in mapping]
    if missing:
        raise ValueError(f"Missing mandatory IMU channels in header {clean_header}: {missing}")
        
    return mapping


def normalize_vehicle_header(header: List[str]) -> Dict[str, int]:
    """Normalizes vehicle CAN-bus CSV header into standardized channel indices."""
    clean_header = [col.strip() for col in header]
    mapping: Dict[str, int] = {}
    
    for idx, col in enumerate(clean_header):
        col_upper = col.upper()
        if "TIME SINCE START OF DAY" in col_upper or col_upper == "TIME (SECONDS)":
            mapping["time_s"] = idx
        elif col_upper == "VELOCITY (KM/HR)" or col_upper == "VELOCITY":
            # Exact match only — must NOT match 'Vertical velocity (km/hr)'
            mapping["velocity_kmh"] = idx
        elif "HEADING" in col_upper:
            mapping["heading_deg"] = idx
        elif "INDICATED VEHICLE SPEED" in col_upper:
            mapping["indicated_speed_kmh"] = idx
            
    if "velocity_kmh" not in mapping:
        raise ValueError(f"Missing Velocity (km/hr) target in vehicle header: {clean_header}")
        
    return mapping


def kmh_to_ms(speed_kmh: Any) -> Any:
    """Converts speed from km/h to m/s."""
    return speed_kmh / 3.6


def euler_to_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Converts Euler angles (yaw, pitch, roll in degrees) to a unit quaternion [qw, qx, qy, qz].
    Uses standard Z-Y-X rotation sequence.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def compute_relative_quaternion(q_ref: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Computes the relative delta quaternion dq = q_ref* x q_target between reference and target.
    Returns 3D imaginary vector components [dq_x, dq_y, dq_z].
    """
    w1, x1, y1, z1 = q_ref
    w2, x2, y2, z2 = q_target
    
    # Conjugate of q_ref: [w1, -x1, -y1, -z1]
    # Quaternion product (q_ref* x q_target):
    dw = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    dx = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
    dy = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
    dz = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2
    
    dq = np.array([dw, dx, dy, dz], dtype=np.float32)
    if dq[0] < 0:
        dq = -dq  # Enforce positive scalar component for unique representation
    return dq[1:]  # Return vector components (x, y, z)


def load_and_parse_csv(
    file_path: Path,
    expected_cols_fn: Any
) -> Tuple[List[str], np.ndarray, Dict[str, int], int]:
    """
    Loads a CSV file, detects malformed rows, handles NaN/Inf values, and returns valid numerical data.
    """
    malformed_count = 0
    valid_rows = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
        reader = csv.reader(fp)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV file: {file_path}")
            
        col_mapping = expected_cols_fn(raw_header)
        num_cols = len(raw_header)
        
        for row in reader:
            if len(row) != num_cols:
                malformed_count += 1
                continue
                
            parsed_row = []
            row_has_error = False
            for idx, val in enumerate(row):
                v_str = val.strip()
                try:
                    v_float = float(v_str)
                    if np.isnan(v_float) or np.isinf(v_float):
                        row_has_error = True
                        break
                    parsed_row.append(v_float)
                except ValueError:
                    # Ignore string date columns if present
                    parsed_row.append(np.nan)
                    
            if row_has_error:
                malformed_count += 1
            else:
                valid_rows.append(parsed_row)
                
    if not valid_rows:
        raise ValueError(f"No valid rows parsed in file: {file_path}")
        
    data_arr = np.array(valid_rows, dtype=np.float64)
    return raw_header, data_arr, col_mapping, malformed_count


def fix_timestamp_resets(timestamps: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects non-monotonic timestamp resets and unwraps timestamps to ensure monotonicity
    while preserving data row order and alignment with vehicle data.
    """
    if len(timestamps) <= 1:
        return timestamps, data
        
    diffs = np.diff(timestamps)
    if np.any(diffs < 0):
        logger.warning("Unwrapping non-monotonic timestamp reset...")
        unwrapped = timestamps.copy().astype(np.float64)
        offset = 0.0
        for i in range(len(diffs)):
            if diffs[i] < 0:
                step = diffs[i-1] if (i > 0 and diffs[i-1] > 0) else 100.0
                offset += (timestamps[i] + step - timestamps[i+1])
            unwrapped[i + 1] += offset
        return unwrapped, data
        
    return timestamps, data


def find_matched_pairs(base_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Discovers all 144 smartphone S-*.csv files and finds their corresponding V-*.csv files.
    Returns list of (smartphone_path, vehicle_path, session_id).
    """
    s_files = sorted(list(base_dir.rglob("S-*.csv")))
    matched_pairs = []
    
    for s_path in s_files:
        filename = s_path.name
        v_filename = "V-" + filename[2:]
        v_path = s_path.parent / v_filename
        
        if not v_path.exists():
            # Search in alternative directories (e.g. Uncategorised V-Dataset)
            v_candidates = list(base_dir.rglob(v_filename))
            if v_candidates:
                v_path = v_candidates[0]
                
        if v_path.exists():
            session_id = filename[:-4]  # Strip .csv extension
            matched_pairs.append((s_path, v_path, session_id))
        else:
            logger.warning(f"Could not find matching vehicle file for: {s_path}")
            
    logger.info(f"Found {len(matched_pairs)} matched S/V recording pairs out of {len(s_files)} S-files.")
    return matched_pairs


def extract_sliding_windows(
    s_data: np.ndarray,
    s_map: Dict[str, int],
    v_data: np.ndarray,
    v_map: Dict[str, int],
    session_id: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE
) -> Dict[str, np.ndarray]:
    """
    Extracts 50-sample (5-second at 10 Hz) sliding windows from paired smartphone and vehicle data.
    """
    # Extract 6 IMU channels
    imu_indices = [s_map[ch] for ch in IMU_CHANNELS]
    imu_raw = s_data[:, imu_indices].astype(np.float32)
    
    # Extract timestamps
    s_time_col = s_map.get("time_ms", 0)
    s_timestamps = s_data[:, s_time_col]
    s_timestamps, imu_raw = fix_timestamp_resets(s_timestamps, imu_raw)
    
    # Extract Ground Truth Speed (km/h -> m/s)
    v_vel_col = v_map["velocity_kmh"]
    v_speed_kmh = v_data[:, v_vel_col]
    v_speed_ms = kmh_to_ms(v_speed_kmh).astype(np.float32)
    
    # Alignment: ensure equal length if minor boundary mismatch
    min_len = min(len(imu_raw), len(v_speed_ms))
    imu_raw = imu_raw[:min_len]
    v_speed_ms = v_speed_ms[:min_len]
    s_timestamps = s_timestamps[:min_len]
    
    # Extract Orientation if available
    has_orient = all(k in s_map for k in ["orient_yaw", "orient_pitch", "orient_roll"])
    quats = None
    if has_orient:
        y_col = s_map["orient_yaw"]
        p_col = s_map["orient_pitch"]
        r_col = s_map["orient_roll"]
        y_arr = s_data[:min_len, y_col]
        p_arr = s_data[:min_len, p_col]
        r_arr = s_data[:min_len, r_col]
        quats = np.zeros((min_len, 4), dtype=np.float32)
        for i in range(min_len):
            quats[i] = euler_to_quaternion(y_arr[i], p_arr[i], r_arr[i])

    num_windows = (min_len - window_size) // stride + 1
    if num_windows <= 0:
        return {}
        
    imu_windows = np.zeros((num_windows, window_size, 6), dtype=np.float32)
    velocity_targets = np.zeros((num_windows,), dtype=np.float32)
    rel_quaternions = np.zeros((num_windows, 3), dtype=np.float32)
    window_timestamps = np.zeros((num_windows, 2), dtype=np.float64)
    session_arr = np.array([session_id] * num_windows, dtype=object)
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        imu_windows[i] = imu_raw[start_idx:end_idx]
        velocity_targets[i] = v_speed_ms[end_idx - 1]  # Speed target at end of window
        window_timestamps[i] = [s_timestamps[start_idx], s_timestamps[end_idx - 1]]
        
        if quats is not None:
            rel_quaternions[i] = compute_relative_quaternion(quats[start_idx], quats[end_idx - 1])
            
    return {
        "imu": imu_windows,
        "velocity": velocity_targets,
        "rel_quaternion": rel_quaternions,
        "timestamps": window_timestamps,
        "session_ids": session_arr
    }


def compute_normalization_stats(imu_windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates channel-wise mean and standard deviation across IMU windows."""
    flat_imu = imu_windows.reshape(-1, 6)
    mean = np.mean(flat_imu, axis=0).astype(np.float32)
    std = np.std(flat_imu, axis=0).astype(np.float32)
    std[std == 0] = 1.0  # Prevent division by zero
    return mean, std


def process_dataset(
    dataset_path: Path,
    output_file: Path,
    limit: Optional[int] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE
) -> Dict[str, Any]:
    """
    Runs the full dataset preprocessing pipeline and saves `.npz` archive.
    """
    base_dir = Path(dataset_path)
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
    pairs = find_matched_pairs(base_dir)
    if limit is not None and limit > 0:
        logger.info(f"Applying limit: processing first {limit} recordings.")
        pairs = pairs[:limit]
        
    all_imu = []
    all_vel = []
    all_rel_quat = []
    all_ts = []
    all_sessions = []
    
    total_malformed = 0
    processed_count = 0
    
    for s_path, v_path, session_id in pairs:
        try:
            _, s_data, s_map, s_malformed = load_and_parse_csv(s_path, normalize_smartphone_header)
            _, v_data, v_map, v_malformed = load_and_parse_csv(v_path, normalize_vehicle_header)
            total_malformed += (s_malformed + v_malformed)
            
            extracted = extract_sliding_windows(
                s_data, s_map, v_data, v_map, session_id, window_size, stride
            )
            
            if extracted:
                all_imu.append(extracted["imu"])
                all_vel.append(extracted["velocity"])
                all_rel_quat.append(extracted["rel_quaternion"])
                all_ts.append(extracted["timestamps"])
                all_sessions.append(extracted["session_ids"])
                processed_count += 1
                logger.info(f"Processed {session_id}: {len(extracted['velocity'])} windows created.")
        except Exception as e:
            logger.error(f"Failed to process pair ({s_path.name}, {v_path.name}): {e}")
            
    if not all_imu:
        raise RuntimeError("No valid windows generated across dataset.")
        
    imu_concat = np.concatenate(all_imu, axis=0)
    vel_concat = np.concatenate(all_vel, axis=0)
    quat_concat = np.concatenate(all_rel_quat, axis=0)
    ts_concat = np.concatenate(all_ts, axis=0)
    sessions_concat = np.concatenate(all_sessions, axis=0)
    
    mean, std = compute_normalization_stats(imu_concat)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "window_size": window_size,
        "stride": stride,
        "channels": IMU_CHANNELS,
        "num_recordings": processed_count,
        "total_windows": len(vel_concat),
        "total_malformed_rows": total_malformed
    }
    
    np.savez_compressed(
        output_file,
        imu=imu_concat,
        velocity=vel_concat,
        rel_quaternion=quat_concat,
        timestamps=ts_concat,
        session_ids=sessions_concat,
        normalize_mean=mean,
        normalize_std=std,
        metadata_json=json.dumps(metadata)
    )
    
    logger.info("==================================================")
    logger.info(f"Successfully saved processed dataset to: {output_file}")
    logger.info(f"Total processed recordings: {processed_count}")
    logger.info(f"Total windows generated: {len(vel_concat)} (shape: {imu_concat.shape})")
    logger.info(f"Normalization Mean: {mean}")
    logger.info(f"Normalization Std:  {std}")
    logger.info("==================================================")
    
    return {
        "imu_shape": imu_concat.shape,
        "total_windows": len(vel_concat),
        "recordings": processed_count,
        "output_path": str(output_file)
    }


def main():
    parser = argparse.ArgumentParser(
        description="IO-VNBD Dataset Preprocessing Pipeline (NAVIGATE 2.0)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to IO-VNBD dataset base directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/iovnbd_dataset.npz",
        help="Output path for processed .npz archive"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of recordings to process for testing"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Window size in samples (default: 50 for 5s at 10 Hz)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Window stride/hop in samples (default: 10 for 1s hop at 10 Hz)"
    )
    
    args = parser.parse_args()
    process_dataset(
        dataset_path=Path(args.dataset),
        output_file=Path(args.output),
        limit=args.limit,
        window_size=args.window_size,
        stride=args.stride
    )


if __name__ == "__main__":
    main()
