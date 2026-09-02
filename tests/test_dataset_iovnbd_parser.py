"""
Unit tests for IO-VNBD Dataset Preprocessing Pipeline (NAVIGATE 2.0).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from navigate.dataset_iovnbd_parser import (
    normalize_smartphone_header,
    normalize_vehicle_header,
    kmh_to_ms,
    euler_to_quaternion,
    compute_relative_quaternion,
    fix_timestamp_resets,
    extract_sliding_windows,
    compute_normalization_stats,
    IMU_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_SIZE
)


def test_schema_normalization_variants():
    """Verifies parsing and mapping of smartphone header variants to standard 6 IMU channels."""
    # Variant 1: GYROSCOPE Yaw/Pitch/Roll, ORIENTATION Yaw/Pitch/Roll
    header1 = [
        "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)",
        "GPS SPEED (Kmh)", "GPS ACCURACY (m)", "GPS ORIENTATION (°)",
        "GPS SATELLITES IN RANGE", "TIME SINCE START (ms)", "DATE (YYYY-MO-DD)",
        "ACCELEROMETER X (m/s)", "ACCELEROMETER Y (m/s)", "ACCELEROMETER Z (m/s)",
        "GRAVITY X (m/s)", "GRAVITY Y (m/s)", "GRAVITY Z (m/s)",
        "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)",
        "MAGNETIC FIELD X (μT)", "MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Z (μT)",
        "ORIENTATION (Yaw) (°)", "ORIENTATION (Pitch) (°)", "ORIENTATION (Roll ) (°)"
    ]
    map1 = normalize_smartphone_header(header1)
    assert map1["accel_x"] == 9
    assert map1["accel_y"] == 10
    assert map1["accel_z"] == 11
    assert map1["gyro_x"] == 15
    assert map1["gyro_y"] == 16
    assert map1["gyro_z"] == 17

    # Variant 2: GYROSCOPE X/Y/Z, ORIENTATION Azimuth/Pitch/Roll
    header2 = [
        "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)",
        "GPS SPEED (Kmh)", "GPS ACCURACY (m)", "GPS ORIENTATION (°)",
        "GPS SATELLITES IN RANGE", "TIME SINCE START (ms)", "DATE (YYYY-MO-DD)",
        "ACCELEROMETER X (m/s)", "ACCELEROMETER Y (m/s)", "ACCELEROMETER Z (m/s)",
        "GRAVITY X (m/s)", "GRAVITY Y (m/s)", "GRAVITY Z (m/s)",
        "GYROSCOPE X (rad/s)", "GYROSCOPE Y (rad/s)", "GYROSCOPE Z (rad/s)",
        "MAGNETIC FIELD X (μT)", "MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Z (μT)",
        "ORIENTATION (Azimuth) (°)", "ORIENTATION (Pitch) (°)", "ORIENTATION (Roll ) (°)"
    ]
    map2 = normalize_smartphone_header(header2)
    assert map2["gyro_x"] == 15
    assert map2["gyro_y"] == 16
    assert map2["gyro_z"] == 17
    assert map2["orient_yaw"] == 21


def test_vehicle_schema_normalization():
    """Verifies vehicle header normalization."""
    v_header = [
        "No of GPS Satellites Available", "Time Since Start of Day (seconds)",
        "Latitude (degrees)", "Longitude (degrees)", "Velocity (km/hr)",
        "Heading (degrees)", "Height (km)"
    ]
    v_map = normalize_vehicle_header(v_header)
    assert v_map["time_s"] == 1
    assert v_map["velocity_kmh"] == 4
    assert v_map["heading_deg"] == 5


def test_kmh_to_ms_conversion():
    """Verifies unit conversion from km/h to m/s."""
    assert pytest.approx(kmh_to_ms(0.0), abs=1e-5) == 0.0
    assert pytest.approx(kmh_to_ms(36.0), abs=1e-5) == 10.0
    assert pytest.approx(kmh_to_ms(117.911), abs=1e-3) == 32.753


def test_euler_and_quaternion_utilities():
    """Verifies Euler-to-Quaternion conversion and relative quaternion delta calculation."""
    q_identity = euler_to_quaternion(0.0, 0.0, 0.0)
    assert pytest.approx(q_identity, abs=1e-5) == np.array([1.0, 0.0, 0.0, 0.0])
    
    q_90_yaw = euler_to_quaternion(90.0, 0.0, 0.0)
    assert pytest.approx(q_90_yaw[0], abs=1e-5) == np.cos(np.pi / 4)
    assert pytest.approx(q_90_yaw[3], abs=1e-5) == np.sin(np.pi / 4)
    
    dq = compute_relative_quaternion(q_identity, q_90_yaw)
    assert len(dq) == 3


def test_timestamp_handling():
    """Verifies non-monotonic timestamp reset handling."""
    ts_reset = np.array([100.0, 200.0, 300.0, 50.0, 400.0])
    data = np.array([[1], [2], [3], [0], [4]])
    
    clean_ts, clean_data = fix_timestamp_resets(ts_reset, data)
    assert np.all(np.diff(clean_ts) >= 0)
    assert clean_data[0][0] == 0


def test_window_creation_50_samples():
    """Verifies 5-second (50-sample at 10 Hz) window extraction and output shapes."""
    num_samples = 120  # 12 seconds of 10 Hz data
    
    # Synthetic Smartphone Data
    s_data = np.zeros((num_samples, 24), dtype=np.float64)
    s_data[:, 7] = np.linspace(0, 11900, num_samples)  # ms timestamps
    s_data[:, 9:12] = np.random.randn(num_samples, 3)  # Accel
    s_data[:, 15:18] = np.random.randn(num_samples, 3) # Gyro
    s_data[:, 21:24] = np.random.randn(num_samples, 3) # Orient
    
    header1 = [
        "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)",
        "GPS SPEED (Kmh)", "GPS ACCURACY (m)", "GPS ORIENTATION (°)",
        "GPS SATELLITES IN RANGE", "TIME SINCE START (ms)", "DATE (YYYY-MO-DD)",
        "ACCELEROMETER X (m/s)", "ACCELEROMETER Y (m/s)", "ACCELEROMETER Z (m/s)",
        "GRAVITY X (m/s)", "GRAVITY Y (m/s)", "GRAVITY Z (m/s)",
        "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)",
        "MAGNETIC FIELD X (μT)", "MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Z (μT)",
        "ORIENTATION (Yaw) (°)", "ORIENTATION (Pitch) (°)", "ORIENTATION (Roll ) (°)"
    ]
    s_map = normalize_smartphone_header(header1)
    
    # Synthetic Vehicle Data
    v_data = np.zeros((num_samples, 7), dtype=np.float64)
    v_data[:, 1] = np.linspace(0, 11.9, num_samples)
    v_data[:, 4] = np.full(num_samples, 36.0)  # 36 km/h = 10 m/s
    v_header = [
        "No of GPS Satellites Available", "Time Since Start of Day (seconds)",
        "Latitude (degrees)", "Longitude (degrees)", "Velocity (km/hr)",
        "Heading (degrees)", "Height (km)"
    ]
    v_map = normalize_vehicle_header(v_header)
    
    # Extract windows: 50 samples, stride 10
    extracted = extract_sliding_windows(
        s_data, s_map, v_data, v_map, session_id="TEST_SESSION", window_size=50, stride=10
    )
    
    # Check shapes
    expected_num_windows = (120 - 50) // 10 + 1  # 8 windows
    assert extracted["imu"].shape == (expected_num_windows, 50, 6)
    assert extracted["velocity"].shape == (expected_num_windows,)
    assert extracted["rel_quaternion"].shape == (expected_num_windows, 3)
    assert extracted["session_ids"].shape == (expected_num_windows,)
    
    # Target value check
    assert pytest.approx(extracted["velocity"][0], abs=1e-5) == 10.0  # 36 km/h -> 10 m/s


def test_no_accidental_50_200hz_assumption():
    """Verifies that sample rate and window size parameters reflect 10 Hz explicitly."""
    assert DEFAULT_SAMPLE_RATE_HZ == 10.0
    assert DEFAULT_WINDOW_SIZE == 50
    assert len(IMU_CHANNELS) == 6
