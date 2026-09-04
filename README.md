# NAVIGATE 2.0: AI-Augmented Smartphone Dead Reckoning & ES-EKF Navigation System

## Objective (SIH26168 NAVIGATE)
NAVIGATE 2.0 is an AI-augmented dead-reckoning and navigation system designed for GNSS-denied environments (e.g., tunnels, urban canyons, GNSS jamming, or signal outages). Using smartphone IMU sensors (accelerometers and gyroscopes), NAVIGATE 2.0 provides continuous, high-accuracy vehicle position and velocity tracking without relying on continuous GNSS signals.

---

## Core System Architecture

1. **AI Deep Velocity Model (`VelocityNet` / `VelocityModelV2`)**
   - 1D ResNet + Bi-GRU network predicting 2D body frame velocities \((v_x, v_y)\) directly from high-rate raw IMU windowed sequences.
   - Mitigates double-integration error explosion inherent in standard IMU dead reckoning.

2. **AI Deep Attitude Model (`AttitudeCorrectionNet`)**
   - Neural model predicting roll and pitch attitude correction angles \((\Delta\phi, \Delta\theta)\) to correct IMU tilt and orientation drift.

3. **Error-State Extended Kalman Filter (`IEKFTracker` / `ErrorStateEKF`)**
   - 15-state Error-State EKF tracking:
     - Position \(\mathbf{p} \in \mathbb{R}^3\)
     - Velocity \(\mathbf{v} \in \mathbb{R}^3\)
     - Attitude Quaternion \(\mathbf{q} \in S^3\)
     - Gyroscope Bias \(\mathbf{b}_g \in \mathbb{R}^3\)
     - Accelerometer Bias \(\mathbf{b}_a \in \mathbb{R}^3\)
   - Fuses high-rate IMU strapdown propagation with low-frequency or AI-predicted pseudo-measurements.

4. **Road-Constrained AI+ES-EKF Pipeline (`AIIEKFRoadPipeline` & `RoadConstraintManager`)**
   - Incorporates road network geometry as pseudo-measurements during GNSS outages.
   - Prevents data leakage by utilizing strictly pre-outage GNSS history to build local road polylines.
   - Dynamically checks distance and heading thresholds before applying EKF update.

5. **Map Matching Engine (`map_matching.py`)**
   - Project vehicle positions onto polyline road networks using spatial projection and heading validation.

---

## Evaluation Results

Evaluated across multiple sessions (S-M, S-Vfa01, S-Vw2) on the IO-VNBD dataset during simulated GNSS blackout intervals (5s, 10s, 30s, 60s):

| Blackout Duration | Baseline Dead Reckoning Final Error (Drift %) | AI + ES-EKF (Version A) Final Error (Drift %) | AI + ES-EKF + Road Constraint (Version B) Final Error (Drift %) | Absolute Improvement vs Baseline | % Error Reduction vs Baseline |
|-------------------|----------------------------------------------|-----------------------------------------------|-----------------------------------------------------------------|----------------------------------|-------------------------------|
| **5s Outage**     | 37.51 m (54.07%)                             | 20.04 m (39.20%)                              | **20.01 m (38.22%)**                                            | 17.47 m                          | **46.57%**                    |
| **10s Outage**    | 71.67 m (56.85%)                             | 41.81 m (42.80%)                              | **41.55 m (39.43%)**                                            | 29.85 m                          | **41.66%**                    |
| **30s Outage**    | 230.57 m (54.14%)                            | 158.14 m (45.98%)                             | **157.73 m (45.69%)**                                           | 72.43 m                          | **31.41%**                    |
| **60s Outage**    | 575.53 m (67.17%)                            | 386.71 m (57.10%)                             | **385.79 m (56.85%)**                                           | 188.82 m                         | **32.81%**                    |

---

## Test Status
- **Suite**: `pytest`
- **Status**: **178 / 178 Passed** (0 failures, 0 errors)
- Tests cover: Dataset parsing, Velocity Model, Attitude Model, Dead Reckoning, IEKF Tracker, AI+IEKF Pipeline, Road Pipeline, Map Matching, and Evaluation Metrics.

---

## Repository Structure

```
NAVIGATE_2.0/
├── data/
│   ├── raw/                        # Raw IO-VNBD dataset (local, excluded from Git)
│   └── processed/
│       ├── iovnbd_full.npz         # Full processed dataset array (local)
│       └── iovnbd_smoke_test.npz   # Lightweight smoke test dataset
├── docs/                           # Documentation and guides
├── models/                         # Trained model weights
│   ├── attitude_model.pt
│   └── velocity_model_v2.pt
├── reference/                      # Reference implementations (QDeepOdo)
├── results/                        # Evaluation metrics and generated figures
│   ├── ai_iekf_blackout_results.json
│   ├── evaluation_summary.json
│   ├── ai_iekf_road/
│   └── figures/                    # Trajectory & performance comparison plots
├── scripts/                        # Pipeline evaluation and figure generation scripts
│   ├── generate_evaluation_plots.py
│   ├── run_ai_iekf_evaluation.py
│   ├── run_ai_iekf_road_evaluation.py
│   └── run_baseline_evaluation.py
├── src/
│   └── navigate/                   # Core Python package
│       ├── models/                 # PyTorch neural network modules
│       ├── ai_iekf_pipeline.py     # AI + ES-EKF integration
│       ├── ai_iekf_road_pipeline.py# AI + ES-EKF + Road constraint integration
│       ├── dataset_iovnbd_parser.py# Dataset parser and window generator
│       ├── dead_reckoning.py       # Baseline strapdown dead reckoning
│       ├── evaluate_blackout.py    # Blackout simulation & metrics
│       ├── iekf_tracker.py         # 15-state Error-State Extended Kalman Filter
│       ├── map_matching.py         # Spatial map matching engine
│       ├── train_attitude.py       # Attitude model training script
│       └── train_velocity.py       # Velocity model training script
├── tests/                          # Complete unit and integration test suite
├── .gitignore                      # Git ignore rules for Python, PyTorch, Android, secrets
├── pytest.ini                      # Pytest runner configuration
└── README.md                       # Project overview and documentation
```

---

## Setup & Verification

### Prerequisites
- Python 3.10+
- PyTorch
- NumPy, SciPy, Scikit-learn, Matplotlib

### Running the Test Suite
Execute all 178 unit and integration tests:
```bash
python -m pytest
```

### Running System Evaluation
Run the full AI + ES-EKF + Road constraint evaluation script:
```bash
python scripts/run_ai_iekf_road_evaluation.py
```

Generate evaluation summary plots:
```bash
python scripts/generate_evaluation_plots.py
```
