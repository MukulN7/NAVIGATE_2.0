# Google Colab Velocity Model Training Guide (NAVIGATE 2.0)

This guide provides step-by-step instructions for executing full IO-VNBD dataset extraction, parsing, CNN+GRU model training, and evaluation on a Google Colab GPU instance.

---

## Exact Colab Execution Sequence

### Step 1: Enable GPU Runtime

In Google Colab menu:
* Click **Runtime** $\rightarrow$ **Change runtime type**.
* Select **T4 GPU** (or any available GPU acceleration).
* Click **Save**.

---

### Step 2: Clone / Upload NAVIGATE 2.0 Repository

Run in a Colab notebook cell:

```bash
# Check GPU availability
!nvidia-smi

# Clone NAVIGATE_2.0 repository
!git clone https://github.com/<your-username>/NAVIGATE_2.0.git
%cd NAVIGATE_2.0
```

---

### Step 3: Mount Google Drive & Make IO-VNBD Dataset Available

Mount Google Drive where your dataset zip file is stored:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### Step 4: Extract Synchronized IO-VNBD Dataset

Extract the raw dataset zip archive:

```bash
!mkdir -p /content/dataset_raw
!unzip -q "/content/drive/MyDrive/IO-VNBD/Synchronised V abd S datasets.zip" -d "/content/dataset_raw"
```

Verify directory structure:
```bash
!ls -la "/content/dataset_raw/Synchronised V abd S datasets"
```

---

### Step 5: Run Full Dataset Preprocessor (All 144 Sessions)

Parse all 144 synchronized smartphone and vehicle recordings into 5-second 10 Hz windows:

```bash
!python src/navigate/dataset_iovnbd_parser.py \
    --dataset "/content/dataset_raw/Synchronised V abd S datasets" \
    --output "data/processed/iovnbd_full.npz" \
    --window-size 50 \
    --stride 10
```

---

### Step 6: Verify the Processed `iovnbd_full.npz` Dataset Archive

```python
import numpy as np

data = np.load('data/processed/iovnbd_full.npz', allow_pickle=True)
print("Keys in NPZ:", list(data.keys()))
print("IMU Tensor Shape:", data['imu'].shape, data['imu'].dtype)
print("Velocity Target Shape:", data['velocity'].shape, data['velocity'].dtype)
print("Total Recordings / Sessions:", len(set(data['session_ids'])))
print("Metadata:", data['metadata_json'])
```

*Expected output: ~213,500 windows across 144 sessions (~130 MB NPZ file, ~260 MB RAM when loaded).*

---

### Step 7: Run GPU Velocity Model Training

Execute training on Colab GPU with deterministic session-wise 80/10/10 split:

```bash
!python src/navigate/train_velocity.py \
    --data "data/processed/iovnbd_full.npz" \
    --checkpoint "models/velocity_model_v2.pt" \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --seed 42 \
    --val-fraction 0.1 \
    --test-fraction 0.1 \
    --num-workers 2 \
    --device cuda
```

> **Memory Note:** If Colab returns GPU Out-Of-Memory (OOM), reduce `--batch-size` to `128` or `64`.

---

### Step 8: Evaluate Held-Out Test Set

The training script automatically:
1. Performs deterministic session partitioning (116 Train / 14 Val / 14 Test sessions).
2. Normalizes using ONLY training set statistics.
3. Saves the best checkpoint to `models/velocity_model_v2.pt` based on Validation MSE.
4. Loads the best checkpoint at the end of training and evaluates the held-out Test set.

Look for the final output report:

```text
==================================================
FINAL EVALUATION METRICS:
  Best Validation MAE : X.XXXX km/h (MSE: X.XXXXXX)
  Held-Out Test MSE   : X.XXXXXX
  Held-Out Test MAE   : X.XXXX m/s
  Held-Out Test MAE   : X.XXXX km/h
==================================================
```

---

### Step 9: Save / Download Best Trained Checkpoint

Copy the trained model checkpoint to Google Drive:

```bash
!cp models/velocity_model_v2.pt "/content/drive/MyDrive/NAVIGATE_2.0/velocity_model_v2.pt"
```
