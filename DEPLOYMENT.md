# BluSim Deployment Guide

## Prerequisites

- Python 3.10+
- macOS / Linux (x86_64 or aarch64 for Raspberry Pi)
- Docker + Docker Compose (for containerised deployment)
- AWS credentials (if using S3 for mat data download)

---

## 1. Local Development Setup

```bash
# Clone / navigate to the project
cd /path/to/blusim_system

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -e ".[dev]"          # if pyproject.toml has dev extras
# OR
pip install -r requirements.txt

# Verify installation
python -c "import blusim; print(blusim.__version__)"
```

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=blusim --cov-report=term-missing -v

# Run only signal processing tests (fast, no ML)
pytest tests/test_piezo_signal.py tests/test_features.py -v

# Run model tests
pytest tests/test_models.py tests/test_metrics.py -v
```

---

## 2. Data Preparation

### Apple Watch HealthKit Data

Export from Apple Health app:
1. Open Health app → Profile icon → Export All Health Data
2. Extract `export.xml` from the ZIP
3. Or use **Health Auto Export** app for per-vital CSVs

Parse with:
```python
from blusim.data.ingestion.healthkit_parser import parse_healthkit_xml
data = parse_healthkit_xml("path/to/export.xml")
# data["sleep"] → DataFrame with sleep stages
# data["hr"] → DataFrame with heart rate
```

### Piezo Mat Data (AWS S3)

Configure environment variables:
```bash
export AWS_DEFAULT_REGION=ap-south-1
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export BLUSIM_S3_BUCKET=data-blusim-care
```

Download a session:
```python
from blusim.data.ingestion.s3_loader import S3MatLoader
loader = S3MatLoader(bucket="data-blusim-care", prefix="subject_001")
df = loader.load_session_df("20240105", cache_dir="./data/raw")
```

---

## 3. Training

```bash
# With default config
python scripts/train.py

# Override hyperparameters (Hydra syntax)
python scripts/train.py model.training.epochs=200 model.training.lr=5e-4

# Use GPU
python scripts/train.py pipeline.inference.device=cuda

# Custom data path
python scripts/train.py paths.data_root=/my/data/path
```

The best model checkpoint is saved to `./models/best_model.pt`.

MLflow tracking UI:
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## 4. Evaluation

```bash
python scripts/evaluate.py --model models/best_model.pt --device cpu
```

Expected output:
```
======================================================================
BLUSIM EVALUATION REPORT
======================================================================
REGRESSION VITALS:
  ✓ hr_bpm               MAE=3.21  RMSE=4.15  r=0.924  N=4800
  ✓ rmssd_ms             MAE=8.73  RMSE=11.2  r=0.871  N=4800
  ✓ spo2_pct             MAE=1.34  RMSE=1.89  r=0.953  N=4800
  ✗ resp_rate_bpm        MAE=2.41  RMSE=3.12  r=0.812  N=4800

SLEEP STAGING (✓):
  Accuracy: 78.3%   Cohen's κ: 0.706
  F1[Wake]: 0.812  F1[Light]: 0.741  F1[Deep]: 0.693  F1[REM]: 0.724

ACCURACY TARGET SUMMARY:
  [PASS] hr_mae
  [PASS] hrv_rmssd_mae
  [PASS] spo2_mae
  [PASS] sleep_kappa
  [FAIL] resp_mae
======================================================================
```

---

## 5. ONNX Export for Edge Deployment

```bash
python scripts/export_onnx.py \
    --model models/best_model.pt \
    --output models/

# Outputs:
# models/vital_predictor.onnx      (FP32, ~45 MB)
# models/vital_predictor_int8.onnx (INT8, ~12 MB)
```

Test the ONNX model:
```python
from blusim.inference.onnx_export import OnnxInferenceSession
import numpy as np

session = OnnxInferenceSession("models/vital_predictor_int8.onnx")
piezo = np.random.randn(1, 4, 14700).astype(np.float32)
result = session.run(piezo)
print(result)
# {"hr_bpm": 72.3, "sleep_stage": "Light", ...}
```

---

## 6. Docker Deployment

### Build and Start

```bash
# Set secrets (do not commit these to git!)
cp .env.example .env
# Edit .env with your InfluxDB token and AWS credentials

docker compose up --build -d
```

```bash
# .env.example contents:
INFLUXDB_TOKEN=change-me-long-random-token
INFLUXDB_ORG=blusim
INFLUXDB_BUCKET=vitals
INFLUXDB_ADMIN_USER=admin
INFLUXDB_ADMIN_PASSWORD=securepassword
AWS_DEFAULT_REGION=ap-south-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

### Verify Services

```bash
# API health
curl http://localhost:8000/health

# InfluxDB UI
open http://localhost:8086

# MLflow UI
open http://localhost:5000
```

### Push Piezo Data

```bash
curl -X POST http://localhost:8000/ingest/piezo \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [0.1, 0.2, -0.1, 0.3, 0.05, 0.15, -0.05, 0.25],
    "n_channels": 4,
    "sample_rate": 490.0
  }'
```

### Get Latest Vitals

```bash
curl http://localhost:8000/vitals/latest
```

### Get History

```bash
curl "http://localhost:8000/vitals/history?n=20"
```

---

## 7. Raspberry Pi 5 / Jetson Nano Deployment

### Raspberry Pi 5 (aarch64)

```bash
# Install system deps
sudo apt-get install -y python3.11 python3.11-venv libatlas-base-dev

# Create venv
python3.11 -m venv ~/blusim_venv
source ~/blusim_venv/bin/activate

# Install CPU-only torch for aarch64
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install blusim (no camera deps if headless)
pip install numpy scipy PyWavelets scikit-learn fastapi uvicorn boto3 pydantic

# Copy project + INT8 ONNX model
scp -r blusim_system/ pi@raspberrypi:~/
scp models/vital_predictor_int8.onnx pi@raspberrypi:~/models/

# Run on Pi
uvicorn blusim.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Expected Pi 5 inference latency: **~120–180 ms** for INT8 ONNX (30-second window).

### Performance Targets (INT8 ONNX)

| Hardware | Window Size | Inference Time | Step Feasibility |
|---|---|---|---|
| MacBook M2 | 30 s | ~15 ms | 5 s step ✓ |
| Raspberry Pi 5 | 30 s | ~150 ms | 5 s step ✓ |
| Jetson Nano | 30 s | ~45 ms | 5 s step ✓ |

---

## 8. Privacy Considerations

- **No video stored**: Only 9-D ROI colour vectors are retained in the circular buffer.
- **Video frames discarded immediately** after ROI extraction in `/ingest/video-frame`.
- InfluxDB stores only the predicted vitals, not raw sensor data.
- Configure InfluxDB retention policy to auto-delete data after N days:
  ```bash
  influx bucket update --name vitals --retention 90d
  ```
