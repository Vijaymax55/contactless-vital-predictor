# BluSim — Contactless Health Vital Prediction System

> **Non-invasive, non-contact monitoring of heart rate, HRV, SpO₂, respiration,
> sleep stages, and stress — using only a mattress sensor and a camera.**

---

## What Is BluSim?

BluSim is a machine-learning system that continuously monitors a person's health vitals
**without attaching anything to their body**. A thin piezoelectric sensor placed under the
mattress picks up the tiny vibrations caused by every heartbeat and breath. A camera captures
subtle colour changes in the face. An Apple Watch worn during sleep provides accurate ground
truth for supervised training.

The system fuses all three signals to predict:

| Vital | Method | Target Accuracy |
|---|---|---|
| Heart Rate (HR) | BCG peak detection + rPPG PSD | MAE < 5 BPM |
| HRV (RMSSD / SDNN) | RR intervals from BCG | MAE < 10 ms |
| SpO₂ | Camera AC/DC ratio | MAE < 2 % |
| Respiration Rate | Piezo respiration band | MAE < 2 br/min |
| Sleep Stage | Multi-modal deep learning | Cohen's κ > 0.65 |
| Stress Level | HRV + sleep context | Low / Medium / High |

---

## Vision

Most health monitoring today requires contact: a finger clip, a chest strap, electrodes
on the skin, or a wristband worn all night. This is inconvenient, uncomfortable, and
often impractical for long-term or clinical monitoring at home.

BluSim's vision is a **completely passive sleep lab** in every bedroom — a sensor under
the mattress, no wearable required, providing the same insights as a hospital sleep study.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                     SENSING                                  │
│                                                              │
│   Piezo Sensor         Camera            Apple Watch        │
│   under mattress       (face)            (ground truth)     │
│   4 channels           30 FPS            HealthKit export   │
│   490 Hz               RGB               HR/HRV/SpO₂/Sleep  │
└──────────┬─────────────────┬──────────────────┬─────────────┘
           │                 │                  │
           ▼                 ▼                  ▼
┌─────────────────┐  ┌───────────────┐  ┌──────────────────┐
│ Signal          │  │ rPPG          │  │ HealthKit Parser │
│ Conditioning    │  │ Processing    │  │                  │
│                 │  │               │  │ Align timestamps │
│ • Bandpass      │  │ • CHROM/POS   │  │ via              │
│   BCG: 1-20 Hz  │  │   algorithm   │  │ cross-correlation│
│   Resp: 0.1-0.5 │  │ • Face mesh   │  │                  │
│   Move: 0.5-5   │  │   ROI track   │  └────────┬─────────┘
│ • Wavelet       │  │ • Motion comp │           │
│   denoising     │  │ • SpO₂ est.   │           │
│ • LMS adaptive  │  └───────┬───────┘           │
│   filter        │          │                   │
└────────┬────────┘          │                   │
         │                  │                   │
         └──────────────────┴───────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │  Feature Extraction    │
               │                        │
               │  Time-domain: RR,      │
               │  RMSSD, IJ amplitude,  │
               │  Hjorth, zero-cross    │
               │                        │
               │  Frequency: Welch PSD, │
               │  LF/HF ratio, entropy  │
               │                        │
               │  Nonlinear: SampEn,    │
               │  DFA alpha, Poincaré   │
               └────────────┬───────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │         Deep Learning Model          │
          │                                     │
          │  PiezoEncoder       RPPGEncoder      │
          │  (CNN+BiLSTM        (CNN+Transformer │
          │   +Attention)        +CLS token)     │
          │       │                   │          │
          │       └────── Fusion ─────┘          │
          │         CrossAttention +              │
          │         Learnable gates               │
          │                │                     │
          │   ┌────────────┼────────────┐        │
          │   ▼            ▼            ▼        │
          │ HR/HRV      Sleep Stage   Stress     │
          │ SpO₂/Resp   4-class       3-class    │
          │ + uncertainty prediction  prediction │
          └─────────────────────────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │     REST API           │
               │  /vitals/latest        │
               │  /vitals/history       │
               │  /ingest/piezo         │
               │  /ingest/video-frame   │
               └────────────────────────┘
```

---

## Repository Layout

```
blusim_system/
│
├── blusim/                          ← Python package (all source code)
│   │
│   ├── data/
│   │   ├── ingestion/
│   │   │   ├── circular_buffer.py   ← Thread-safe ring buffer for live streams
│   │   │   ├── healthkit_parser.py  ← Parse Apple Watch CSV / export.xml
│   │   │   └── s3_loader.py         ← Download piezo CSV files from AWS S3
│   │   │
│   │   └── preprocessing/
│   │       ├── piezo_signal.py      ← Bandpass, wavelet, LMS, BCG peaks, SQI
│   │       ├── rppg_signal.py       ← CHROM/POS rPPG, MediaPipe ROI, SpO₂
│   │       └── synchronization.py   ← Align piezo + rPPG + Apple Watch timestamps
│   │
│   ├── features/
│   │   ├── piezo_features.py        ← Time/frequency/nonlinear feature extraction
│   │   └── rppg_features.py         ← HR/HRV proxy and SpO₂ from rPPG waveform
│   │
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── piezo_encoder.py     ← 1D-CNN → BiLSTM → Multi-head Attention
│   │   │   └── rppg_encoder.py      ← 1D-CNN → Positional Enc → Transformer
│   │   ├── fusion/
│   │   │   └── multimodal_fusion.py ← Late fusion + Cross-attention fusion
│   │   ├── heads/
│   │   │   └── output_heads.py      ← Regression, SleepStage, Stress heads
│   │   └── vital_predictor.py       ← Top-level nn.Module tying everything together
│   │
│   ├── training/
│   │   ├── augmentation.py          ← Noise, time-warp, scale, channel dropout
│   │   ├── loss.py                  ← GaussNLL + Focal + Temporal + Kendall weighting
│   │   └── trainer.py               ← Training loop, subject-independent splits, MLflow
│   │
│   ├── evaluation/
│   │   └── metrics.py               ← MAE, RMSE, Pearson r, Cohen's κ, Bland-Altman
│   │
│   ├── inference/
│   │   ├── pipeline.py              ← Real-time sliding window with circular buffer
│   │   └── onnx_export.py           ← ONNX FP32 export + INT8 quantization
│   │
│   └── api/
│       └── main.py                  ← FastAPI REST endpoints (ingest + vitals)
│
├── config/                          ← Hydra configuration (YAML, no hardcoded values)
│   ├── config.yaml                  ← Root config (paths, MLflow, logging)
│   ├── data/default.yaml            ← Sampling rates, band limits, split settings
│   ├── model/default.yaml           ← Architecture hyperparameters
│   └── pipeline/default.yaml        ← Inference window, API port, ONNX settings
│
├── scripts/
│   ├── train.py                     ← Entry point: load data → train → checkpoint
│   ├── evaluate.py                  ← Load checkpoint → run evaluation suite
│   └── export_onnx.py               ← Export FP32 ONNX → quantize to INT8
│
├── tests/
│   ├── test_piezo_signal.py         ← Unit tests: filtering, denoising, peak detection
│   ├── test_features.py             ← Unit tests: all feature extraction functions
│   ├── test_models.py               ← Unit tests: encoders, fusion, heads, losses
│   └── test_metrics.py              ← Unit tests: MAE, kappa, Bland-Altman, reports
│
├── Dockerfile                       ← Multi-stage production container
├── docker-compose.yml               ← API + InfluxDB + MLflow services
├── requirements.txt                 ← All Python dependencies
├── pyproject.toml                   ← Package config, pytest, ruff, mypy settings
│
├── legacy/                          ← Original research code (preserved for reference)
│   ├── god/                         ← GOD dashboard: live data viewer + S3 downloader
│   └── sleep_cycle/                 ← First sleep model: pixel colour → XGBoost
│
├── README.md                        ← This file
├── ARCHITECTURE.md                  ← Deep-dive: data flow, model diagrams, design rationale
├── IMPROVEMENTS.md                  ← Complete audit: every bug fixed + every change made
├── RESULTS.md                       ← Evaluation results, accuracy vs targets, latency
└── DEPLOYMENT.md                    ← Step-by-step setup, Docker, Raspberry Pi, API usage
```

---

## Quick Start

### 1. Install

```bash
# Python 3.10+ required
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Tests

```bash
pytest tests/ -v
# → 79 passed
```

### 3. Train a Model

```bash
# Uses synthetic data by default — swap in your real data in scripts/train.py
python scripts/train.py

# Override any hyperparameter (Hydra syntax)
python scripts/train.py model.training.epochs=200 model.training.lr=5e-4
```

The best checkpoint is saved to `models/best_model.pt`.

### 4. Evaluate

```bash
python scripts/evaluate.py --model models/best_model.pt
```

### 5. Start the REST API

```bash
uvicorn blusim.api.main:app --host 0.0.0.0 --port 8000
```

Send piezo data:
```bash
curl -X POST http://localhost:8000/ingest/piezo \
  -H "Content-Type: application/json" \
  -d '{"samples": [0.1,0.2,-0.1,0.3, 0.05,0.15,-0.05,0.25], "n_channels": 4}'
```

Get current vitals:
```bash
curl http://localhost:8000/vitals/latest
```

### 6. Export to ONNX (for Raspberry Pi / Jetson)

```bash
python scripts/export_onnx.py --model models/best_model.pt --output models/
# Produces: models/vital_predictor.onnx (FP32) + models/vital_predictor_int8.onnx (INT8)
```

### 7. Full Docker Stack

```bash
# Copy and fill in secrets
cp .env.example .env

docker compose up --build -d
# API:     http://localhost:8000
# InfluxDB: http://localhost:8086
# MLflow:  http://localhost:5000
```

---

## Using Real Data

### Apple Watch (HealthKit)

Export from the **Health** app → Profile → Export All Health Data, then:

```python
from blusim.data.ingestion.healthkit_parser import parse_healthkit_xml

data = parse_healthkit_xml("path/to/export.xml")
sleep_df   = data["sleep"]    # columns: time, sleep_stage, sleep_stage_int
hr_df      = data["hr"]       # columns: time, hr_bpm
hrv_df     = data["hrv"]      # columns: time, hrv_sdnn_ms
spo2_df    = data["spo2"]     # columns: time, spo2_pct
resp_df    = data["resp"]     # columns: time, resp_rate_bpm
```

Or with **Health Auto Export** CSV files:
```python
from blusim.data.ingestion.healthkit_parser import parse_healthkit_sleep_csv, parse_healthkit_hr_csv

sleep_df = parse_healthkit_sleep_csv("Sleep Analysis.csv")
hr_df    = parse_healthkit_hr_csv("Heart Rate.csv")
```

### Piezo Mat Data (AWS S3)

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=ap-south-1
```

```python
from blusim.data.ingestion.s3_loader import S3MatLoader

loader = S3MatLoader(bucket="data-blusim-care", prefix="subject_001")
df = loader.load_session_df("20240105", cache_dir="./data/raw")
# Returns DataFrame with columns: Time, Chn_1, Chn_2, Chn_3, Chn_4
```

### Synchronize All Modalities

```python
from blusim.data.preprocessing.synchronization import synchronize_session

session = synchronize_session(
    piezo_df=piezo_df,
    apple_watch_dfs={
        "hr": hr_df,
        "sleep": sleep_df,
        "spo2": spo2_df,
    },
    rppg_df=rppg_df,     # optional
)
# session.piezo → (n_seconds, 4) aligned to 1s grid
# session.apple_watch["hr"] → (n_seconds,) interpolated HR
```

---

## Signal Processing Details

### Piezoelectric (BCG) Pipeline

```
Raw ADC → DC removal → Bandpass filter → Wavelet denoise → LMS filter → Peak detection → Features
```

| Band | Range | Captures |
|---|---|---|
| BCG | 1 – 20 Hz | Heartbeat mechanical impulse |
| Respiration | 0.1 – 0.5 Hz | Chest/diaphragm movement |
| Micro-movement | 0.5 – 5 Hz | Body position shifts |

**Noise removal** uses two stages:
1. Wavelet soft-thresholding (PyWavelets `db4`, 5 levels) for broadband noise
2. LMS adaptive filter for correlated reference-channel interference

**Peak detection** adapts the Pan-Tompkins ECG algorithm for BCG: square → integrate → adaptive threshold → physiological refractory period.

### rPPG Pipeline

```
Camera frame → MediaPipe Face Mesh → ROI crop (forehead + cheeks)
→ Optical flow motion correction → CHROM / POS algorithm → Bandpass → HR / SpO₂
```

CHROM and POS algorithms project skin colour changes onto axes that maximise
haemoglobin signal and cancel specular reflections and illumination changes.

---

## Model Architecture Summary

The deep learning model is a **multi-task, multi-modal transformer-based network**:

```
PiezoEncoder   →  ──┐
                    ├── CrossAttentionFusion ──► HR, HRV, SpO₂, Resp
RPPGEncoder    →  ──┘                       ──► Sleep Stage (4-class)
                                            ──► Stress (3-class)
```

Key design choices:

- **Missing modality robustness**: if the camera is off, the model falls back to piezo-only using a learned `[MISSING]` token. Works in both directions.
- **Aleatoric uncertainty**: the regression head outputs a predicted variance per vital, so you know how confident the model is in each number.
- **Kendall multi-task weighting**: loss weights are learned automatically — no manual tuning of `λ₁, λ₂, λ₃`.
- **Stochastic modality dropout** (p=0.15 during training): randomly masks a modality to force the model to work robustly without it.

---

## Configuration

All hyperparameters live in `config/` — no hardcoded values anywhere in the code.

```bash
# Change window size
model.training.batch_size=64

# Use GPU
pipeline.inference.device=cuda

# Adjust loss weights
model.training.loss_weights.spo2=3.0

# Use late fusion instead of cross-attention
model.fusion.strategy=late
```

---

## Hardware Targets

| Platform | Inference (INT8 ONNX) | Step Feasible? |
|---|---|---|
| MacBook M2 | ~15 ms | Yes (5 s step) |
| Raspberry Pi 5 | ~150 ms | Yes (5 s step) |
| NVIDIA Jetson Nano | ~45 ms | Yes (5 s step) |

---

## Documentation

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | Complete system diagrams, data flow, all design decisions |
| `IMPROVEMENTS.md` | Every bug found in the original code + every change made |
| `RESULTS.md` | Evaluation metrics, accuracy vs targets, latency benchmarks |
| `DEPLOYMENT.md` | Full setup guide: local, Docker, Raspberry Pi, API reference |

---

## License

Private — BluSim internal project. Not for public distribution.
