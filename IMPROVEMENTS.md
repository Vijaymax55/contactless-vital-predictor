# Improvements & Changes Log

## PHASE 1 — Audit Findings

### CRITICAL Issues

| ID | File | Issue | Status |
|----|------|-------|--------|
| C1 | `sleep_cycle/SC_model.py:29` | Predicts `Pose` column, not sleep stage `label` — model output is wrong | Fixed |
| C2 | `sleep_cycle/SC_to_csv.py:17` | `LightSleep1` and `LightSleep2` share identical RGB `[85,139,247]` — indistinguishable | Fixed |
| C3 | `sleep_cycle/SC_to_csv.py:146-148` | `ET, _ = ET.split("\n")` crashes if OCR returns unexpected whitespace/newline format | Fixed |
| C4 | `god/mat_data_get.py:6` | Hardcoded Windows absolute path `C:\Users\Amarjith CK\...` at module level | Fixed |
| C5 | `sleep_cycle/SC_preprocessing.py:5-7` | Hardcoded local filesystem paths as module-level globals | Fixed |
| C6 | `sleep_cycle/SC_to_csv.py:52-67` | `assign_sleep_mode` state machine: `current_mode in (1..5)` branch returns 0 instead of the newly detected stage | Fixed |
| C7 | `god/execution.py:49-64` | Folder check logic: wrong string `"accuracy_prediction"` tested but never in loop; `cam()` `process` list bug | Fixed |
| C8 | All models | No signal processing — raw 4-channel ADC values fed directly to XGBoost/GBM without filtering | Fixed |
| C9 | `sc_model_v2.ipynb` | Flattening 4×490 samples → 1960 features fed to XGBoost is extremely memory-intensive with no temporal context | Fixed |
| C10 | All models | Random train/test split with no subject-independent separation → severe data leakage | Fixed |

### MAJOR Issues

| ID | File | Issue | Status |
|----|------|-------|--------|
| M1 | All | No BCG peak detection; no HR/HRV/SpO2/Respiration extraction from piezo signal | Fixed |
| M2 | All | No rPPG pipeline; camera modality completely unimplemented | Fixed |
| M3 | All | No Apple Watch HealthKit CSV parsing; labels scraped from screenshot pixels (brittle) | Fixed |
| M4 | `god/dashboard_script.py` | Windows path separators `\` hardcoded throughout on macOS/Linux | Fixed |
| M5 | All | No timestamp synchronization across modalities | Fixed |
| M6 | All | No multi-modal fusion; models train on single modality | Fixed |
| M7 | All | No uncertainty quantification on predictions | Fixed |
| M8 | All | No real-time sliding window inference pipeline | Fixed |
| M9 | All | No tests whatsoever | Fixed |
| M10 | All | No Docker/containerization | Fixed |

### MINOR Issues

| ID | File | Issue | Status |
|----|------|-------|--------|
| m1 | `god/requirement.txt` | `tempfile`, `os-sys`, `sklearn` are not valid pip package names | Fixed |
| m2 | `god/requirement.txt` | `custom_oem` tesseract config string erroneously placed in requirements.txt | Fixed |
| m3 | All | No type hints | Fixed |
| m4 | All | No docstrings | Fixed |
| m5 | All | No logging (print statements only) | Fixed |
| m6 | All | No configuration management (Hydra/Pydantic) | Fixed |
| m7 | `god/accuracy_csv.py` | `create_csv_file` called every time `add_accuracy` runs, overwriting existing data | Fixed |
| m8 | `sleep_cycle/SC_to_csv.py` | `datetime` imported twice (stdlib and module) — name collision | Fixed |

---

## PHASE 2 — Signal Processing Improvements

### Piezoelectric Signal Processing (`blusim/data/preprocessing/piezo_signal.py`)
- **Added adaptive bandpass filters**: BCG (1–20 Hz), Respiration (0.1–0.5 Hz), micro-movement (0.5–5 Hz)
- **Added motion artifact removal**: PyWavelets soft-thresholding (db4, level 5) and LMS adaptive filter
- **Added BCG peak detection**: Pan-Tompkins-adapted algorithm with adaptive threshold
- **Added quality score**: Signal quality index (SQI) based on spectral entropy
- **Added segment rejection**: Automatic rejection of segments with SQI < threshold

### rPPG Signal Processing (`blusim/data/preprocessing/rppg_signal.py`)
- **Added CHROM algorithm**: Chrominance-based rPPG (De Haan & Jeanne, 2013)
- **Added POS algorithm**: Plane-Orthogonal-to-Skin (Wang et al., 2017)
- **Added MediaPipe face mesh ROI tracking**: Forehead + cheek ROIs
- **Added Lucas-Kanade optical flow** motion compensation
- **Added SpO2 estimation**: AC/DC ratio from red/infrared channels

### Feature Extraction (`blusim/features/piezo_features.py`, `blusim/features/rppg_features.py`)
- **Time-domain**: IJ amplitude, JK slope, peak-to-peak intervals, RMS, kurtosis, skewness, Hjorth parameters
- **Frequency-domain**: Welch PSD, dominant frequency, spectral entropy, LF/HF power ratio
- **Time-frequency**: STFT spectrogram statistics, CWT scalogram energy
- **Nonlinear**: Sample Entropy, Approximate Entropy, DFA scaling exponent, Poincaré SD1/SD2

---

## PHASE 3 — Model Architecture Improvements

### Per-Modality Encoders
- **Piezo encoder** (`blusim/models/encoders/piezo_encoder.py`): 1D-CNN → BiLSTM → Multi-head Self-Attention
- **rPPG encoder** (`blusim/models/encoders/rppg_encoder.py`): 1D-CNN + Transformer blocks

### Multi-Modal Fusion (`blusim/models/fusion/multimodal_fusion.py`)
- **Three fusion strategies implemented**: Late fusion, Early fusion, Cross-attention fusion
- **Learnable modality weights**: Gating network learns to trust piezo vs camera dynamically
- **Missing-modality robustness**: Zero-masking + learned missing-modal token

### Multi-Task Output Heads (`blusim/models/heads/output_heads.py`)
- **Regression head**: HR, HRV (RMSSD, SDNN), SpO2, Respiration Rate
- **Classification head**: Sleep stage (4-class), Stress level (3-class)
- **Uncertainty head**: Predicted variance per regression output (aleatoric uncertainty)

### Loss Function (`blusim/training/loss.py`)
- **Multi-task loss** with Kendall uncertainty weighting
- **Temporal consistency loss**: Penalizes large consecutive prediction jumps
- **Focal loss** for imbalanced sleep stage classification

---

## PHASE 4 — Training Pipeline Improvements

- **Subject-independent splits**: GroupShuffleSplit on subject ID — no subject appears in both train and test
- **Data augmentation**: Gaussian noise, magnitude scaling, time warping, channel dropout, DC shift
- **Curriculum learning**: 3-stage curriculum from clean → noisy samples
- **Calibration**: Platt scaling for classification, conformal prediction intervals for regression
- **MLflow integration**: All experiments tracked with metrics, params, and artifacts

---

## PHASE 5 — Evaluation Suite

- **Per-vital metrics**: MAE, RMSE, Pearson r for regression; accuracy, Cohen's κ for classification
- **Bland-Altman plots**: HR and HRV vs Apple Watch ground truth
- **Robustness tests**: Motion artifact levels, body position, lighting conditions
- **Latency profiling**: End-to-end inference time measurement

---

## PHASE 6 — Real-Time & Deployment

- **Sliding window inference**: 30s window, 5s step, async pipeline
- **Circular buffer**: Efficient streaming data management
- **ONNX export**: Model exported to ONNX with INT8 quantization
- **FastAPI REST API**: `/ingest/piezo`, `/ingest/video-frame`, `/vitals/latest`, `/vitals/history`, `/health`
- **Docker**: Dockerfile + docker-compose with InfluxDB

---

## PHASE 7 — Code Quality

- Python 3.10+ type hints throughout
- Google-style docstrings on all public functions/classes
- Pydantic Settings for configuration management
- Structured JSON logging
- pytest test suite (>80% coverage target)
- No hardcoded paths or magic numbers
