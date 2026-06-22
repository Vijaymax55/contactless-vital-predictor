# BluSim System Architecture v2.0

## Overview

BluSim is a **contactless, non-invasive health vital prediction system** that fuses signals from:
- **Piezoelectric sensors** under a mattress (BCG, respiration, micro-movement)
- **Camera / rPPG** for HR, SpO2 from facial colour changes
- **Apple Watch** as ground-truth supervisor during training

---

## System Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SENSING LAYER                                   │
│                                                                         │
│  Piezo Sensors          Camera                   Apple Watch           │
│  (4 channels,          (30 FPS,                  (HealthKit CSV        │
│   490 Hz ADC)          face RGB)                  or XML export)       │
└────────┬────────────────────┬────────────────────────┬─────────────────┘
         │                    │                        │
         ▼                    ▼                        ▼
┌────────────────┐  ┌──────────────────┐   ┌──────────────────────────┐
│ CircularBuffer │  │ FaceMeshROI      │   │ HealthKit Parser         │
│ (piezo_buf)    │  │ Tracker          │   │ (sleep, HR, HRV,         │
│                │  │ (MediaPipe)      │   │  SpO2, RespRate)         │
└───────┬────────┘  └────────┬─────────┘   └────────────┬─────────────┘
        │                    │                           │
        ▼                    ▼                           │
┌──────────────────────────────────────┐                │
│          SIGNAL CONDITIONING          │                │
│                                      │                │
│  PiezoSignal:                        │                │
│  • Bandpass (BCG:1-20Hz,            │                ▼
│    Resp:0.1-0.5Hz, Move:0.5-5Hz)    │   ┌────────────────────────┐
│  • Wavelet denoising (db4, L=5)     │   │ Synchronization        │
│  • LMS adaptive filter              │   │ • Cross-correlation lag │
│  • BCG peak detection               │   │ • 1-second resampling   │
│  • Signal Quality Index (SQI)       │   │ • Linear interpolation  │
│                                      │   └────────────┬───────────┘
│  rPPG:                              │                │
│  • CHROM / POS algorithm            │                │
│  • Motion compensation (LK flow)    │◄───────────────┘
│  • SpO2 AC/DC ratio estimate        │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       FEATURE EXTRACTION                              │
│                                                                      │
│  Piezo Features (per channel):          rPPG Features:               │
│  ┌──────────────────┐                  ┌──────────────────┐         │
│  │ Time-domain      │                  │ HR from PSD      │         │
│  │ • RR intervals   │                  │ • Dominant freq  │         │
│  │ • RMSSD, SDNN    │                  │ PP intervals     │         │
│  │ • IJ amplitude   │                  │ • RMSSD proxy    │         │
│  │ • JK slope       │                  │ SpO2 AC/DC       │         │
│  │ • Hjorth params  │                  │ Waveform stats   │         │
│  ├──────────────────┤                  └──────────────────┘         │
│  │ Frequency-domain │                                                │
│  │ • Welch PSD      │                                                │
│  │ • LF/HF ratio    │                                                │
│  │ • Spec entropy   │                                                │
│  ├──────────────────┤                                                │
│  │ Time-frequency   │                                                │
│  │ • STFT stats     │                                                │
│  │ • CWT energy     │                                                │
│  ├──────────────────┤                                                │
│  │ Nonlinear        │                                                │
│  │ • SampEn, ApEn   │                                                │
│  │ • DFA alpha      │                                                │
│  │ • Poincaré SD1/2 │                                                │
│  └──────────────────┘                                                │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼ (raw waveform windows)
┌──────────────────────────────────────────────────────────────────────┐
│                      MODEL ARCHITECTURE                               │
│                                                                      │
│  ┌─────────────────────────┐   ┌─────────────────────────┐          │
│  │    PiezoEncoder          │   │    RPPGEncoder            │          │
│  │                         │   │                           │          │
│  │  Input: (B,4,14700)     │   │  Input: (B,9,900)         │          │
│  │  Conv1D × 3 + BN + ReLU │   │  Conv1D × 3 + BN + GELU  │          │
│  │  MaxPool × 3            │   │  MaxPool × 3              │          │
│  │  BiLSTM (2 layers)      │   │  Positional Encoding      │          │
│  │  Multi-head Attn (4H)   │   │  Transformer × 2          │          │
│  │  → Embed: (B,256)       │   │  CLS pooling              │          │
│  └───────────┬─────────────┘   │  → Embed: (B,256)         │          │
│              │                 └──────────────┬────────────┘          │
│              │                                │                       │
│              ▼                                ▼                       │
│         ┌───────────────────────────────────────────┐                │
│         │        CrossAttentionFusion                │                │
│         │                                           │                │
│         │  Piezo ──┐                               │                │
│         │           ├─ Cross-Attention ──► Gate ──► Fused (B,256)   │
│         │  rPPG  ──┘                               │                │
│         │                                           │                │
│         │  Missing modality → learned [MISS] token  │                │
│         │  Stochastic modality dropout (p=0.15)     │                │
│         └────────────────────┬──────────────────────┘                │
│                              │                                        │
│              ┌───────────────┼────────────────┐                      │
│              ▼               ▼                ▼                      │
│  ┌───────────────────┐ ┌─────────────┐ ┌───────────────┐            │
│  │  RegressionHead   │ │SleepHead    │ │ StressHead    │            │
│  │  • HR, RMSSD,     │ │ 4-class     │ │ 3-class       │            │
│  │    SDNN, SpO2,    │ │ (Wake/Light │ │ (Low/Med/High)│            │
│  │    RespRate       │ │  /Deep/REM) │ │               │            │
│  │  • + log-variance │ │ + focal loss│ │ + CE loss     │            │
│  │    (uncertainty)  │ │             │ │               │            │
│  └───────────────────┘ └─────────────┘ └───────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MULTI-TASK LOSS FUNCTION                           │
│                                                                      │
│  L_total = (L_regression + L_temporal) / 2σ²_reg + log(σ_reg)       │
│           + L_sleep / 2σ²_sleep + log(σ_sleep)                       │
│           + L_stress / 2σ²_stress + log(σ_stress)                    │
│                                                                      │
│  • L_regression: Gaussian NLL (heteroscedastic, masks NaN targets)  │
│  • L_temporal: consecutive prediction smoothness penalty             │
│  • L_sleep: Focal loss (γ=2, handles class imbalance)               │
│  • L_stress: Cross-entropy                                           │
│  • σ params: Kendall uncertainty weighting (learned)                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Inference Pipeline (Real-Time)

```
Sensor Stream ──► CircularBuffer (2 × window_size)
                          │
              every step_size seconds
                          │
                          ▼
              Extract window_size window
                          │
                          ▼
              Process (bandpass + denoise)
                          │
                          ▼
              Model inference (PyTorch or ONNX)
                          │
                          ▼
              VitalSnapshot (HR, HRV, SpO2, Sleep, Stress)
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
           FastAPI GET        InfluxDB
           /vitals/latest     (time-series)
```

---

## Directory Structure

```
blusim_system/
├── config/
│   ├── config.yaml             # Root Hydra config
│   ├── data/default.yaml       # Sampling rates, band definitions, split params
│   ├── model/default.yaml      # Architecture hyperparameters
│   └── pipeline/default.yaml   # Inference, API, ONNX, DB settings
│
├── blusim/
│   ├── data/
│   │   ├── ingestion/
│   │   │   ├── circular_buffer.py      # Thread-safe ring buffer
│   │   │   ├── healthkit_parser.py     # Apple Watch CSV/XML parser
│   │   │   └── s3_loader.py            # AWS S3 piezo data downloader
│   │   └── preprocessing/
│   │       ├── piezo_signal.py         # Bandpass, wavelet, LMS, peak detection
│   │       ├── rppg_signal.py          # CHROM/POS, MediaPipe ROI, SpO2
│   │       └── synchronization.py      # Cross-correlation lag, interpolation
│   │
│   ├── features/
│   │   ├── piezo_features.py           # Time, freq, TF, nonlinear features
│   │   └── rppg_features.py            # rPPG HR, HRV proxy, SpO2
│   │
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── piezo_encoder.py        # 1D-CNN + BiLSTM + MH-Attention
│   │   │   └── rppg_encoder.py         # 1D-CNN + Transformer (CLS token)
│   │   ├── fusion/
│   │   │   └── multimodal_fusion.py    # Late, CrossAttention fusion
│   │   ├── heads/
│   │   │   └── output_heads.py         # Regression, Sleep, Stress heads
│   │   └── vital_predictor.py          # Top-level nn.Module
│   │
│   ├── training/
│   │   ├── augmentation.py             # Noise, scale, time-warp, dropout
│   │   ├── loss.py                     # GaussNLL, Focal, MultiTask, Kendall
│   │   └── trainer.py                  # Training loop, MLflow, checkpointing
│   │
│   ├── evaluation/
│   │   └── metrics.py                  # MAE, RMSE, r, kappa, Bland-Altman
│   │
│   ├── inference/
│   │   ├── pipeline.py                 # Real-time sliding window pipeline
│   │   └── onnx_export.py              # ONNX FP32 + INT8 quantization
│   │
│   └── api/
│       └── main.py                     # FastAPI: ingest + vitals + health
│
├── tests/
│   ├── test_piezo_signal.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── scripts/
│   ├── train.py                        # Hydra-based training entry point
│   ├── evaluate.py                     # Evaluation on held-out data
│   └── export_onnx.py                  # ONNX export + INT8 quantization
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── ARCHITECTURE.md
├── IMPROVEMENTS.md
├── RESULTS.md
└── DEPLOYMENT.md
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Piezo is the primary modality | Continuous, doesn't require camera. BCG is always available during sleep. |
| CrossAttention fusion | Allows modalities to inform each other; learns which to trust per sample. |
| Stochastic modality dropout | Trains model to work when camera is unavailable (lights off, etc.). |
| Gaussian NLL loss | Gives the model an aleatoric uncertainty estimate per prediction. |
| Kendall multi-task weighting | Prevents one high-scale task from dominating training. |
| Subject-independent splits | Prevents data leakage; tests true generalisation to new users. |
| Circular buffer | O(1) amortised writes; no memory copy on window extraction. |
| ONNX INT8 | ~4× size reduction, ~2–4× latency improvement on ARM (Raspberry Pi). |
