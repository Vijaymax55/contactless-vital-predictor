# Sleep Cycle — Original Sleep Stage Prediction

The first version of sleep stage prediction, built before the production system.
Uses pixel colour extraction from Apple Watch sleep dashboard screenshots to label
sleep stages, then trains an XGBoost/GBM classifier on raw piezo features.

## Files

| File | Purpose |
|---|---|
| `SC_to_csv.py` | Reads sleep dashboard screenshots, maps pixel colours → sleep stage labels, writes CSV |
| `SC_preprocessing.py` | Merges piezo CSV with sleep stage labels; PCHIP interpolation to 490 Hz |
| `SC_model.py` | Trains Gradient Boosting Classifier on flattened 4×490=1960 feature vector |
| `sc_model_v2.ipynb` | Iterative experiments: Butterworth filter, Apple Watch HealthKit CSV parsing, Altair plots |

## Sleep Stage Colour Map (from Apple Watch dashboard screenshots)

| Stage | RGB |
|---|---|
| Wake | `[255, 255, 255]` |
| REM | `[54, 162, 235]` |
| Light Sleep | `[85, 139, 247]` |
| Deep Sleep | `[22, 50, 115]` |

## Known Issues (superseded by production system)

- `SC_model.py` line 42: `y = master_data['Pose']` — trains on Pose column, not sleep stage
- `SC_to_csv.py`: `LightSleep1` and `LightSleep2` both assigned identical RGB `[85, 139, 247]`
- `SC_to_csv.py`: `assign_sleep_mode()` state machine returns `0` instead of the new stage
- `SC_preprocessing.py`: hardcoded path `/Users/vijayprakash/Downloads/...` as module global
- No subject-independent train/test split — data leakage across sessions
- No signal processing — raw ADC fed to XGBoost with no bandpass, denoising, or SQI gating

These issues are fully resolved in the production system (`blusim/` package).
