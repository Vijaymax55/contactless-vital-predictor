# GOD — Grow Own Data Dashboard

The original operational dashboard used during live data collection sessions.
It displays piezoelectric sensor streams from S3, computes accuracy against Apple Watch
ground truth, and tracks model versions via git.

## Files

| File | Purpose |
|---|---|
| `dashboard_script.py` | Dash web app — plots raw piezo channels + model accuracy per session |
| `execution.py` | Orchestrator — launches dashboard, S3 downloader, and accuracy tracker as parallel processes |
| `mat_data_get.py` | Downloads hourly piezo CSV files from AWS S3 for a given date |
| `accuracy_csv.py` | Reads Apple Watch export and writes per-session accuracy metrics to CSV |
| `files_git.py` | Monitors the git repo for new model commits and logs diffs |
| `requirement.txt` | Original dependencies (Windows environment) |

## Known Issues (superseded by production system)

- Hardcoded Windows path `C:\Users\Amarjith CK\...` in `mat_data_get.py`
- Dashboard uses `fr'{root}\{date}\...'` backslash separators (Windows-only)
- `accuracy_csv.py` calls `create_csv_file()` on every run, overwriting previous results
- No signal processing — raw ADC values fed directly to the model

These issues are fully resolved in the production system (`blusim/` package).
