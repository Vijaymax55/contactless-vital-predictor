# Legacy Code

This directory preserves the original codebase developed during early research and data
collection phases at Blusim Tech. It is kept here for reference and to show the evolution
of the system.

| Folder | What it was | Replaced by |
|---|---|---|
| `god/` | Live data dashboard + S3 downloader + accuracy tracker | `blusim/api/`, `blusim/data/ingestion/s3_loader.py`, `blusim/evaluation/` |
| `sleep_cycle/` | First sleep stage prediction model (pixel colour → XGBoost) | `blusim/models/`, `blusim/training/`, `blusim/data/preprocessing/` |

The production system in `blusim/` addresses all known bugs and architectural limitations
of this legacy code. See `IMPROVEMENTS.md` in the root for a full audit.
