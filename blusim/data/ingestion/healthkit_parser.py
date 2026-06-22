"""Apple Watch HealthKit CSV export parser.

Parses exported CSV files from the Health Auto Export app or the Apple Health
XML export. Produces a unified DataFrame with second-level timestamps and
sleep stage labels aligned to the mat/piezo timeline.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# HealthKit sleep stage value → canonical label
_SLEEP_VALUE_MAP: dict[str, str] = {
    "HKCategoryValueSleepAnalysisAwake": "Wake",
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "Light",
    "HKCategoryValueSleepAnalysisAsleepCore": "Light",
    "HKCategoryValueSleepAnalysisAsleepDeep": "Deep",
    "HKCategoryValueSleepAnalysisAsleepREM": "REM",
    "HKCategoryValueSleepAnalysisInBed": "InBed",
    # Short-form from Health Auto Export CSV
    "Awake": "Wake",
    "Core": "Light",
    "LS": "Light",
    "Deep": "Deep",
    "REM": "REM",
    "InBed": "InBed",
}

SLEEP_STAGE_INT: dict[str, int] = {
    "Wake": 0,
    "Light": 1,
    "Deep": 2,
    "REM": 3,
}


def parse_healthkit_sleep_csv(path: Path | str) -> pd.DataFrame:
    """Parse a Health Auto Export sleep CSV into a second-resolution DataFrame.

    Args:
        path: Path to CSV exported from Health Auto Export (or Apple Health).
              Expected columns: ``Source``, ``Start``, ``End``, ``Value`` or
              ``Sleep Analysis``.

    Returns:
        DataFrame with columns ``[time, sleep_stage, sleep_stage_int]`` at
        1-second resolution covering the full sleep period. ``time`` is a
        timezone-naive UTC datetime64.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names from different export formats
    col_map: dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if "start" in lc:
            col_map[c] = "Start"
        elif "end" in lc:
            col_map[c] = "End"
        elif "value" in lc or "sleep analysis" in lc or "stage" in lc:
            col_map[c] = "Value"
        elif "source" in lc:
            col_map[c] = "Source"
    df = df.rename(columns=col_map)

    required = {"Start", "End", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in HealthKit CSV: {missing}")

    # Drop InBed rows (not a sleep stage)
    df = df[~df["Value"].str.contains("InBed", na=False)].copy()

    df["Start"] = pd.to_datetime(df["Start"], utc=True).dt.tz_localize(None)
    df["End"] = pd.to_datetime(df["End"], utc=True).dt.tz_localize(None)
    df["stage"] = df["Value"].map(_SLEEP_VALUE_MAP).fillna("Unknown")

    rows: list[dict] = []
    for _, row in df.iterrows():
        ts_range = pd.date_range(row["Start"], row["End"], freq="1s")
        for ts in ts_range:
            rows.append({"time": ts, "sleep_stage": row["stage"]})

    expanded = pd.DataFrame(rows)
    expanded["sleep_stage_int"] = expanded["sleep_stage"].map(SLEEP_STAGE_INT).fillna(-1).astype(int)
    expanded = expanded.sort_values("time").reset_index(drop=True)
    expanded = expanded.drop_duplicates(subset="time", keep="last")
    logger.info("Parsed %d sleep stage seconds from %s", len(expanded), path.name)
    return expanded


def parse_healthkit_hr_csv(path: Path | str) -> pd.DataFrame:
    """Parse Apple Watch heart rate export CSV.

    Args:
        path: CSV with columns ``Start``, ``Finish``, ``Value`` (BPM).

    Returns:
        DataFrame ``[time, hr_bpm]``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    time_col = next((c for c in df.columns if "start" in c.lower() or "time" in c.lower()), None)
    val_col = next((c for c in df.columns if "value" in c.lower() or "bpm" in c.lower() or "heart" in c.lower()), None)
    if not time_col or not val_col:
        raise ValueError(f"Cannot find time/value columns in {path.name}. Columns: {list(df.columns)}")

    df = df.rename(columns={time_col: "time", val_col: "hr_bpm"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["hr_bpm"] = pd.to_numeric(df["hr_bpm"], errors="coerce")
    df = df.dropna(subset=["hr_bpm"]).sort_values("time").reset_index(drop=True)
    return df[["time", "hr_bpm"]]


def parse_healthkit_hrv_csv(path: Path | str) -> pd.DataFrame:
    """Parse Apple Watch HRV (SDNN) export CSV.

    Args:
        path: CSV with columns ``Start``, ``Value`` (ms).

    Returns:
        DataFrame ``[time, hrv_sdnn_ms]``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    time_col = next((c for c in df.columns if "start" in c.lower() or "time" in c.lower()), None)
    val_col = next((c for c in df.columns if "value" in c.lower() or "hrv" in c.lower() or "sdnn" in c.lower()), None)
    if not time_col or not val_col:
        raise ValueError(f"Cannot identify columns in {path.name}")

    df = df.rename(columns={time_col: "time", val_col: "hrv_sdnn_ms"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["hrv_sdnn_ms"] = pd.to_numeric(df["hrv_sdnn_ms"], errors="coerce")
    df = df.dropna(subset=["hrv_sdnn_ms"]).sort_values("time").reset_index(drop=True)
    return df[["time", "hrv_sdnn_ms"]]


def parse_healthkit_spo2_csv(path: Path | str) -> pd.DataFrame:
    """Parse Apple Watch Blood Oxygen export CSV.

    Args:
        path: CSV with columns ``Start``, ``Value`` (percentage 0–100 or 0–1).

    Returns:
        DataFrame ``[time, spo2_pct]`` with SpO2 in percent (0–100).
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    time_col = next((c for c in df.columns if "start" in c.lower() or "time" in c.lower()), None)
    val_col = next((c for c in df.columns if "value" in c.lower() or "oxygen" in c.lower() or "spo2" in c.lower()), None)
    if not time_col or not val_col:
        raise ValueError(f"Cannot identify columns in {path.name}")

    df = df.rename(columns={time_col: "time", val_col: "spo2_pct"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["spo2_pct"] = pd.to_numeric(df["spo2_pct"], errors="coerce")
    # Normalise 0–1 range to percentage
    if df["spo2_pct"].dropna().max() <= 1.0:
        df["spo2_pct"] *= 100.0
    df = df.dropna(subset=["spo2_pct"]).sort_values("time").reset_index(drop=True)
    return df[["time", "spo2_pct"]]


def parse_healthkit_resp_csv(path: Path | str) -> pd.DataFrame:
    """Parse Apple Watch Respiratory Rate export CSV.

    Args:
        path: CSV with columns ``Start``, ``Value`` (breaths/min).

    Returns:
        DataFrame ``[time, resp_rate_bpm]``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    time_col = next((c for c in df.columns if "start" in c.lower() or "time" in c.lower()), None)
    val_col = next((c for c in df.columns if "value" in c.lower() or "resp" in c.lower() or "breath" in c.lower()), None)
    if not time_col or not val_col:
        raise ValueError(f"Cannot identify columns in {path.name}")

    df = df.rename(columns={time_col: "time", val_col: "resp_rate_bpm"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["resp_rate_bpm"] = pd.to_numeric(df["resp_rate_bpm"], errors="coerce")
    df = df.dropna(subset=["resp_rate_bpm"]).sort_values("time").reset_index(drop=True)
    return df[["time", "resp_rate_bpm"]]


def parse_healthkit_xml(xml_path: Path | str) -> dict[str, pd.DataFrame]:
    """Parse Apple Health export.xml and extract all relevant record types.

    Args:
        xml_path: Path to Apple Health ``export.xml``.

    Returns:
        Dict mapping vital name to DataFrame:
        ``{sleep, hr, hrv, spo2, resp}``.
    """
    xml_path = Path(xml_path)
    logger.info("Parsing Apple Health XML: %s", xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sleep_rows: list[dict] = []
    hr_rows: list[dict] = []
    hrv_rows: list[dict] = []
    spo2_rows: list[dict] = []
    resp_rows: list[dict] = []

    for record in root.iter("Record"):
        rtype = record.get("type", "")
        start = pd.to_datetime(record.get("startDate"))
        end = pd.to_datetime(record.get("endDate"))
        value = record.get("value", "")

        if rtype == "HKCategoryTypeIdentifierSleepAnalysis":
            stage = _SLEEP_VALUE_MAP.get(value, "Unknown")
            if stage not in ("InBed", "Unknown"):
                for ts in pd.date_range(start, end, freq="1s"):
                    sleep_rows.append({"time": ts, "sleep_stage": stage,
                                       "sleep_stage_int": SLEEP_STAGE_INT.get(stage, -1)})
        elif rtype == "HKQuantityTypeIdentifierHeartRate":
            hr_rows.append({"time": start, "hr_bpm": float(value)})
        elif rtype == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
            hrv_rows.append({"time": start, "hrv_sdnn_ms": float(value)})
        elif rtype == "HKQuantityTypeIdentifierOxygenSaturation":
            spo2_rows.append({"time": start, "spo2_pct": float(value) * 100})
        elif rtype == "HKQuantityTypeIdentifierRespiratoryRate":
            resp_rows.append({"time": start, "resp_rate_bpm": float(value)})

    def _to_df(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        return df.sort_values("time").reset_index(drop=True)

    return {
        "sleep": _to_df(sleep_rows),
        "hr": _to_df(hr_rows),
        "hrv": _to_df(hrv_rows),
        "spo2": _to_df(spo2_rows),
        "resp": _to_df(resp_rows),
    }
