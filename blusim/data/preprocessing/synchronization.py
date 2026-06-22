"""Cross-modality timestamp synchronization.

Aligns piezo, rPPG (camera), and Apple Watch streams to a common time axis
using cross-correlation lag estimation and linear interpolation.

Design notes:
- Piezo data is the primary clock (highest rate, most continuous).
- Apple Watch and rPPG signals are resampled / interpolated to piezo timeline.
- Lag between streams is estimated by cross-correlating the low-frequency
  HR-band envelopes (a signal present in all modalities).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import correlate

logger = logging.getLogger(__name__)


@dataclass
class SyncedSession:
    """All modalities aligned to the same second-resolution time axis.

    Attributes:
        timestamps: Shared datetime64 array (1-second resolution).
        piezo: Piezo channel data of shape ``(n_seconds, n_channels)``.
        rppg: rPPG waveform of shape ``(n_seconds,)`` or ``None``.
        apple_watch: Dict mapping vital name → array aligned to timestamps.
        lag_piezo_watch_s: Estimated time lag (piezo − watch) in seconds.
        lag_piezo_rppg_s: Estimated time lag (piezo − rPPG) in seconds.
    """

    timestamps: np.ndarray
    piezo: np.ndarray
    rppg: Optional[np.ndarray]
    apple_watch: dict[str, np.ndarray]
    lag_piezo_watch_s: float = 0.0
    lag_piezo_rppg_s: float = 0.0


def estimate_lag_cross_correlation(
    ref: np.ndarray,
    target: np.ndarray,
    max_lag_samples: int = 300,
) -> int:
    """Estimate time lag between two signals via normalised cross-correlation.

    Returns the lag (in samples) such that ``target[lag:] ≈ ref``.
    Positive lag means target is delayed relative to reference.

    Args:
        ref: Reference signal (1-D array).
        target: Target signal to align to reference.
        max_lag_samples: Maximum lag to search (±max_lag_samples).

    Returns:
        Estimated lag in samples.
    """
    # Normalise
    ref_n = (ref - ref.mean()) / (ref.std() + 1e-8)
    tgt_n = (target - target.mean()) / (target.std() + 1e-8)

    corr = correlate(ref_n, tgt_n, mode="full")
    lags = np.arange(-(len(tgt_n) - 1), len(ref_n))

    # Restrict search window
    mask = np.abs(lags) <= max_lag_samples
    best_idx = np.argmax(np.abs(corr[mask]))
    lag = lags[mask][best_idx]
    logger.debug("Cross-correlation peak lag = %d samples", lag)
    return int(lag)


def interpolate_to_grid(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    method: str = "linear",
    fill_value: float = float("nan"),
) -> np.ndarray:
    """Resample ``values`` from ``source_times`` onto ``target_times``.

    Args:
        values: 1-D array of values at ``source_times``.
        source_times: Source timestamps as float seconds or datetime64.
        target_times: Target timestamps (same units).
        method: Interpolation kind: ``"linear"``, ``"nearest"``, ``"cubic"``.
        fill_value: Value for extrapolation outside the source range.

    Returns:
        Interpolated values at ``target_times``.
    """
    # Convert datetime64 to float seconds
    def to_float(t: np.ndarray) -> np.ndarray:
        if np.issubdtype(t.dtype, np.datetime64):
            return t.astype("datetime64[ns]").astype(np.float64) / 1e9
        return t.astype(np.float64)

    src_f = to_float(source_times)
    tgt_f = to_float(target_times)

    if len(src_f) < 2:
        return np.full(len(tgt_f), fill_value)

    interp_fn = interp1d(
        src_f, values, kind=method, bounds_error=False, fill_value=fill_value
    )
    return interp_fn(tgt_f)


def synchronize_session(
    piezo_df: pd.DataFrame,
    apple_watch_dfs: dict[str, pd.DataFrame],
    rppg_df: Optional[pd.DataFrame] = None,
    max_lag_seconds: float = 5.0,
    piezo_time_col: str = "Time",
    piezo_channel_cols: Optional[list[str]] = None,
) -> SyncedSession:
    """Align all modalities to the piezo time grid.

    Args:
        piezo_df: Piezo sensor DataFrame with a ``Time`` column (datetime).
        apple_watch_dfs: Dict mapping vital name → Apple Watch DataFrame with
                         a ``time`` column and one value column.
        rppg_df: Optional rPPG DataFrame with ``time`` (float s or datetime)
                 and ``rppg`` columns.
        max_lag_seconds: Maximum lag to search during cross-correlation.
        piezo_time_col: Name of the timestamp column in ``piezo_df``.
        piezo_channel_cols: Channel column names; inferred if ``None``.

    Returns:
        :class:`SyncedSession` with all data on a shared second-resolution grid.
    """
    # --- Piezo time grid ---
    piezo_df = piezo_df.copy()
    piezo_df[piezo_time_col] = pd.to_datetime(piezo_df[piezo_time_col])
    piezo_df = piezo_df.sort_values(piezo_time_col).reset_index(drop=True)

    if piezo_channel_cols is None:
        piezo_channel_cols = [c for c in piezo_df.columns if c.lower().startswith(("chn", "channel", "ch_"))]

    if not piezo_channel_cols:
        raise ValueError("No piezo channel columns found. Pass ``piezo_channel_cols`` explicitly.")

    # Resample piezo to 1-second averages
    piezo_df["_ts"] = piezo_df[piezo_time_col].dt.floor("1s")
    piezo_1s = (
        piezo_df.groupby("_ts")[piezo_channel_cols]
        .mean()
        .reset_index()
        .rename(columns={"_ts": "time"})
    )

    shared_times = piezo_1s["time"].values  # dtype datetime64[ns]
    piezo_arr = piezo_1s[piezo_channel_cols].values.astype(np.float32)

    # --- Apple Watch vitals ---
    watch_aligned: dict[str, np.ndarray] = {}
    for vital_name, watch_df in apple_watch_dfs.items():
        watch_df = watch_df.copy()
        time_col = "time" if "time" in watch_df.columns else watch_df.columns[0]
        val_col = [c for c in watch_df.columns if c != time_col][0]

        watch_df[time_col] = pd.to_datetime(watch_df[time_col])
        watch_df = watch_df.sort_values(time_col).reset_index(drop=True)

        interpolated = interpolate_to_grid(
            watch_df[val_col].values.astype(np.float64),
            watch_df[time_col].values,
            shared_times,
        )
        watch_aligned[vital_name] = interpolated.astype(np.float32)
        logger.debug("Interpolated Apple Watch '%s' onto piezo grid", vital_name)

    # --- rPPG ---
    rppg_aligned: Optional[np.ndarray] = None
    lag_rppg = 0.0
    if rppg_df is not None:
        rppg_df = rppg_df.copy()
        rppg_time_col = "time" if "time" in rppg_df.columns else rppg_df.columns[0]
        rppg_val_col = "rppg" if "rppg" in rppg_df.columns else [c for c in rppg_df.columns if c != rppg_time_col][0]
        rppg_df[rppg_time_col] = pd.to_datetime(rppg_df[rppg_time_col])

        rppg_vals = rppg_df[rppg_val_col].values.astype(np.float64)
        rppg_times = rppg_df[rppg_time_col].values

        # Cross-correlate rPPG with piezo ch 0 envelope to find lag
        piezo_env = np.abs(piezo_1s[piezo_channel_cols[0]].values.astype(np.float64))
        rppg_resampled = interpolate_to_grid(rppg_vals, rppg_times, shared_times)
        rppg_env = np.abs(rppg_resampled)

        valid = ~np.isnan(rppg_env)
        if np.sum(valid) > 10:
            lag_samples = estimate_lag_cross_correlation(
                piezo_env[valid],
                rppg_env[valid],
                max_lag_samples=int(max_lag_seconds),
            )
            lag_rppg = float(lag_samples)
            logger.info("Estimated piezo-rPPG lag: %.1f s", lag_rppg)

        rppg_aligned = rppg_resampled.astype(np.float32)

    return SyncedSession(
        timestamps=shared_times,
        piezo=piezo_arr,
        rppg=rppg_aligned,
        apple_watch=watch_aligned,
        lag_piezo_watch_s=0.0,
        lag_piezo_rppg_s=lag_rppg,
    )
