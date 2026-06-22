"""Feature extraction from rPPG waveforms.

Features extracted per window:
- HR from dominant PSD frequency
- Amplitude statistics of the rPPG waveform
- HRV-proxy from rPPG peak-to-peak intervals
- SpO2 estimate from AC/DC ratio
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew

from blusim.data.preprocessing.rppg_signal import ROITrace, estimate_spo2_from_rois

logger = logging.getLogger(__name__)


def extract_rppg_features(
    rppg_signal: np.ndarray,
    fps: float = 30.0,
    roi_trace: ROITrace | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute rPPG feature vector for one window.

    Args:
        rppg_signal: Bandpass-filtered rPPG waveform (1-D, normalised).
        fps: Video frame rate.
        roi_trace: Original ROI colour trace for SpO2 estimation.

    Returns:
        ``(feature_vector, feature_names)`` float32 array + name list.
    """
    feats: dict[str, float] = {}

    # --- HR from PSD ---
    nperseg = min(256, len(rppg_signal))
    freqs, psd = welch(rppg_signal, fs=fps, nperseg=nperseg)
    mask_hr = (freqs >= 0.67) & (freqs <= 3.33)
    if np.any(mask_hr):
        peak_freq = freqs[mask_hr][np.argmax(psd[mask_hr])]
        feats["rppg_hr_bpm"] = float(peak_freq * 60.0)
    else:
        feats["rppg_hr_bpm"] = float("nan")

    # --- Spectral features ---
    total_power = float(np.sum(psd)) + 1e-12
    psd_norm = psd / total_power
    feats["rppg_spec_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    feats["rppg_dom_freq_hz"] = float(freqs[np.argmax(psd)])

    # --- Peak-to-peak HRV proxy ---
    peaks, _ = find_peaks(rppg_signal, distance=int(fps * 0.4))
    if len(peaks) >= 2:
        pp_intervals_ms = np.diff(peaks) / fps * 1000.0
        feats["rppg_rmssd_ms"] = float(np.sqrt(np.mean(np.diff(pp_intervals_ms) ** 2)))
        feats["rppg_sdnn_ms"] = float(np.std(pp_intervals_ms))
        feats["rppg_mean_pp_ms"] = float(np.mean(pp_intervals_ms))
    else:
        feats["rppg_rmssd_ms"] = float("nan")
        feats["rppg_sdnn_ms"] = float("nan")
        feats["rppg_mean_pp_ms"] = float("nan")

    # --- Waveform statistics ---
    feats["rppg_rms"] = float(np.sqrt(np.mean(rppg_signal ** 2)))
    feats["rppg_kurtosis"] = float(kurtosis(rppg_signal))
    feats["rppg_skewness"] = float(skew(rppg_signal))
    feats["rppg_zero_cross"] = float(np.mean(np.diff(np.sign(rppg_signal)) != 0))

    # --- SpO2 ---
    if roi_trace is not None:
        feats["rppg_spo2_est"] = float(estimate_spo2_from_rois(roi_trace))
    else:
        feats["rppg_spo2_est"] = float("nan")

    names = list(feats.keys())
    return np.array(list(feats.values()), dtype=np.float32), names
