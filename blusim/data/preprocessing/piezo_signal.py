"""Piezoelectric sensor signal conditioning pipeline.

Implements:
- Adaptive bandpass filtering (BCG, respiration, movement bands)
- Motion artifact removal (wavelet soft-thresholding + LMS adaptive filter)
- BCG peak detection (Pan-Tompkins-inspired with adaptive threshold)
- Signal Quality Index (SQI) assessment
- Segment rejection for low-quality windows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pywt
from scipy.signal import (
    butter,
    find_peaks,
    lfilter,
    sosfilt,
    sosfiltfilt,
    butter,
    welch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def butter_bandpass_sos(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Design a zero-phase Butterworth bandpass filter in SOS form.

    Args:
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (applied twice for zero-phase → effective 2×order).

    Returns:
        Second-order sections array suitable for :func:`scipy.signal.sosfiltfilt`.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to (0, 1) exclusive to avoid numerical errors at the boundary
    low = max(low, 1e-4)
    high = min(high, 1.0 - 1e-4)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def apply_bandpass(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply zero-phase bandpass filter to a 1-D signal.

    Args:
        signal: 1-D array of samples.
        lowcut: Lower cutoff in Hz.
        highcut: Upper cutoff in Hz.
        fs: Sampling rate in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered signal of same shape.
    """
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    return sosfiltfilt(sos, signal)


# ---------------------------------------------------------------------------
# Multi-band decomposition
# ---------------------------------------------------------------------------


@dataclass
class PiezoBands:
    """Frequency-decomposed piezo signal components.

    Attributes:
        bcg: BCG band (1–20 Hz) — heartbeat mechanical vibration.
        resp: Respiration band (0.1–0.5 Hz) — chest/diaphragm movement.
        movement: Micro-movement band (0.5–5 Hz) — body position shifts.
        raw: Zero-mean de-trended raw signal.
    """

    bcg: np.ndarray
    resp: np.ndarray
    movement: np.ndarray
    raw: np.ndarray


def decompose_bands(
    signal: np.ndarray,
    fs: float = 490.0,
    bcg_band: tuple[float, float] = (1.0, 20.0),
    resp_band: tuple[float, float] = (0.1, 0.5),
    movement_band: tuple[float, float] = (0.5, 5.0),
    filter_order: int = 4,
) -> PiezoBands:
    """Decompose a piezo channel into physiologically meaningful bands.

    Args:
        signal: 1-D raw piezo signal.
        fs: Sampling rate (Hz).
        bcg_band: (low, high) Hz for BCG band.
        resp_band: (low, high) Hz for respiration band.
        movement_band: (low, high) Hz for movement band.
        filter_order: Butterworth filter order.

    Returns:
        :class:`PiezoBands` containing filtered components.
    """
    raw = signal - np.mean(signal)          # remove DC offset
    bcg = apply_bandpass(raw, *bcg_band, fs, filter_order)
    resp = apply_bandpass(raw, *resp_band, fs, filter_order)
    movement = apply_bandpass(raw, *movement_band, fs, filter_order)
    return PiezoBands(bcg=bcg, resp=resp, movement=movement, raw=raw)


# ---------------------------------------------------------------------------
# Motion artifact removal
# ---------------------------------------------------------------------------


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """Remove motion artifacts via wavelet soft-thresholding (Donoho & Johnstone).

    Uses universal threshold ``σ × √(2 ln N)`` where σ is estimated from the
    finest-scale coefficients via the MAD estimator.

    Args:
        signal: 1-D signal array.
        wavelet: PyWavelets wavelet name.
        level: Decomposition depth.
        threshold_mode: ``"soft"`` or ``"hard"``.

    Returns:
        Denoised signal of the same shape.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Estimate noise std from the finest-level detail coefficients
    detail_finest = coeffs[-1]
    sigma = np.median(np.abs(detail_finest)) / 0.6745  # MAD estimator
    n = len(signal)
    threshold = sigma * np.sqrt(2 * np.log(n)) if n > 1 else sigma

    denoised_coeffs = [coeffs[0]]  # keep approximation unchanged
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)]


class LMSAdaptiveFilter:
    """Least Mean Squares (LMS) adaptive filter for reference-signal cancellation.

    When a reference motion signal (e.g. from an accelerometer channel) is
    available, LMS cancels correlated interference from the target BCG channel.

    Args:
        n_taps: Filter length (number of taps).
        mu: Step-size / learning rate (default 0.01).
    """

    def __init__(self, n_taps: int = 32, mu: float = 0.01) -> None:
        self._n = n_taps
        self._mu = mu
        self._w = np.zeros(n_taps)

    def filter(self, desired: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Run the LMS filter online, returning the error (cleaned) signal.

        Args:
            desired: Target signal contaminated with noise (BCG channel).
            reference: Reference noise signal correlated with interference.

        Returns:
            Error signal ``e[n] = desired[n] - y[n]`` which approximates a
            clean BCG signal.
        """
        n = len(desired)
        self._w = np.zeros(self._n)
        output = np.zeros(n)
        ref_padded = np.concatenate([np.zeros(self._n - 1), reference])

        for i in range(n):
            x = ref_padded[i: i + self._n][::-1]   # tap-delay line
            y = np.dot(self._w, x)
            e = desired[i] - y
            self._w += 2 * self._mu * e * x
            output[i] = e
        return output


def remove_motion_artifacts(
    bcg: np.ndarray,
    reference: Optional[np.ndarray] = None,
    wavelet: str = "db4",
    wavelet_level: int = 5,
    lms_taps: int = 32,
    lms_mu: float = 0.005,
) -> np.ndarray:
    """Two-stage motion artifact removal: wavelet → LMS.

    Stage 1: Wavelet soft-thresholding removes broadband noise.
    Stage 2 (optional): LMS removes correlated reference interference.

    Args:
        bcg: BCG-band signal.
        reference: Reference channel correlated with motion (e.g., piezo
                   channel furthest from the body). If ``None``, only wavelet
                   denoising is applied.
        wavelet: Wavelet basis for stage 1.
        wavelet_level: Decomposition depth for stage 1.
        lms_taps: Number of LMS taps for stage 2.
        lms_mu: LMS learning rate.

    Returns:
        Cleaned BCG signal.
    """
    cleaned = wavelet_denoise(bcg, wavelet=wavelet, level=wavelet_level)
    if reference is not None:
        lms = LMSAdaptiveFilter(n_taps=lms_taps, mu=lms_mu)
        cleaned = lms.filter(cleaned, reference)
    return cleaned


# ---------------------------------------------------------------------------
# BCG peak detection (Pan-Tompkins-adapted)
# ---------------------------------------------------------------------------


def detect_bcg_peaks(
    bcg: np.ndarray,
    fs: float = 490.0,
    min_hr_bpm: float = 30.0,
    max_hr_bpm: float = 200.0,
    adaptive_window_seconds: float = 5.0,
) -> np.ndarray:
    """Detect BCG J-peaks using a Pan-Tompkins-inspired algorithm.

    Steps:
    1. Square the signal to emphasise peaks and make all values positive.
    2. Integrate with a moving window (150 ms).
    3. Apply adaptive threshold updated every ``adaptive_window_seconds``.
    4. Enforce physiological refractory period based on max HR.

    Args:
        bcg: BCG-band filtered signal (1–20 Hz).
        fs: Sampling rate.
        min_hr_bpm: Minimum physiologically plausible HR.
        max_hr_bpm: Maximum physiologically plausible HR.
        adaptive_window_seconds: Window over which threshold is updated.

    Returns:
        1-D array of peak sample indices.
    """
    # Step 1: Square
    squared = bcg ** 2

    # Step 2: Moving-average integration (150 ms window)
    win_len = max(1, int(0.15 * fs))
    kernel = np.ones(win_len) / win_len
    integrated = np.convolve(squared, kernel, mode="same")

    # Step 3: Adaptive threshold
    window_samples = int(adaptive_window_seconds * fs)
    threshold = np.zeros_like(integrated)
    for i in range(0, len(integrated), window_samples):
        seg = integrated[i: i + window_samples]
        thr = 0.3 * np.max(seg) + 0.7 * np.mean(seg)
        threshold[i: i + window_samples] = thr

    # Step 4: Find peaks above threshold with refractory period
    min_dist = int(60.0 / max_hr_bpm * fs)  # minimum samples between peaks
    max_dist = int(60.0 / min_hr_bpm * fs)  # maximum samples between peaks

    above = (integrated > threshold).astype(int)
    peaks_raw, _ = find_peaks(integrated, height=threshold, distance=min_dist)

    return peaks_raw


def compute_rr_intervals(peaks: np.ndarray, fs: float = 490.0) -> np.ndarray:
    """Compute RR-intervals (peak-to-peak intervals) in milliseconds.

    Args:
        peaks: Sample indices of detected J-peaks.
        fs: Sampling rate (Hz).

    Returns:
        Array of RR intervals in milliseconds (length = len(peaks) - 1).
    """
    if len(peaks) < 2:
        return np.array([])
    diff_samples = np.diff(peaks)
    return diff_samples / fs * 1000.0  # convert to ms


# ---------------------------------------------------------------------------
# Signal Quality Index
# ---------------------------------------------------------------------------


def compute_sqi(
    signal: np.ndarray,
    fs: float = 490.0,
    bcg_band: tuple[float, float] = (1.0, 20.0),
    nperseg: int = 512,
) -> float:
    """Compute a Signal Quality Index (SQI) for a piezo window.

    SQI is defined as the fraction of PSD power in the BCG frequency band
    (1–20 Hz) relative to total power. Higher ≈ better signal quality.

    Args:
        signal: Raw or BCG-filtered 1-D signal.
        fs: Sampling rate (Hz).
        bcg_band: (low, high) Hz defining the BCG band.
        nperseg: Welch segment length.

    Returns:
        SQI value in [0, 1].
    """
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd) + 1e-12
    band_mask = (freqs >= bcg_band[0]) & (freqs <= bcg_band[1])
    band_power = np.sum(psd[band_mask])
    return float(band_power / total_power)


# ---------------------------------------------------------------------------
# Full processing pipeline for a single window
# ---------------------------------------------------------------------------


@dataclass
class PiezoWindowResult:
    """Output of processing one piezo window.

    Attributes:
        bands: Frequency-decomposed signal components per channel.
        peaks: BCG peak indices per channel.
        rr_intervals: RR intervals in ms per channel.
        sqi: Signal quality index per channel.
        is_valid: True if mean SQI ≥ threshold.
    """

    bands: list[PiezoBands]
    peaks: list[np.ndarray]
    rr_intervals: list[np.ndarray]
    sqi: list[float]
    is_valid: bool


def process_piezo_window(
    window: np.ndarray,
    fs: float = 490.0,
    sqi_threshold: float = 0.4,
    bcg_band: tuple[float, float] = (1.0, 20.0),
    resp_band: tuple[float, float] = (0.1, 0.5),
    movement_band: tuple[float, float] = (0.5, 5.0),
    filter_order: int = 4,
    wavelet: str = "db4",
    wavelet_level: int = 5,
) -> PiezoWindowResult:
    """Full conditioning pipeline for a multi-channel piezo window.

    Args:
        window: Array of shape ``(n_samples, n_channels)``.
        fs: Sampling rate (Hz).
        sqi_threshold: Minimum SQI for a window to be considered valid.
        bcg_band: BCG bandpass limits.
        resp_band: Respiration bandpass limits.
        movement_band: Movement bandpass limits.
        filter_order: Butterworth filter order.
        wavelet: Wavelet name for denoising.
        wavelet_level: Decomposition level for wavelet denoising.

    Returns:
        :class:`PiezoWindowResult` with per-channel processed data.
    """
    n_channels = window.shape[1] if window.ndim > 1 else 1
    if window.ndim == 1:
        window = window[:, np.newaxis]

    # Use the channel with the lowest variance as the motion reference
    channel_vars = np.var(window, axis=0)
    ref_ch_idx = int(np.argmin(channel_vars))

    bands_list: list[PiezoBands] = []
    peaks_list: list[np.ndarray] = []
    rr_list: list[np.ndarray] = []
    sqi_list: list[float] = []

    for ch in range(n_channels):
        sig = window[:, ch].astype(np.float64)
        reference = window[:, ref_ch_idx].astype(np.float64) if ref_ch_idx != ch else None

        # Band decomposition
        b = decompose_bands(sig, fs, bcg_band, resp_band, movement_band, filter_order)

        # Motion artifact removal on BCG band
        clean_bcg = remove_motion_artifacts(
            b.bcg, reference=reference, wavelet=wavelet, wavelet_level=wavelet_level
        )
        b = PiezoBands(bcg=clean_bcg, resp=b.resp, movement=b.movement, raw=b.raw)

        # SQI
        sqi = compute_sqi(sig, fs, bcg_band)

        # Peak detection on cleaned BCG
        peaks = detect_bcg_peaks(clean_bcg, fs)
        rr = compute_rr_intervals(peaks, fs)

        bands_list.append(b)
        peaks_list.append(peaks)
        rr_list.append(rr)
        sqi_list.append(sqi)

    is_valid = bool(np.mean(sqi_list) >= sqi_threshold)
    return PiezoWindowResult(
        bands=bands_list,
        peaks=peaks_list,
        rr_intervals=rr_list,
        sqi=sqi_list,
        is_valid=is_valid,
    )
