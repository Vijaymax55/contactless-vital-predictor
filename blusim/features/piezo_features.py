"""Feature extraction from processed piezoelectric BCG/respiration signals.

Feature taxonomy:
- Time-domain: IJ amplitude, JK slope, peak-to-peak intervals, RMS, moments
- Frequency-domain: Welch PSD, dominant frequency, spectral entropy, LF/HF
- Time-frequency: STFT statistics, CWT energy per scale
- Nonlinear: Sample Entropy, Approximate Entropy, DFA, Poincaré SD1/SD2
- HRV: RMSSD, SDNN, pNN50, mean RR

All features return a flat 1-D NumPy array and a corresponding list of feature
names to ensure full traceability.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pywt
from scipy.signal import stft, welch
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_log(x: float) -> float:
    return np.log(x + 1e-10)


def _normalize_segment(s: np.ndarray) -> np.ndarray:
    std = s.std()
    return (s - s.mean()) / (std if std > 1e-8 else 1.0)


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------


def time_domain_features(
    signal: np.ndarray,
    peaks: np.ndarray,
    rr_intervals_ms: np.ndarray,
    fs: float = 490.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute time-domain BCG/HRV features.

    Args:
        signal: BCG-band filtered signal.
        peaks: J-peak sample indices.
        rr_intervals_ms: Peak-to-peak intervals in milliseconds.
        fs: Sampling rate.

    Returns:
        ``(features, names)`` tuple.
    """
    feats: dict[str, float] = {}

    # --- RR interval / HRV features ---
    if len(rr_intervals_ms) >= 2:
        feats["rr_mean_ms"] = float(np.mean(rr_intervals_ms))
        feats["rr_std_ms"] = float(np.std(rr_intervals_ms))
        feats["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2)))
        feats["sdnn_ms"] = float(np.std(rr_intervals_ms))
        pnn50 = np.mean(np.abs(np.diff(rr_intervals_ms)) > 50.0)
        feats["pnn50"] = float(pnn50)
        feats["hr_bpm"] = float(60000.0 / (np.mean(rr_intervals_ms) + 1e-8))
    else:
        for k in ["rr_mean_ms", "rr_std_ms", "rmssd_ms", "sdnn_ms", "pnn50", "hr_bpm"]:
            feats[k] = float("nan")

    # --- Peak morphology (IJ amplitude, JK slope) ---
    ij_amplitudes: list[float] = []
    jk_slopes: list[float] = []
    for pk in peaks:
        half_win = int(0.1 * fs)  # ±100 ms window around J-peak
        left = max(0, pk - half_win)
        right = min(len(signal) - 1, pk + half_win)
        seg = signal[left:right]
        if len(seg) > 4:
            ij_amplitudes.append(float(np.max(seg) - np.min(seg)))
            # JK slope: slope from J-peak to 50 ms after
            jk_end = min(len(signal) - 1, pk + int(0.05 * fs))
            if jk_end > pk:
                jk_slopes.append(float((signal[jk_end] - signal[pk]) / ((jk_end - pk) / fs + 1e-8)))

    feats["ij_amplitude_mean"] = float(np.mean(ij_amplitudes)) if ij_amplitudes else float("nan")
    feats["ij_amplitude_std"] = float(np.std(ij_amplitudes)) if ij_amplitudes else float("nan")
    feats["jk_slope_mean"] = float(np.mean(jk_slopes)) if jk_slopes else float("nan")

    # --- Waveform statistics ---
    feats["rms"] = float(np.sqrt(np.mean(signal ** 2)))
    feats["kurtosis"] = float(kurtosis(signal))
    feats["skewness"] = float(skew(signal))
    feats["zero_cross_rate"] = float(np.mean(np.diff(np.sign(signal)) != 0))

    # --- Hjorth parameters ---
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    activity = float(np.var(signal))
    mobility = float(np.sqrt(np.var(d1) / (np.var(signal) + 1e-12)))
    complexity_num = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)))
    complexity = complexity_num / (mobility + 1e-8)
    feats["hjorth_activity"] = activity
    feats["hjorth_mobility"] = mobility
    feats["hjorth_complexity"] = complexity

    names = list(feats.keys())
    return np.array(list(feats.values()), dtype=np.float32), names


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------

_LF_BAND = (0.04, 0.15)   # Hz — low-frequency HRV band
_HF_BAND = (0.15, 0.40)   # Hz — high-frequency HRV band (parasympathetic)


def frequency_domain_features(
    signal: np.ndarray,
    fs: float = 490.0,
    nperseg: int = 512,
) -> tuple[np.ndarray, list[str]]:
    """Compute frequency-domain PSD features.

    Args:
        signal: BCG or respiration signal.
        fs: Sampling rate (Hz).
        nperseg: Welch segment length.

    Returns:
        ``(features, names)`` tuple.
    """
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    total_power = float(np.sum(psd)) + 1e-12

    # Dominant frequency
    dom_freq = float(freqs[np.argmax(psd)])

    # Spectral entropy
    psd_norm = psd / total_power
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    # Band powers
    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs <= hi)
        return float(np.sum(psd[mask]) / total_power)

    lf = band_power(*_LF_BAND)
    hf = band_power(*_HF_BAND)
    lf_hf_ratio = lf / (hf + 1e-8)

    bcg_power = band_power(1.0, 20.0)
    resp_power = band_power(0.1, 0.5)

    feats = {
        "dom_freq_hz": dom_freq,
        "spec_entropy": spec_entropy,
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": lf_hf_ratio,
        "bcg_band_power": bcg_power,
        "resp_band_power": resp_power,
        "total_power_log": float(np.log(total_power + 1e-10)),
    }
    names = list(feats.keys())
    return np.array(list(feats.values()), dtype=np.float32), names


# ---------------------------------------------------------------------------
# Time-frequency features
# ---------------------------------------------------------------------------


def stft_features(
    signal: np.ndarray,
    fs: float = 490.0,
    nperseg: int = 128,
    noverlap: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Extract summary statistics from the STFT spectrogram.

    Args:
        signal: 1-D signal.
        fs: Sampling rate.
        nperseg: STFT window length.
        noverlap: STFT overlap.

    Returns:
        ``(features, names)`` tuple.
    """
    _, _, Zxx = stft(signal, fs=fs, nperseg=min(nperseg, len(signal)), noverlap=noverlap)
    power = np.abs(Zxx) ** 2

    feats = {
        "stft_mean": float(np.mean(power)),
        "stft_std": float(np.std(power)),
        "stft_max": float(np.max(power)),
        "stft_energy": float(np.sum(power)),
        "stft_spectral_flux": float(np.mean(np.diff(power, axis=1) ** 2)),
    }
    names = list(feats.keys())
    return np.array(list(feats.values()), dtype=np.float32), names


def cwt_features(
    signal: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    scales: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract energy features from the Continuous Wavelet Transform scalogram.

    Args:
        signal: 1-D signal.
        wavelet: CWT wavelet name (complex Morlet for good time-frequency).
        scales: CWT scales; defaults to 1–64.

    Returns:
        ``(features, names)`` tuple.
    """
    if scales is None:
        scales = np.arange(1, 33)

    try:
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
    except Exception as exc:
        logger.warning("CWT failed (%s); returning zeros", exc)
        dummy = np.zeros(6, dtype=np.float32)
        return dummy, [f"cwt_{i}" for i in range(6)]

    power = np.abs(coeffs) ** 2  # (n_scales, n_time)
    energy_per_scale = power.mean(axis=1)  # (n_scales,)

    feats = {
        "cwt_total_energy": float(energy_per_scale.sum()),
        "cwt_peak_scale": float(scales[np.argmax(energy_per_scale)]),
        "cwt_energy_entropy": float(-np.sum((energy_per_scale / (energy_per_scale.sum() + 1e-12)) *
                                            np.log(energy_per_scale / (energy_per_scale.sum() + 1e-12) + 1e-12))),
        "cwt_low_scale_energy": float(energy_per_scale[:len(scales)//3].mean()),
        "cwt_mid_scale_energy": float(energy_per_scale[len(scales)//3:2*len(scales)//3].mean()),
        "cwt_high_scale_energy": float(energy_per_scale[2*len(scales)//3:].mean()),
    }
    names = list(feats.keys())
    return np.array(list(feats.values()), dtype=np.float32), names


# ---------------------------------------------------------------------------
# Nonlinear features
# ---------------------------------------------------------------------------


def sample_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute Sample Entropy (SampEn) of a signal.

    Args:
        signal: 1-D array (at least m+1 samples).
        m: Template length.
        r_factor: Tolerance as a fraction of signal standard deviation.

    Returns:
        Sample entropy value (or NaN on failure).
    """
    n = len(signal)
    if n < m + 2:
        return float("nan")

    r = r_factor * np.std(signal)
    sig = np.array(signal, dtype=np.float64)

    def _count_matches(template_len: int) -> int:
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(sig[i:i+template_len] - sig[j:j+template_len])) < r:
                    count += 1
        return count

    # Limit computation to first 500 samples for speed
    sig = sig[:500] if n > 500 else sig
    n = len(sig)

    a = _count_matches(m + 1)
    b = _count_matches(m)
    if b == 0:
        return float("nan")
    return float(-np.log((a + 1e-10) / (b + 1e-10)))


def approximate_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute Approximate Entropy (ApEn) of a signal.

    Args:
        signal: 1-D array.
        m: Template length.
        r_factor: Tolerance as fraction of std.

    Returns:
        ApEn value.
    """
    n = len(signal)
    if n < m + 1:
        return float("nan")

    r = r_factor * np.std(signal)
    sig = np.array(signal[:500] if n > 500 else signal, dtype=np.float64)
    n = len(sig)

    def phi(template_len: int) -> float:
        count = 0
        for i in range(n - template_len + 1):
            matches = 0
            for j in range(n - template_len + 1):
                if np.max(np.abs(sig[i:i+template_len] - sig[j:j+template_len])) < r:
                    matches += 1
            count += np.log(matches / (n - template_len + 1) + 1e-10)
        return count / (n - template_len + 1)

    return float(phi(m) - phi(m + 1))


def dfa_scaling_exponent(signal: np.ndarray, min_box: int = 4, max_box: int = 64) -> float:
    """Detrended Fluctuation Analysis (DFA) scaling exponent alpha.

    alpha ≈ 0.5 → white noise; alpha > 0.5 → long-range correlations.

    Args:
        signal: 1-D signal (at least ``max_box`` samples).
        min_box: Minimum box size.
        max_box: Maximum box size.

    Returns:
        DFA scaling exponent, or NaN if computation fails.
    """
    n = len(signal)
    if n < max_box:
        return float("nan")

    # Cumulative sum (profile)
    y = np.cumsum(signal - np.mean(signal))

    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(min(max_box, n // 4)), num=20).astype(int))
    fluctuations = []

    for box in box_sizes:
        n_boxes = n // box
        if n_boxes < 2:
            continue
        rms_list = []
        for k in range(n_boxes):
            seg = y[k * box: (k + 1) * box]
            x = np.arange(len(seg))
            poly = np.polyfit(x, seg, 1)
            trend = np.polyval(poly, x)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations.append(np.mean(rms_list))

    if len(fluctuations) < 2:
        return float("nan")

    log_boxes = np.log(box_sizes[:len(fluctuations)])
    log_fluct = np.log(np.array(fluctuations) + 1e-12)
    alpha = float(np.polyfit(log_boxes, log_fluct, 1)[0])
    return alpha


def poincare_features(rr_intervals_ms: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute Poincaré plot SD1 and SD2 features.

    Args:
        rr_intervals_ms: RR intervals in milliseconds.

    Returns:
        ``(features, names)`` tuple.
    """
    if len(rr_intervals_ms) < 2:
        return np.array([float("nan"), float("nan")], dtype=np.float32), ["sd1_ms", "sd2_ms"]

    rr1 = rr_intervals_ms[:-1]
    rr2 = rr_intervals_ms[1:]
    sd1 = float(np.std((rr2 - rr1) / np.sqrt(2)))
    sd2 = float(np.std((rr2 + rr1) / np.sqrt(2)))
    return np.array([sd1, sd2], dtype=np.float32), ["sd1_ms", "sd2_ms"]


# ---------------------------------------------------------------------------
# Complete feature vector for one window
# ---------------------------------------------------------------------------


def extract_all_features(
    signal: np.ndarray,
    peaks: np.ndarray,
    rr_intervals_ms: np.ndarray,
    fs: float = 490.0,
    include_nonlinear: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute the full feature vector for one BCG window.

    Args:
        signal: BCG-band filtered signal (1-D).
        peaks: J-peak indices.
        rr_intervals_ms: RR intervals in ms.
        fs: Sampling rate.
        include_nonlinear: Whether to include slow nonlinear features.

    Returns:
        ``(feature_vector, feature_names)`` where vector is float32 1-D.
    """
    all_feats = []
    all_names = []

    f, n = time_domain_features(signal, peaks, rr_intervals_ms, fs)
    all_feats.append(f); all_names.extend(n)

    f, n = frequency_domain_features(signal, fs)
    all_feats.append(f); all_names.extend(n)

    f, n = stft_features(signal, fs)
    all_feats.append(f); all_names.extend(n)

    f, n = cwt_features(signal)
    all_feats.append(f); all_names.extend(n)

    f, n = poincare_features(rr_intervals_ms)
    all_feats.append(f); all_names.extend(n)

    if include_nonlinear and len(signal) >= 50:
        samp_en = sample_entropy(signal)
        approx_en = approximate_entropy(signal)
        dfa = dfa_scaling_exponent(signal)
        all_feats.append(np.array([samp_en, approx_en, dfa], dtype=np.float32))
        all_names.extend(["sample_entropy", "approx_entropy", "dfa_alpha"])

    feature_vector = np.concatenate(all_feats)
    return feature_vector, all_names
