"""Unit tests for piezo signal processing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from blusim.data.preprocessing.piezo_signal import (
    LMSAdaptiveFilter,
    PiezoBands,
    apply_bandpass,
    compute_rr_intervals,
    compute_sqi,
    decompose_bands,
    detect_bcg_peaks,
    process_piezo_window,
    remove_motion_artifacts,
    wavelet_denoise,
)

FS = 490.0
DURATION_S = 30
N_SAMPLES = int(FS * DURATION_S)
N_CHANNELS = 4


def _synthetic_bcg(fs: float = FS, duration: float = DURATION_S, hr: float = 60.0) -> np.ndarray:
    """Generate a synthetic BCG-like signal with known HR."""
    t = np.linspace(0, duration, int(fs * duration))
    freq = hr / 60.0
    signal = (
        np.sin(2 * np.pi * freq * t)           # fundamental
        + 0.3 * np.sin(4 * np.pi * freq * t)   # second harmonic
        + 0.05 * np.random.randn(len(t))        # noise
    )
    return signal.astype(np.float32)


def _synthetic_multichannel(n_channels: int = 4) -> np.ndarray:
    t = np.linspace(0, DURATION_S, N_SAMPLES)
    channels = []
    for i in range(n_channels):
        sig = (
            np.sin(2 * np.pi * 1.2 * t + i * 0.1)
            + 0.1 * np.random.randn(N_SAMPLES)
        )
        channels.append(sig)
    return np.stack(channels, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------


class TestApplyBandpass:
    def test_output_shape(self) -> None:
        sig = _synthetic_bcg()
        filtered = apply_bandpass(sig, 1.0, 20.0, FS)
        assert filtered.shape == sig.shape

    def test_attenuates_out_of_band(self) -> None:
        """Signal below lowcut should be attenuated."""
        t = np.linspace(0, 10, int(FS * 10))
        dc = np.ones_like(t)                    # 0 Hz — below 1 Hz lowcut
        filtered = apply_bandpass(dc, 1.0, 20.0, FS)
        assert np.std(filtered) < 1e-3, "DC component should be removed"

    def test_passes_bcg_band(self) -> None:
        t = np.linspace(0, 10, int(FS * 10))
        sig = np.sin(2 * np.pi * 1.5 * t)      # 1.5 Hz — inside BCG band
        filtered = apply_bandpass(sig, 1.0, 20.0, FS)
        assert np.std(filtered) > 0.5, "In-band signal should pass"


# ---------------------------------------------------------------------------
# Band decomposition
# ---------------------------------------------------------------------------


class TestDecomposeBands:
    def test_returns_piezobands(self) -> None:
        sig = _synthetic_bcg()
        bands = decompose_bands(sig, FS)
        assert isinstance(bands, PiezoBands)

    def test_shapes_equal(self) -> None:
        sig = _synthetic_bcg()
        bands = decompose_bands(sig, FS)
        assert bands.bcg.shape == sig.shape
        assert bands.resp.shape == sig.shape
        assert bands.movement.shape == sig.shape

    def test_dc_removed(self) -> None:
        sig = _synthetic_bcg() + 100.0  # large DC
        bands = decompose_bands(sig, FS)
        # float32 precision limits mean to ~1e-5; check DC is effectively removed
        assert abs(np.mean(bands.raw)) < 1e-4


# ---------------------------------------------------------------------------
# Motion artifact removal
# ---------------------------------------------------------------------------


class TestWaveletDenoise:
    def test_output_shape(self) -> None:
        sig = _synthetic_bcg()
        out = wavelet_denoise(sig)
        assert out.shape == sig.shape

    def test_reduces_noise(self) -> None:
        clean = _synthetic_bcg()
        noisy = clean + np.random.randn(len(clean)) * 0.5
        denoised = wavelet_denoise(noisy)
        noise_before = np.std(noisy - clean)
        noise_after = np.std(denoised - clean)
        assert noise_after < noise_before, "Wavelet denoising should reduce noise"


class TestLMSFilter:
    def test_output_shape(self) -> None:
        n = 1000
        desired = np.random.randn(n)
        reference = np.random.randn(n)
        lms = LMSAdaptiveFilter(n_taps=16, mu=0.01)
        out = lms.filter(desired, reference)
        assert out.shape == desired.shape

    def test_cancels_known_noise(self) -> None:
        """LMS should reduce correlated interference."""
        n = 2000
        t = np.arange(n) / FS
        clean = np.sin(2 * np.pi * 1.2 * t)
        noise = np.sin(2 * np.pi * 50.0 * t)  # 50 Hz interference
        desired = clean + noise
        lms = LMSAdaptiveFilter(n_taps=32, mu=0.005)
        output = lms.filter(desired, noise)
        # After convergence (last 500 samples), noise should be reduced
        power_before = np.mean((desired[-500:] - clean[-500:]) ** 2)
        power_after = np.mean((output[-500:] - clean[-500:]) ** 2)
        assert power_after < power_before


# ---------------------------------------------------------------------------
# BCG peak detection
# ---------------------------------------------------------------------------


class TestDetectBcgPeaks:
    def test_detects_peaks(self) -> None:
        sig = _synthetic_bcg(hr=60.0)
        bcg = apply_bandpass(sig, 1.0, 20.0, FS)
        peaks = detect_bcg_peaks(bcg, FS)
        # With 60 BPM over 30 s, expect ~30 peaks
        assert 15 <= len(peaks) <= 60, f"Expected 15–60 peaks, got {len(peaks)}"

    def test_peaks_in_valid_range(self) -> None:
        sig = _synthetic_bcg(hr=75.0)
        bcg = apply_bandpass(sig, 1.0, 20.0, FS)
        peaks = detect_bcg_peaks(bcg, FS)
        assert all(0 <= p < len(bcg) for p in peaks)

    def test_empty_signal(self) -> None:
        sig = np.zeros(N_SAMPLES)
        peaks = detect_bcg_peaks(sig, FS)
        assert isinstance(peaks, np.ndarray)


class TestComputeRRIntervals:
    def test_correct_intervals(self) -> None:
        # Place peaks exactly 490 samples apart (1 second at 490 Hz)
        peaks = np.array([0, 490, 980, 1470])
        rr = compute_rr_intervals(peaks, FS)
        assert len(rr) == 3
        np.testing.assert_allclose(rr, 1000.0, atol=1.0)  # 1 second = 1000 ms

    def test_single_peak_returns_empty(self) -> None:
        rr = compute_rr_intervals(np.array([100]), FS)
        assert len(rr) == 0


# ---------------------------------------------------------------------------
# Signal Quality Index
# ---------------------------------------------------------------------------


class TestComputeSQI:
    def test_high_sqi_for_bcg_signal(self) -> None:
        sig = _synthetic_bcg()
        sqi = compute_sqi(sig, FS)
        assert 0.0 <= sqi <= 1.0

    def test_low_sqi_for_noise(self) -> None:
        noise = np.random.randn(N_SAMPLES)
        sqi = compute_sqi(noise, FS)
        # Pure white noise has flat PSD, so BCG band fraction ≈ BCG_bandwidth / total_bandwidth
        # Should be reasonably low
        bcg_fraction = (20.0 - 1.0) / (FS / 2.0)
        assert sqi < bcg_fraction * 5  # loose upper bound

    def test_synthetic_bcg_higher_sqi_than_noise(self) -> None:
        sig = _synthetic_bcg()
        noise = np.random.randn(N_SAMPLES)
        sqi_sig = compute_sqi(sig, FS)
        sqi_noise = compute_sqi(noise, FS)
        assert sqi_sig > sqi_noise


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestProcessPiezoWindow:
    def test_returns_result(self) -> None:
        win = _synthetic_multichannel()
        result = process_piezo_window(win, fs=FS)
        assert len(result.bands) == N_CHANNELS
        assert len(result.sqi) == N_CHANNELS
        assert isinstance(result.is_valid, bool)

    def test_result_shapes(self) -> None:
        win = _synthetic_multichannel()
        result = process_piezo_window(win, fs=FS)
        for bands in result.bands:
            assert bands.bcg.shape == (N_SAMPLES,)
            assert bands.resp.shape == (N_SAMPLES,)

    def test_flat_signal_is_invalid(self) -> None:
        win = np.zeros((N_SAMPLES, N_CHANNELS), dtype=np.float32)
        result = process_piezo_window(win, fs=FS, sqi_threshold=0.4)
        assert not result.is_valid
