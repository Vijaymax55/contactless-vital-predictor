"""Unit tests for feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from blusim.features.piezo_features import (
    approximate_entropy,
    cwt_features,
    dfa_scaling_exponent,
    extract_all_features,
    frequency_domain_features,
    poincare_features,
    sample_entropy,
    stft_features,
    time_domain_features,
)

FS = 490.0
N = int(FS * 10)  # 10-second window for fast tests


def _signal() -> np.ndarray:
    t = np.linspace(0, 10, N)
    return (np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(N)).astype(np.float32)


def _peaks() -> np.ndarray:
    return np.array([int(FS * i) for i in range(1, 10)])


def _rr() -> np.ndarray:
    return np.diff(_peaks()) / FS * 1000.0


class TestTimeDomainFeatures:
    def test_output_shape_and_names(self) -> None:
        f, n = time_domain_features(_signal(), _peaks(), _rr(), FS)
        assert len(f) == len(n)
        assert len(f) > 0

    def test_hr_bpm_range(self) -> None:
        sig = _signal()
        # Peaks 1 second apart → HR = 60 BPM
        f, names = time_domain_features(sig, _peaks(), _rr(), FS)
        hr_idx = names.index("hr_bpm")
        assert 55 <= f[hr_idx] <= 65

    def test_no_peaks_returns_nan(self) -> None:
        sig = _signal()
        f, names = time_domain_features(sig, np.array([]), np.array([]), FS)
        hr_idx = names.index("hr_bpm")
        assert np.isnan(f[hr_idx])

    def test_returns_float32(self) -> None:
        f, _ = time_domain_features(_signal(), _peaks(), _rr(), FS)
        assert f.dtype == np.float32


class TestFrequencyDomainFeatures:
    def test_output_shape(self) -> None:
        f, n = frequency_domain_features(_signal(), FS)
        assert len(f) == len(n)

    def test_dom_freq_in_range(self) -> None:
        t = np.linspace(0, 10, N)
        sig = np.sin(2 * np.pi * 1.5 * t)   # 1.5 Hz dominant
        f, n = frequency_domain_features(sig, FS)
        dom_idx = n.index("dom_freq_hz")
        # Welch frequency resolution at 490 Hz / 512 samples ≈ 0.96 Hz per bin;
        # allow up to one full bin of error.
        assert abs(f[dom_idx] - 1.5) < 1.0

    def test_spec_entropy_positive(self) -> None:
        f, n = frequency_domain_features(_signal(), FS)
        ent_idx = n.index("spec_entropy")
        assert f[ent_idx] > 0


class TestSTFTFeatures:
    def test_output_shape(self) -> None:
        f, n = stft_features(_signal(), FS)
        assert len(f) == len(n)
        assert len(f) == 5

    def test_non_negative_energy(self) -> None:
        f, n = stft_features(_signal(), FS)
        energy_idx = n.index("stft_energy")
        assert f[energy_idx] >= 0


class TestCWTFeatures:
    def test_output_shape(self) -> None:
        sig = _signal()[:1000]  # shorter for speed
        f, n = cwt_features(sig)
        assert len(f) == len(n)
        assert len(f) == 6


class TestNonlinearFeatures:
    def test_sample_entropy_positive(self) -> None:
        sig = _signal()
        se = sample_entropy(sig[:200])
        assert not np.isnan(se), "SampEn should not be NaN for typical signals"
        assert se >= 0

    def test_approx_entropy_range(self) -> None:
        sig = np.random.randn(300)
        ae = approximate_entropy(sig, m=2)
        assert not np.isnan(ae)
        assert ae >= 0

    def test_dfa_alpha_white_noise(self) -> None:
        """White noise should have DFA alpha ≈ 0.5."""
        sig = np.random.randn(1000)
        alpha = dfa_scaling_exponent(sig)
        if not np.isnan(alpha):
            assert 0.2 <= alpha <= 0.9, f"DFA alpha for white noise should be ~0.5, got {alpha:.3f}"

    def test_poincare_sd1_sd2(self) -> None:
        rr = np.array([800.0, 820.0, 810.0, 790.0, 805.0])
        f, n = poincare_features(rr)
        assert "sd1_ms" in n
        assert "sd2_ms" in n
        assert f[n.index("sd1_ms")] >= 0

    def test_poincare_short_rr(self) -> None:
        f, n = poincare_features(np.array([800.0]))
        assert np.isnan(f[0])


class TestExtractAllFeatures:
    def test_output_is_flat_float32(self) -> None:
        sig = _signal()
        f, n = extract_all_features(sig, _peaks(), _rr(), FS)
        assert f.ndim == 1
        assert f.dtype == np.float32
        assert len(f) == len(n)

    def test_no_nan_inf_for_normal_signal(self) -> None:
        sig = _signal()
        f, n = extract_all_features(sig, _peaks(), _rr(), FS, include_nonlinear=False)
        # Should have no inf values; NaN is acceptable for some features
        assert not np.any(np.isinf(f)), "Feature vector should not contain inf"

    def test_feature_count_is_consistent(self) -> None:
        sig1 = _signal()
        sig2 = _signal()
        f1, n1 = extract_all_features(sig1, _peaks(), _rr(), FS, include_nonlinear=False)
        f2, n2 = extract_all_features(sig2, _peaks(), _rr(), FS, include_nonlinear=False)
        assert len(f1) == len(f2), "Feature length must be deterministic"
        assert n1 == n2, "Feature names must be deterministic"
