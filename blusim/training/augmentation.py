"""Data augmentation transforms for piezo and rPPG time-series windows.

All transforms operate on numpy arrays and return numpy arrays.
They are designed to be applied during training via a transform pipeline.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Piezo augmentations
# ---------------------------------------------------------------------------


def add_gaussian_noise(signal: np.ndarray, snr_db_range: tuple[float, float] = (15.0, 40.0)) -> np.ndarray:
    """Add white Gaussian noise at a random SNR drawn from ``snr_db_range``.

    Args:
        signal: Array of shape ``(n_samples, n_channels)`` or ``(n_samples,)``.
        snr_db_range: Min/max SNR in decibels.

    Returns:
        Noisy signal of same shape.
    """
    snr_db = np.random.uniform(*snr_db_range)
    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def magnitude_scale(signal: np.ndarray, scale_range: tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Scale signal amplitude by a random factor.

    Args:
        signal: Input signal.
        scale_range: Min/max scale factor.

    Returns:
        Scaled signal.
    """
    scale = np.random.uniform(*scale_range)
    return signal * scale


def time_warp(signal: np.ndarray, max_warp_frac: float = 0.1) -> np.ndarray:
    """Apply smooth time-axis warping (interpolation-based).

    Simulates variation in recording speed or heartbeat timing shift.

    Args:
        signal: ``(n_samples,)`` or ``(n_samples, n_channels)`` array.
        max_warp_frac: Maximum fractional displacement at warp knots.

    Returns:
        Warped signal of the same length.
    """
    n = signal.shape[0]
    # Create 5 control points, displace them randomly
    n_knots = 5
    knots = np.linspace(0, n - 1, n_knots)
    knot_disp = np.random.uniform(-max_warp_frac * n, max_warp_frac * n, n_knots)
    knot_disp[0] = 0.0
    knot_disp[-1] = 0.0
    warped_knots = np.clip(knots + knot_disp, 0, n - 1)

    src_x = np.linspace(0, n - 1, n)
    # Interpolate warp map from knots
    warp_map = np.interp(src_x, knots, warped_knots)
    warp_map = np.clip(warp_map, 0, n - 1)

    if signal.ndim == 1:
        warped = np.interp(warp_map, src_x, signal)
    else:
        warped = np.stack([np.interp(warp_map, src_x, signal[:, c]) for c in range(signal.shape[1])], axis=1)
    return warped.astype(signal.dtype)


def channel_dropout(signal: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
    """Randomly zero out entire channels to simulate sensor failure.

    Args:
        signal: ``(n_samples, n_channels)`` array.
        drop_prob: Probability of dropping each channel.

    Returns:
        Signal with some channels zeroed.
    """
    if signal.ndim == 1:
        return signal
    out = signal.copy()
    for c in range(signal.shape[1]):
        if np.random.rand() < drop_prob:
            out[:, c] = 0.0
    return out


def dc_shift(signal: np.ndarray, shift_range: tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
    """Add a random DC offset to simulate baseline drift.

    Args:
        signal: Input signal.
        shift_range: Range for additive offset.

    Returns:
        Shifted signal.
    """
    shift = np.random.uniform(*shift_range) * np.std(signal)
    return signal + shift


# ---------------------------------------------------------------------------
# Camera/rPPG augmentations
# ---------------------------------------------------------------------------


def brightness_jitter(rgb_trace: np.ndarray, jitter_range: tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Scale all RGB channels by a random brightness factor.

    Args:
        rgb_trace: Array of shape ``(n_frames, 3)`` with R/G/B values.
        jitter_range: Multiplicative jitter range.

    Returns:
        Jittered RGB trace.
    """
    factor = np.random.uniform(*jitter_range)
    return np.clip(rgb_trace * factor, 0, 255)


def contrast_jitter(rgb_trace: np.ndarray, jitter_range: tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Apply random contrast scaling around each channel mean.

    Args:
        rgb_trace: ``(n_frames, 3)`` array.
        jitter_range: Contrast scale range.

    Returns:
        Contrast-adjusted trace.
    """
    factor = np.random.uniform(*jitter_range)
    mean = rgb_trace.mean(axis=0, keepdims=True)
    return np.clip((rgb_trace - mean) * factor + mean, 0, 255)


def add_motion_blur(signal: np.ndarray, max_blur: int = 5) -> np.ndarray:
    """Simulate motion blur by applying a short moving average.

    Args:
        signal: rPPG or RGB trace.
        max_blur: Maximum blur kernel length.

    Returns:
        Blurred signal.
    """
    k = np.random.randint(1, max_blur + 1)
    if k <= 1:
        return signal
    kernel = np.ones(k) / k
    if signal.ndim == 1:
        return np.convolve(signal, kernel, mode="same")
    return np.stack([np.convolve(signal[:, c], kernel, mode="same") for c in range(signal.shape[1])], axis=1)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class PiezoAugmentation:
    """Stochastic augmentation pipeline for piezo windows.

    Each augmentation is applied independently with probability ``p``.

    Args:
        p: Probability of applying each individual transform.
        noise_snr_range: SNR range for Gaussian noise.
        scale_range: Amplitude scale range.
        time_warp_frac: Maximum time-warp displacement fraction.
        channel_drop_prob: Per-channel drop probability.
    """

    def __init__(
        self,
        p: float = 0.5,
        noise_snr_range: tuple[float, float] = (15.0, 40.0),
        scale_range: tuple[float, float] = (0.85, 1.15),
        time_warp_frac: float = 0.08,
        channel_drop_prob: float = 0.1,
    ) -> None:
        self.p = p
        self.noise_snr_range = noise_snr_range
        self.scale_range = scale_range
        self.time_warp_frac = time_warp_frac
        self.channel_drop_prob = channel_drop_prob

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            signal = add_gaussian_noise(signal, self.noise_snr_range)
        if np.random.rand() < self.p:
            signal = magnitude_scale(signal, self.scale_range)
        if np.random.rand() < self.p:
            signal = time_warp(signal, self.time_warp_frac)
        if np.random.rand() < self.p:
            signal = channel_dropout(signal, self.channel_drop_prob)
        if np.random.rand() < self.p:
            signal = dc_shift(signal)
        return signal


class RPPGAugmentation:
    """Stochastic augmentation pipeline for rPPG windows."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, trace: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            trace = brightness_jitter(trace)
        if np.random.rand() < self.p:
            trace = contrast_jitter(trace)
        if np.random.rand() < self.p:
            trace = add_motion_blur(trace)
        return trace
