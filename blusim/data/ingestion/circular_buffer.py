"""Lock-free circular ring buffer for streaming sensor data.

Each modality (piezo, rPPG) has its own typed buffer. The buffer supports
concurrent writer and reader threads via a simple lock so it can be used
with Python asyncio + a thread-pool executor.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class CircularBuffer:
    """Fixed-capacity circular buffer for multi-channel time-series data.

    Args:
        capacity: Number of samples to store (e.g. ``sample_rate * buffer_seconds``).
        n_channels: Number of signal channels.
        dtype: NumPy dtype for the sample array.
    """

    def __init__(self, capacity: int, n_channels: int = 1, dtype: np.dtype = np.float32) -> None:
        self._capacity = capacity
        self._n_channels = n_channels
        self._buf = np.zeros((capacity, n_channels), dtype=dtype)
        self._timestamps = np.zeros(capacity, dtype="datetime64[ns]")
        self._head = 0          # next write position
        self._size = 0          # current number of valid samples
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Append samples to the buffer, overwriting oldest if full.

        Args:
            samples: Array of shape ``(n, n_channels)`` or ``(n,)`` for single channel.
            timestamps: Array of shape ``(n,)`` with numpy datetime64 values.
        """
        samples = np.atleast_2d(samples)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        n = len(samples)
        if n > self._capacity:
            # Keep only the most recent `capacity` samples
            samples = samples[-self._capacity:]
            timestamps = timestamps[-self._capacity:]
            n = self._capacity

        with self._lock:
            end = (self._head + n) % self._capacity
            if end > self._head:
                self._buf[self._head:end] = samples
                self._timestamps[self._head:end] = timestamps
            else:
                split = self._capacity - self._head
                self._buf[self._head:] = samples[:split]
                self._buf[:end] = samples[split:]
                self._timestamps[self._head:] = timestamps[:split]
                self._timestamps[:end] = timestamps[split:]
            self._head = end
            self._size = min(self._size + n, self._capacity)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``n`` most-recent samples without consuming them.

        Args:
            n: Number of samples to read (≤ ``capacity``).

        Returns:
            Tuple of ``(samples, timestamps)`` both in chronological order.
            If fewer than ``n`` samples are available, returns all available.
        """
        with self._lock:
            available = min(n, self._size)
            if available == 0:
                return (
                    np.zeros((0, self._n_channels), dtype=self._buf.dtype),
                    np.array([], dtype="datetime64[ns]"),
                )
            start = (self._head - available) % self._capacity
            if start + available <= self._capacity:
                s = self._buf[start:start + available].copy()
                t = self._timestamps[start:start + available].copy()
            else:
                split = self._capacity - start
                s = np.concatenate([self._buf[start:], self._buf[:available - split]])
                t = np.concatenate([self._timestamps[start:], self._timestamps[:available - split]])
        return s, t

    def read_window(self, window_seconds: float, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the most-recent ``window_seconds`` of data.

        Args:
            window_seconds: Length of the window in seconds.
            sample_rate: Samples per second of the stream.

        Returns:
            ``(samples, timestamps)`` of shape ``(n_samples, n_channels)``
            and ``(n_samples,)`` respectively.
        """
        n = int(window_seconds * sample_rate)
        return self.read_latest(n)

    @property
    def size(self) -> int:
        """Current number of valid samples in the buffer."""
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        """Reset the buffer to empty."""
        with self._lock:
            self._head = 0
            self._size = 0
