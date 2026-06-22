"""Real-time inference pipeline with sliding window and circular buffer.

The pipeline:
1. Accepts streaming piezo and video frame data via asyncio queues.
2. Maintains a circular buffer per modality.
3. Every ``step_size`` seconds, extracts a ``window_size``-second window,
   pre-processes it, runs the model, and emits a :class:`VitalSnapshot`.
4. Results are stored in an in-memory deque and optionally written to InfluxDB.

Designed for concurrent writer (sensor acquisition) + reader (inference) threads.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

from blusim.data.ingestion.circular_buffer import CircularBuffer
from blusim.data.preprocessing.piezo_signal import process_piezo_window
from blusim.models.vital_predictor import VitalPredictor, VitalPredictions

logger = logging.getLogger(__name__)

# Maximum number of vital snapshots kept in memory
_HISTORY_MAXLEN = 1000


@dataclass
class VitalSnapshot:
    """One set of predicted vitals at a point in time.

    Attributes:
        timestamp: UTC datetime when this prediction was made.
        hr_bpm: Heart rate (beats per minute).
        hr_bpm_std: Prediction uncertainty (1σ).
        rmssd_ms: HRV RMSSD in milliseconds.
        sdnn_ms: HRV SDNN in milliseconds.
        spo2_pct: SpO2 percentage.
        resp_rate_bpm: Respiration rate (breaths per minute).
        sleep_stage: Predicted sleep stage string.
        sleep_probabilities: Dict of stage name → probability.
        stress_level: Predicted stress level string.
        piezo_sqi: Mean signal quality index from piezo.
        modality_sources: Which modalities contributed to this prediction.
    """

    timestamp: str
    hr_bpm: float
    hr_bpm_std: float
    rmssd_ms: float
    sdnn_ms: float
    spo2_pct: float
    resp_rate_bpm: float
    sleep_stage: str
    sleep_probabilities: dict[str, float]
    stress_level: str
    piezo_sqi: float = 1.0
    modality_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RealtimePipeline:
    """Sliding-window real-time inference pipeline.

    Args:
        model: Trained :class:`VitalPredictor` (in eval mode).
        fs_piezo: Piezo sampling rate (Hz).
        fps_rppg: Camera frame rate.
        window_size: Analysis window length (seconds).
        step_size: Inference step interval (seconds).
        device: PyTorch device.
        n_piezo_channels: Number of piezo channels.
    """

    def __init__(
        self,
        model: VitalPredictor,
        fs_piezo: float = 490.0,
        fps_rppg: float = 30.0,
        window_size: float = 30.0,
        step_size: float = 5.0,
        device: str = "cpu",
        n_piezo_channels: int = 4,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.fs_piezo = fs_piezo
        self.fps_rppg = fps_rppg
        self.window_size = window_size
        self.step_size = step_size
        self.n_piezo_channels = n_piezo_channels

        # Circular buffers: capacity = 2 × window_size (headroom)
        capacity_piezo = int(fs_piezo * window_size * 2)
        capacity_rppg = int(fps_rppg * window_size * 2)
        self._piezo_buf = CircularBuffer(capacity_piezo, n_channels=n_piezo_channels)
        self._rppg_buf = CircularBuffer(capacity_rppg, n_channels=9)  # 3 ROIs × RGB

        self._history: deque[VitalSnapshot] = deque(maxlen=_HISTORY_MAXLEN)
        self._last_inference_time: float = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def push_piezo(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Append piezo samples to the circular buffer.

        Args:
            samples: ``(n, n_channels)`` float32 array.
            timestamps: ``(n,)`` datetime64 timestamps.
        """
        self._piezo_buf.push(samples, timestamps)

    def push_rppg_frame(self, rgb_roi: np.ndarray, timestamp: np.ndarray) -> None:
        """Append one rPPG frame (9-dimensional: 3 ROIs × RGB) to buffer.

        Args:
            rgb_roi: ``(9,)`` array with concatenated ROI means.
            timestamp: ``(1,)`` datetime64 scalar.
        """
        self._rppg_buf.push(rgb_roi.reshape(1, -1), timestamp.reshape(1))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def maybe_run_inference(self) -> Optional[VitalSnapshot]:
        """Run inference if ``step_size`` seconds have elapsed since the last run.

        Returns:
            :class:`VitalSnapshot` if inference ran, ``None`` otherwise.
        """
        now = time.monotonic()
        if now - self._last_inference_time < self.step_size:
            return None
        self._last_inference_time = now
        return self._run_inference()

    def _run_inference(self) -> Optional[VitalSnapshot]:
        """Extract window, preprocess, run model, return snapshot."""
        n_piezo = int(self.fs_piezo * self.window_size)
        piezo_win, _ = self._piezo_buf.read_latest(n_piezo)
        if len(piezo_win) < n_piezo // 2:
            logger.debug("Insufficient piezo data (%d / %d samples)", len(piezo_win), n_piezo)
            return None

        # Pad with zeros if slightly short
        if len(piezo_win) < n_piezo:
            pad = np.zeros((n_piezo - len(piezo_win), self.n_piezo_channels), dtype=np.float32)
            piezo_win = np.concatenate([pad, piezo_win], axis=0)

        # Signal conditioning
        result = process_piezo_window(piezo_win, fs=self.fs_piezo)
        sqi = float(np.mean(result.sqi))

        # Prepare tensor: (1, n_channels, n_samples)
        piezo_tensor = torch.from_numpy(piezo_win.T[np.newaxis]).float().to(self.device)

        # rPPG (optional)
        n_rppg = int(self.fps_rppg * self.window_size)
        rppg_win, _ = self._rppg_buf.read_latest(n_rppg)
        rppg_tensor = None
        modalities = ["piezo"]
        if len(rppg_win) >= n_rppg // 2:
            if len(rppg_win) < n_rppg:
                pad_r = np.zeros((n_rppg - len(rppg_win), 9), dtype=np.float32)
                rppg_win = np.concatenate([pad_r, rppg_win], axis=0)
            rppg_tensor = torch.from_numpy(rppg_win.T[np.newaxis]).float().to(self.device)
            modalities.append("rppg")

        with torch.no_grad():
            preds: VitalPredictions = self.model(piezo=piezo_tensor, rppg=rppg_tensor)

        pred_dict = preds.to_dict()

        snapshot = VitalSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hr_bpm=pred_dict.get("hr_bpm", float("nan")),
            hr_bpm_std=pred_dict.get("hr_bpm_std", float("nan")),
            rmssd_ms=pred_dict.get("rmssd_ms", float("nan")),
            sdnn_ms=pred_dict.get("sdnn_ms", float("nan")),
            spo2_pct=pred_dict.get("spo2_pct", float("nan")),
            resp_rate_bpm=pred_dict.get("resp_rate_bpm", float("nan")),
            sleep_stage=pred_dict.get("sleep_stage", "Unknown"),
            sleep_probabilities={
                k.replace("sleep_prob_", ""): v
                for k, v in pred_dict.items()
                if k.startswith("sleep_prob_")
            },
            stress_level=pred_dict.get("stress_level", "Unknown"),
            piezo_sqi=sqi,
            modality_sources=modalities,
        )

        self._history.append(snapshot)
        logger.info("Inference: HR=%.1f±%.1f BPM | Sleep=%s | SQI=%.2f",
                    snapshot.hr_bpm, snapshot.hr_bpm_std, snapshot.sleep_stage, sqi)
        return snapshot

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[VitalSnapshot]:
        """Return the most-recent vital snapshot."""
        return self._history[-1] if self._history else None

    def get_history(self, n: int = 100) -> list[VitalSnapshot]:
        """Return the last ``n`` snapshots in chronological order."""
        items = list(self._history)
        return items[-n:]

    # ------------------------------------------------------------------
    # Async continuous loop
    # ------------------------------------------------------------------

    async def run_async(self, poll_interval: float = 1.0) -> None:
        """Run inference loop asynchronously in the background.

        Args:
            poll_interval: How often to check if inference should run (seconds).
        """
        self._running = True
        logger.info("Real-time pipeline started (window=%.0fs, step=%.0fs)", self.window_size, self.step_size)
        while self._running:
            self.maybe_run_inference()
            await asyncio.sleep(poll_interval)

    def stop(self) -> None:
        """Signal the async loop to stop."""
        self._running = False
