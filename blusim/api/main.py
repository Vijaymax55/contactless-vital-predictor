"""FastAPI REST API for the BluSim vital prediction service.

Endpoints:
  POST /ingest/piezo          — receive a piezo stream chunk
  POST /ingest/video-frame    — receive a camera frame as base64 JPEG
  GET  /vitals/latest         — return current predicted vitals
  GET  /vitals/history        — return time-range vital history
  GET  /health                — system health check

The RealtimePipeline runs as a background asyncio task started on startup.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from blusim.inference.pipeline import RealtimePipeline, VitalSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class PiezoChunk(BaseModel):
    """Incoming piezo sensor data chunk.

    Attributes:
        samples: Flat list of ADC values interleaved channel-by-channel.
                 Layout: [ch0_s0, ch1_s0, ch2_s0, ch3_s0, ch0_s1, ...].
        n_channels: Number of channels.
        sample_rate: Sampling rate (Hz).
        timestamp_iso: ISO-8601 UTC timestamp for the first sample.
    """

    samples: list[float]
    n_channels: int = 4
    sample_rate: float = 490.0
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VideoFrame(BaseModel):
    """Incoming video frame encoded as base64 JPEG.

    Attributes:
        frame_b64: Base64-encoded JPEG bytes.
        timestamp_iso: ISO-8601 UTC timestamp for this frame.
        fps: Camera frame rate.
    """

    frame_b64: str
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fps: float = 30.0


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    n_history: int
    uptime_s: float


class IngestResponse(BaseModel):
    accepted: bool
    buffer_size: int
    message: str


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(pipeline: RealtimePipeline) -> FastAPI:
    """Construct the FastAPI application with an injected inference pipeline.

    Args:
        pipeline: Initialised :class:`RealtimePipeline` instance.

    Returns:
        Configured FastAPI application.
    """
    _start_time = time.monotonic()
    _pipeline = pipeline

    app = FastAPI(
        title="BluSim Vital Prediction API",
        version="2.0.0",
        description="Contactless health vital prediction from piezo and camera sensors.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Start inference loop on startup
    @app.on_event("startup")
    async def _start_pipeline() -> None:
        asyncio.create_task(_pipeline.run_async(poll_interval=1.0))
        logger.info("BluSim inference pipeline started")

    # ------------------------------------------------------------------
    # Ingestion endpoints
    # ------------------------------------------------------------------

    @app.post("/ingest/piezo", response_model=IngestResponse, tags=["Ingestion"])
    async def ingest_piezo(chunk: PiezoChunk) -> IngestResponse:
        """Receive a chunk of piezo sensor data and push to the inference buffer.

        The samples should be flat-interleaved by channel:
        ``[ch0_s0, ch1_s0, …, chN_s0, ch0_s1, …]``.
        """
        try:
            arr = np.array(chunk.samples, dtype=np.float32)
            n_samples = len(arr) // chunk.n_channels
            if n_samples == 0:
                raise ValueError("Empty samples array")
            arr = arr[: n_samples * chunk.n_channels].reshape(n_samples, chunk.n_channels)

            # Generate timestamps for each sample
            t0 = np.datetime64(chunk.timestamp_iso)
            dt_ns = int(1e9 / chunk.sample_rate)
            timestamps = t0 + np.arange(n_samples, dtype="timedelta64[ns]") * dt_ns

            _pipeline.push_piezo(arr, timestamps)
            return IngestResponse(accepted=True, buffer_size=_pipeline._piezo_buf.size,
                                  message=f"Received {n_samples} samples on {chunk.n_channels} channels")
        except Exception as exc:
            logger.exception("Failed to ingest piezo chunk")
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/ingest/video-frame", response_model=IngestResponse, tags=["Ingestion"])
    async def ingest_video_frame(frame: VideoFrame) -> IngestResponse:
        """Receive a base64-encoded JPEG frame and push rPPG features to buffer.

        This endpoint decodes the frame, extracts forehead + cheek ROI mean
        RGB values, and pushes a 9-D feature vector to the rPPG buffer.
        Privacy: no video is stored; only the 9-D colour summary is kept.
        """
        try:
            import cv2  # type: ignore

            raw = base64.b64decode(frame.frame_b64)
            nparr = np.frombuffer(raw, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not decode image")

            # Simple ROI: top-third (forehead), mid-left, mid-right (cheeks)
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            forehead = rgb[h//10: h//4, w//4: 3*w//4].mean(axis=(0, 1))
            left_cheek = rgb[h//3: h//2, w//8: w//3].mean(axis=(0, 1))
            right_cheek = rgb[h//3: h//2, 2*w//3: 7*w//8].mean(axis=(0, 1))

            roi_vec = np.concatenate([forehead, left_cheek, right_cheek])  # (9,)
            ts = np.array([np.datetime64(frame.timestamp_iso)])
            _pipeline.push_rppg_frame(roi_vec, ts)

            return IngestResponse(accepted=True, buffer_size=_pipeline._rppg_buf.size,
                                  message="Frame processed; 9-D ROI vector stored (no video retained)")
        except Exception as exc:
            logger.exception("Failed to ingest video frame")
            raise HTTPException(status_code=422, detail=str(exc))

    # ------------------------------------------------------------------
    # Vital retrieval endpoints
    # ------------------------------------------------------------------

    @app.get("/vitals/latest", tags=["Vitals"])
    async def get_latest_vitals() -> dict:
        """Return the most-recently predicted vital signs."""
        snapshot = _pipeline.get_latest()
        if snapshot is None:
            raise HTTPException(status_code=503, detail="No predictions available yet — collecting data")
        return snapshot.to_dict()

    @app.get("/vitals/history", tags=["Vitals"])
    async def get_vital_history(
        n: int = Query(default=60, ge=1, le=1000, description="Number of recent snapshots to return"),
        start: Optional[str] = Query(default=None, description="ISO-8601 start time filter"),
        end: Optional[str] = Query(default=None, description="ISO-8601 end time filter"),
    ) -> list[dict]:
        """Return a list of past vital snapshots.

        Args:
            n: Maximum number of snapshots to return.
            start: Optional ISO-8601 start timestamp (inclusive).
            end: Optional ISO-8601 end timestamp (inclusive).
        """
        history = _pipeline.get_history(n=n)
        if start or end:
            history = [
                s for s in history
                if (not start or s.timestamp >= start) and (not end or s.timestamp <= end)
            ]
        return [s.to_dict() for s in history]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """Return system health status."""
        return HealthResponse(
            status="ok",
            pipeline_ready=True,
            n_history=len(_pipeline._history),
            uptime_s=time.monotonic() - _start_time,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from blusim.models.vital_predictor import VitalPredictor

    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = VitalPredictor.from_config(cfg)
    pipeline = RealtimePipeline(model, device="cpu")

    api_cfg = cfg.get("pipeline", {}).get("api", {})
    app = create_app(pipeline)
    uvicorn.run(app, host=api_cfg.get("host", "0.0.0.0"), port=api_cfg.get("port", 8000))
