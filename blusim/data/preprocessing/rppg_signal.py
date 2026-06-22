"""Remote Photoplethysmography (rPPG) signal processing pipeline.

Implements:
- CHROM algorithm (De Haan & Jeanne, IEEE TBME 2013)
- POS algorithm (Wang et al., IEEE TBME 2017)
- MediaPipe-based face mesh ROI tracking
- Lucas-Kanade optical flow motion compensation
- SpO2 estimation from AC/DC channel ratios
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch

logger = logging.getLogger(__name__)

# Try to import mediapipe gracefully — it's optional at import time
try:
    import mediapipe as mp  # type: ignore
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logger.warning("mediapipe not installed — face detection disabled; rPPG will not run")


# ---------------------------------------------------------------------------
# ROI Extraction
# ---------------------------------------------------------------------------

# MediaPipe landmark indices for anatomical ROIs
_FOREHEAD_LANDMARKS = [10, 108, 107, 66, 105, 104, 103, 67, 109, 10]
_LEFT_CHEEK_LANDMARKS = [205, 187, 123, 50, 101, 100, 142, 203, 205]
_RIGHT_CHEEK_LANDMARKS = [425, 411, 352, 280, 330, 329, 371, 423, 425]


@dataclass
class ROITrace:
    """Time series of mean RGB values from a single facial ROI.

    Attributes:
        name: ROI identifier (e.g. ``"forehead"``).
        r: Red channel mean per frame.
        g: Green channel mean per frame.
        b: Blue channel mean per frame.
        timestamps_s: Timestamps in seconds.
    """

    name: str
    r: np.ndarray
    g: np.ndarray
    b: np.ndarray
    timestamps_s: np.ndarray


class FaceMeshROITracker:
    """Tracks facial ROIs across frames using MediaPipe Face Mesh.

    Args:
        min_detection_confidence: Minimum detection confidence threshold.
        min_tracking_confidence: Minimum tracking confidence threshold.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError("mediapipe is required for FaceMeshROITracker")

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._rois = {
            "forehead": _FOREHEAD_LANDMARKS,
            "left_cheek": _LEFT_CHEEK_LANDMARKS,
            "right_cheek": _RIGHT_CHEEK_LANDMARKS,
        }
        # Per-ROI running accumulators
        self._traces: dict[str, list[np.ndarray]] = {k: [] for k in self._rois}
        self._timestamps: list[float] = []

    def process_frame(self, frame: np.ndarray, timestamp_s: float) -> bool:
        """Extract mean RGB from each ROI in one video frame.

        Args:
            frame: BGR frame from OpenCV (H×W×3 uint8).
            timestamp_s: Frame timestamp in seconds.

        Returns:
            ``True`` if face was detected; ``False`` otherwise.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False

        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark

        for roi_name, indices in self._rois.items():
            pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            mean_rgb = cv2.mean(rgb, mask=mask)[:3]  # (R, G, B)
            self._traces[roi_name].append(np.array(mean_rgb))

        self._timestamps.append(timestamp_s)
        return True

    def get_roi_traces(self) -> list[ROITrace]:
        """Return accumulated ROI traces as :class:`ROITrace` objects."""
        ts = np.array(self._timestamps)
        traces = []
        for name, values in self._traces.items():
            if not values:
                continue
            arr = np.array(values)  # (n_frames, 3)
            traces.append(ROITrace(name=name, r=arr[:, 0], g=arr[:, 1], b=arr[:, 2], timestamps_s=ts))
        return traces

    def reset(self) -> None:
        """Clear accumulated frame data."""
        for k in self._traces:
            self._traces[k].clear()
        self._timestamps.clear()


def extract_roi_traces_from_video(
    video_path: str,
    min_detection_confidence: float = 0.7,
) -> list[ROITrace]:
    """Convenience function: run ROI tracker on an entire video file.

    Args:
        video_path: Path to video file.
        min_detection_confidence: MediaPipe confidence threshold.

    Returns:
        List of :class:`ROITrace` for each detected ROI.
    """
    tracker = FaceMeshROITracker(min_detection_confidence=min_detection_confidence)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        tracker.process_frame(frame, ts)
        frame_idx += 1

    cap.release()
    return tracker.get_roi_traces()


# ---------------------------------------------------------------------------
# Motion compensation via Lucas-Kanade optical flow
# ---------------------------------------------------------------------------


def motion_compensate_trace(
    trace: ROITrace,
    frames: list[np.ndarray],
    roi_landmarks: Optional[np.ndarray] = None,
) -> ROITrace:
    """Subtract motion-induced luminance changes from an ROI trace.

    Estimates per-frame displacement via Lucas-Kanade sparse optical flow and
    regresses it out from each colour channel.

    Args:
        trace: Original ROI trace to compensate.
        frames: List of BGR frames corresponding to the trace.
        roi_landmarks: Optional reference points for optical flow.

    Returns:
        Motion-compensated :class:`ROITrace`.
    """
    if len(frames) < 2:
        return trace

    # Estimate motion magnitude for each frame
    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    h, w = gray_prev.shape
    pts = np.array([[w // 2, h // 4]], dtype=np.float32).reshape(-1, 1, 2)  # forehead centroid

    motion_x = np.zeros(len(frames))
    motion_y = np.zeros(len(frames))

    for i, frame in enumerate(frames[1:], 1):
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts, None)
        if next_pts is not None and status[0][0]:
            motion_x[i] = next_pts[0][0][0] - pts[0][0][0]
            motion_y[i] = next_pts[0][0][1] - pts[0][0][1]
            pts = next_pts
        gray_prev = gray_curr

    # Regress motion out of each colour channel
    motion = np.stack([motion_x, motion_y], axis=1)
    motion_norm = (motion - motion.mean(axis=0)) / (motion.std(axis=0) + 1e-8)

    def regress_out(channel: np.ndarray) -> np.ndarray:
        channel_norm = (channel - channel.mean()) / (channel.std() + 1e-8)
        coeff = np.linalg.lstsq(motion_norm, channel_norm, rcond=None)[0]
        residual = channel_norm - motion_norm @ coeff
        return residual * channel.std() + channel.mean()

    return ROITrace(
        name=trace.name,
        r=regress_out(trace.r),
        g=regress_out(trace.g),
        b=regress_out(trace.b),
        timestamps_s=trace.timestamps_s,
    )


# ---------------------------------------------------------------------------
# rPPG algorithms
# ---------------------------------------------------------------------------


def chrom_rppg(trace: ROITrace, fps: float = 30.0, lowcut: float = 0.7, highcut: float = 4.0) -> np.ndarray:
    """CHROM algorithm for rPPG (De Haan & Jeanne, 2013).

    Constructs two chrominance signals, combines them with a bandpass filter to
    suppress specular reflection and motion noise.

    Args:
        trace: ROI colour trace.
        fps: Video frame rate.
        lowcut: Bandpass lower cutoff in Hz (≈ 42 BPM).
        highcut: Bandpass upper cutoff in Hz (≈ 240 BPM).

    Returns:
        Normalised rPPG waveform array.
    """
    r = trace.r.astype(np.float64)
    g = trace.g.astype(np.float64)
    b = trace.b.astype(np.float64)

    # Normalise each channel by its mean
    r_n = r / (np.mean(r) + 1e-8)
    g_n = g / (np.mean(g) + 1e-8)
    b_n = b / (np.mean(b) + 1e-8)

    # CHROM chrominance channels
    xs = 3 * r_n - 2 * g_n
    ys = 1.5 * r_n + g_n - 1.5 * b_n

    # Combine with frequency-domain alpha
    xs_std = np.std(xs) + 1e-8
    ys_std = np.std(ys) + 1e-8
    alpha = xs_std / ys_std
    rppg_raw = xs - alpha * ys

    # Bandpass filter
    nyq = fps / 2.0
    b_coef, a_coef = butter(2, [lowcut / nyq, min(highcut / nyq, 0.99)], btype="band")
    rppg = filtfilt(b_coef, a_coef, rppg_raw)
    return (rppg - np.mean(rppg)) / (np.std(rppg) + 1e-8)


def pos_rppg(trace: ROITrace, fps: float = 30.0, lowcut: float = 0.7, highcut: float = 4.0) -> np.ndarray:
    """POS algorithm for rPPG (Wang et al., 2017).

    Projects skin colours onto a plane orthogonal to the skin-tone vector in
    normalised RGB space.

    Args:
        trace: ROI colour trace.
        fps: Video frame rate.
        lowcut: Bandpass lower cutoff (Hz).
        highcut: Bandpass upper cutoff (Hz).

    Returns:
        Normalised rPPG waveform.
    """
    r = trace.r.astype(np.float64)
    g = trace.g.astype(np.float64)
    b = trace.b.astype(np.float64)

    # Normalise by mean (divide by temporal mean)
    denom = (r + g + b) / 3.0 + 1e-8
    r_n = r / denom
    g_n = g / denom
    b_n = b / denom

    # POS projection matrix
    p1 = g_n - b_n
    p2 = -2 * r_n + g_n + b_n
    std1 = np.std(p1) + 1e-8
    std2 = np.std(p2) + 1e-8
    rppg_raw = p1 + (std1 / std2) * p2

    # Bandpass filter
    nyq = fps / 2.0
    b_coef, a_coef = butter(2, [lowcut / nyq, min(highcut / nyq, 0.99)], btype="band")
    rppg = filtfilt(b_coef, a_coef, rppg_raw)
    return (rppg - np.mean(rppg)) / (np.std(rppg) + 1e-8)


def extract_hr_from_rppg(rppg: np.ndarray, fps: float = 30.0, nperseg: int = 256) -> float:
    """Estimate heart rate from rPPG waveform via Welch PSD.

    Args:
        rppg: Bandpass-filtered rPPG signal.
        fps: Frame rate.
        nperseg: Welch segment length.

    Returns:
        Estimated HR in beats per minute.
    """
    nperseg = min(nperseg, len(rppg))
    freqs, psd = welch(rppg, fs=fps, nperseg=nperseg)
    # Restrict to physiological HR range (40–200 BPM = 0.67–3.33 Hz)
    mask = (freqs >= 0.67) & (freqs <= 3.33)
    if not np.any(mask):
        return float("nan")
    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return peak_freq * 60.0


# ---------------------------------------------------------------------------
# SpO2 estimation
# ---------------------------------------------------------------------------


def estimate_spo2_from_rois(
    red_trace: ROITrace,
    *,
    r_channel_weight: float = 0.8,
    ir_proxy_weight: float = 0.2,
) -> float:
    """Estimate SpO2 from the AC/DC ratio in red and green channels.

    The red/green channel in a standard camera acts as a proxy for the
    red/infrared ratio in a pulse oximeter. This method is approximate;
    accuracy depends on lighting conditions and camera sensor spectral response.

    The calibration curve ``SpO2 = A - B × R`` uses empirical constants
    ``A=110``, ``B=25`` (typical values from the literature for RGB cameras).

    Args:
        red_trace: ROI trace with R and G channel data.
        r_channel_weight: Weight for red AC/DC term.
        ir_proxy_weight: Weight for green (IR-proxy) AC/DC term.

    Returns:
        Estimated SpO2 percentage (clamped to [70, 100]).
    """
    def ac_dc_ratio(channel: np.ndarray) -> float:
        dc = np.mean(channel) + 1e-8
        ac = np.std(channel)
        return ac / dc

    r_ratio = ac_dc_ratio(red_trace.r)
    g_ratio = ac_dc_ratio(red_trace.g)

    # Perfusion index ratio (R = AC_red/DC_red ÷ AC_ir/DC_ir)
    ratio_of_ratios = (r_ratio + 1e-8) / (g_ratio + 1e-8)

    # Empirical SpO2 calibration (Beer-Lambert approximation)
    A, B = 110.0, 25.0
    spo2 = A - B * ratio_of_ratios
    return float(np.clip(spo2, 70.0, 100.0))
