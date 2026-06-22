"""ONNX model export and INT8 quantization for edge deployment.

Supports export to ONNX opset 17 and optional INT8 post-training quantization
via onnxruntime, suitable for Raspberry Pi 5 / Jetson Nano.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path | str,
    piezo_shape: tuple[int, int, int] = (1, 4, 14700),
    rppg_shape: tuple[int, int, int] = (1, 9, 900),
    opset_version: int = 17,
) -> Path:
    """Export the VitalPredictor to ONNX format.

    Args:
        model: Trained model (will be set to eval mode).
        output_path: Destination ``.onnx`` file path.
        piezo_shape: Dummy input shape ``(batch, n_channels, n_samples)``
                     for 30-second window at 490 Hz.
        rppg_shape: Dummy input shape ``(batch, n_rppg_channels, n_frames)``
                    for 30-second window at 30 FPS.
        opset_version: ONNX opset to target.

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_piezo = torch.zeros(*piezo_shape, dtype=torch.float32)
    dummy_rppg = torch.zeros(*rppg_shape, dtype=torch.float32)

    # Wrap forward to return flat tensors (ONNX doesn't support dataclasses)
    class _OnnxWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, piezo: torch.Tensor, rppg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            out = self.inner(piezo=piezo, rppg=rppg)
            return out.regression_means, out.regression_log_vars, out.sleep_logits, out.stress_logits

    wrapper = _OnnxWrapper(model)

    torch.onnx.export(
        wrapper,
        (dummy_piezo, dummy_rppg),
        str(output_path),
        opset_version=opset_version,
        input_names=["piezo", "rppg"],
        output_names=["regression_means", "regression_log_vars", "sleep_logits", "stress_logits"],
        dynamic_axes={
            "piezo": {0: "batch"},
            "rppg": {0: "batch"},
            "regression_means": {0: "batch"},
            "regression_log_vars": {0: "batch"},
            "sleep_logits": {0: "batch"},
            "stress_logits": {0: "batch"},
        },
        do_constant_folding=True,
    )
    logger.info("Exported ONNX model → %s", output_path)
    return output_path


def quantize_onnx(onnx_path: Path | str, output_path: Path | str) -> Path:
    """Apply INT8 post-training quantization to an ONNX model.

    Uses onnxruntime's static quantization with a dynamic calibration
    (no calibration dataset required for dynamic quantization mode).

    Args:
        onnx_path: Path to FP32 ONNX model.
        output_path: Destination for quantized ONNX model.

    Returns:
        Path to quantized model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore
    except ImportError:
        raise RuntimeError("onnxruntime is required for quantization. Install: pip install onnxruntime")

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    logger.info("Quantized ONNX model → %s", output_path)
    return output_path


class OnnxInferenceSession:
    """ONNX Runtime inference session for the VitalPredictor.

    Args:
        model_path: Path to ONNX model file.
        providers: List of ONNX Runtime execution providers.
        regression_targets: Ordered list of regression target names.
    """

    def __init__(
        self,
        model_path: Path | str,
        providers: list[str] | None = None,
        regression_targets: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            raise RuntimeError("onnxruntime is required. Install: pip install onnxruntime")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self._providers = providers or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=self._providers)
        self.regression_targets = regression_targets or [
            "hr_bpm", "rmssd_ms", "sdnn_ms", "spo2_pct", "resp_rate_bpm"
        ]
        logger.info("Loaded ONNX model from %s (providers: %s)", model_path, self._providers)

    def run(
        self,
        piezo: np.ndarray,
        rppg: np.ndarray | None = None,
    ) -> dict:
        """Run inference on one window.

        Args:
            piezo: ``(1, n_channels, n_samples)`` float32 array.
            rppg: ``(1, 9, n_frames)`` float32 array or ``None``.

        Returns:
            Dict with predicted vitals (same structure as
            :meth:`VitalPredictions.to_dict`).
        """
        if rppg is None:
            rppg = np.zeros((piezo.shape[0], 9, 900), dtype=np.float32)

        outputs = self._session.run(
            None,
            {"piezo": piezo.astype(np.float32), "rppg": rppg.astype(np.float32)},
        )
        reg_means, reg_log_vars, sleep_logits, stress_logits = outputs

        import torch.nn.functional as F
        import torch

        sleep_probs = F.softmax(torch.from_numpy(sleep_logits), dim=-1).numpy()
        stress_probs = F.softmax(torch.from_numpy(stress_logits), dim=-1).numpy()

        result: dict = {}
        for i, name in enumerate(self.regression_targets):
            result[name] = float(reg_means[0, i])
            result[f"{name}_std"] = float(np.exp(0.5 * reg_log_vars[0, i]))

        sleep_names = ["Wake", "Light", "Deep", "REM"]
        result["sleep_stage"] = sleep_names[int(sleep_probs[0].argmax())]
        for i, sn in enumerate(sleep_names):
            result[f"sleep_prob_{sn}"] = float(sleep_probs[0, i])

        stress_names = ["Low", "Medium", "High"]
        result["stress_level"] = stress_names[int(stress_probs[0].argmax())]

        return result
