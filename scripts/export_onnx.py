"""Export trained model to ONNX and quantize to INT8.

Usage:
    python scripts/export_onnx.py --model models/best_model.pt --output models/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from blusim.inference.onnx_export import export_to_onnx, quantize_onnx
from blusim.models.vital_predictor import VitalPredictor

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BluSim model to ONNX")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="./models", help="Output directory")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    cfg = {
        "piezo_encoder": {"in_channels": 4},
        "rppg_encoder": {"in_channels": 9},
        "fusion": {"strategy": "cross_attention", "embed_dim": 256},
        "heads": {"regression": {}, "classification": {}},
    }
    model = VitalPredictor.from_config(cfg)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    # FP32 export
    fp32_path = output_dir / "vital_predictor.onnx"
    export_to_onnx(model, fp32_path)
    logger.info("FP32 ONNX: %s (%.1f MB)", fp32_path, fp32_path.stat().st_size / 1e6)

    if not args.no_quantize:
        int8_path = output_dir / "vital_predictor_int8.onnx"
        quantize_onnx(fp32_path, int8_path)
        logger.info("INT8 ONNX: %s (%.1f MB)", int8_path, int8_path.stat().st_size / 1e6)
        size_ratio = fp32_path.stat().st_size / int8_path.stat().st_size
        logger.info("Compression ratio: %.1f×", size_ratio)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
