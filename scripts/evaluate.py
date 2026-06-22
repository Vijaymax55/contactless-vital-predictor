"""Evaluation script: load a trained model and run the evaluation suite.

Usage:
    python scripts/evaluate.py --model models/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from blusim.evaluation.metrics import build_evaluation_report
from blusim.models.vital_predictor import VitalPredictor

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VitalPredictor on held-out data")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────
    cfg = {
        "piezo_encoder": {"in_channels": 4},
        "rppg_encoder": {"in_channels": 9},
        "fusion": {"strategy": "cross_attention", "embed_dim": 256},
        "heads": {"regression": {}, "classification": {}},
    }
    model = VitalPredictor.from_config(cfg)
    state = torch.load(args.model, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded model from %s", args.model)

    # ── Synthetic evaluation data ───────────────────────────────────────
    # In production: load your test set from disk here.
    n_test = 100
    np.random.seed(99)
    piezo = torch.randn(n_test, 4, 14700)
    hr_true = 60.0 + np.random.randn(n_test) * 10
    sleep_true = np.random.randint(0, 4, n_test)

    hr_pred: list[float] = []
    sleep_pred: list[int] = []

    for i in range(n_test):
        with torch.no_grad():
            out = model(piezo=piezo[i:i+1])
        hr_pred.append(float(out.regression_means[0, 0]))
        sleep_pred.append(int(out.sleep_logits[0].argmax()))

    # ── Build and print report ──────────────────────────────────────────
    report = build_evaluation_report(
        reg_predictions={"hr_bpm": (hr_true, np.array(hr_pred))},
        sleep_true=sleep_true,
        sleep_pred=np.array(sleep_pred),
    )
    report.print_summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
