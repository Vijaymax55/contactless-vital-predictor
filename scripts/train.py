"""Training entry point.

Usage:
    python scripts/train.py                          # default config
    python scripts/train.py model.training.epochs=50 # Hydra override
    python scripts/train.py --config-path ../config  # custom config dir
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Ensure blusim package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from blusim.models.vital_predictor import VitalPredictor
from blusim.training.augmentation import PiezoAugmentation, RPPGAugmentation
from blusim.training.trainer import Trainer, VitalDataset

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load data, build model, train, and save checkpoint."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # ── Model ──────────────────────────────────────────────────────────
    model = VitalPredictor.from_config(cfg_dict)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── Data ───────────────────────────────────────────────────────────
    # Replace with real data loading from your synced session files.
    # This is a minimal synthetic example to demonstrate the pipeline works.
    data_cfg = cfg_dict.get("data", {})
    n_subjects = 5
    n_windows_per_subject = 50
    n_total = n_subjects * n_windows_per_subject

    fs = data_cfg.get("piezo", {}).get("sample_rate", 490)
    window_s = data_cfg.get("piezo", {}).get("window_size", 30)
    n_samples = int(fs * window_s)
    n_ch = data_cfg.get("piezo", {}).get("n_channels", 4)
    n_reg_targets = 5

    logger.info("Generating synthetic training data (%d windows × %d channels × %d samples)",
                n_total, n_ch, n_samples)

    np.random.seed(42)
    piezo = np.random.randn(n_total, n_ch, n_samples).astype(np.float32)
    rppg = np.random.randn(n_total, 9, int(30 * 30)).astype(np.float32)
    reg_targets = np.random.randn(n_total, n_reg_targets).astype(np.float32)
    sleep_labels = np.random.randint(0, 4, n_total)
    stress_labels = np.random.randint(0, 3, n_total)
    subject_ids = np.repeat(np.arange(n_subjects), n_windows_per_subject)

    # Subject-independent split
    unique_subjects = np.unique(subject_ids)
    n_val = max(1, int(len(unique_subjects) * 0.2))
    val_subjects = np.random.choice(unique_subjects, n_val, replace=False)
    val_mask = np.isin(subject_ids, val_subjects)
    train_mask = ~val_mask

    logger.info("Train windows: %d | Val windows: %d | Val subjects: %s",
                train_mask.sum(), val_mask.sum(), val_subjects.tolist())

    aug_piezo = PiezoAugmentation(p=0.5)
    aug_rppg = RPPGAugmentation(p=0.5)

    train_ds = VitalDataset(
        piezo[train_mask], rppg[train_mask], reg_targets[train_mask],
        sleep_labels[train_mask], stress_labels[train_mask], subject_ids[train_mask],
        piezo_transform=aug_piezo, rppg_transform=aug_rppg,
    )
    val_ds = VitalDataset(
        piezo[val_mask], rppg[val_mask], reg_targets[val_mask],
        sleep_labels[val_mask], stress_labels[val_mask], subject_ids[val_mask],
    )

    # ── Training ───────────────────────────────────────────────────────
    checkpoint_dir = Path(cfg_dict.get("paths", {}).get("models_dir", "./models"))
    device = cfg_dict.get("pipeline", {}).get("inference", {}).get("device", "cpu")

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=cfg_dict,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    history = trainer.train()
    logger.info("Training complete. Best val HR MAE: %.2f BPM", trainer.best_val_hr_mae)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
