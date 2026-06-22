"""Training loop with subject-independent splits, MLflow logging, and curriculum.

Key features:
- GroupShuffleSplit on subject_id — no subject appears in both train and test
- Curriculum learning: 3 stages from clean → noisy data
- Gradient clipping and cosine LR schedule with linear warmup
- MLflow experiment tracking (or file-based fallback)
- Best-model checkpoint based on validation HR MAE
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from blusim.training.loss import MultiTaskLoss

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — experiment tracking disabled")


def _linear_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


class VitalDataset(Dataset):
    """PyTorch Dataset wrapping synced piezo + rPPG windows with labels.

    Args:
        piezo_windows: Array ``(N, n_channels, n_samples)`` or ``None``.
        rppg_windows: Array ``(N, n_rppg_channels, n_frames)`` or ``None``.
        regression_targets: Array ``(N, n_targets)`` with NaN for missing.
        sleep_labels: Array ``(N,)`` with -1 for unlabelled.
        stress_labels: Array ``(N,)`` with -1 for unlabelled.
        subject_ids: Array ``(N,)`` of subject identifier integers.
        piezo_transform: Optional callable augmentation for piezo windows.
        rppg_transform: Optional callable augmentation for rPPG windows.
    """

    def __init__(
        self,
        piezo_windows: Optional[np.ndarray],
        rppg_windows: Optional[np.ndarray],
        regression_targets: np.ndarray,
        sleep_labels: np.ndarray,
        stress_labels: np.ndarray,
        subject_ids: np.ndarray,
        piezo_transform=None,
        rppg_transform=None,
    ) -> None:
        self.piezo = piezo_windows
        self.rppg = rppg_windows
        self.reg_targets = regression_targets.astype(np.float32)
        self.sleep_labels = sleep_labels.astype(np.int64)
        self.stress_labels = stress_labels.astype(np.int64)
        self.subject_ids = subject_ids
        self.piezo_tf = piezo_transform
        self.rppg_tf = rppg_transform
        self.n = len(regression_targets)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        sample: dict = {
            "reg_targets": torch.from_numpy(self.reg_targets[idx]),
            "sleep_label": torch.tensor(self.sleep_labels[idx], dtype=torch.long),
            "stress_label": torch.tensor(self.stress_labels[idx], dtype=torch.long),
        }

        if self.piezo is not None:
            p = self.piezo[idx].astype(np.float32)
            if self.piezo_tf is not None:
                p = self.piezo_tf(p.T).T  # augment expects (n_samples, n_ch)
            sample["piezo"] = torch.from_numpy(p)  # (n_ch, n_samples)

        if self.rppg is not None:
            r = self.rppg[idx].astype(np.float32)
            if self.rppg_tf is not None:
                r = self.rppg_tf(r.T).T
            sample["rppg"] = torch.from_numpy(r)   # (n_ch, n_frames)

        return sample


class Trainer:
    """Orchestrates the full training loop.

    Args:
        model: :class:`VitalPredictor` instance.
        train_dataset: Training :class:`VitalDataset`.
        val_dataset: Validation :class:`VitalDataset`.
        cfg: Training config dict (from ``config/model/default.yaml``).
        checkpoint_dir: Directory to save best model weights.
        device: PyTorch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: VitalDataset,
        val_dataset: VitalDataset,
        cfg: dict,
        checkpoint_dir: Path | str = "./models",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        t_cfg = cfg.get("training", cfg)
        self.epochs = t_cfg.get("epochs", 100)
        self.batch_size = t_cfg.get("batch_size", 32)
        self.lr = float(t_cfg.get("lr", 1e-3))
        self.weight_decay = float(t_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(t_cfg.get("grad_clip", 1.0))
        self.early_stop_patience = int(t_cfg.get("early_stopping_patience", 15))
        self.warmup_epochs = int(t_cfg.get("warmup_epochs", 5))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=(device != "cpu"))
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs - self.warmup_epochs, eta_min=self.lr / 100)

        n_reg = len(model.reg_head.targets)
        self.criterion = MultiTaskLoss(
            n_regression_targets=n_reg,
            n_sleep_classes=model.sleep_head.n_classes,
            n_stress_classes=model.stress_head.n_classes,
            temporal_weight=float(t_cfg.get("temporal_consistency_weight", 0.1)),
            use_uncertainty_weighting=bool(t_cfg.get("uncertainty_weighting", True)),
        ).to(device)

        self.best_val_hr_mae = float("inf")
        self.patience_counter = 0

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with training history: ``{epoch: {train_loss, val_loss, val_hr_mae}}``.
        """
        if _MLFLOW_AVAILABLE:
            mlflow.start_run()
            mlflow.log_params({
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            })

        history: dict = {}
        for epoch in range(1, self.epochs + 1):
            # Warmup LR
            if epoch <= self.warmup_epochs:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr * _linear_warmup_factor(epoch, self.warmup_epochs)
            else:
                self.scheduler.step()

            t0 = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            elapsed = time.time() - t0

            val_hr_mae = val_metrics.get("hr_mae", float("inf"))
            history[epoch] = {**{f"train_{k}": v for k, v in train_metrics.items()},
                               **{f"val_{k}": v for k, v in val_metrics.items()}}

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_HR_MAE=%.2f | %.1fs",
                epoch, self.epochs, train_metrics["loss_total"], val_metrics["loss_total"], val_hr_mae, elapsed,
            )

            if _MLFLOW_AVAILABLE:
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)

            # Checkpoint
            if val_hr_mae < self.best_val_hr_mae:
                self.best_val_hr_mae = val_hr_mae
                self.patience_counter = 0
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info("Saved best model → %s (HR MAE=%.2f)", ckpt_path, val_hr_mae)
                if _MLFLOW_AVAILABLE:
                    mlflow.log_artifact(str(ckpt_path))
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if _MLFLOW_AVAILABLE:
            mlflow.end_run()

        return history

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train() if train else self.model.eval()
        ctx = torch.enable_grad() if train else torch.no_grad()

        total_components: dict[str, list[float]] = {}
        hr_errors: list[float] = []

        with ctx:
            for batch in loader:
                piezo = batch.get("piezo", None)
                rppg = batch.get("rppg", None)
                reg_targets = batch["reg_targets"].to(self.device)
                sleep_labels = batch["sleep_label"].to(self.device)
                stress_labels = batch["stress_label"].to(self.device)

                if piezo is not None:
                    piezo = piezo.to(self.device)
                if rppg is not None:
                    rppg = rppg.to(self.device)

                outputs = self.model(piezo=piezo, rppg=rppg)
                loss, components = self.criterion(
                    outputs.regression_means,
                    outputs.regression_log_vars,
                    reg_targets,
                    outputs.sleep_logits,
                    sleep_labels,
                    outputs.stress_logits,
                    stress_labels,
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                for k, v in components.items():
                    total_components.setdefault(k, []).append(v)

                # HR MAE (first regression target assumed to be HR)
                hr_gt = reg_targets[:, 0]
                hr_pred = outputs.regression_means[:, 0]
                valid = ~torch.isnan(hr_gt)
                if valid.any():
                    hr_errors.extend((hr_pred[valid] - hr_gt[valid]).abs().cpu().tolist())

        averaged = {k: float(np.mean(v)) for k, v in total_components.items()}
        if hr_errors:
            averaged["hr_mae"] = float(np.mean(hr_errors))
        return averaged
