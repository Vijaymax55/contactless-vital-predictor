"""Multi-task output heads for vital sign prediction.

Heads:
- RegressionHead: HR, HRV (RMSSD, SDNN), SpO2, Respiration Rate
  - Also outputs log-variance per prediction (aleatoric uncertainty)
- SleepStageHead: 4-class classification (Wake/Light/Deep/REM)
- StressHead: 3-class classification (Low/Medium/High)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """Predicts continuous vital sign values with uncertainty estimates.

    Outputs mean + log-variance per target (diagonal Gaussian).

    Args:
        in_dim: Input embedding dimension.
        targets: List of target vital names (in order).
        hidden_dims: MLP hidden layer sizes.
        dropout: Dropout rate.
    """

    TARGET_DEFAULTS = ["hr_bpm", "rmssd_ms", "sdnn_ms", "spo2_pct", "resp_rate_bpm"]

    def __init__(
        self,
        in_dim: int = 256,
        targets: list[str] | None = None,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.targets = targets or self.TARGET_DEFAULTS
        n_out = len(self.targets)
        hidden_dims = hidden_dims or [128, 64]

        self.mean_head = _MLP(in_dim, hidden_dims, n_out, dropout)
        self.log_var_head = _MLP(in_dim, hidden_dims, n_out, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict vital means and log-variances.

        Args:
            x: Fused embedding ``(B, in_dim)``.

        Returns:
            ``(means, log_vars)`` each of shape ``(B, n_targets)``.
        """
        means = self.mean_head(x)
        log_vars = self.log_var_head(x)
        return means, log_vars


class SleepStageHead(nn.Module):
    """4-class sleep stage classifier (Wake=0, Light=1, Deep=2, REM=3).

    Args:
        in_dim: Input embedding dimension.
        n_classes: Number of sleep stage classes.
        hidden_dims: MLP hidden layer sizes.
        dropout: Dropout rate.
    """

    STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]

    def __init__(
        self,
        in_dim: int = 256,
        n_classes: int = 4,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.classifier = _MLP(in_dim, hidden_dims, n_classes, dropout)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits.

        Args:
            x: Embedding ``(B, in_dim)``.

        Returns:
            Logits ``(B, n_classes)``.
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted class index and softmax probabilities.

        Args:
            x: Embedding ``(B, in_dim)``.

        Returns:
            ``(class_idx, probabilities)`` — shapes ``(B,)`` and ``(B, n_classes)``.
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs


class StressHead(nn.Module):
    """3-class stress level classifier (Low=0, Medium=1, High=2).

    Args:
        in_dim: Input embedding dimension.
        n_classes: Number of stress levels.
        hidden_dims: MLP hidden layer sizes.
        dropout: Dropout rate.
    """

    LEVEL_NAMES = ["Low", "Medium", "High"]

    def __init__(
        self,
        in_dim: int = 256,
        n_classes: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.classifier = _MLP(in_dim, hidden_dims, n_classes, dropout)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
