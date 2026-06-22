"""Multi-task loss function with Kendall uncertainty weighting.

Implements:
- Gaussian NLL loss for regression with learned variance (aleatoric uncertainty)
- Focal loss for imbalanced sleep stage classification
- Temporal consistency regularisation
- Kendall et al. (2018) multi-task uncertainty weighting

Reference:
    Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using
    uncertainty to weigh losses in scene geometry and semantics. CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss for handling class imbalance in sleep staging.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        n_classes: Number of classes.
        gamma: Focusing parameter (default 2.0).
        alpha: Per-class weight tensor or ``None`` for uniform weights.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        n_classes: int = 4,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(B, n_classes)`` raw class logits.
            targets: ``(B,)`` integer class labels.

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Gather probabilities for correct class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean() if self.reduction == "mean" else loss.sum()


class GaussianNLLLoss(nn.Module):
    """Heteroscedastic Gaussian NLL loss for regression with uncertainty.

    L = 0.5 * exp(-log_var) * (y - ŷ)² + 0.5 * log_var

    Args:
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        means: torch.Tensor,
        log_vars: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute heteroscedastic NLL.

        Args:
            means: Predicted means ``(B, n_targets)``.
            log_vars: Predicted log-variances ``(B, n_targets)``.
            targets: Ground-truth values ``(B, n_targets)``.

        Returns:
            Scalar loss.
        """
        # Mask NaN targets (vitals not available for every window)
        valid = ~torch.isnan(targets)
        if valid.sum() == 0:
            return means.sum() * 0.0  # zero-gradient pass-through

        m = means[valid]
        lv = log_vars[valid]
        t = targets[valid]

        precision = torch.exp(-lv)
        loss = 0.5 * precision * (t - m) ** 2 + 0.5 * lv
        return loss.mean() if self.reduction == "mean" else loss.sum()


class TemporalConsistencyLoss(nn.Module):
    """Penalise large jumps in consecutive regression predictions.

    L_temp = mean(|ŷ_t - ŷ_{t-1}|) over the batch dimension interpreted
    as time steps (requires batch size ≥ 2 to be meaningful).

    Args:
        weight: Scalar multiplier for this loss term.
    """

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, means: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency penalty.

        Args:
            means: Predictions ``(B, n_targets)`` where B is a temporal batch.

        Returns:
            Scalar penalty.
        """
        if means.size(0) < 2:
            return means.sum() * 0.0
        delta = torch.abs(means[1:] - means[:-1])
        return self.weight * delta.mean()


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss with Kendall uncertainty weighting.

    Learnable log-sigma parameters weight each sub-loss automatically.
    The effective weight for task k is ``1 / (2σ_k²)``, and the total loss
    includes a ``log(σ_k)`` regularisation term that prevents σ → ∞.

    Args:
        n_regression_targets: Number of regression targets.
        n_sleep_classes: Number of sleep stage classes.
        n_stress_classes: Number of stress level classes.
        temporal_weight: Weight for temporal consistency loss.
        use_uncertainty_weighting: Whether to use learned Kendall weighting.
        sleep_focal_gamma: Gamma for focal loss on sleep staging.
    """

    def __init__(
        self,
        n_regression_targets: int = 5,
        n_sleep_classes: int = 4,
        n_stress_classes: int = 3,
        temporal_weight: float = 0.1,
        use_uncertainty_weighting: bool = True,
        sleep_focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Learnable log-sigma² for each task group
        if use_uncertainty_weighting:
            self.log_sigma_reg = nn.Parameter(torch.zeros(1))
            self.log_sigma_sleep = nn.Parameter(torch.zeros(1))
            self.log_sigma_stress = nn.Parameter(torch.zeros(1))

        self.regression_loss = GaussianNLLLoss()
        self.sleep_loss = FocalLoss(n_classes=n_sleep_classes, gamma=sleep_focal_gamma)
        self.stress_loss = nn.CrossEntropyLoss()
        self.temporal_loss = TemporalConsistencyLoss(weight=temporal_weight)

    def forward(
        self,
        reg_means: torch.Tensor,
        reg_log_vars: torch.Tensor,
        reg_targets: torch.Tensor,
        sleep_logits: torch.Tensor,
        sleep_targets: torch.Tensor,
        stress_logits: torch.Tensor,
        stress_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total multi-task loss.

        Args:
            reg_means: ``(B, n_reg)`` regression predictions.
            reg_log_vars: ``(B, n_reg)`` log-variance predictions.
            reg_targets: ``(B, n_reg)`` ground-truth regression values.
            sleep_logits: ``(B, n_sleep)`` sleep stage logits.
            sleep_targets: ``(B,)`` sleep stage labels (-1 to ignore).
            stress_logits: ``(B, n_stress)`` stress level logits.
            stress_targets: ``(B,)`` stress labels (-1 to ignore).

        Returns:
            ``(total_loss, component_dict)`` for logging.
        """
        l_reg = self.regression_loss(reg_means, reg_log_vars, reg_targets)
        l_temp = self.temporal_loss(reg_means)

        # Sleep loss — ignore samples with label -1
        valid_sleep = sleep_targets >= 0
        l_sleep = (
            self.sleep_loss(sleep_logits[valid_sleep], sleep_targets[valid_sleep])
            if valid_sleep.any()
            else sleep_logits.sum() * 0.0
        )

        # Stress loss — ignore -1 labels
        valid_stress = stress_targets >= 0
        l_stress = (
            self.stress_loss(stress_logits[valid_stress], stress_targets[valid_stress])
            if valid_stress.any()
            else stress_logits.sum() * 0.0
        )

        if self.use_uncertainty_weighting:
            # Kendall weighting: L_total = L/σ² + log(σ)
            s_reg = torch.exp(self.log_sigma_reg)
            s_sleep = torch.exp(self.log_sigma_sleep)
            s_stress = torch.exp(self.log_sigma_stress)

            total = (
                (l_reg + l_temp) / (2 * s_reg ** 2) + torch.log(s_reg)
                + l_sleep / (2 * s_sleep ** 2) + torch.log(s_sleep)
                + l_stress / (2 * s_stress ** 2) + torch.log(s_stress)
            )
        else:
            total = l_reg + l_temp + l_sleep + l_stress

        components = {
            "loss_regression": float(l_reg.detach()),
            "loss_temporal": float(l_temp.detach()),
            "loss_sleep": float(l_sleep.detach()),
            "loss_stress": float(l_stress.detach()),
            "loss_total": float(total.detach()),
        }
        return total, components
