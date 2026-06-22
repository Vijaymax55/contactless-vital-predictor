"""Evaluation metrics for all predicted vitals.

Implements:
- HR/HRV/SpO2/Respiration: MAE, RMSE, Pearson r, Bland-Altman statistics
- Sleep staging: accuracy, Cohen's κ, per-class F1, confusion matrix
- Stress: accuracy, macro-F1
- Accuracy target checks vs. clinical thresholds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accuracy targets (from project spec)
# ---------------------------------------------------------------------------

ACCURACY_TARGETS = {
    "hr_mae": 5.0,          # BPM
    "hrv_rmssd_mae": 10.0,  # ms
    "spo2_mae": 2.0,        # %
    "sleep_kappa": 0.65,
    "resp_mae": 2.0,        # breaths/min
}


# ---------------------------------------------------------------------------
# Dataclasses for results
# ---------------------------------------------------------------------------


@dataclass
class RegressionMetrics:
    """Metrics for one regression vital.

    Attributes:
        target_name: Name of the target variable.
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        pearson_r: Pearson correlation coefficient.
        n_valid: Number of valid (non-NaN) samples.
        bland_altman_bias: Mean of (prediction − reference).
        bland_altman_loa_upper: Upper limit of agreement (+1.96σ).
        bland_altman_loa_lower: Lower limit of agreement (−1.96σ).
        meets_target: Whether MAE meets the clinical accuracy target.
    """

    target_name: str
    mae: float
    rmse: float
    pearson_r: float
    n_valid: int
    bland_altman_bias: float
    bland_altman_loa_upper: float
    bland_altman_loa_lower: float
    meets_target: bool = False


@dataclass
class ClassificationMetrics:
    """Metrics for sleep stage or stress classification.

    Attributes:
        task_name: Name of the classification task.
        accuracy: Overall accuracy.
        cohen_kappa: Cohen's kappa.
        per_class_f1: Per-class F1 scores.
        class_names: Class name labels.
        confusion_matrix: Confusion matrix as a 2-D array.
        n_samples: Total samples evaluated.
        meets_target: Whether kappa meets the project target (sleep only).
    """

    task_name: str
    accuracy: float
    cohen_kappa: float
    per_class_f1: list[float]
    class_names: list[str]
    confusion_matrix: np.ndarray
    n_samples: int
    meets_target: bool = False


@dataclass
class EvaluationReport:
    """Full evaluation report for all modalities and vitals.

    Attributes:
        regression: Dict mapping vital name → :class:`RegressionMetrics`.
        sleep_stage: :class:`ClassificationMetrics` for sleep staging.
        stress: :class:`ClassificationMetrics` for stress.
        latency_ms_mean: Mean inference latency per sample in ms.
        latency_ms_p95: 95th-percentile latency.
        target_summary: Dict of ``{target_name: passed}`` booleans.
    """

    regression: dict[str, RegressionMetrics] = field(default_factory=dict)
    sleep_stage: Optional[ClassificationMetrics] = None
    stress: Optional[ClassificationMetrics] = None
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    target_summary: dict[str, bool] = field(default_factory=dict)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print("\n" + "=" * 70)
        print("BLUSIM EVALUATION REPORT")
        print("=" * 70)

        print("\nREGRESSION VITALS:")
        for name, m in self.regression.items():
            status = "✓" if m.meets_target else "✗"
            print(f"  {status} {name:20s}  MAE={m.mae:.2f}  RMSE={m.rmse:.2f}  r={m.pearson_r:.3f}  N={m.n_valid}")

        if self.sleep_stage:
            s = self.sleep_stage
            status = "✓" if s.meets_target else "✗"
            print(f"\nSLEEP STAGING ({status}):")
            print(f"  Accuracy: {s.accuracy:.1%}   Cohen's κ: {s.cohen_kappa:.3f}")
            for cls, f1 in zip(s.class_names, s.per_class_f1):
                print(f"  F1[{cls}]: {f1:.3f}")

        print("\nACCURACY TARGET SUMMARY:")
        for target, passed in self.target_summary.items():
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {target}")

        print(f"\nLATENCY: mean={self.latency_ms_mean:.1f} ms  p95={self.latency_ms_p95:.1f} ms")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    mae_target: Optional[float] = None,
) -> RegressionMetrics:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground-truth values (may contain NaN).
        y_pred: Predicted values.
        target_name: Descriptive name for the target.
        mae_target: Accuracy target for MAE; used to set ``meets_target``.

    Returns:
        :class:`RegressionMetrics`.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid.sum() < 2:
        logger.warning("Insufficient valid samples for %s metrics", target_name)
        return RegressionMetrics(target_name=target_name, mae=float("nan"), rmse=float("nan"),
                                 pearson_r=float("nan"), n_valid=int(valid.sum()),
                                 bland_altman_bias=float("nan"), bland_altman_loa_upper=float("nan"),
                                 bland_altman_loa_lower=float("nan"))

    yt = y_true[valid]
    yp = y_pred[valid]

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
        r = float(np.corrcoef(yt, yp)[0, 1])
    else:
        r = float("nan")

    # Bland-Altman
    diff = yp - yt
    bias = float(np.mean(diff))
    loa_std = float(np.std(diff))
    loa_upper = bias + 1.96 * loa_std
    loa_lower = bias - 1.96 * loa_std

    meets = mae <= mae_target if mae_target is not None else False

    return RegressionMetrics(
        target_name=target_name,
        mae=mae,
        rmse=rmse,
        pearson_r=r,
        n_valid=int(valid.sum()),
        bland_altman_bias=bias,
        bland_altman_loa_upper=loa_upper,
        bland_altman_loa_lower=loa_lower,
        meets_target=meets,
    )


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Compute Cohen's kappa for multi-class classification.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        n_classes: Number of classes.

    Returns:
        Cohen's kappa in [-1, 1].
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    n = cm.sum()
    if n == 0:
        return 0.0
    p_o = np.trace(cm) / n
    p_e = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n ** 2)
    if abs(1 - p_e) < 1e-10:
        return 1.0 if p_o == 1.0 else 0.0
    return float((p_o - p_e) / (1 - p_e))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels (integers 0..n_classes-1).
        y_pred: Predicted labels.
        n_classes: Number of classes.

    Returns:
        ``(n_classes, n_classes)`` confusion matrix (rows=true, cols=pred).
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def per_class_f1(cm: np.ndarray) -> list[float]:
    """Compute per-class F1 from a confusion matrix.

    Args:
        cm: ``(n, n)`` confusion matrix.

    Returns:
        List of F1 scores, one per class.
    """
    f1s = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(float(f1))
    return f1s


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str,
    class_names: list[str],
    kappa_target: Optional[float] = None,
) -> ClassificationMetrics:
    """Compute classification evaluation metrics.

    Args:
        y_true: True class labels (integers). -1 labels are ignored.
        y_pred: Predicted class labels.
        task_name: Descriptive task name.
        class_names: Human-readable class names.
        kappa_target: If provided, used to set ``meets_target``.

    Returns:
        :class:`ClassificationMetrics`.
    """
    # Filter out -1 (unlabelled)
    valid = (y_true >= 0) & (y_pred >= 0)
    yt = y_true[valid]
    yp = y_pred[valid]
    n = len(yt)

    if n == 0:
        return ClassificationMetrics(task_name=task_name, accuracy=0.0, cohen_kappa=0.0,
                                     per_class_f1=[], class_names=class_names,
                                     confusion_matrix=np.zeros((len(class_names), len(class_names))),
                                     n_samples=0)

    n_classes = len(class_names)
    cm = confusion_matrix(yt, yp, n_classes)
    acc = float(np.trace(cm) / cm.sum())
    kappa = cohen_kappa(yt, yp, n_classes)
    f1s = per_class_f1(cm)
    meets = kappa >= kappa_target if kappa_target is not None else False

    return ClassificationMetrics(
        task_name=task_name,
        accuracy=acc,
        cohen_kappa=kappa,
        per_class_f1=f1s,
        class_names=class_names,
        confusion_matrix=cm,
        n_samples=n,
        meets_target=meets,
    )


def build_evaluation_report(
    reg_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    sleep_true: Optional[np.ndarray],
    sleep_pred: Optional[np.ndarray],
    stress_true: Optional[np.ndarray] = None,
    stress_pred: Optional[np.ndarray] = None,
    latency_ms: Optional[list[float]] = None,
) -> EvaluationReport:
    """Build a full :class:`EvaluationReport` from raw predictions.

    Args:
        reg_predictions: Dict ``{target_name: (y_true, y_pred)}``.
        sleep_true: True sleep stage labels.
        sleep_pred: Predicted sleep stage labels.
        stress_true: True stress labels (optional).
        stress_pred: Predicted stress labels (optional).
        latency_ms: List of per-sample inference latencies in ms.

    Returns:
        :class:`EvaluationReport`.
    """
    mae_targets = {
        "hr_bpm": ACCURACY_TARGETS["hr_mae"],
        "rmssd_ms": ACCURACY_TARGETS["hrv_rmssd_mae"],
        "spo2_pct": ACCURACY_TARGETS["spo2_mae"],
        "resp_rate_bpm": ACCURACY_TARGETS["resp_mae"],
    }

    reg_metrics: dict[str, RegressionMetrics] = {}
    for name, (yt, yp) in reg_predictions.items():
        m = regression_metrics(np.array(yt, dtype=float), np.array(yp, dtype=float), name, mae_targets.get(name))
        reg_metrics[name] = m

    sleep_m = None
    if sleep_true is not None and sleep_pred is not None:
        sleep_m = classification_metrics(
            np.array(sleep_true), np.array(sleep_pred),
            task_name="sleep_stage",
            class_names=["Wake", "Light", "Deep", "REM"],
            kappa_target=ACCURACY_TARGETS["sleep_kappa"],
        )

    stress_m = None
    if stress_true is not None and stress_pred is not None:
        stress_m = classification_metrics(
            np.array(stress_true), np.array(stress_pred),
            task_name="stress",
            class_names=["Low", "Medium", "High"],
        )

    lat_arr = np.array(latency_ms) if latency_ms else np.array([0.0])
    target_summary = {
        "hr_mae": reg_metrics.get("hr_bpm", RegressionMetrics("", 999, 999, 0, 0, 0, 0, 0)).meets_target,
        "hrv_rmssd_mae": reg_metrics.get("rmssd_ms", RegressionMetrics("", 999, 999, 0, 0, 0, 0, 0)).meets_target,
        "spo2_mae": reg_metrics.get("spo2_pct", RegressionMetrics("", 999, 999, 0, 0, 0, 0, 0)).meets_target,
        "sleep_kappa": sleep_m.meets_target if sleep_m else False,
        "resp_mae": reg_metrics.get("resp_rate_bpm", RegressionMetrics("", 999, 999, 0, 0, 0, 0, 0)).meets_target,
    }

    return EvaluationReport(
        regression=reg_metrics,
        sleep_stage=sleep_m,
        stress=stress_m,
        latency_ms_mean=float(np.mean(lat_arr)),
        latency_ms_p95=float(np.percentile(lat_arr, 95)),
        target_summary=target_summary,
    )
