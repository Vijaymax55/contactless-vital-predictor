"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from blusim.evaluation.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    build_evaluation_report,
    classification_metrics,
    cohen_kappa,
    confusion_matrix,
    per_class_f1,
    regression_metrics,
)


class TestRegressionMetrics:
    def test_perfect_prediction(self) -> None:
        y = np.array([70.0, 72.0, 68.0, 75.0])
        m = regression_metrics(y, y, "hr_bpm", mae_target=5.0)
        assert m.mae == pytest.approx(0.0, abs=1e-5)
        assert m.rmse == pytest.approx(0.0, abs=1e-5)
        assert m.pearson_r == pytest.approx(1.0, abs=1e-5)
        assert m.meets_target

    def test_known_mae(self) -> None:
        y_true = np.array([70.0, 70.0, 70.0, 70.0])
        y_pred = np.array([73.0, 73.0, 73.0, 73.0])
        m = regression_metrics(y_true, y_pred, "hr_bpm", mae_target=5.0)
        assert m.mae == pytest.approx(3.0, abs=1e-5)
        assert m.meets_target

    def test_exceeds_target(self) -> None:
        y_true = np.array([70.0, 70.0])
        y_pred = np.array([80.0, 80.0])
        m = regression_metrics(y_true, y_pred, "hr_bpm", mae_target=5.0)
        assert m.mae == pytest.approx(10.0)
        assert not m.meets_target

    def test_nan_handling(self) -> None:
        y_true = np.array([70.0, float("nan"), 72.0])
        y_pred = np.array([70.0, 70.0, 72.0])
        m = regression_metrics(y_true, y_pred, "hr_bpm")
        assert m.n_valid == 2

    def test_bland_altman_bias(self) -> None:
        y_true = np.array([70.0, 72.0, 74.0])
        y_pred = y_true + 3.0    # constant bias of +3
        m = regression_metrics(y_true, y_pred, "hr_bpm")
        assert m.bland_altman_bias == pytest.approx(3.0, abs=1e-5)


class TestCohenKappa:
    def test_perfect_agreement(self) -> None:
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        k = cohen_kappa(y, y, n_classes=4)
        assert k == pytest.approx(1.0, abs=1e-5)

    def test_chance_level(self) -> None:
        np.random.seed(42)
        y_true = np.repeat([0, 1, 2, 3], 100)
        y_pred = np.random.choice([0, 1, 2, 3], 400)
        k = cohen_kappa(y_true, y_pred, n_classes=4)
        assert abs(k) < 0.2    # near zero for random

    def test_all_wrong(self) -> None:
        # When all predictions are wrong AND both classes are represented in truth,
        # kappa is negative.  With only one truth class (all 0s predicted as 1s),
        # p_e = 0 so kappa = 0. Use a balanced two-class case to get negative kappa.
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])   # perfectly inverted
        k = cohen_kappa(y_true, y_pred, n_classes=2)
        assert k <= 0    # inverted predictions → kappa ≤ 0


class TestConfusionMatrix:
    def test_diagonal_for_perfect(self) -> None:
        y = np.array([0, 1, 2, 3])
        cm = confusion_matrix(y, y, 4)
        np.testing.assert_array_equal(cm, np.eye(4, dtype=np.int64))

    def test_shape(self) -> None:
        cm = confusion_matrix(np.array([0, 1]), np.array([1, 0]), n_classes=4)
        assert cm.shape == (4, 4)

    def test_counts(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix(y_true, y_pred, 2)
        assert cm[0, 0] == 1   # true pos class 0
        assert cm[0, 1] == 1   # class 0 predicted as 1
        assert cm[1, 0] == 1   # class 1 predicted as 0
        assert cm[1, 1] == 1   # true pos class 1


class TestPerClassF1:
    def test_perfect_f1(self) -> None:
        cm = np.eye(4, dtype=np.int64) * 10
        f1s = per_class_f1(cm)
        for f in f1s:
            assert f == pytest.approx(1.0, abs=1e-4)

    def test_zero_class(self) -> None:
        cm = np.array([[10, 0], [10, 0]], dtype=np.int64)
        f1s = per_class_f1(cm)
        assert f1s[1] == pytest.approx(0.0, abs=1e-5)


class TestBuildEvaluationReport:
    def test_full_report(self) -> None:
        reg_preds = {
            "hr_bpm": (np.array([70.0, 72.0, 75.0]), np.array([71.0, 73.0, 74.0])),
            "spo2_pct": (np.array([98.0, 97.5, 99.0]), np.array([98.5, 97.0, 98.9])),
        }
        sleep_true = np.array([0, 1, 2, 3, 0, 1])
        sleep_pred = np.array([0, 1, 2, 3, 1, 1])
        report = build_evaluation_report(reg_preds, sleep_true, sleep_pred)
        assert "hr_bpm" in report.regression
        assert report.sleep_stage is not None
        assert "hr_mae" in report.target_summary

    def test_no_sleep_labels(self) -> None:
        reg_preds = {"hr_bpm": (np.array([70.0]), np.array([71.0]))}
        report = build_evaluation_report(reg_preds, None, None)
        assert report.sleep_stage is None
