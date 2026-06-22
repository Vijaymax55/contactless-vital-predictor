"""Unit tests for model architecture components."""

from __future__ import annotations

import torch
import pytest

from blusim.models.encoders.piezo_encoder import PiezoEncoder
from blusim.models.encoders.rppg_encoder import RPPGEncoder
from blusim.models.fusion.multimodal_fusion import CrossAttentionFusion, LateFusion, build_fusion
from blusim.models.heads.output_heads import RegressionHead, SleepStageHead, StressHead
from blusim.models.vital_predictor import VitalPredictor
from blusim.training.loss import FocalLoss, GaussianNLLLoss, MultiTaskLoss, TemporalConsistencyLoss

BATCH = 4
N_CHANNELS_PIEZO = 4
N_SAMPLES_PIEZO = 14700   # 30 s × 490 Hz
N_CHANNELS_RPPG = 9
N_FRAMES_RPPG = 900       # 30 s × 30 FPS
EMBED_DIM = 64            # small for tests


@pytest.fixture
def piezo_input() -> torch.Tensor:
    return torch.randn(BATCH, N_CHANNELS_PIEZO, N_SAMPLES_PIEZO)


@pytest.fixture
def rppg_input() -> torch.Tensor:
    return torch.randn(BATCH, N_CHANNELS_RPPG, N_FRAMES_RPPG)


@pytest.fixture
def piezo_encoder() -> PiezoEncoder:
    return PiezoEncoder(
        in_channels=N_CHANNELS_PIEZO,
        cnn_channels=[8, 16, 32],
        cnn_kernel_size=5,
        lstm_hidden=16,
        lstm_layers=1,
        attn_heads=2,
        dropout=0.0,
        embed_dim=EMBED_DIM,
    )


@pytest.fixture
def rppg_encoder() -> RPPGEncoder:
    return RPPGEncoder(
        in_channels=N_CHANNELS_RPPG,
        cnn_channels=[8, 16, 32],
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff_dim=64,
        dropout=0.0,
        embed_dim=EMBED_DIM,
    )


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class TestPiezoEncoder:
    def test_output_shape(self, piezo_encoder: PiezoEncoder, piezo_input: torch.Tensor) -> None:
        out = piezo_encoder(piezo_input)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_missing_mask(self, piezo_encoder: PiezoEncoder, piezo_input: torch.Tensor) -> None:
        mask = torch.zeros(BATCH, dtype=torch.bool)
        mask[0] = True  # first sample is missing
        out = piezo_encoder(piezo_input, mask=mask)
        assert out.shape == (BATCH, EMBED_DIM)
        assert torch.all(out[0] == 0), "Masked sample should produce zero embedding"

    def test_no_nans(self, piezo_encoder: PiezoEncoder, piezo_input: torch.Tensor) -> None:
        out = piezo_encoder(piezo_input)
        assert not torch.any(torch.isnan(out))


class TestRPPGEncoder:
    def test_output_shape(self, rppg_encoder: RPPGEncoder, rppg_input: torch.Tensor) -> None:
        out = rppg_encoder(rppg_input)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_no_nans(self, rppg_encoder: RPPGEncoder, rppg_input: torch.Tensor) -> None:
        out = rppg_encoder(rppg_input)
        assert not torch.any(torch.isnan(out))


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class TestLateFusion:
    def test_output_shape(self) -> None:
        fusion = LateFusion(embed_dim=EMBED_DIM, n_modalities=2, output_dim=EMBED_DIM)
        p = torch.randn(BATCH, EMBED_DIM)
        r = torch.randn(BATCH, EMBED_DIM)
        out = fusion([p, r])
        assert out.shape == (BATCH, EMBED_DIM)

    def test_handles_none_modality(self) -> None:
        fusion = LateFusion(embed_dim=EMBED_DIM, n_modalities=2, output_dim=EMBED_DIM)
        p = torch.randn(BATCH, EMBED_DIM)
        out = fusion([p, None])   # rPPG absent
        assert out.shape == (BATCH, EMBED_DIM)


class TestCrossAttentionFusion:
    def test_output_shape(self) -> None:
        fusion = CrossAttentionFusion(embed_dim=EMBED_DIM, n_heads=2, output_dim=EMBED_DIM)
        p = torch.randn(BATCH, EMBED_DIM)
        r = torch.randn(BATCH, EMBED_DIM)
        out = fusion(p, r)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_handles_none_piezo(self) -> None:
        fusion = CrossAttentionFusion(embed_dim=EMBED_DIM, n_heads=2, output_dim=EMBED_DIM)
        r = torch.randn(BATCH, EMBED_DIM)
        out = fusion(None, r)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_handles_none_rppg(self) -> None:
        fusion = CrossAttentionFusion(embed_dim=EMBED_DIM, n_heads=2, output_dim=EMBED_DIM)
        p = torch.randn(BATCH, EMBED_DIM)
        out = fusion(p, None)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_per_sample_mask(self) -> None:
        fusion = CrossAttentionFusion(embed_dim=EMBED_DIM, n_heads=2, output_dim=EMBED_DIM)
        p = torch.randn(BATCH, EMBED_DIM)
        r = torch.randn(BATCH, EMBED_DIM)
        pmask = torch.zeros(BATCH, dtype=torch.bool)
        pmask[1] = True
        out = fusion(p, r, piezo_mask=pmask)
        assert out.shape == (BATCH, EMBED_DIM)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------


class TestRegressionHead:
    def test_output_shapes(self) -> None:
        head = RegressionHead(in_dim=EMBED_DIM, targets=["hr", "spo2"], hidden_dims=[32])
        x = torch.randn(BATCH, EMBED_DIM)
        means, log_vars = head(x)
        assert means.shape == (BATCH, 2)
        assert log_vars.shape == (BATCH, 2)


class TestSleepStageHead:
    def test_output_shapes(self) -> None:
        head = SleepStageHead(in_dim=EMBED_DIM, n_classes=4, hidden_dims=[32])
        x = torch.randn(BATCH, EMBED_DIM)
        logits = head(x)
        assert logits.shape == (BATCH, 4)

    def test_predict_returns_valid_class(self) -> None:
        head = SleepStageHead(in_dim=EMBED_DIM, n_classes=4, hidden_dims=[32])
        x = torch.randn(BATCH, EMBED_DIM)
        cls, probs = head.predict(x)
        assert all(0 <= c < 4 for c in cls.tolist())
        assert probs.shape == (BATCH, 4)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TestGaussianNLLLoss:
    def test_finite_loss(self) -> None:
        loss_fn = GaussianNLLLoss()
        means = torch.randn(BATCH, 5)
        log_vars = torch.zeros(BATCH, 5)
        targets = torch.randn(BATCH, 5)
        loss = loss_fn(means, log_vars, targets)
        assert torch.isfinite(loss)

    def test_nan_targets_skipped(self) -> None:
        loss_fn = GaussianNLLLoss()
        means = torch.randn(BATCH, 5)
        log_vars = torch.zeros(BATCH, 5)
        targets = torch.randn(BATCH, 5)
        targets[0, :] = float("nan")
        loss = loss_fn(means, log_vars, targets)
        assert torch.isfinite(loss)

    def test_all_nan_targets(self) -> None:
        loss_fn = GaussianNLLLoss()
        means = torch.randn(BATCH, 5)
        log_vars = torch.zeros(BATCH, 5)
        targets = torch.full((BATCH, 5), float("nan"))
        loss = loss_fn(means, log_vars, targets)
        assert float(loss) == 0.0


class TestFocalLoss:
    def test_finite_loss(self) -> None:
        loss_fn = FocalLoss(n_classes=4, gamma=2.0)
        logits = torch.randn(BATCH, 4)
        targets = torch.randint(0, 4, (BATCH,))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_perfect_predictions_low_loss(self) -> None:
        loss_fn = FocalLoss(n_classes=4, gamma=2.0)
        targets = torch.zeros(BATCH, dtype=torch.long)
        logits = torch.zeros(BATCH, 4)
        logits[:, 0] = 100.0   # strongly predict class 0
        loss = loss_fn(logits, targets)
        assert float(loss) < 0.01


class TestMultiTaskLoss:
    def test_returns_scalar(self) -> None:
        loss_fn = MultiTaskLoss(n_regression_targets=5, n_sleep_classes=4, n_stress_classes=3)
        reg_means = torch.randn(BATCH, 5)
        reg_log_vars = torch.zeros(BATCH, 5)
        reg_targets = torch.randn(BATCH, 5)
        sleep_logits = torch.randn(BATCH, 4)
        sleep_targets = torch.randint(0, 4, (BATCH,))
        stress_logits = torch.randn(BATCH, 3)
        stress_targets = torch.randint(0, 3, (BATCH,))
        total, components = loss_fn(reg_means, reg_log_vars, reg_targets, sleep_logits, sleep_targets, stress_logits, stress_targets)
        assert torch.isfinite(total)
        assert "loss_total" in components

    def test_ignored_labels(self) -> None:
        loss_fn = MultiTaskLoss(n_regression_targets=5, use_uncertainty_weighting=False)
        reg_means = torch.randn(BATCH, 5)
        reg_log_vars = torch.zeros(BATCH, 5)
        reg_targets = torch.randn(BATCH, 5)
        sleep_logits = torch.randn(BATCH, 4)
        sleep_targets = torch.full((BATCH,), -1, dtype=torch.long)   # all unlabelled
        stress_logits = torch.randn(BATCH, 3)
        stress_targets = torch.full((BATCH,), -1, dtype=torch.long)
        total, _ = loss_fn(reg_means, reg_log_vars, reg_targets, sleep_logits, sleep_targets, stress_logits, stress_targets)
        assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# Full VitalPredictor
# ---------------------------------------------------------------------------


class TestVitalPredictor:
    @pytest.fixture
    def model(self) -> VitalPredictor:
        cfg = {
            "piezo_encoder": {"in_channels": 4, "cnn_channels": [8, 16, 32], "lstm_hidden": 16, "lstm_layers": 1, "attn_heads": 2, "dropout": 0.0},
            "rppg_encoder": {"in_channels": 9, "cnn_channels": [8, 16, 32], "transformer_heads": 2, "transformer_layers": 1, "transformer_ff_dim": 64, "dropout": 0.0},
            "fusion": {"strategy": "cross_attention", "embed_dim": EMBED_DIM, "attn_heads": 2, "dropout": 0.0, "modality_dropout_prob": 0.0},
            "heads": {
                "regression": {"targets": ["hr_bpm", "spo2_pct"], "hidden_dims": [32]},
                "classification": {"sleep_stages": 4, "stress_levels": 3, "hidden_dims": [32]},
            },
        }
        return VitalPredictor.from_config(cfg)

    def test_forward_both_modalities(self, model: VitalPredictor, piezo_input: torch.Tensor, rppg_input: torch.Tensor) -> None:
        with torch.no_grad():
            out = model(piezo=piezo_input, rppg=rppg_input)
        assert out.regression_means.shape == (BATCH, 2)
        assert out.sleep_logits.shape == (BATCH, 4)
        assert out.stress_logits.shape == (BATCH, 3)

    def test_forward_piezo_only(self, model: VitalPredictor, piezo_input: torch.Tensor) -> None:
        with torch.no_grad():
            out = model(piezo=piezo_input, rppg=None)
        assert out.regression_means.shape == (BATCH, 2)

    def test_forward_rppg_only(self, model: VitalPredictor, rppg_input: torch.Tensor) -> None:
        with torch.no_grad():
            out = model(piezo=None, rppg=rppg_input)
        assert out.regression_means.shape == (BATCH, 2)

    def test_to_dict(self, model: VitalPredictor, piezo_input: torch.Tensor) -> None:
        with torch.no_grad():
            out = model(piezo=piezo_input[:1], rppg=None)
        d = out.to_dict()
        assert "hr_bpm" in d
        assert "sleep_stage" in d
        assert "stress_level" in d
        assert isinstance(d["sleep_stage"], str)
