"""Top-level VitalPredictor model: encodes → fuses → predicts all vitals.

This is the single nn.Module that wraps the entire prediction graph:
  PiezoEncoder + RPPGEncoder → CrossAttentionFusion → {RegressionHead,
  SleepStageHead, StressHead}

Usage:
    model = VitalPredictor.from_config(cfg)
    outputs = model(piezo=x_piezo, rppg=x_rppg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from blusim.models.encoders.piezo_encoder import PiezoEncoder
from blusim.models.encoders.rppg_encoder import RPPGEncoder
from blusim.models.fusion.multimodal_fusion import build_fusion
from blusim.models.heads.output_heads import RegressionHead, SleepStageHead, StressHead


@dataclass
class VitalPredictions:
    """Container for all model outputs.

    Attributes:
        regression_means: Shape ``(B, n_regression_targets)``.
        regression_log_vars: Shape ``(B, n_regression_targets)``.
        sleep_logits: Shape ``(B, 4)`` — Wake/Light/Deep/REM.
        stress_logits: Shape ``(B, 3)`` — Low/Medium/High.
        regression_targets: Ordered list of regression target names.
    """

    regression_means: torch.Tensor
    regression_log_vars: torch.Tensor
    sleep_logits: torch.Tensor
    stress_logits: torch.Tensor
    regression_targets: list[str]

    def to_dict(self, apply_softmax: bool = True) -> dict:
        """Convert predictions to a CPU-side Python dict for API responses."""
        import torch.nn.functional as F

        means = self.regression_means.detach().cpu()
        log_vars = self.regression_log_vars.detach().cpu()
        std = (0.5 * log_vars).exp()

        sleep_probs = F.softmax(self.sleep_logits, dim=-1).detach().cpu()
        stress_probs = F.softmax(self.stress_logits, dim=-1).detach().cpu()

        result: dict = {}
        for i, name in enumerate(self.regression_targets):
            result[name] = float(means[0, i])
            result[f"{name}_std"] = float(std[0, i])

        sleep_names = ["Wake", "Light", "Deep", "REM"]
        result["sleep_stage"] = sleep_names[int(sleep_probs[0].argmax())]
        for i, sn in enumerate(sleep_names):
            result[f"sleep_prob_{sn}"] = float(sleep_probs[0, i])

        stress_names = ["Low", "Medium", "High"]
        result["stress_level"] = stress_names[int(stress_probs[0].argmax())]
        for i, sn in enumerate(stress_names):
            result[f"stress_prob_{sn}"] = float(stress_probs[0, i])

        return result


class VitalPredictor(nn.Module):
    """Full multi-modal, multi-task vital prediction model.

    Args:
        piezo_encoder: Configured :class:`PiezoEncoder` instance.
        rppg_encoder: Configured :class:`RPPGEncoder` instance.
        fusion: Fusion module from :func:`build_fusion`.
        regression_head: :class:`RegressionHead`.
        sleep_head: :class:`SleepStageHead`.
        stress_head: :class:`StressHead`.
        embed_dim: Shared embedding dimension.
    """

    def __init__(
        self,
        piezo_encoder: PiezoEncoder,
        rppg_encoder: RPPGEncoder,
        fusion: nn.Module,
        regression_head: RegressionHead,
        sleep_head: SleepStageHead,
        stress_head: StressHead,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.piezo_enc = piezo_encoder
        self.rppg_enc = rppg_encoder
        self.fusion = fusion
        self.reg_head = regression_head
        self.sleep_head = sleep_head
        self.stress_head = stress_head
        self.embed_dim = embed_dim

    def forward(
        self,
        piezo: Optional[torch.Tensor] = None,
        rppg: Optional[torch.Tensor] = None,
        piezo_mask: Optional[torch.Tensor] = None,
        rppg_mask: Optional[torch.Tensor] = None,
    ) -> VitalPredictions:
        """Run the full prediction graph.

        Args:
            piezo: ``(B, n_channels, n_samples)`` or ``None``.
            rppg: ``(B, in_channels, n_frames)`` or ``None``.
            piezo_mask: ``(B,)`` bool — True if piezo missing for that sample.
            rppg_mask: ``(B,)`` bool — True if rPPG missing for that sample.

        Returns:
            :class:`VitalPredictions` with all model outputs.
        """
        piezo_emb = self.piezo_enc(piezo, mask=piezo_mask) if piezo is not None else None
        rppg_emb = self.rppg_enc(rppg, mask=rppg_mask) if rppg is not None else None

        # Fusion — handle all-None case gracefully
        if isinstance(self.fusion, __import__("torch").nn.Module):
            from blusim.models.fusion.multimodal_fusion import CrossAttentionFusion, LateFusion
            if isinstance(self.fusion, CrossAttentionFusion):
                fused = self.fusion(piezo_emb, rppg_emb, piezo_mask, rppg_mask)
            else:
                fused = self.fusion([piezo_emb, rppg_emb], [piezo_mask, rppg_mask])
        else:
            fused = self.fusion([piezo_emb, rppg_emb])

        # Heads
        reg_means, reg_log_vars = self.reg_head(fused)
        sleep_logits = self.sleep_head(fused)
        stress_logits = self.stress_head(fused)

        return VitalPredictions(
            regression_means=reg_means,
            regression_log_vars=reg_log_vars,
            sleep_logits=sleep_logits,
            stress_logits=stress_logits,
            regression_targets=self.reg_head.targets,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "VitalPredictor":
        """Construct a VitalPredictor from a nested config dictionary.

        Args:
            cfg: Nested dict with keys ``piezo_encoder``, ``rppg_encoder``,
                 ``fusion``, ``heads`` (see ``config/model/default.yaml``).

        Returns:
            Initialised :class:`VitalPredictor`.
        """
        mc = cfg.get("model", cfg)  # allow nested or flat config
        embed_dim = mc.get("fusion", {}).get("embed_dim", 256)

        pe_cfg = mc.get("piezo_encoder", {})
        piezo_enc = PiezoEncoder(
            in_channels=pe_cfg.get("in_channels", 4),
            cnn_channels=pe_cfg.get("cnn_channels", [32, 64, 128]),
            cnn_kernel_size=pe_cfg.get("cnn_kernel_size", 7),
            lstm_hidden=pe_cfg.get("lstm_hidden", 128),
            lstm_layers=pe_cfg.get("lstm_layers", 2),
            attn_heads=pe_cfg.get("attn_heads", 4),
            attn_dropout=pe_cfg.get("attn_dropout", 0.1),
            dropout=pe_cfg.get("dropout", 0.2),
            embed_dim=embed_dim,
        )

        rc_cfg = mc.get("rppg_encoder", {})
        rppg_enc = RPPGEncoder(
            in_channels=rc_cfg.get("in_channels", 9),
            cnn_channels=rc_cfg.get("cnn_channels", [32, 64, 128]),
            transformer_heads=rc_cfg.get("transformer_heads", 4),
            transformer_layers=rc_cfg.get("transformer_layers", 2),
            transformer_ff_dim=rc_cfg.get("transformer_ff_dim", 256),
            dropout=rc_cfg.get("dropout", 0.1),
            embed_dim=embed_dim,
        )

        f_cfg = mc.get("fusion", {})
        fusion = build_fusion(
            strategy=f_cfg.get("strategy", "cross_attention"),
            embed_dim=embed_dim,
            n_heads=f_cfg.get("attn_heads", 4),
            output_dim=embed_dim,
            dropout=f_cfg.get("dropout", 0.1),
            modality_dropout_prob=f_cfg.get("modality_dropout_prob", 0.15),
        )

        h_cfg = mc.get("heads", {})
        reg_cfg = h_cfg.get("regression", {})
        reg_head = RegressionHead(
            in_dim=embed_dim,
            targets=reg_cfg.get("targets", None),
            hidden_dims=reg_cfg.get("hidden_dims", [128, 64]),
            dropout=reg_cfg.get("dropout", 0.1),
        )

        cls_cfg = h_cfg.get("classification", {})
        sleep_head = SleepStageHead(
            in_dim=embed_dim,
            n_classes=cls_cfg.get("sleep_stages", 4),
            hidden_dims=cls_cfg.get("hidden_dims", [128, 64]),
            dropout=cls_cfg.get("dropout", 0.1),
        )
        stress_head = StressHead(
            in_dim=embed_dim,
            n_classes=cls_cfg.get("stress_levels", 3),
            hidden_dims=cls_cfg.get("hidden_dims", [128, 64]),
            dropout=cls_cfg.get("dropout", 0.1),
        )

        return cls(
            piezo_encoder=piezo_enc,
            rppg_encoder=rppg_enc,
            fusion=fusion,
            regression_head=reg_head,
            sleep_head=sleep_head,
            stress_head=stress_head,
            embed_dim=embed_dim,
        )
