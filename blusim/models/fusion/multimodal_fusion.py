"""Multi-modal fusion strategies for combining piezo and rPPG embeddings.

Three strategies are implemented and can be selected via config:

1. **Late fusion** — concatenate per-modality embeddings, apply MLP.
2. **Early fusion** — concatenate along feature axis before encoding (not here;
   early fusion is handled at the dataset level).
3. **Cross-attention fusion** — modalities attend to each other via
   cross-attention, then the outputs are gated with learned modality weights.

All strategies handle the missing-modality case by replacing the absent
modality's embedding with a learned ``[MISSING]`` token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusion(nn.Module):
    """Concatenate per-modality embeddings → shared MLP.

    Args:
        embed_dim: Dimensionality of each modality embedding.
        n_modalities: Number of input modalities.
        hidden_dim: Hidden size of the fusion MLP.
        output_dim: Output embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_modalities: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.missing_token = nn.ParameterList(
            [nn.Parameter(torch.randn(embed_dim)) for _ in range(n_modalities)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.n_modalities = n_modalities
        self.embed_dim = embed_dim

    def forward(
        self,
        embeddings: list[torch.Tensor | None],
        masks: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Fuse a list of modality embeddings.

        Args:
            embeddings: List of per-modality tensors ``(B, embed_dim)`` or
                        ``None`` if the modality is absent for the whole batch.
            masks: Optional per-modality boolean masks ``(B,)`` where True
                   means that particular sample is missing.

        Returns:
            Fused embedding ``(B, output_dim)``.
        """
        filled: list[torch.Tensor] = []
        batch_size = next(e.size(0) for e in embeddings if e is not None)

        for i, emb in enumerate(embeddings):
            token = self.missing_token[i].unsqueeze(0).expand(batch_size, -1)
            if emb is None:
                filled.append(token)
            elif masks is not None and masks[i] is not None:
                # Per-sample replacement
                m = masks[i].float().unsqueeze(1)          # (B, 1)
                filled.append(emb * (1 - m) + token * m)
            else:
                filled.append(emb)

        concat = torch.cat(filled, dim=-1)                 # (B, embed_dim * n)
        return self.mlp(concat)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion with learnable modality confidence gates.

    Each modality embedding attends to the other's embedding, then
    a gating network (softmax over learned scalars) weights the contributions.

    Args:
        embed_dim: Dimension of modality embeddings (must be equal for all).
        n_heads: Number of cross-attention heads.
        output_dim: Output dimension.
        dropout: Dropout rate.
        modality_dropout_prob: Probability of randomly masking a modality
                               during training (data augmentation for
                               missing-modality robustness).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 4,
        output_dim: int = 256,
        dropout: float = 0.1,
        modality_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_prob = modality_dropout_prob

        # Learned missing-modality tokens (one per modality: 0=piezo, 1=rppg)
        self.missing_piezo = nn.Parameter(torch.randn(embed_dim))
        self.missing_rppg = nn.Parameter(torch.randn(embed_dim))

        # Cross-attention: piezo attends to rPPG and vice versa
        self.cross_attn_p2r = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_r2p = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(embed_dim)
        self.norm_r = nn.LayerNorm(embed_dim)

        # Learned per-sample gate (predicts confidence in each modality)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        piezo_emb: torch.Tensor | None,
        rppg_emb: torch.Tensor | None,
        piezo_mask: torch.Tensor | None = None,
        rppg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse piezo and rPPG embeddings via cross-attention.

        Args:
            piezo_emb: Piezo embedding ``(B, embed_dim)`` or ``None``.
            rppg_emb: rPPG embedding ``(B, embed_dim)`` or ``None``.
            piezo_mask: Boolean mask ``(B,)`` — True if piezo is missing per sample.
            rppg_mask: Boolean mask ``(B,)`` — True if rPPG is missing per sample.

        Returns:
            Fused tensor ``(B, output_dim)``.
        """
        B = (piezo_emb if piezo_emb is not None else rppg_emb).size(0)

        # Replace absent modality with learned missing token
        if piezo_emb is None:
            piezo_emb = self.missing_piezo.unsqueeze(0).expand(B, -1)
        if rppg_emb is None:
            rppg_emb = self.missing_rppg.unsqueeze(0).expand(B, -1)

        # Per-sample replacement
        def _apply_mask(emb: torch.Tensor, mask: torch.Tensor | None, token: nn.Parameter) -> torch.Tensor:
            if mask is None:
                return emb
            t = token.unsqueeze(0).expand(B, -1)
            m = mask.float().unsqueeze(1)
            return emb * (1 - m) + t * m

        piezo_emb = _apply_mask(piezo_emb, piezo_mask, self.missing_piezo)
        rppg_emb = _apply_mask(rppg_emb, rppg_mask, self.missing_rppg)

        # Stochastic modality dropout during training
        if self.training and self.dropout_prob > 0:
            if torch.rand(1).item() < self.dropout_prob:
                piezo_emb = self.missing_piezo.unsqueeze(0).expand(B, -1).detach().clone()
            if torch.rand(1).item() < self.dropout_prob:
                rppg_emb = self.missing_rppg.unsqueeze(0).expand(B, -1).detach().clone()

        # Cross-attention (treat each embedding as a sequence of length 1)
        p = piezo_emb.unsqueeze(1)   # (B, 1, D)
        r = rppg_emb.unsqueeze(1)    # (B, 1, D)

        p_cross, _ = self.cross_attn_p2r(p, r, r)     # piezo queries, rPPG keys/values
        r_cross, _ = self.cross_attn_r2p(r, p, p)     # rPPG queries, piezo keys/values

        p_out = self.norm_p(p + p_cross).squeeze(1)   # (B, D)
        r_out = self.norm_r(r + r_cross).squeeze(1)   # (B, D)

        # Learnable confidence gate
        gate_input = torch.cat([p_out, r_out], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)   # (B, 2)
        gated = gate_weights[:, 0:1] * p_out + gate_weights[:, 1:2] * r_out

        fused = torch.cat([gated, p_out + r_out], dim=-1)         # (B, 2D)
        return self.proj(fused)


def build_fusion(
    strategy: str = "cross_attention",
    embed_dim: int = 256,
    n_heads: int = 4,
    output_dim: int = 256,
    dropout: float = 0.1,
    modality_dropout_prob: float = 0.15,
) -> nn.Module:
    """Factory for fusion modules.

    Args:
        strategy: One of ``"late"``, ``"cross_attention"``.
        embed_dim: Per-modality embedding size.
        n_heads: Attention heads (cross_attention only).
        output_dim: Fused output dimension.
        dropout: Dropout rate.
        modality_dropout_prob: Stochastic modality dropout (cross_attention).

    Returns:
        Fusion module instance.
    """
    if strategy == "late":
        return LateFusion(embed_dim=embed_dim, n_modalities=2, hidden_dim=output_dim, output_dim=output_dim, dropout=dropout)
    elif strategy == "cross_attention":
        return CrossAttentionFusion(embed_dim=embed_dim, n_heads=n_heads, output_dim=output_dim, dropout=dropout, modality_dropout_prob=modality_dropout_prob)
    else:
        raise ValueError(f"Unknown fusion strategy: '{strategy}'. Choose 'late' or 'cross_attention'.")
