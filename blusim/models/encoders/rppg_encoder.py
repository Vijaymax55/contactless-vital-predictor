"""rPPG signal encoder: 1D-CNN + Transformer blocks.

Accepts the rPPG waveform (and optionally stacked ROI colour traces)
and produces a fixed-length embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.drop(x)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class RPPGEncoder(nn.Module):
    """Encoder for rPPG waveform data.

    Architecture:
      - 1D-CNN stem for local pattern extraction
      - Positional encoding
      - Stack of Transformer blocks for global temporal modelling
      - CLS-token pooling → projection to embed_dim

    Args:
        in_channels: Input channels (1 for single rPPG, 9 for 3 ROI × RGB).
        cnn_channels: Feature channels per CNN stage.
        transformer_heads: Attention heads per Transformer block.
        transformer_layers: Number of Transformer blocks.
        transformer_ff_dim: Feedforward dimension inside Transformer.
        dropout: Dropout rate.
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 9,
        cnn_channels: list[int] | None = None,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        # CNN stem
        cnn_layers: list[nn.Module] = []
        prev = in_channels
        for ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(prev, ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            prev = ch
        self.cnn = nn.Sequential(*cnn_layers)
        d_model = cnn_channels[-1]

        # Input projection + CLS token
        self.input_proj = nn.Linear(d_model, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_enc = _PositionalEncoding(embed_dim, dropout=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, transformer_heads, transformer_ff_dim, dropout)
             for _ in range(transformer_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode rPPG input.

        Args:
            x: Tensor of shape ``(batch, in_channels, n_frames)``.
            mask: Optional missing-modality boolean mask ``(batch,)``.

        Returns:
            Embedding ``(batch, embed_dim)``.
        """
        # CNN: (B, C, T) → (B, d_cnn, T//pool)
        h = self.cnn(x)
        h = h.permute(0, 2, 1)                       # (B, T', d_cnn)
        h = self.input_proj(h)                        # (B, T', embed_dim)

        # Prepend CLS token
        B = h.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)               # (B, 1+T', embed_dim)

        h = self.pos_enc(h)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        cls_out = h[:, 0]                            # (B, embed_dim) — CLS token
        out = self.proj(cls_out)

        if mask is not None:
            out = out * (~mask).float().unsqueeze(1)

        return out
