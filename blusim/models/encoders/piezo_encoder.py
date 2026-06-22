"""Piezoelectric signal temporal encoder: 1D-CNN → BiLSTM → Self-Attention.

Architecture:
  Input: (batch, n_channels, n_samples)
  ├─ 3 × (Conv1d + BN + ReLU + MaxPool)   → local pattern extraction
  ├─ BiLSTM (2 layers)                     → sequential context
  └─ Multi-head self-attention              → global dependency modelling
  Output: (batch, embed_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.relu(self.bn(self.conv(x))))


class PiezoEncoder(nn.Module):
    """1D-CNN + BiLSTM + Multi-head Self-Attention encoder for BCG signals.

    Args:
        in_channels: Number of piezo sensor channels.
        cnn_channels: Feature maps at each CNN stage.
        cnn_kernel_size: Conv kernel width (same for all layers).
        lstm_hidden: BiLSTM hidden size per direction.
        lstm_layers: Number of stacked LSTM layers.
        attn_heads: Number of self-attention heads.
        attn_dropout: Dropout in attention.
        dropout: General dropout rate.
        embed_dim: Final embedding dimension (projected output).
    """

    def __init__(
        self,
        in_channels: int = 4,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 7,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        dropout: float = 0.2,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        # CNN backbone
        cnn_layers: list[nn.Module] = []
        prev_ch = in_channels
        for out_ch in cnn_channels:
            cnn_layers.append(_CausalConvBlock(prev_ch, out_ch, cnn_kernel_size, dropout))
            cnn_layers.append(nn.MaxPool1d(2))
            prev_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)

        # Projection to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode piezo input to a fixed-length embedding.

        Args:
            x: Tensor of shape ``(batch, n_channels, n_samples)``.
            mask: Optional boolean mask ``(batch,)`` — True means this sample
                  should be zeroed out (missing modality).

        Returns:
            Embedding tensor ``(batch, embed_dim)``.
        """
        # CNN: (B, C, T) → (B, cnn_out_ch, T//pool)
        h = self.cnn(x)

        # LSTM expects (B, T, C)
        h = h.permute(0, 2, 1)
        h, _ = self.lstm(h)                       # (B, T', 2*lstm_hidden)

        # Self-attention with temporal average pooling
        h_attn, _ = self.attn(h, h, h)
        h = self.attn_norm(h + h_attn)            # residual
        h = h.mean(dim=1)                         # (B, 2*lstm_hidden)

        out = self.proj(h)                        # (B, embed_dim)

        # Zero-out missing modality samples
        if mask is not None:
            out = out * (~mask).float().unsqueeze(1)

        return out
