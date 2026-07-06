"""Audio-visual self-attention fusion module.

Concatenates audio bottleneck tokens with per-source gesture tokens and runs a
stack of Transformer encoder layers so audio queries can attend to the gesture
cues of the source to be separated. Returns the refined audio tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class AudioVisualFusion(nn.Module):
    def __init__(self, dim: int = 512, depth: int = 3, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.audio_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.visual_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_tokens: torch.Tensor,
                visual_tokens: torch.Tensor) -> torch.Tensor:
        """audio_tokens: [B, Na, D] ; visual_tokens: [B, Nv, D] -> [B, Na, D]."""
        na = audio_tokens.shape[1]
        a = audio_tokens + self.audio_type
        v = visual_tokens + self.visual_type
        x = torch.cat([a, v], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, :na]  # refined audio tokens only
