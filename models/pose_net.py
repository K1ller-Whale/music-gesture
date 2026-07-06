"""Context-aware Graph CNN (CT-GCN) over body + hand keypoints.

Spatial-temporal graph convolution (ST-GCN style) over the human skeleton
(body 18 + 2x hand 21 = 60 joints), modulated by the semantic context feature
through FiLM (feature-wise linear modulation). The output is a temporal sequence
of gesture tokens used by the audio-visual fusion module.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from utils.pose import build_skeleton_adjacency


class GraphConv(nn.Module):
    """Spatial graph convolution: aggregates over normalized adjacency A."""

    def __init__(self, in_c: int, out_c: int, num_subsets: int):
        super().__init__()
        self.num_subsets = num_subsets
        self.conv = nn.Conv2d(in_c, out_c * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V] ; A: [num_subsets, V, V]
        b, c, t, v = x.shape
        x = self.conv(x)
        x = x.view(b, self.num_subsets, -1, t, v)
        # einsum over the partition subsets and joints
        x = torch.einsum("bkctv,kvw->bctw", x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_subsets: int,
                 temporal_kernel: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, num_subsets)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (temporal_kernel, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout, inplace=True),
        )
        if in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, (stride, 1)),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class FiLM(nn.Module):
    """Feature-wise linear modulation from the semantic context vector."""

    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(context_dim, feature_dim)
        self.to_beta = nn.Linear(context_dim, feature_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V] ; context: [B, context_dim]
        gamma = self.to_gamma(context)[:, :, None, None]
        beta = self.to_beta(context)[:, :, None, None]
        return (1 + gamma) * x + beta


class ContextAwareGraphCNN(nn.Module):
    def __init__(self, in_channels: int = 3, graph_layers=(64, 64, 64, 128, 128, 256),
                 temporal_kernel: int = 9, context_dim: int = 512,
                 embed_dim: int = 512, dropout: float = 0.1,
                 body_joints: int = 18, hand_joints: int = 21):
        super().__init__()
        A = build_skeleton_adjacency(body_joints, hand_joints)
        self.register_buffer("A", A)
        num_subsets = A.shape[0]

        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        c_prev = in_channels
        for i, c in enumerate(graph_layers):
            stride = 2 if (i in (2, 4)) else 1
            self.blocks.append(
                STGCNBlock(c_prev, c, num_subsets, temporal_kernel, stride, dropout)
            )
            self.films.append(FiLM(context_dim, c))
            c_prev = c
        self.to_tokens = nn.Conv2d(c_prev, embed_dim, kernel_size=1)

    def forward(self, keypoints: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """keypoints: [B, C, T, V] ; context: [B, context_dim].

        Returns gesture tokens [B, T', embed_dim] (mean-pooled over joints).
        """
        b, c, t, v = keypoints.shape
        x = keypoints.permute(0, 3, 1, 2).contiguous().view(b, v * c, t)
        x = self.data_bn(x).view(b, v, c, t).permute(0, 2, 3, 1).contiguous()
        for block, film in zip(self.blocks, self.films):
            x = block(x, self.A)
            x = film(x, context)
        x = self.to_tokens(x)                 # [B, D, T', V]
        tokens = x.mean(dim=3).transpose(1, 2)  # [B, T', D]
        return tokens
