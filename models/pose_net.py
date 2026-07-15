"""Context-aware Graph CNN (CT-GCN) over body + hand keypoints.

Spatial-temporal graph convolution (ST-GCN style) over the human skeleton
(body 18 + 2x hand 21 = 60 joints), conditioned on the semantic context feature.

Two context-conditioning modes are supported (``context_mode``):

* ``film``  -- feature-wise linear modulation applied after *every* graph block
               (the repo's original, efficient default).
* ``concat``-- the paper-faithful scheme: the semantic context vector is
               broadcast over the (time, joint) grid and concatenated onto the
               node features (Music Gesture, Sec. 3.1: "we concatenated the
               visual appearance context features to each ... node feature").

P2 (context injection): in ``concat`` mode the context is concatenated onto the
*encoded* node features -- after the first ``context_inject_after`` graph blocks
have turned the raw (x, y, conf) coordinates into a learned pose representation --
rather than onto the raw 3-channel input. Concatenating 2048 context channels
directly onto 3 coordinate channels lets the graph net minimise the loss from
appearance alone and ignore the keypoints; encoding first (and optionally
projecting the wide context down via ``context_proj_dim``) keeps the two streams
at a comparable width/scale so gradients actually reach the coordinates. This
matches the paper's wording ("node feature", i.e. the encoded representation)
more literally than a raw-input concat.

The output is a temporal sequence of gesture tokens used by the audio-visual
fusion module.
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
                 body_joints: int = 18, hand_joints: int = 21,
                 context_mode: str = "film", stride_layers=(2, 4),
                 context_inject_after: int = 1, context_proj_dim: int = 0):
        super().__init__()
        if context_mode not in ("film", "concat"):
            raise ValueError(f"context_mode must be 'film' or 'concat', got {context_mode!r}")
        self.context_mode = context_mode
        self.stride_layers = tuple(stride_layers)
        A = build_skeleton_adjacency(body_joints, hand_joints)
        self.register_buffer("A", A)
        num_subsets = A.shape[0]

        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])

        # --- P2: concat-mode context injection controls ---------------------
        # `context_inject_after` = how many graph blocks encode the raw keypoints
        # BEFORE the context is concatenated. 0 reproduces the old raw-input
        # concat; 1-2 encodes coords into node features first (paper-faithful,
        # recommended). `context_proj_dim` optionally projects the wide context
        # (e.g. 2048-d ResNet-50) down before concat so it does not dominate the
        # encoded keypoint channels; 0 keeps the raw context width.
        self.context_inject_after = 0
        self.context_proj = None
        ctx_used = context_dim
        if context_mode == "concat":
            self.context_inject_after = max(0, min(int(context_inject_after),
                                                   len(graph_layers) - 1))
            if context_proj_dim and context_proj_dim > 0 and context_proj_dim != context_dim:
                self.context_proj = nn.Linear(context_dim, context_proj_dim)
                ctx_used = int(context_proj_dim)
        self.context_ctx_used = ctx_used

        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        c_prev = in_channels
        for i, c in enumerate(graph_layers):
            stride = 2 if (i in self.stride_layers) else 1
            in_c = c_prev
            # In concat mode the context is concatenated just before block
            # `context_inject_after`, so that block sees the extra channels.
            if context_mode == "concat" and i == self.context_inject_after:
                in_c = c_prev + ctx_used
            self.blocks.append(
                STGCNBlock(in_c, c, num_subsets, temporal_kernel, stride, dropout)
            )
            if context_mode == "film":
                # One FiLM per block; unused (and not created) in concat mode.
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
        if self.context_mode == "concat":
            # Optionally project the (possibly 2048-d) context down first.
            ctx_vec = self.context_proj(context) if self.context_proj is not None else context
            for i, block in enumerate(self.blocks):
                if i == self.context_inject_after:
                    # Broadcast context over the CURRENT (T', V) grid and concat
                    # onto the *encoded* node features (paper Sec. 3.1). Injected
                    # once, so context is not double-counted per block.
                    ctx = ctx_vec[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
                    x = torch.cat([x, ctx], dim=1)
                x = block(x, self.A)
        else:
            for block, film in zip(self.blocks, self.films):
                x = block(x, self.A)
                x = film(x, context)
        x = self.to_tokens(x)                 # [B, D, T', V]
        tokens = x.mean(dim=3).transpose(1, 2)  # [B, T', D]
        return tokens
