"""PWC-Net optical-flow estimator (Sun et al., CVPR 2018).

The Sound of Motions (SoM, Zhao et al., ICCV 2019) builds its Deep Dense
Trajectory (DDT) on top of a PWC-Net that is *pretrained on Sintel* and kept
(optionally fine-tuned) inside the motion branch. This module provides a
faithful, self-contained PWC-Net that:

  * consumes two RGB frames in [0, 1], NCHW, and returns dense forward flow at
    1/4 input resolution (the native PWC-Net output); the DDT bilinearly
    upsamples it to the trajectory grid;
  * can load an official PyTorch PWC-Net checkpoint via ``load_pretrained``
    (tolerant name remap, with a shape-ordered fallback);
  * is differentiable end-to-end; ``freeze=True`` (default, matching SoM) keeps
    the flow weights fixed so only the audio/motion/appearance heads train.

Architecture: 6-level shared feature pyramid (16..196 ch), cost-volume
correlation with search radius 4, densely-connected optical-flow estimators at
levels 6..2, and a dilated context refinement network at the finest level. It is
pure PyTorch (no custom CUDA correlation), so it runs on CPU for smoke tests.

Fidelity note: a from-scratch module cannot bit-match third-party weight names.
For exact Sintel-quality flow, either (a) drop the official ``pwc-net.pytorch``
module in and point ``DDTConfig.flow_impl='external'`` at it, or (b) load
weights here with ``load_pretrained`` (positional shape match) and keep frozen.
The DDT/appearance/fusion/audio heads do not depend on which flow backend is
used, only on the [B,2,H,W] flow contract.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-level multiplier applied to the flow before it warps the next feature
# level (matches the standard PWC-Net backward-warp scaling for levels 6..2).
_BACKWARP_SCALE = {5: 0.625, 4: 1.25, 3: 2.5, 2: 5.0, 1: 10.0}


def _conv(in_c: int, out_c: int, k: int = 3, s: int = 1, d: int = 1) -> nn.Module:
    pad = ((k - 1) * d) // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, pad, dilation=d),
        nn.LeakyReLU(0.1, inplace=True),
    )


class _FeaturePyramid(nn.Module):
    """6-level shared feature pyramid (channels 16..196, scales 1/2..1/64)."""

    CHS = [16, 32, 64, 96, 128, 196]

    def __init__(self) -> None:
        super().__init__()
        in_c = 3
        self.levels = nn.ModuleList()
        for c in self.CHS:
            self.levels.append(
                nn.Sequential(_conv(in_c, c, 3, 2), _conv(c, c, 3, 1), _conv(c, c, 3, 1))
            )
            in_c = c

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for lvl in self.levels:
            x = lvl(x)
            feats.append(x)
        return feats  # indices 0..5 -> scales 1/2 .. 1/64


def _warp(x: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``x`` toward the reference frame by optical flow ``flo``."""
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=x.dtype),
        torch.arange(W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    vgrid = grid + flo
    vgrid_x = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    vgrid = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(x, vgrid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def _correlation(feat1: torch.Tensor, feat2: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Naive cost volume over a (2r+1)^2 search window (pure PyTorch)."""
    B, C, H, W = feat1.shape
    f2 = F.pad(feat2, [radius] * 4)
    cost = []
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            shifted = f2[:, :, dy:dy + H, dx:dx + W]
            cost.append((feat1 * shifted).mean(dim=1, keepdim=True))
    return F.leaky_relu(torch.cat(cost, dim=1), 0.1)


class _FlowEstimator(nn.Module):
    """Densely-connected optical-flow estimator. Returns (flow, dense_feat)."""

    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.c1 = _conv(in_c, 128)
        self.c2 = _conv(in_c + 128, 128)
        self.c3 = _conv(in_c + 256, 96)
        self.c4 = _conv(in_c + 352, 64)
        self.c5 = _conv(in_c + 416, 32)
        self.feat_c = in_c + 448  # width of the dense feature feeding predict
        self.predict = nn.Conv2d(self.feat_c, 2, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        x = torch.cat([self.c1(x), x], dim=1)
        x = torch.cat([self.c2(x), x], dim=1)
        x = torch.cat([self.c3(x), x], dim=1)
        x = torch.cat([self.c4(x), x], dim=1)
        feat = torch.cat([self.c5(x), x], dim=1)  # [B, feat_c, H, W]
        return self.predict(feat), feat


class _ContextNetwork(nn.Module):
    """Dilated refinement network applied at the finest decoded level."""

    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv(in_c, 128, 3, 1, 1),
            _conv(128, 128, 3, 1, 2),
            _conv(128, 128, 3, 1, 4),
            _conv(128, 96, 3, 1, 8),
            _conv(96, 64, 3, 1, 16),
            _conv(64, 32, 3, 1, 1),
        )
        self.predict = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(self.net(x))


class PWCNet(nn.Module):
    """PWC-Net returning dense forward flow at 1/4 input resolution."""

    def __init__(self, radius: int = 4, freeze: bool = True) -> None:
        super().__init__()
        self.radius = radius
        self.pyramid = _FeaturePyramid()
        chs = _FeaturePyramid.CHS
        corr_c = (2 * radius + 1) ** 2

        self.decoders = nn.ModuleList()
        self.upflow = nn.ModuleList()
        self.upfeat = nn.ModuleList()
        finest_feat_c = None
        # Decode pyramid indices 5 (1/64) down to 1 (1/4).
        for i in range(5, 0, -1):
            in_c = corr_c if i == 5 else corr_c + chs[i] + 2 + 2
            est = _FlowEstimator(in_c)
            self.decoders.append(est)
            if i > 1:
                self.upflow.append(nn.ConvTranspose2d(2, 2, 4, 2, 1))
                self.upfeat.append(nn.ConvTranspose2d(est.feat_c, 2, 4, 2, 1))
            finest_feat_c = est.feat_c
        self.context = _ContextNetwork(finest_feat_c)

        self._frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """frame1, frame2: [B,3,H,W] in [0,1]. Returns flow [B,2,H/4,W/4]."""
        c1 = self.pyramid(frame1)
        c2 = self.pyramid(frame2)
        flow = None
        feat_up = None
        feat = None
        for k, i in enumerate(range(5, 0, -1)):
            f1, f2 = c1[i], c2[i]
            if flow is None:
                corr = _correlation(f1, f2, self.radius)
                dec_in = corr
            else:
                # Transposed-conv upsampling can be off by a pixel versus the
                # next pyramid level when the input size is not a clean power of
                # two (e.g. 48 -> 24 -> 12 -> 6 -> 3, but a stride-2 deconv of a
                # 2-wide map yields 4). Align the upsampled flow/feature to the
                # actual feature-map size before warping and concatenating.
                size = f1.shape[-2:]
                if flow.shape[-2:] != size:
                    flow = F.interpolate(flow, size=size, mode="bilinear",
                                         align_corners=False)
                if feat_up.shape[-2:] != size:
                    feat_up = F.interpolate(feat_up, size=size, mode="bilinear",
                                            align_corners=False)
                warped = _warp(f2, flow * _BACKWARP_SCALE[i])
                corr = _correlation(f1, warped, self.radius)
                dec_in = torch.cat([corr, f1, flow, feat_up], dim=1)
            res_flow, feat = self.decoders[k](dec_in)
            flow = res_flow if flow is None else flow + res_flow
            if i > 1:
                flow = self.upflow[k](flow)
                feat_up = self.upfeat[k](feat)
        flow = flow + self.context(feat)
        return flow

    def load_pretrained(self, path: str, strict: bool = False) -> dict:
        """Load an official PWC-Net checkpoint with tolerant key remapping.

        Accepts a raw ``state_dict`` or a checkpoint carrying ``state_dict`` /
        ``model``. If key names match this module they are loaded directly;
        otherwise weights are copied positionally into shape-compatible tensors
        (enough to seed a strong flow prior when kept frozen). Returns whatever
        ``load_state_dict`` reports so callers can log missing/unexpected keys.
        """
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]
        obj = {k.replace("module.", ""): v for k, v in obj.items()}
        own = self.state_dict()
        if set(obj.keys()) & set(own.keys()):
            return self.load_state_dict(obj, strict=strict)
        remapped, src_items, si = {}, list(obj.items()), 0
        for name, tensor in own.items():
            while si < len(src_items) and src_items[si][1].shape != tensor.shape:
                si += 1
            if si < len(src_items):
                remapped[name] = src_items[si][1]
                si += 1
        return self.load_state_dict(remapped, strict=False)
