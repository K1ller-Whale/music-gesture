"""Adapter: official I3D (piergiaj/pytorch-i3d) as the SoM motion feature net.

Why: the clean-room models/i3d.py cannot faithfully absorb the official
rgb_imagenet.pt (different layer names -> only a partial positional transfer).
This adapter uses piergiaj's *own* InceptionI3d class, so rgb_imagenet.pt loads
exactly, then adapts it to the SoM motion contract.

Usage (configs/som_paper_faithful.yaml):
    model:
      motion:
        i3d_impl: external
        i3d_factory: som_backends.i3d:build_official_i3d
        i3d_weights: weights/rgb_imagenet.pt

Setup:
    git clone https://github.com/piergiaj/pytorch-i3d   # into the project root
    # weights ship in the repo: pytorch-i3d/models/rgb_imagenet.pt

This file auto-adds ./pytorch-i3d to sys.path, so if you cloned it into the
project root (next to train.py) it just works; otherwise put pytorch_i3d.py on
your PYTHONPATH.

Two gotchas handled here:
  1. The official stem is 3-channel (RGB). The DDT feeds a 2-channel (dx, dy)
     trajectory volume, so we load the 3-channel weights and then inflate the
     stem conv to 2 channels (mean over input dim, repeated).
  2. The official extract_features average-pools spatially. SoM fusion wants a
     spatial map [B, C, T', H', W'], so we run the endpoints up to Mixed_5c and
     skip the final avg-pool.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Make a project-root clone importable without extra PYTHONPATH fiddling.
for _cand in (os.path.join(os.getcwd(), "pytorch-i3d"),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pytorch-i3d")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

_VALID = (
    "Conv3d_1a_7x7", "MaxPool3d_2a_3x3", "Conv3d_2b_1x1", "Conv3d_2c_3x3",
    "MaxPool3d_3a_3x3", "Mixed_3b", "Mixed_3c", "MaxPool3d_4a_3x3",
    "Mixed_4b", "Mixed_4c", "Mixed_4d", "Mixed_4e", "Mixed_4f",
    "MaxPool3d_5a_2x2", "Mixed_5b", "Mixed_5c",
)


class OfficialI3D(nn.Module):
    def __init__(self, weights: str | None = None, in_channels: int = 2) -> None:
        super().__init__()
        from pytorch_i3d import InceptionI3d  # type: ignore  # piergiaj/pytorch-i3d
        net = InceptionI3d(num_classes=400, in_channels=3)   # matches rgb_imagenet.pt
        if weights:
            sd = torch.load(weights, map_location="cpu")
            missing, unexpected = net.load_state_dict(sd, strict=False)
            print(f"[backbone] official I3D: missing {len(missing)}, unexpected {len(unexpected)}")
        self._inflate_stem(net, in_channels)
        self.net = net
        self.out_channels = 1024

    @staticmethod
    def _inflate_stem(net, in_channels: int) -> None:
        unit = net.end_points["Conv3d_1a_7x7"] if hasattr(net, "end_points") \
            else net._modules["Conv3d_1a_7x7"]
        conv = unit.conv3d
        if conv.in_channels == in_channels:
            return
        w = conv.weight.data                                 # [out, 3, t, h, w]
        # [FIX #13] Scale-preserving inflation. Taking the mean over the 3 RGB
        # input channels and repeating it across `in_channels` multiplies the
        # layer's total response by (in_channels / old_in): with old_in=3 and
        # in_channels=2 the stem output was 2/3 of the pretrained scale, which
        # shifts every downstream BatchNorm off its pretrained statistics.
        # Rescaling by old_in / in_channels keeps the summed weight -- and
        # therefore the activation scale -- equal to the pretrained network's.
        old_in = w.shape[1]
        neww = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1, 1)
        neww = neww * (float(old_in) / float(in_channels))
        new_conv = nn.Conv3d(in_channels, conv.out_channels, conv.kernel_size,
                             conv.stride, conv.padding, bias=conv.bias is not None)
        new_conv.weight.data = neww
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data
        unit.conv3d = new_conv

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 2, T, H, W] -> spatial feature map [B, 1024, T', H', W']."""
        for name in _VALID:
            x = self.net._modules[name](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)


def build_official_i3d(weights: str | None = None) -> nn.Module:
    return OfficialI3D(weights)
