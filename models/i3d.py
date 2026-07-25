"""Inflated Inception (I3D) backbone (Carreira & Zisserman, CVPR 2017).

The Sound of Motions motion branch runs an I3D over the Deep Dense Trajectory
volume to extract spatio-temporal motion features. This is the widely-used
InceptionI3d architecture (piergiaj/pytorch-i3d), adapted so that:

  * ``in_channels`` is configurable -- 2 for the (dx, dy) trajectory volume used
    by the DDT, or 3 for raw RGB;
  * ``extract_features`` returns the pre-logit spatio-temporal feature map
    [B, 1024, T', H', W'] that the SoM fusion module consumes (rather than
    classification logits);
  * ``load_pretrained`` can ingest the official ``rgb_imagenet.pt`` weights
    (ImageNet-inflated), inflating / trimming the stem conv if ``in_channels``
    differs from 3.

Pure PyTorch; runs on CPU for smoke tests.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel=(1, 1, 1), stride=(1, 1, 1),
                 use_bn: bool = True, activation=F.relu, use_bias: bool = False) -> None:
        super().__init__()
        self._activation = activation
        pad = tuple(k // 2 for k in kernel)
        self.conv3d = nn.Conv3d(in_c, out_c, kernel, stride, padding=pad, bias=use_bias)
        self.bn = nn.BatchNorm3d(out_c, eps=1e-3, momentum=0.01) if use_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        if self.bn is not None:
            x = self.bn(x)
        if self._activation is not None:
            x = self._activation(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_c: int, out_c) -> None:
        super().__init__()
        self.b0 = Unit3D(in_c, out_c[0], (1, 1, 1))
        self.b1a = Unit3D(in_c, out_c[1], (1, 1, 1))
        self.b1b = Unit3D(out_c[1], out_c[2], (3, 3, 3))
        self.b2a = Unit3D(in_c, out_c[3], (1, 1, 1))
        self.b2b = Unit3D(out_c[3], out_c[4], (3, 3, 3))
        self.b3a = nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1), padding=1)
        self.b3b = Unit3D(in_c, out_c[5], (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """I3D feature extractor. ``extract_features`` -> [B, 1024, T', H', W']."""

    def __init__(self, in_channels: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv3d_1a = Unit3D(in_channels, 64, (7, 7, 7), (2, 2, 2))
        self.maxpool_2a = nn.MaxPool3d((1, 3, 3), (1, 2, 2), padding=(0, 1, 1))
        self.conv3d_2b = Unit3D(64, 64, (1, 1, 1))
        self.conv3d_2c = Unit3D(64, 192, (3, 3, 3))
        self.maxpool_3a = nn.MaxPool3d((1, 3, 3), (1, 2, 2), padding=(0, 1, 1))
        self.mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.maxpool_4a = nn.MaxPool3d((3, 3, 3), (2, 2, 2), padding=(1, 1, 1))
        self.mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.maxpool_5a = nn.MaxPool3d((2, 2, 2), (2, 2, 2), padding=(0, 0, 0))
        self.mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        self.dropout = nn.Dropout(dropout)
        self.out_channels = 1024

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W] -> feature map [B, 1024, T', H', W']."""
        x = self.conv3d_1a(x)
        x = self.maxpool_2a(x)
        x = self.conv3d_2b(x)
        x = self.conv3d_2c(x)
        x = self.maxpool_3a(x)
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool_4a(x)
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)

    def load_pretrained(self, path: str, strict: bool = False) -> dict:
        """Load official I3D weights; adapt the stem conv to ``in_channels``.

        The official ``rgb_imagenet.pt`` uses a 3-channel stem. When this model
        runs on the 2-channel trajectory volume, the stem weights are averaged
        over the input dim and repeated to ``in_channels`` (standard inflation),
        so the ImageNet motion prior is still transferred.
        """
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        obj = {k.replace("module.", ""): v for k, v in obj.items()}
        own = self.state_dict()
        fixed = {}
        for k, v in obj.items():
            if k not in own:
                continue
            tv = own[k]
            if v.shape == tv.shape:
                fixed[k] = v
            elif v.dim() == 5 and tv.dim() == 5 and v.shape[1] == 3 and tv.shape[1] == self.in_channels:
                mean = v.mean(dim=1, keepdim=True)
                fixed[k] = mean.repeat(1, self.in_channels, 1, 1, 1)
        return self.load_state_dict(fixed, strict=strict)
