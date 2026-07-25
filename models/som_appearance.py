"""Appearance branch of Sound of Motions.

SoM conditions separation not only on motion but on *appearance*: a ResNet-18
over a single (first) frame of the source, kept as a spatial feature map so the
fusion module can derive a per-location attention that says *where* the sounding
object is. This mirrors SoM Sec. 3.3 ("appearance feature ... a single video
frame ... ResNet-18").

Returns both:
  * ``feat_map``  [B, C, h, w] : the pre-pool conv feature map (spatial), and
  * ``feat_vec``  [B, C]       : globally pooled appearance vector,
so downstream code can build a spatial attention gate and/or a clip-level FiLM
conditioning vector.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision


class AppearanceNet(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 out_dim: int = 512) -> None:
        super().__init__()
        net = getattr(torchvision.models, backbone)(pretrained=pretrained)
        # Keep everything up to (and including) the last conv stage; drop
        # avgpool + fc so we retain the [B, C, h, w] spatial map.
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.backbone_dim = net.fc.in_features   # 512 for resnet18
        self.out_dim = out_dim
        if out_dim != self.backbone_dim:
            self.proj = nn.Conv2d(self.backbone_dim, out_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """frame: [B, 3, H, W] -> (feat_map [B, out_dim, h, w], feat_vec [B, out_dim])."""
        x = self.stem(frame)
        x = self.proj(x)
        vec = self.pool(x).flatten(1)
        return x, vec
