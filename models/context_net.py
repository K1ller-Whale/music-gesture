"""Semantic context extractor.

The paper conditions the gesture graph network on a semantic *context* feature
of each musician so the model knows *which* instrument the keypoints belong to.
We use a ResNet-50 backbone over a cropped frame of each musician.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ContextNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 feat_dim: int = 512):
        super().__init__()
        net = getattr(torchvision.models, backbone)(pretrained=pretrained)
        modules = list(net.children())[:-1]  # drop the classifier fc
        self.backbone = nn.Sequential(*modules)
        in_dim = net.fc.in_features
        self.proj = nn.Linear(in_dim, feat_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, 3, H, W] crop of the musician -> [B, feat_dim]."""
        feat = self.backbone(frames).flatten(1)
        return self.proj(feat)
