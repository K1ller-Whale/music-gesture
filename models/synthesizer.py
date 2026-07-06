"""Mask head + spectrogram masking utilities."""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskHead(nn.Module):
    """Maps the decoder output to a separation mask.

    ``ratio`` masks use a sigmoid in [0, 1]; ``binary`` masks use the same
    sigmoid at train time and are thresholded at inference.
    """

    def __init__(self, mask_type: str = "ratio"):
        super().__init__()
        self.mask_type = mask_type
        self.act = nn.Sigmoid()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.act(logits)


def apply_mask(mixture_mag: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Element-wise apply a predicted mask to the mixture magnitude."""
    if mask.shape[-2:] != mixture_mag.shape[-2:]:
        mask = torch.nn.functional.interpolate(
            mask, size=mixture_mag.shape[-2:], mode="bilinear", align_corners=False
        )
    return mixture_mag * mask
