"""Audio U-Net for spectrogram analysis / synthesis.

Follows the encoder-decoder used across the MIT visual-sound-separation line of
work (Sound of Pixels / Sound of Motions / Music Gesture). The encoder produces
a bottleneck feature that is conditioned on the visual (gesture) stream through
the fusion module; the decoder produces a per-source mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _down_block(in_c: int, out_c: int, norm: bool = True) -> nn.Module:
    layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up_block(in_c: int, out_c: int, dropout: bool = False) -> nn.Module:
    layers = [
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class AudioUNet(nn.Module):
    """U-Net over the (log-)magnitude spectrogram.

    The ``encode`` method returns the bottleneck plus skip connections so the
    fusion module can inject visual conditioning before ``decode``.
    """

    def __init__(self, ngf: int = 64, num_downs: int = 7, input_nc: int = 1,
                 output_nc: int = 1, bottleneck_dim: int = 512):
        super().__init__()
        self.num_downs = num_downs
        self.bottleneck_dim = bottleneck_dim

        chans = [input_nc] + [min(ngf * (2 ** i), 512) for i in range(num_downs)]
        self.downs = nn.ModuleList()
        for i in range(num_downs):
            self.downs.append(_down_block(chans[i], chans[i + 1], norm=(i > 0)))

        # Project bottleneck to the fusion dimension and back.
        self.to_bottleneck = nn.Conv2d(chans[-1], bottleneck_dim, kernel_size=1)
        self.from_bottleneck = nn.Conv2d(bottleneck_dim, chans[-1], kernel_size=1)

        self.ups = nn.ModuleList()
        rev = list(reversed(chans[1:]))  # deepest -> shallowest
        for i in range(num_downs - 1):
            in_c = rev[i] * (2 if i > 0 else 1)
            out_c = rev[i + 1]
            self.ups.append(_up_block(in_c, out_c, dropout=(i < 2)))
        self.final = nn.ConvTranspose2d(rev[-1] * 2, output_nc,
                                        kernel_size=4, stride=2, padding=1)

    def encode(self, spec: torch.Tensor):
        """Return (bottleneck [B, C], skips list)."""
        skips = []
        x = spec
        for down in self.downs:
            x = down(x)
            skips.append(x)
        bottleneck_map = self.to_bottleneck(x)          # [B, D, h, w]
        b, d, h, w = bottleneck_map.shape
        tokens = bottleneck_map.flatten(2).transpose(1, 2)  # [B, h*w, D]
        return tokens, (h, w), skips

    def decode(self, tokens: torch.Tensor, hw, skips) -> torch.Tensor:
        h, w = hw
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, h, w)
        x = self.from_bottleneck(x)
        x = self.ups[0](x)
        for i in range(1, len(self.ups)):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i](x)
        x = torch.cat([x, skips[0]], dim=1)
        return self.final(x)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        tokens, hw, skips = self.encode(spec)
        return self.decode(tokens, hw, skips)
