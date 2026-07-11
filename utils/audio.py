"""Audio utilities: STFT / iSTFT, log-frequency warping, masks, mix-and-separate."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def stft(waveform: torch.Tensor, n_fft: int, hop_length: int,
         win_length: int) -> torch.Tensor:
    """Return complex STFT [..., F, T] using a Hann window."""
    window = torch.hann_window(win_length, device=waveform.device)
    return torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=True)


def istft(spec: torch.Tensor, n_fft: int, hop_length: int, win_length: int,
          length: int | None = None) -> torch.Tensor:
    window = torch.hann_window(win_length, device=spec.device)
    return torch.istft(spec, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window, length=length)


def magnitude_phase(spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return spec.abs(), torch.angle(spec)


def log_magnitude(mag: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.log(mag + eps)


def ideal_ratio_mask(source_mag: torch.Tensor,
                     mixture_mag: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Ground-truth ratio mask for Mix-and-Separate supervision."""
    return source_mag / (mixture_mag + eps)


def ideal_binary_mask(source_mag: torch.Tensor,
                      other_mag: torch.Tensor) -> torch.Tensor:
    """1 where this source is at least as loud as the other sources.

    This is the ideal binary mask used as the ground-truth target in Music
    Gesture (Eq. 4): the label is the dominant source at each time-frequency
    bin. Because the two per-source masks are (near-)complementary, the targets
    are balanced ~50/50, so a constant prediction cannot minimise the loss.
    """
    return (source_mag >= other_mag).float()


def mix_and_separate(waveforms: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Given N solo waveforms, return (mixture, sources).

    The mixture is the sum of the sources (standard Mix-and-Separate, as used
    in Sound of Pixels / Music Gesture); each source is the individual
    waveform. This is the self-supervised signal used to train separation.
    """
    stacked = torch.stack(waveforms, dim=0)
    mixture = stacked.sum(dim=0)
    return mixture, waveforms


def build_log_freq_matrix(n_freq: int, n_log_freq: int,
                          sample_rate: int) -> np.ndarray:
    """Linear->log frequency remap matrix (mel-like), shape [n_log_freq, n_freq]."""
    f_max = sample_rate / 2
    lin = np.linspace(0, f_max, n_freq)
    log_pts = np.logspace(np.log10(max(lin[1], 1.0)), np.log10(f_max), n_log_freq)
    mat = np.zeros((n_log_freq, n_freq), dtype=np.float32)
    for i, center in enumerate(log_pts):
        idx = np.argmin(np.abs(lin - center))
        mat[i, idx] = 1.0
    return mat
