#!/usr/bin/env python3
"""Self-contained shape/gradient smoke test for the Sound of Motions model.

Needs ONLY torch + torchvision (no dataset, no audio files, no network -- the
appearance backbone is built with pretrained=False). It builds a small
SoundOfMotions, pushes random tensors through both motion paths, and checks:

  * the cached-trajectory path  (motion = [B, 2, T-1, H, W]) forward + backward;
  * the raw-frames path         (motion = [B, T, 3, H, W], flow computed on the
    fly by PWC-Net) forward;
  * every predicted mask has shape [B, 1, F, T], lies in [0, 1], and is finite;
  * gradients actually flow into the audio U-Net, the FiLM layer, the I3D motion
    net, the appearance net, and the fusion module.

Run on the GPU box the moment torch is installed:

    python scripts/smoke_test_som.py            # CPU is fine
    python scripts/smoke_test_som.py --cuda     # if a GPU is available

Exit code 0 == all checks passed.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model  # noqa: E402


def small_cfg() -> dict:
    """A deliberately tiny SoM config so the test runs on CPU in seconds."""
    return {
        "audio": {"mask_type": "binary"},
        "model": {
            "type": "som",
            "audio": {"ngf": 16, "num_downs": 4, "input_nc": 1, "output_nc": 8,
                      "conv_kernel": 3, "up_kernel": 3, "dilation": 1},
            "motion": {"freeze_flow": True, "freeze_i3d": False, "corr_radius": 4,
                       "i3d_dropout": 0.0, "traj_grid_stride": 1},
            # pretrained=False keeps the test offline / self-contained.
            "appearance": {"backbone": "resnet18", "pretrained": False, "out_dim": 128},
            "fusion": {"dim": 64, "visual_dim": 64},
        },
    }


def _check_masks(masks, B, Fdim, Tdim, tag):
    assert len(masks) == 2, f"{tag}: expected 2 source masks, got {len(masks)}"
    for i, mk in enumerate(masks):
        assert tuple(mk.shape) == (B, 1, Fdim, Tdim), \
            f"{tag}: mask {i} shape {tuple(mk.shape)} != {(B, 1, Fdim, Tdim)}"
        md = mk.detach()
        assert torch.isfinite(md).all(), f"{tag}: mask {i} has non-finite values"
        assert float(md.min()) >= 0.0 and float(md.max()) <= 1.0, \
            f"{tag}: mask {i} outside [0,1] (min {float(md.min())}, max {float(md.max())})"
    print(f"  [ok] {tag}: 2 masks, shape {tuple(masks[0].shape)}, range in [0,1], finite")


def _named_grad_present(model, prefix):
    return any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
              for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    model = build_model(small_cfg()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built SoundOfMotions ({n_params/1e6:.1f}M params)")

    B, Fdim, Tdim = 1, 64, 64          # spectrogram (multiple of 2**num_downs)
    H = W = 48

    # ---- 1. cached-trajectory path: forward + backward ------------------
    print("\n[1] cached-trajectory path (motion = [B,2,T-1,H,W])")
    T_traj = 16                        # >= 6 so the I3D temporal dim survives
    spec = torch.rand(B, 1, Fdim, Tdim, device=device)
    motion = [torch.randn(B, 2, T_traj - 1, H, W, device=device) for _ in range(2)]
    first_frames = [torch.randn(B, 3, H, W, device=device) for _ in range(2)]

    model.train()
    masks = model(spec, motion, first_frames)
    _check_masks(masks, B, Fdim, Tdim, "trajectory forward")

    loss = sum(m.mean() for m in masks)
    model.zero_grad(set_to_none=True)
    loss.backward()
    for prefix, label in [("audio_net", "audio U-Net"), ("film", "FiLM"),
                          ("motion_net.i3d", "I3D motion"),
                          ("appearance_net", "appearance"), ("som_fusion", "fusion"),
                          ("vis_gate_w", "vision-gated head")]:
        ok = _named_grad_present(model, prefix)
        assert ok, f"no gradient reached {label} ({prefix})"
        print(f"  [ok] gradient flows into {label}")
    # PWC-Net is frozen on this path (trajectories are precomputed) -> no grad expected.
    print("  [ok] trajectory backward complete")

    # ---- 2. raw-frames path: forward (flow computed on the fly) ---------
    print("\n[2] raw-frames path (motion = [B,T,3,H,W], PWC-Net flow on the fly)")
    T_frames = 8                       # 7 flow fields -> I3D temporal dim survives
    model.eval()
    with torch.no_grad():
        frames = [torch.rand(B, T_frames, 3, H, W, device=device) for _ in range(2)]
        ff = [torch.rand(B, 3, H, W, device=device) for _ in range(2)]
        masks2 = model(spec, frames, ff)
    _check_masks(masks2, B, Fdim, Tdim, "frames forward")

    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
