"""Deep Dense Trajectory (DDT) motion branch of Sound of Motions.

Pipeline (SoM Zhao et al., ICCV 2019, Sec. 3.2):

  1. Estimate dense optical flow omega_t between consecutive frames with a
     (Sintel-pretrained) PWC-Net.
  2. Track a regular pixel grid through the flow to obtain dense trajectories:
         G_0   = regular grid of (x, y) pixel coordinates
         G_{t+1} = G_t + grid_sample(omega_t, G_t)
     Stacking the per-step displacements yields a trajectory volume
     [B, 2, T-1, H, W] (the (dx, dy) each tracked point moves at each step).
  3. Feed the trajectory volume to an I3D to extract spatio-temporal motion
     features [B, 1024, T', H', W'] used by the SoM fusion module.

The flow net and I3D are swappable; only the [B, 2, T-1, H, W] trajectory
contract and the I3D feature-map contract matter downstream. Optical flow and
trajectories are the expensive, deterministic part of the pipeline, so callers
should cache them per clip (see scripts/prepare_music21_som.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .i3d import InceptionI3d
from .pwcnet import PWCNet


@dataclass
class DDTConfig:
    freeze_flow: bool = True       # keep PWC-Net fixed (SoM default)
    freeze_i3d: bool = False       # fine-tune I3D motion features
    corr_radius: int = 4
    i3d_dropout: float = 0.5
    traj_grid_stride: int = 1      # subsample the tracked grid for speed (1 = dense)
    flow_chunk_size: int = 8       # frame-pairs batched per flow_net call (see flow_from_frames)
    # [FIX #2] Trajectory displacements are in PIXELS (can be tens of px at
    # 224x224). The I3D stem is inflated from RGB weights trained on ~[-1, 1]
    # inputs, so raw displacements are far out of distribution and saturate the
    # backbone. Divide by this scale, then optionally clamp to [-1, 1].
    traj_norm_scale: float = 20.0
    traj_clamp: bool = True


def _regular_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """Return a [B, 2, H, W] grid of (x, y) pixel coordinates."""
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    return grid.contiguous()


def _sample_flow_at(flow: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``flow`` [B,2,H,W] at ``points`` [B,2,H,W] (pixel xy)."""
    B, _, H, W = flow.shape
    px = 2.0 * points[:, 0] / max(W - 1, 1) - 1.0
    py = 2.0 * points[:, 1] / max(H - 1, 1) - 1.0
    grid = torch.stack((px, py), dim=3)
    return F.grid_sample(flow, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def compute_trajectories(flows: torch.Tensor) -> torch.Tensor:
    """Turn a sequence of dense flows into a trajectory displacement volume.

    flows: [B, T-1, 2, H, W] forward flow for each consecutive frame pair.
    Returns [B, 2, T-1, H, W]: the displacement of each originally-gridded point
    at each step, following the tracked location G_t (SoM trajectory tracking).
    """
    B, Tm1, _, H, W = flows.shape
    device, dtype = flows.device, flows.dtype
    G = _regular_grid(B, H, W, device, dtype)          # [B,2,H,W] current positions
    disps = []
    for t in range(Tm1):
        omega = flows[:, t]                            # [B,2,H,W]
        step = _sample_flow_at(omega, G)               # flow at tracked points
        disps.append(step)
        G = G + step                                   # advance the trajectory
    return torch.stack(disps, dim=2)                   # [B,2,T-1,H,W]


class DDTMotionNet(nn.Module):
    """Full DDT motion branch: frames -> trajectory volume -> I3D features."""

    def __init__(self, cfg: DDTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DDTConfig()
        self.flow_net = PWCNet(radius=self.cfg.corr_radius, freeze=self.cfg.freeze_flow)
        self.i3d = InceptionI3d(in_channels=2, dropout=self.cfg.i3d_dropout)
        self.out_channels = self.i3d.out_channels
        if self.cfg.freeze_i3d:
            for p in self.i3d.parameters():
                p.requires_grad_(False)

    def flow_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] in [0,1] -> flows [B, T-1, 2, H, W] at H,W.

        Frame-pairs are batched in chunks of ``cfg.flow_chunk_size`` instead of
        one Python-loop iteration (and one flow_net kernel launch) per pair.
        There is no cross-pair state in flow estimation itself (only the later
        trajectory-tracking step is sequential), so any batch of independent
        pairs can be pushed through flow_net together. This is the dominant
        real-time/live-inference bottleneck: at batch size 1, a T=24 clip used
        to cost 23 sequential kernel launches; chunking collapses that to
        ceil(23/chunk_size) launches (3 at the default chunk_size=8), while
        chunk_size still bounds peak VRAM (set it to 1 to reproduce the old,
        fully-sequential, lowest-memory behavior; raise it toward T-1 for
        maximum throughput if VRAM allows a fully-batched call).
        """
        B, T, C, H, W = frames.shape
        if T < 2:
            return frames.new_zeros(B, 0, 2, H, W)
        chunk_size = max(1, self.cfg.flow_chunk_size)
        flow_ctx = torch.no_grad() if self.cfg.freeze_flow else _nullctx()
        outs = []
        with flow_ctx:
            for start in range(0, T - 1, chunk_size):
                end = min(start + chunk_size, T - 1)
                n = end - start
                f1 = frames[:, start:end].reshape(B * n, C, H, W)
                f2 = frames[:, start + 1:end + 1].reshape(B * n, C, H, W)
                flow = self.flow_net(f1, f2)                         # [B*n,2,H/4,W/4]
                flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
                outs.append(flow.reshape(B, n, 2, H, W))
        return torch.cat(outs, dim=1)

    def forward(self, frames: torch.Tensor | None = None,
                trajectories: torch.Tensor | None = None) -> torch.Tensor:
        """Return I3D motion features [B, 1024, T', H', W'].

        Provide either raw ``frames`` [B,T,3,H,W] (flow + tracking computed on
        the fly) or precomputed ``trajectories`` [B,2,T-1,H,W] (cached, the fast
        path used in training once flows are on disk).
        """
        if trajectories is None:
            assert frames is not None, "provide frames or trajectories"
            flows = self.flow_from_frames(frames)
            trajectories = compute_trajectories(flows)
        s = self.cfg.traj_grid_stride
        if s > 1:
            trajectories = trajectories[:, :, :, ::s, ::s]
        # [FIX #2] Normalize BEFORE the I3D, on both the on-the-fly and the
        # cached-trajectory path, so training and eval always see the same
        # input scale regardless of which path produced the volume.
        trajectories = self.normalize_trajectories(trajectories)
        return self.i3d.extract_features(trajectories)

    def normalize_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Scale raw pixel displacements into the I3D's expected input range.

        The trajectory volume holds per-step (dx, dy) in PIXELS. The I3D stem is
        inflated from ``rgb_imagenet.pt``, which was pretrained on RGB inputs
        normalized to roughly [-1, 1]. Feeding raw displacements put the
        backbone 10-50x outside its training distribution, saturating it and
        making the motion features uninformative. Dividing by a characteristic
        displacement (``traj_norm_scale``) and clamping restores that range.
        """
        scale = float(self.cfg.traj_norm_scale)
        if scale > 0:
            trajectories = trajectories / scale
        if self.cfg.traj_clamp:
            trajectories = trajectories.clamp(-1.0, 1.0)
        return trajectories


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
