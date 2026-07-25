"""Sound of Motions visual fusion: gate motion by appearance attention.

This is the SoM-specific fusion (NOT the Music Gesture cross-modal transformer).
Given the DDT motion feature map [B, Cm, T', Hm, Wm] and the appearance feature
map [B, Ca, Ha, Wa], it:

  1. derives a spatial attention map a(x) = sigmoid(conv(appearance)) of shape
     [B, 1, H, W] -- *where* the sounding object is;
  2. inflates that attention over time and channels and uses it to gate the
     motion features (suppressing motion off the object);
  3. concatenates the (broadcast) appearance features with the gated motion
     features, fuses them with a 1x1x1 conv, and spatially max-pools, yielding a
     per-time visual feature f_v of shape [B, T', Kv];
  4. also returns a clip-level visual vector (temporal mean) [B, Kv] for the
     FiLM bottleneck conditioning (SoM Eq. 2) and the vision-gated output head.

All spatial mismatches between the motion and appearance grids are resolved by
bilinearly resizing the attention/appearance maps onto the motion grid.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoMFusion(nn.Module):
    def __init__(self, motion_dim: int, appearance_dim: int, out_dim: int = 512) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(appearance_dim, appearance_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(appearance_dim // 4, 1, 1),
        )
        self.fuse = nn.Conv3d(motion_dim + appearance_dim, out_dim, kernel_size=1)
        self.out_dim = out_dim

    def forward(self, motion_feat: torch.Tensor,
                appearance_map: torch.Tensor,
                return_attn: bool = False,
                ablate_motion: bool = False,
                ablate_appearance: bool = False):
        """motion_feat: [B, Cm, T', Hm, Wm]; appearance_map: [B, Ca, Ha, Wa].

        Returns (f_v_time [B, T', out_dim], f_v_vec [B, out_dim]). If
        ``return_attn`` is set, also returns ``attn_raw`` [B, 1, Ha, Wa]: the
        raw-resolution sigmoid attention over the appearance map -- the
        "where is the sounding object" localization heatmap. It depends only
        on the single appearance frame (not on motion or audio), so callers
        that just want a live heatmap overlay can compute it without running
        the rest of separation every frame.
        """
        B, Cm, Tp, Hm, Wm = motion_feat.shape
        # [FIX #9] True ablations, applied to the FEATURES rather than to the
        # network inputs. Zeroing the input frame (what eval_som.py used to do)
        # is not a zero ablation: conv biases and BatchNorm shifts mean a zero
        # image still produces a non-zero appearance map and an attention map of
        # sigmoid(conv(.)) ~ 0.5, so the branch was still contributing. Zeroing
        # here removes the appearance contribution exactly and replaces the gate
        # with a uniform pass-through, which is what "no appearance" means.
        if ablate_motion:
            motion_feat = torch.zeros_like(motion_feat)
        # Resize appearance + attention onto the motion spatial grid.
        app = F.interpolate(appearance_map, size=(Hm, Wm), mode="bilinear",
                            align_corners=False)                    # [B, Ca, Hm, Wm]
        attn_raw = torch.sigmoid(self.attn(appearance_map))        # [B, 1, Ha, Wa]
        attn = F.interpolate(attn_raw, size=(Hm, Wm), mode="bilinear",
                             align_corners=False)                   # [B, 1, Hm, Wm]
        if ablate_appearance:
            app = torch.zeros_like(app)
            attn = torch.ones_like(attn)
        # Inflate over time.
        attn_t = attn.unsqueeze(2).expand(-1, -1, Tp, -1, -1)      # [B,1,T',Hm,Wm]
        app_t = app.unsqueeze(2).expand(-1, -1, Tp, -1, -1)        # [B,Ca,T',Hm,Wm]
        gated_motion = motion_feat * attn_t                        # gate motion by attn
        fused = torch.cat([gated_motion, app_t], dim=1)            # [B,Cm+Ca,T',Hm,Wm]
        fused = self.fuse(fused)                                   # [B,out_dim,T',Hm,Wm]
        # Spatial max-pool -> per-time visual feature.
        f_v_time = F.adaptive_max_pool3d(fused, (Tp, 1, 1)).flatten(2).transpose(1, 2)
        f_v_vec = f_v_time.mean(dim=1)                             # clip-level vector
        if return_attn:
            return f_v_time, f_v_vec, attn_raw
        return f_v_time, f_v_vec
