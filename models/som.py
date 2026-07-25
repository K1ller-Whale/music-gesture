"""Sound of Motions (SoM, Zhao et al., ICCV 2019) separation model.

Given a mixture spectrogram and, per source, that source's motion input (video
frames or precomputed dense trajectories) and a single appearance frame, predict
one separation mask per source.

Architecture (paper-faithful, config-driven):

  * audio analysis/synthesis U-Net (reused from the repo, models/audio_net.py);
  * DDT motion branch: PWC-Net flow -> dense trajectory volume -> I3D
    (models/ddt.py);
  * appearance branch: ResNet-18 over the first frame (models/som_appearance.py);
  * SoM fusion: appearance-derived spatial attention gates the motion features,
    which are fused and pooled into a per-source visual feature (models/
    som_fusion.py);
  * FiLM conditioning of the audio bottleneck by the visual feature
    (SoM Eq. 2: FiLM(f_s) = gamma(f_v) * f_s + beta(f_v));
  * vision-gated output head (Sound-of-Pixels synthesizer) that also gives the
    visual feature a direct per-pixel path to the mask -- this is the same head
    that fixed collapse in the sibling Music Gesture reproduction, kept here.

Input contract for the per-source motion tensor (auto-detected by shape):
  * frames:       [B, T, 3, H, W]  (flow + tracking computed on the fly)
  * trajectories: [B, 2, T-1, H, W] (cached; the fast training path)
"""
from __future__ import annotations

import importlib
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_net import AudioUNet
from .ddt import DDTConfig, DDTMotionNet
from .som_appearance import AppearanceNet
from .som_fusion import SoMFusion
from .synthesizer import MaskHead


def _load_callable(spec: str):
    """Import a factory given as 'package.module:function'."""
    mod_name, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    return getattr(importlib.import_module(mod_name), attr)


def _report_load(name: str, module: nn.Module, res) -> dict:
    """Print/return how many tensors of ``module`` a load actually populated."""
    total = len(module.state_dict())
    missing = len(getattr(res, "missing_keys", []) or [])
    unexpected = len(getattr(res, "unexpected_keys", []) or [])
    loaded = total - missing
    print(f"[backbone] {name}: loaded {loaded}/{total} tensors "
          f"(missing {missing}, unexpected {unexpected})")
    if loaded == 0:
        print(f"[backbone] WARNING: {name} loaded 0 tensors -- check the path/checkpoint format")
    elif loaded < 0.5 * total:
        print(f"[backbone] WARNING: {name} only loaded {loaded}/{total} tensors -- "
              f"transfer is likely incomplete (clean-room name mismatch?)")
    return {"loaded": loaded, "total": total, "missing": missing, "unexpected": unexpected}


class FiLMConditioner(nn.Module):
    """Feature-wise linear modulation of the audio bottleneck by f_v."""

    def __init__(self, visual_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(visual_dim, feature_dim)
        self.to_beta = nn.Linear(visual_dim, feature_dim)

    def forward(self, tokens: torch.Tensor, f_v: torch.Tensor,
                f_v_time: torch.Tensor | None = None, hw=None) -> torch.Tensor:
        """tokens: [B, N, D]; f_v: [B, visual_dim].

        [FIX #3] When ``f_v_time`` [B, T', Kv] and ``hw`` are supplied, the
        modulation is TIME-VARYING. The bottleneck token grid is [B, h, w, D]
        with h over frequency and w over TIME (audio_net.encode flattens a
        [B, D, h, w] map, and the spectrogram is [B, 1, F, T]). We resample the
        visual feature from its T' steps onto those w bottleneck time steps and
        emit a separate (gamma, beta) per time step, broadcast over frequency.

        Previously only the clip-level mean f_v was used, so every bottleneck
        token got an identical modulation and all information about *when* a
        source moved was destroyed before it could influence the mask -- which
        is exactly the cue SoM needs to separate two same-instrument sources.
        Passing f_v_time=None reproduces the old global behavior.
        """
        if f_v_time is None or hw is None:
            gamma = self.to_gamma(f_v).unsqueeze(1)   # [B, 1, D]
            beta = self.to_beta(f_v).unsqueeze(1)     # [B, 1, D]
            return (1 + gamma) * tokens + beta
        h, w = hw
        B, N, D = tokens.shape
        # Resample the visual time axis T' -> w bottleneck time steps.
        fvt = f_v_time.transpose(1, 2)                          # [B, Kv, T']
        fvt = F.interpolate(fvt, size=w, mode="linear", align_corners=False)
        fvt = fvt.transpose(1, 2)                               # [B, w, Kv]
        gamma = self.to_gamma(fvt).unsqueeze(1)                 # [B, 1, w, D]
        beta = self.to_beta(fvt).unsqueeze(1)                   # [B, 1, w, D]
        grid = tokens.view(B, h, w, D)
        out = (1 + gamma) * grid + beta
        return out.reshape(B, N, D)


class SoundOfMotions(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]
        a = m["audio"]
        dim = m["fusion"]["dim"]

        self.audio_net = AudioUNet(
            ngf=a["ngf"], num_downs=a["num_downs"],
            input_nc=a["input_nc"], output_nc=a["output_nc"],
            bottleneck_dim=dim,
            conv_kernel=a.get("conv_kernel", 4), up_kernel=a.get("up_kernel", 4),
            dilation=a.get("dilation", 1),
        )

        mo = m.get("motion", {})
        ddt_cfg = DDTConfig(
            freeze_flow=mo.get("freeze_flow", True),
            freeze_i3d=mo.get("freeze_i3d", False),
            corr_radius=mo.get("corr_radius", 4),
            i3d_dropout=mo.get("i3d_dropout", 0.5),
            traj_grid_stride=mo.get("traj_grid_stride", 1),
            flow_chunk_size=mo.get("flow_chunk_size", 8),
            # [FIX #2] scale raw pixel displacements into the I3D's input range
            traj_norm_scale=mo.get("traj_norm_scale", 20.0),
            traj_clamp=mo.get("traj_clamp", True),
        )
        self.motion_net = DDTMotionNet(ddt_cfg)

        ap = m.get("appearance", {})
        self.appearance_net = AppearanceNet(
            backbone=ap.get("backbone", "resnet18"),
            pretrained=ap.get("pretrained", True),
            out_dim=ap.get("out_dim", 512),
        )

        fu = m["fusion"]
        self.som_fusion = SoMFusion(
            motion_dim=self.motion_net.out_channels,
            appearance_dim=self.appearance_net.out_dim,
            out_dim=fu.get("visual_dim", dim),
        )
        visual_dim = self.som_fusion.out_dim
        self.film = FiLMConditioner(visual_dim, dim)
        # [FIX #3] route the per-timestep visual feature into FiLM
        self.temporal_film = bool(fu.get("temporal_film", True))
        # [FIX #12] freeze pretrained BN running stats (small-batch training)
        self.freeze_bn = bool(m.get("freeze_bn", True))

        self.mask_head = MaskHead(mask_type=cfg["audio"]["mask_type"])
        self.synth_channels = a["output_nc"]
        self.vis_gate_w = nn.Linear(visual_dim, self.synth_channels)
        self.vis_gate_b = nn.Linear(visual_dim, 1)

    def _motion_features(self, motion: torch.Tensor) -> torch.Tensor:
        """Dispatch on tensor shape: frames [B,T,3,H,W] or traj [B,2,T-1,H,W]."""
        if motion.dim() == 5 and motion.shape[2] == 3:
            return self.motion_net(frames=motion)
        if motion.dim() == 5 and motion.shape[1] == 2:
            return self.motion_net(trajectories=motion)
        raise ValueError(f"unrecognised motion tensor shape {tuple(motion.shape)}")

    def _encode_mixture(self, mixture_spec: torch.Tensor):
        """Pad + run the audio encoder ONCE per mixture.

        Every source in a mixture shares the exact same input spectrogram, but
        the old code called ``audio_net.encode(spec)`` once *per source* from
        inside ``separate_one`` -- i.e. num_mix redundant, identical encoder
        passes. That's pure wasted latency for real-time/live inference, which
        runs at batch size 1 with no cross-source batching to amortize it away.
        Hoisting the encode here makes ``forward`` do exactly one audio-encoder
        pass per mixture regardless of how many sources are separated from it.
        """
        _, _, Fdim, Tdim = mixture_spec.shape
        m = 2 ** self.audio_net.num_downs
        pad_f = (m - Fdim % m) % m
        pad_t = (m - Tdim % m) % m
        spec = mixture_spec
        if pad_f or pad_t:
            spec = F.pad(spec, (0, pad_t, 0, pad_f))
        audio_tokens, hw, skips = self.audio_net.encode(spec)      # [B, N, D]
        return audio_tokens, hw, skips, Fdim, Tdim

    def separate_one(self, audio_tokens: torch.Tensor, hw, skips,
                     Fdim: int, Tdim: int, motion: torch.Tensor,
                     first_frame: torch.Tensor, return_heatmap: bool = False,
                     ablate: str | None = None):
        """``ablate``: None | 'motion' | 'appearance'.

        [FIX #9] Ablations are applied to the FUSED FEATURES, not by zeroing the
        network inputs. A zeroed input frame still yields non-zero appearance
        features (conv biases + BatchNorm shift), so the old input-zeroing
        ablation never actually removed the branch.
        """
        motion_feat = self._motion_features(motion)                # [B, Cm, T', h, w]
        app_map, _ = self.appearance_net(first_frame)              # [B, Ca, ha, wa]
        fusion_out = self.som_fusion(motion_feat, app_map, return_attn=return_heatmap,
                                     ablate_motion=(ablate == "motion"),
                                     ablate_appearance=(ablate == "appearance"))
        if return_heatmap:
            f_v_time, f_v_vec, heatmap = fusion_out                # heatmap: [B,1,Ha,Wa]
        else:
            f_v_time, f_v_vec = fusion_out

        # FiLM (SoM Eq.2). [FIX #3] time-varying when temporal_film is enabled.
        if self.temporal_film:
            fused_tokens = self.film(audio_tokens, f_v_vec, f_v_time=f_v_time, hw=hw)
        else:
            fused_tokens = self.film(audio_tokens, f_v_vec)
        featmap = self.audio_net.decode(fused_tokens, hw, skips)   # [B, K, F, T]
        featmap = featmap[..., :Fdim, :Tdim]

        w = self.vis_gate_w(f_v_vec)                               # [B, K]
        b = self.vis_gate_b(f_v_vec)                               # [B, 1]
        logits = torch.einsum("bkft,bk->bft", featmap, w) + b.unsqueeze(-1)
        mask = self.mask_head(logits.unsqueeze(1))                 # [B, 1, F, T]
        if return_heatmap:
            return mask, heatmap
        return mask

    def forward(self, mixture_spec: torch.Tensor,
                motion: List[torch.Tensor],
                first_frames: List[torch.Tensor],
                return_heatmaps: bool = False,
                ablate: str | None = None):
        """Returns masks (List[Tensor]), or (masks, heatmaps) if return_heatmaps.

        Default behavior/signature is unchanged for existing callers (train.py,
        eval_som.py): with return_heatmaps=False this still just returns the
        List[Tensor] of masks.
        """
        audio_tokens, hw, skips, Fdim, Tdim = self._encode_mixture(mixture_spec)
        masks: List[torch.Tensor] = []
        heatmaps: List[torch.Tensor] = []
        for mo, ff in zip(motion, first_frames):
            out = self.separate_one(audio_tokens, hw, skips, Fdim, Tdim, mo, ff,
                                    return_heatmap=return_heatmaps, ablate=ablate)
            if return_heatmaps:
                mask, hm = out
                masks.append(mask)
                heatmaps.append(hm)
            else:
                masks.append(out)
        if return_heatmaps:
            return masks, heatmaps
        return masks

    # ------------------------------------------------------------------
    # Pretrained backbone handling (config-driven).
    #
    # Two independent concerns:
    #   * setup_backbones()          -- STRUCTURAL. If motion.flow_impl or
    #     motion.i3d_impl == 'external', replace the clean-room PWC-Net / I3D
    #     with an official module returned by a user factory. Must run for every
    #     stage (before loading any checkpoint) so module keys stay consistent.
    #   * load_pretrained_backbones()-- VALUES. Load Sintel / rgb_imagenet
    #     weights into the built-in modules via their load_pretrained(). Only run
    #     on a fresh start; later curriculum stages inherit fine-tuned weights
    #     through the stage checkpoint (init_from).
    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_bn_stats(self, enabled: bool | None = None) -> None:
        """[FIX #12] Put backbone BatchNorm layers into eval mode.

        At batch_size 2 (forced by a 6 GB card) BatchNorm estimates its
        statistics from 2 samples per step. Those estimates are so noisy that
        fine-tuning destroys the pretrained ImageNet / Kinetics running stats
        and destabilizes training. Holding the backbone BN layers in eval mode
        keeps the pretrained statistics and uses them at both train and test
        time -- standard practice for small-batch backbone fine-tuning.

        Affine weights still receive gradients; only the running statistics and
        the use of batch statistics are frozen. Must be re-applied after every
        ``model.train()`` call, since that resets all submodules to train mode.
        """
        if enabled is None:
            enabled = getattr(self, "freeze_bn", False)
        if not enabled:
            return
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.SyncBatchNorm)
        for mod in list(self.appearance_net.modules()) + list(self.motion_net.modules()):
            if isinstance(mod, bn_types):
                mod.eval()

    def setup_backbones(self, cfg: dict) -> None:
        """Swap in external flow / I3D backbones when requested by the config.

        External factory contract (config value 'pkg.module:factory'):
          * flow factory  -> returns an nn.Module whose forward(f1, f2) maps two
            [B,3,H,W] frames in [0,1] to a flow field [B,2,h,w] (any h,w; the
            DDT resizes it to the frame resolution).
          * i3d factory   -> returns an nn.Module exposing
            extract_features(x[B,2,T,H,W]) -> [B, C, T', H', W'] and an
            int attribute out_channels.
        The factory is called with the weights path (or None) as its only arg,
        so it can load its own official checkpoint.
        """
        mo = cfg.get("model", {}).get("motion", {})
        dev = self._device()
        if mo.get("flow_impl") == "external" and mo.get("flow_factory"):
            ctor = _load_callable(mo["flow_factory"])
            # [FIX #14] Propagate freeze_flow to the external backbone. The
            # official PWC wrapper used to hardcode @torch.no_grad() on forward,
            # so `freeze_flow: false` silently did nothing on this path. Older
            # factories that don't accept the kwarg still work (TypeError
            # fallback), and we set the attribute directly when present.
            freeze_flow = mo.get("freeze_flow", True)
            try:
                flow_net = ctor(mo.get("flow_weights"), freeze=freeze_flow)
            except TypeError:
                flow_net = ctor(mo.get("flow_weights"))
                if hasattr(flow_net, "freeze"):
                    flow_net.freeze = bool(freeze_flow)
            self.motion_net.flow_net = flow_net.to(dev)
            if mo.get("freeze_flow", True):
                for p in self.motion_net.flow_net.parameters():
                    p.requires_grad_(False)
            print(f"[backbone] flow: external via {mo['flow_factory']}")
        if mo.get("i3d_impl") == "external" and mo.get("i3d_factory"):
            ctor = _load_callable(mo["i3d_factory"])
            ext = ctor(mo.get("i3d_weights")).to(dev)
            self.motion_net.i3d = ext
            if getattr(ext, "out_channels", None):
                self.motion_net.out_channels = ext.out_channels
            if mo.get("freeze_i3d", False):
                for p in self.motion_net.i3d.parameters():
                    p.requires_grad_(False)
            print(f"[backbone] i3d: external via {mo['i3d_factory']}")

    def load_pretrained_backbones(self, cfg: dict) -> dict:
        """Load Sintel / ImageNet weights into the built-in flow / I3D modules.

        No-op for backbones configured as 'external' (they load their own
        weights in setup_backbones). ResNet-18 appearance weights are handled by
        torchvision at construction (appearance.pretrained). Returns a per-
        backbone report of how many tensors were populated.
        """
        mo = cfg.get("model", {}).get("motion", {})
        report = {}
        i3d_w = mo.get("i3d_weights")
        if i3d_w and mo.get("i3d_impl", "builtin") != "external":
            res = self.motion_net.i3d.load_pretrained(i3d_w)
            report["i3d"] = _report_load("I3D (rgb_imagenet)", self.motion_net.i3d, res)
            if mo.get("freeze_i3d", False):
                for p in self.motion_net.i3d.parameters():
                    p.requires_grad_(False)
        flow_w = mo.get("flow_weights")
        if flow_w and mo.get("flow_impl", "builtin") != "external":
            res = self.motion_net.flow_net.load_pretrained(flow_w)
            report["flow"] = _report_load("PWC-Net (Sintel)", self.motion_net.flow_net, res)
            if mo.get("freeze_flow", True):
                for p in self.motion_net.flow_net.parameters():
                    p.requires_grad_(False)
        flow_ext = mo.get("flow_impl", "builtin") == "external"
        i3d_ext = mo.get("i3d_impl", "builtin") == "external"
        if flow_ext:
            print("[backbone] flow: external (loaded in setup_backbones, PWC-Net Sintel)")
        elif not flow_w:
            print("[backbone] flow: builtin with no weights -- random init "
                  "(NOT paper-faithful)")
        if i3d_ext:
            print("[backbone] i3d: external (loaded in setup_backbones, rgb_imagenet)")
        elif not i3d_w:
            print("[backbone] i3d: builtin with no weights -- random init "
                  "(NOT paper-faithful)")
        return report
