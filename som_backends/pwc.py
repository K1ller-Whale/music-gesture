"""Adapter: official PWC-Net (sniklaus/pytorch-pwc) as the SoM flow backbone.

Usage (configs/som_paper_faithful.yaml):
    model:
      motion:
        flow_impl: external
        flow_factory: som_backends.pwc:build_official_pwc
        flow_weights: weights/network-default.pytorch

Setup:
    git clone https://github.com/sniklaus/pytorch-pwc
    # put the repo on PYTHONPATH (or copy run.py's `Network` into your project)

Weights: the sniklaus repo does NOT ship a weights file -- its ``Network`` self-
downloads ``network-default.pytorch`` via torch.hub on construction (cached to
~/.cache/torch/hub/checkpoints). So ``flow_weights`` can stay null; just have
internet the first time. To pre-cache for an offline box, download it directly:
    curl -L http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch \
         -o weights/network-default.pytorch

IMPORTANT: sniklaus PWC-Net uses a custom cupy CUDA correlation kernel -- it
requires a GPU and `pip install cupy-cuda12x` (match your CUDA) and does NOT run
on CPU. If you need CPU flow, use the built-in models/pwcnet.py instead.

TEMPLATE: verify the import path and class name against the commit you cloned.
The sniklaus repo exposes the network in ``run.py`` as ``Network``.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make a project-root clone importable without extra PYTHONPATH fiddling.
_PWC_DIR = None
for _cand in (os.path.join(os.getcwd(), "pytorch-pwc"),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pytorch-pwc")):
    if os.path.isdir(_cand):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        if _PWC_DIR is None:
            _PWC_DIR = _cand


def _ensure_correlation() -> None:
    """Make `correlation.FunctionCorrelation` resolvable inside sniklaus run.py.

    run.py does a bare ``import correlation`` and calls
    ``correlation.FunctionCorrelation(...)``. With pytorch-pwc/ on sys.path a
    bare ``import correlation`` resolves to the correlation/ DIRECTORY (an
    implicit namespace package) which has no ``FunctionCorrelation`` -> the
    AttributeError you hit. Fix: pre-register the top-level ``correlation``
    module bound to the real file correlation/correlation.py, and alias
    ``correlation.correlation`` to itself so BOTH ``import correlation`` and
    ``from correlation import correlation`` expose the kernel factory.
    """
    existing = sys.modules.get("correlation")
    if existing is not None and hasattr(existing, "FunctionCorrelation"):
        return
    search = ([_PWC_DIR] if _PWC_DIR else []) + [
        p for p in sys.path if os.path.basename(p.rstrip("/\\")) == "pytorch-pwc"
    ]
    for base in search:
        if not base:
            continue
        corr_py = os.path.join(base, "correlation", "correlation.py")
        if os.path.isfile(corr_py):
            spec = importlib.util.spec_from_file_location("correlation", corr_py)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["correlation"] = mod
            spec.loader.exec_module(mod)      # cupy kernels JIT lazily at call time
            setattr(mod, "correlation", mod)  # support `from correlation import correlation`
            return


class OfficialPWC(nn.Module):
    """Wrap the official PWC-Net so forward(f1, f2) -> [B, 2, h, w].

    Gotcha: PWC-Net requires spatial dims divisible by 64. Following the
    official run.py, we RESIZE to the next multiple of 64, run the network,
    resize the flow back to (H, W), and rescale the flow VECTORS by the
    resize ratio (W/Wp, H/Hp) -- a flow field is not resolution-invariant, so
    resampling it without rescaling its magnitudes is wrong.

    (The previous docstring said "pad ... and crop", but the code resizes;
    the comment has been corrected to match the actual implementation.)
    """

    def __init__(self, weights: str | None = None, freeze: bool = True) -> None:
        super().__init__()
        # sniklaus run.py parses sys.argv with getopt AT IMPORT TIME, so importing
        # it while our own CLI flags (e.g. --config) are present raises
        # GetoptError. Neutralize argv during the import, then restore it.
        # Resolve the custom cupy correlation module BEFORE importing run.py.
        _ensure_correlation()
        _saved_argv = sys.argv
        # sniklaus run.py runs `torch.set_grad_enabled(False)` at MODULE level
        # "for performance". Importing it here (during setup_backbones, before
        # training) would globally disable autograd for the whole process, so
        # every subsequent training forward is detached and loss.backward()
        # raises "element 0 ... does not require grad". Save/restore the global
        # grad flag around the import (same pattern as sys.argv). The
        # @torch.no_grad() on forward still keeps the frozen flow grad-free.
        _saved_grad = torch.is_grad_enabled()
        try:
            sys.argv = sys.argv[:1]
            from run import Network  # type: ignore  # sniklaus/pytorch-pwc
        finally:
            sys.argv = _saved_argv
            torch.set_grad_enabled(_saved_grad)
        # Network.__init__ self-loads network-default.pytorch via torch.hub, so
        # construction alone gives you the official Sintel weights.
        self.net = Network()
        if weights:
            # Optional offline override. The raw checkpoint keys start with
            # 'module...'; sniklaus remaps 'module' -> 'net' to match its params.
            sd = torch.load(weights, map_location="cpu")
            sd = {k.replace("module", "net"): v for k, v in sd.items()}
            self.net.load_state_dict(sd, strict=False)
        self.net.eval()
        # [FIX #14] Was a hardcoded @torch.no_grad() on forward, which silently
        # made config `freeze_flow: false` a no-op on the external path -- flow
        # could never be fine-tuned and no error was raised. Now the grad
        # context is chosen from this flag, which models/som.py sets from
        # `model.motion.freeze_flow`.
        self.freeze = bool(freeze)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f1.shape
        Hp = ((H + 63) // 64) * 64
        Wp = ((W + 63) // 64) * 64
        ctx = torch.no_grad() if self.freeze else _nullctx()
        with ctx:
            f1p = F.interpolate(f1, size=(Hp, Wp), mode="bilinear", align_corners=False)
            f2p = F.interpolate(f2, size=(Hp, Wp), mode="bilinear", align_corners=False)
            flow = self.net(f1p, f2p)                  # [B, 2, Hp/?, Wp/?]
            flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
            # [FIX #4] Rescale the flow VECTORS by the resize ratio between the
            # padded resolution the network saw and the output resolution.
            # The old code divided by flow.shape[-1], but that is exactly W
            # after the interpolate above -- so the ratio was always 1.0 and the
            # rescale was a silent no-op. At frame_size 224 -> Hp = Wp = 256,
            # so flow magnitudes were ~14% too large (256/224). Official run.py
            # scales by (original / padded), which is what we do here.
            scale = flow.new_tensor([float(W) / float(Wp),
                                     float(H) / float(Hp)]).view(1, 2, 1, 1)
            flow = flow * scale
        return flow


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def build_official_pwc(weights: str | None = None, freeze: bool = True) -> nn.Module:
    return OfficialPWC(weights, freeze=freeze)
