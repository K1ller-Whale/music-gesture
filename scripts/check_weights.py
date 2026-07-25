#!/usr/bin/env python3
"""Report how well the pretrained backbones load into the SoM model.

No dataset or audio needed -- it just builds the model from the config, runs the
same backbone setup/loading path as training, and prints how many tensors each
backbone actually populated. Use this to decide between the built-in loader and
the external-backbone path (som_backends/).

    python scripts/check_weights.py \
        --config configs/som_paper_faithful.yaml \
        --i3d weights/rgb_imagenet.pt \
        --flow weights/network-default.pytorch

Interpretation:
  * loaded == total            -> perfect transfer.
  * loaded >= ~80% of total    -> good; safe to train.
  * loaded 0 or a low fraction -> clean-room name mismatch; prefer the external
    backbone (set *_impl: external in the config, see docs/PRETRAINED_WEIGHTS.md).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--i3d", default=None, help="override model.motion.i3d_weights")
    ap.add_argument("--flow", default=None, help="override model.motion.flow_weights")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    mo = cfg.setdefault("model", {}).setdefault("motion", {})
    if args.i3d:
        mo["i3d_weights"] = args.i3d
    if args.flow:
        mo["flow_weights"] = args.flow

    print(f"i3d_impl={mo.get('i3d_impl', 'builtin')}  i3d_weights={mo.get('i3d_weights')}")
    print(f"flow_impl={mo.get('flow_impl', 'builtin')}  flow_weights={mo.get('flow_weights')}")
    for key in ("i3d_weights", "flow_weights"):
        p = mo.get(key)
        if p and not os.path.isfile(p):
            print(f"  WARNING: {key} path does not exist: {p}")

    model = build_model(cfg)          # CPU is fine; this only constructs + loads
    if hasattr(model, "setup_backbones"):
        model.setup_backbones(cfg)
    report = model.load_pretrained_backbones(cfg) if hasattr(
        model, "load_pretrained_backbones") else {}

    print("\n=== summary ===")
    flow_ext = mo.get("flow_impl", "builtin") == "external"
    i3d_ext = mo.get("i3d_impl", "builtin") == "external"
    if i3d_ext:
        print("  i3d : external official I3D  -> OK (check 'missing 0' above)")
    if flow_ext:
        print("  flow: external official PWC-Net -> OK (self-loaded Sintel weights)")
    if not report and not (flow_ext or i3d_ext):
        print("  no backbone weights were loaded (all unset -- random init).")
    for name, r in report.items():
        frac = r["loaded"] / r["total"] if r["total"] else 0.0
        verdict = "OK" if frac >= 0.8 else ("PARTIAL" if frac > 0 else "FAILED")
        print(f"  {name:5s}: {r['loaded']}/{r['total']} ({frac:5.1%})  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
