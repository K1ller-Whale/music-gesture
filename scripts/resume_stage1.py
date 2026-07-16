"""Resume ONLY stage 1 (homo finetune) of the curriculum.

Why this exists
---------------
``train.py``'s ``main()`` always reruns *every* curriculum stage from scratch
and ignores ``--resume`` when ``train.stages`` is set. So once stage 0
(hetero pretrain) has finished, there is no built-in way to start stage 1
without redoing stage 0. This script does exactly that: it initialises the
stage-1 run from the stage-0 checkpoint and trains only stage 1, then surfaces
the final checkpoint at ``<output_dir>/last.pth`` (same as ``main()`` would).

Two modes
---------
1. Start stage 1 from the finished stage-0 weights (the normal case)::

     python scripts/resume_stage1.py --config configs/urmp_paper_faithful.yaml

   This calls ``train_model(..., init_from=<stage0 last.pth>)`` -> loads model
   weights only, fresh optimizer, epoch 0, and runs the full stage-1 schedule.

2. Resume a stage 1 that was itself interrupted partway (a real
   ``stage1_finetune_homo/last.pth`` with epochs already done exists)::

     python scripts/resume_stage1.py --resume-ckpt runs/stage1_finetune_homo/last.pth

   This calls ``train_model(..., resume=<ckpt>)`` -> restores model + optimizer
   + epoch and continues from where it stopped.

The two modes are mutually exclusive; ``--resume-ckpt`` wins if both would
apply.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch
import yaml

# Make the repo root importable when run as ``python scripts/resume_stage1.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train import apply_stage, set_seed, train_model  # noqa: E402


def _stage_dir(base_out: str, index: int, stage: dict) -> str:
    """Reproduce train.main()'s directory convention: ``stage{i}_{name}``."""
    name = stage.get("name", f"stage{index}")
    return os.path.join(base_out, f"stage{index}_{name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume only stage 1 (homo finetune) of the curriculum.")
    parser.add_argument("--config", default="configs/urmp_paper_faithful.yaml",
                        help="Path to the training config (must define train.stages).")
    parser.add_argument("--stage0-ckpt", default=None,
                        help="Stage-0 checkpoint to initialise stage 1 from. "
                             "Defaults to <output_dir>/stage0_<name>/last.pth.")
    parser.add_argument("--resume-ckpt", default=None,
                        help="Resume an interrupted stage-1 run (restores "
                             "model + optimizer + epoch) instead of starting fresh "
                             "from stage 0.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stages = cfg["train"].get("stages")
    if not stages or len(stages) < 2:
        raise SystemExit(
            "This config has no stage 1 to resume (train.stages must have >= 2 "
            "entries).")

    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_out = cfg["experiment"]["output_dir"]
    os.makedirs(base_out, exist_ok=True)

    stage0 = stages[0]
    stage1 = stages[1]
    stage1_cfg = apply_stage(cfg, stage1)
    out_dir = _stage_dir(base_out, 1, stage1)

    print(f"=== resume curriculum stage 1: {stage1.get('name')} "
          f"(mix_policy={stage1_cfg['data'].get('mix_policy')}, "
          f"epochs={stage1_cfg['train']['epochs']}) ===")

    if args.resume_ckpt:
        # Mode 2: continue an interrupted stage-1 run (model + optimizer + epoch).
        if not os.path.isfile(args.resume_ckpt):
            raise SystemExit(f"--resume-ckpt not found: {args.resume_ckpt}")
        print(f"[resume] continuing interrupted stage 1 from {args.resume_ckpt}")
        ckpt = train_model(stage1_cfg, device, out_dir, resume=args.resume_ckpt)
    else:
        # Mode 1: start stage 1 fresh from the finished stage-0 weights.
        stage0_ckpt = args.stage0_ckpt or os.path.join(
            _stage_dir(base_out, 0, stage0), "last.pth")
        if not os.path.isfile(stage0_ckpt):
            raise SystemExit(
                f"stage-0 checkpoint not found: {stage0_ckpt}\n"
                "Run stage 0 first, or pass --stage0-ckpt explicitly.")
        print(f"[init] starting stage 1 from stage-0 weights: {stage0_ckpt}")
        ckpt = train_model(stage1_cfg, device, out_dir, init_from=stage0_ckpt)

    # Surface the final stage-1 checkpoint at the top level, exactly like main().
    if ckpt and os.path.isfile(ckpt):
        final = os.path.join(base_out, "last.pth")
        shutil.copyfile(ckpt, final)
        print(f"stage 1 done: {ckpt}\nfinal checkpoint copied to: {final}")
    else:
        print("stage 1 finished but no checkpoint was produced.")


if __name__ == "__main__":
    main()
