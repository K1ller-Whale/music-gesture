"""Evaluate a Sound of Motions checkpoint: SDR/SIR/SAR, ablations, per-N,
per-instrument.

Reproduces the SoM result tables:
  * N-source sweep (N = 2, 3, 4 mixtures) via ``--n-sources``;
  * conditioning ablations ``full`` / ``zero_motion`` / ``zero_appearance``
    (and the ``mask_0.5`` collapse floor) -- the SoM analogues of the sibling
    reproduction's zero_pose / zero_ctx diagnostics;
  * per-instrument SDR breakdown (violin, cello, congas, erhu, xylophone, ...).

Inference is paper-faithful: the predicted mask is warped back to the linear
STFT grid and thresholded (binary masks) before reconstruction, matching
test.py. Use --soft to disable thresholding for a collapse-sensitive diagnostic.

Usage:
  python scripts/eval_som.py --config configs/som_paper_faithful.yaml \
      --checkpoint runs/som_music21/last.pth --n-sources 2 3 4 --per-instrument
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_dataset  # noqa: E402
from models import build_model  # noqa: E402
from models.synthesizer import apply_mask  # noqa: E402
from utils.audio import istft, stft, warp_freq, build_inv_log_freq_matrix  # noqa: E402
from utils.metrics import compute_sdr, compute_sdr_per_source  # noqa: E402,F401

CONDITIONS = ["full", "zero_motion", "zero_appearance", "mask_0.5"]


def masks_for_condition(model, net_input, motion, first_frames, cond):
    """Predicted masks per source under an ablation condition.

    [FIX #9] Ablations are now requested from the model, which zeroes the
    branch's FEATURES. Zeroing the inputs here (the old approach) was not a
    true ablation: a zeroed frame still produces non-zero appearance features
    because of conv biases and BatchNorm shifts, and the attention map became a
    constant ~0.5 rather than being switched off. That made
    ``full - zero_appearance`` understate the appearance branch's contribution.
    """
    if cond == "mask_0.5":
        return [torch.full_like(net_input, 0.5) for _ in motion]
    ablate = None
    if cond == "zero_motion":
        ablate = "motion"
    elif cond == "zero_appearance":
        ablate = "appearance"
    return model(net_input, motion, first_frames, ablate=ablate)


def reconstruct(mix_wav, mix_mag, mask, phase, cfg, inv_warp, threshold):
    c = cfg["audio"]
    if inv_warp is not None:
        mask = warp_freq(mask, inv_warp)
    if threshold is not None:
        mask = (mask >= threshold).float()
    est_mag = apply_mask(mix_mag, mask)
    spec = est_mag.squeeze(1) * torch.exp(1j * phase)
    return istft(spec, c["n_fft"], c["hop_length"], c["win_length"],
                 length=mix_wav.shape[-1])


@torch.no_grad()
def evaluate_n(cfg, model, device, n_sources, args):
    cfg = copy.deepcopy(cfg)
    cfg["data"]["num_mix"] = n_sources
    val_set, collate_fn = build_dataset(cfg, cfg["data"]["val_index"], "val")
    loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    c = cfg["audio"]
    inv_warp = None
    if c.get("log_freq"):
        inv_warp = torch.from_numpy(build_inv_log_freq_matrix(
            c["n_freq"], c["n_log_freq"], c["sample_rate"])).to(device)
    threshold = None if args.soft else c.get("mask_threshold", 0.5)

    metrics = {cond: defaultdict(list) for cond in CONDITIONS}
    per_inst = defaultdict(list)   # category -> full-condition SDR
    evaluated = skipped = 0

    for batch in loader:
        if evaluated >= args.n:
            break
        refs = [batch["source_wavs"][i].squeeze(0).numpy()
                for i in range(len(batch["source_wavs"]))]
        ref_rms = [float(np.sqrt(np.mean(r ** 2) + 1e-12)) for r in refs]
        if min(ref_rms) < args.min_ref_rms:
            skipped += 1
            continue
        ref_arr = np.stack(refs)

        net_input = batch["net_input"].to(device)
        motion = [m.to(device) for m in batch["motion"]]
        first_frames = [f.to(device) for f in batch["first_frames"]]

        mix_wav = batch["mixture_wav"].to(device)
        mix_spec = stft(mix_wav.squeeze(0), c["n_fft"], c["hop_length"], c["win_length"])
        mix_mag = mix_spec.abs().unsqueeze(0).unsqueeze(0).to(device)
        phase = torch.angle(mix_spec).unsqueeze(0).to(device)

        for cond in CONDITIONS:
            masks = masks_for_condition(model, net_input, motion, first_frames, cond)
            ests = [reconstruct(mix_wav, mix_mag, mk, phase, cfg, inv_warp, threshold)
                    .squeeze(0).cpu().numpy() for mk in masks]
            # [FIX #8] Per-SOURCE scores. The summary table still reports the
            # mixture-level mean, but the per-instrument breakdown must use each
            # source's own score. Previously the mixture mean was appended to
            # EVERY category in the mixture, so a violin mixed with a tuba
            # credited both with the same value and the per-instrument table
            # could only ever reproduce the overall mean.
            ps = compute_sdr_per_source(ref_arr, np.stack(ests))
            for k in ("sdr", "sir", "sar"):
                metrics[cond][k].append(float(np.mean(ps[k])))
            if cond == "full" and args.per_instrument and "categories" in batch:
                cats = [batch["categories"][i][0] for i in range(len(batch["categories"]))]
                for i, cat in enumerate(cats):
                    if i < len(ps["sdr"]):
                        per_inst[cat].append(float(ps["sdr"][i]))
        evaluated += 1

    print(f"\n===== N = {n_sources} sources =====")
    print(f"Evaluated {evaluated}; skipped {skipped} near-silent "
          f"(< {args.min_ref_rms} RMS). Inference: "
          f"{'soft mask' if args.soft else f'threshold {threshold}'}\n")
    header = f"{'condition':<16} {'SDR':>7} {'SIR':>7} {'SAR':>7}"
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        vals = [float(np.mean(metrics[cond][k])) if metrics[cond][k] else float("nan")
                for k in ("sdr", "sir", "sar")]
        print(f"{cond:<16} {vals[0]:>7.3f} {vals[1]:>7.3f} {vals[2]:>7.3f}")

    f_sdr = np.mean(metrics["full"]["sdr"]) if metrics["full"]["sdr"] else float("nan")
    zm = np.mean(metrics["zero_motion"]["sdr"]) if metrics["zero_motion"]["sdr"] else float("nan")
    za = np.mean(metrics["zero_appearance"]["sdr"]) if metrics["zero_appearance"]["sdr"] else float("nan")
    print("\nAblation deltas (dB):")
    print(f"  full - zero_motion     = {f_sdr - zm:+.3f}  (>0 => motion helps)")
    print(f"  full - zero_appearance = {f_sdr - za:+.3f}  (>0 => appearance helps)")

    if args.per_instrument and per_inst:
        print("\nPer-instrument SDR (full):")
        for cat in sorted(per_inst):
            print(f"  {cat:<16} {np.mean(per_inst[cat]):>7.3f}  (n={len(per_inst[cat])})")


def main():
    ap = argparse.ArgumentParser(description="SoM evaluation: SDR/SIR/SAR + ablations.")
    ap.add_argument("--config", default="configs/som_paper_faithful.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=300, help="validation samples per N")
    ap.add_argument("--n-sources", type=int, nargs="+", default=[2])
    ap.add_argument("--per-instrument", action="store_true")
    ap.add_argument("--soft", action="store_true", help="disable mask thresholding")
    ap.add_argument("--min-ref-rms", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for mixture partner sampling "
                         "(default: experiment.seed from the config)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # [FIX #7] Seed the evaluation. The val dataset picks each mixture's partner
    # sources with random.sample on every __getitem__ (only start_sec was
    # deterministic), so an unseeded eval scored a DIFFERENT set of mixtures on
    # every run. Two back-to-back runs of the same checkpoint differed by
    # 0.33 dB SDR -- 7-13x the ablation deltas being measured, which made small
    # deltas meaningless. Seeding makes runs reproducible and A/B tests valid.
    seed = args.seed if args.seed is not None else \
        cfg.get("experiment", {}).get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[eval] seed {seed}; mix_policy "
          f"{cfg.get('data', {}).get('mix_policy')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    # Structural backbone swaps (external flow/I3D) must happen BEFORE loading
    # the checkpoint, exactly as train.py does, or the state_dict keys won't
    # match: a checkpoint trained with flow_impl/i3d_impl == 'external' stores
    # the official module keys (motion_net.flow_net.net.netExtractor.*,
    # motion_net.i3d.net.Mixed_*) while a freshly built model still has the
    # clean-room ones (motion_net.flow_net.pyramid.*, motion_net.i3d.mixed_*).
    # NOTE: do NOT call load_pretrained_backbones here -- the checkpoint already
    # carries the fine-tuned flow/I3D weights.
    if hasattr(model, "setup_backbones"):
        model.setup_backbones(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for n_sources in args.n_sources:
        evaluate_n(cfg, model, device, n_sources, args)


if __name__ == "__main__":
    main()
