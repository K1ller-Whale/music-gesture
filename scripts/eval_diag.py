"""Diagnose separation collapse via SDR + conditioning ablations.

Runs a trained Music Gesture checkpoint over the validation set under four
conditions and reports SDR/SIR/SAR plus predicted-mask statistics:

    full       normal pose + context conditioning
    zero_pose  keypoints zeroed      (measures gesture dependence)
    zero_ctx   context frame zeroed  (measures appearance dependence)
    mask_0.5   ignore the model, apply a constant 0.5 mask (collapse floor)

If the model has genuinely learned to separate using gestures, `full` should
beat `zero_pose`/`zero_ctx`, and every model condition should beat `mask_0.5`.
If `full` ~= `zero_pose` ~= `mask_0.5` and mask_std ~= 0, the model has
collapsed to a near-constant, condition-independent mask.

Masks are applied *softly* here (no 0.7 threshold) so the diagnostic is
sensitive to small conditioning differences and comparable across conditions,
including the constant baseline. This is intentionally different from test.py's
paper-faithful thresholded inference metric.

Usage:
    python scripts/eval_diag.py --config configs/urmp.yaml \\
        --checkpoint runs/exp/ckpt.pth --n 80
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow running as `python scripts/eval_diag.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.music_dataset import MusicMixDataset, collate  # noqa: E402
from models import MusicGesture  # noqa: E402
from models.synthesizer import apply_mask  # noqa: E402
from utils.audio import istft, stft, warp_freq, build_inv_log_freq_matrix  # noqa: E402
from utils.metrics import compute_sdr  # noqa: E402

CONDITIONS = ["full", "zero_pose", "zero_ctx", "mask_0.5"]


def masks_for_condition(model, mix, keypoints, contexts, cond):
    """Return one predicted mask per source for the given ablation condition."""
    if cond == "mask_0.5":
        # Constant, condition-independent mask: the trivial "no separation"
        # baseline that a collapsed model effectively reproduces.
        return [torch.full_like(mix, 0.5) for _ in keypoints]
    if cond == "zero_pose":
        keypoints = [torch.zeros_like(k) for k in keypoints]
    elif cond == "zero_ctx":
        contexts = [torch.zeros_like(ct) for ct in contexts]
    return model(mix, keypoints, contexts)


def reconstruct(mix_wav, mix_mag, mask, phase, cfg, inv_warp=None):
    if inv_warp is not None:
        # Warp the (soft) log-frequency mask back to the linear STFT grid.
        mask = warp_freq(mask, inv_warp)
    est_mag = apply_mask(mix_mag, mask)
    spec = est_mag.squeeze(1) * torch.exp(1j * phase)
    c = cfg["audio"]
    return istft(spec, c["n_fft"], c["hop_length"], c["win_length"],
                 length=mix_wav.shape[-1])


def mask_stats(masks):
    """(mean value, mean per-sample spatial std) over a list of [B,1,F,T] masks.

    A spatial std near 0 means the mask is (near-)constant across the
    spectrogram -- the signature of collapse.
    """
    means, stds = [], []
    for m in masks:
        m = m.detach()
        means.append(m.mean().item())
        stds.append(m.flatten(1).std(dim=1).mean().item())
    return float(np.mean(means)), float(np.mean(stds))


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="SDR + pose/context ablation diagnostics for Music Gesture.")
    parser.add_argument("--config", default="configs/urmp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=80,
                        help="number of validation samples to evaluate")
    parser.add_argument("--min-ref-rms", type=float, default=1e-4,
                        help="skip samples whose quietest reference is below "
                             "this RMS (bss_eval is unstable on near-silence)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_set = MusicMixDataset(cfg["data"]["val_index"], cfg, split="val")
    loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate)

    model = MusicGesture(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    c = cfg["audio"]
    inv_warp = None
    if c.get("log_freq"):
        inv_warp = torch.from_numpy(build_inv_log_freq_matrix(
            c["n_freq"], c["n_log_freq"], c["sample_rate"])).to(device)
    metrics = {cond: {"sdr": [], "sir": [], "sar": []} for cond in CONDITIONS}
    mstats = {cond: {"mean": [], "std": []} for cond in CONDITIONS}
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

        net_input = batch["net_input"].to(device)
        keypoints = [k.to(device) for k in batch["keypoints"]]
        contexts = [ct.to(device) for ct in batch["contexts"]]

        mix_wav = batch["mixture_wav"].to(device)
        mix_spec = stft(mix_wav.squeeze(0), c["n_fft"], c["hop_length"], c["win_length"])
        mix_mag = mix_spec.abs().unsqueeze(0).unsqueeze(0).to(device)
        phase = torch.angle(mix_spec).unsqueeze(0).to(device)

        ref_arr = np.stack(refs)
        for cond in CONDITIONS:
            masks = masks_for_condition(model, net_input, keypoints, contexts, cond)
            if cond != "mask_0.5":
                mean, std = mask_stats(masks)
                mstats[cond]["mean"].append(mean)
                mstats[cond]["std"].append(std)
            ests = [reconstruct(mix_wav, mix_mag, mask, phase, cfg, inv_warp).squeeze(0).cpu().numpy()
                    for mask in masks]
            m = compute_sdr(ref_arr, np.stack(ests))
            for k in metrics[cond]:
                metrics[cond][k].append(m[k])
        evaluated += 1

    # ---- report ----
    print(f"\nEvaluated {evaluated} sample(s); skipped {skipped} near-silent "
          f"(< {args.min_ref_rms} RMS).\n")
    header = (f"{'condition':<10} {'SDR':>7} {'SIR':>7} {'SAR':>7} "
              f"{'mask_mean':>10} {'mask_std':>9}")
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        sdr, sir, sar = (_mean(metrics[cond][k]) for k in ("sdr", "sir", "sar"))
        if cond == "mask_0.5":
            mm, ms = 0.5, 0.0
        else:
            mm, ms = _mean(mstats[cond]["mean"]), _mean(mstats[cond]["std"])
        print(f"{cond:<10} {sdr:>7.3f} {sir:>7.3f} {sar:>7.3f} "
              f"{mm:>10.4f} {ms:>9.4f}")

    # ---- interpretation hints ----
    full_sdr = _mean(metrics["full"]["sdr"])
    zp_sdr = _mean(metrics["zero_pose"]["sdr"])
    zc_sdr = _mean(metrics["zero_ctx"]["sdr"])
    base_sdr = _mean(metrics["mask_0.5"]["sdr"])
    full_std = _mean(mstats["full"]["std"])
    print("\nInterpretation:")
    print(f"  full - zero_pose = {full_sdr - zp_sdr:+.3f} dB  "
          "(>0 => pose conditioning helps)")
    print(f"  full - zero_ctx  = {full_sdr - zc_sdr:+.3f} dB  "
          "(>0 => appearance/context helps)")
    print(f"  full - mask_0.5  = {full_sdr - base_sdr:+.3f} dB  "
          "(>0 => beats a constant mask)")
    print(f"  full mask_std    = {full_std:.4f}  "
          "(~0 => collapsed near-constant mask)")


if __name__ == "__main__":
    main()
