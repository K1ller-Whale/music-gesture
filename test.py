"""Evaluate a trained Music Gesture model with SDR/SIR/SAR."""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.music_dataset import MusicMixDataset, collate
from models import MusicGesture
from models.synthesizer import apply_mask
from utils.audio import istft
from utils.metrics import compute_sdr


def reconstruct(mix_wav, mix_mag, mask, phase_spec, cfg):
    est_mag = apply_mask(mix_mag, mask)
    spec = est_mag.squeeze(1) * torch.exp(1j * phase_spec)
    c = cfg["audio"]
    return istft(spec, c["n_fft"], c["hop_length"], c["win_length"], length=mix_wav.shape[-1])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
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

    all_metrics = {"sdr": [], "sir": [], "sar": []}
    c = cfg["audio"]
    for batch in loader:
        mix = batch["mixture_mag"].to(device)
        keypoints = [k.to(device) for k in batch["keypoints"]]
        contexts = [ct.to(device) for ct in batch["contexts"]]
        masks = model(mix, keypoints, contexts)

        mix_wav = batch["mixture_wav"]
        from utils.audio import stft
        mix_spec = stft(mix_wav.squeeze(0), c["n_fft"], c["hop_length"], c["win_length"])
        phase = torch.angle(mix_spec).unsqueeze(0).to(device)

        estimates, refs = [], []
        for i, mask in enumerate(masks):
            est = reconstruct(mix_wav.to(device), mix, mask, phase, cfg)
            estimates.append(est.squeeze(0).cpu().numpy())
            refs.append(batch["source_wavs"][i].squeeze(0).numpy())
        m = compute_sdr(np.stack(refs), np.stack(estimates))
        for k in all_metrics:
            all_metrics[k].append(m[k])

    for k, v in all_metrics.items():
        print(f"{k.upper()}: {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
