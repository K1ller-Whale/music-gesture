"""Run separation on a single real mixture video.

Extracts pose per detected musician, runs the model conditioned on each
musician's gestures, and writes one separated waveform per musician.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from models import MusicGesture
from models.synthesizer import apply_mask
from utils.audio import stft, istft


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="out")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicGesture(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # NOTE: pose extraction + audio loading for an arbitrary video is delegated to
    # scripts/extract_pose.py; here we assume a prepared bundle exists.
    raise SystemExit(
        "Prepare the video with scripts/extract_pose.py, then load the bundle here.\n"
        "This entry point is a template for single-video inference."
    )


if __name__ == "__main__":
    main()
