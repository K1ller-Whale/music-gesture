#!/usr/bin/env python3
"""Fabricate a tiny SYNTHETIC SoM dataset for an end-to-end pipeline smoke test.

This lets you run the *entire* real training loop -- dataset -> STFT / log-freq
warp -> mix-and-separate -> external I3D + PWC-Net -> fusion -> loss.backward ->
SGD step -> checkpoint -- WITHOUT downloading MUSIC-21. It writes a handful of
solo clips in the exact schema datasets/som_dataset.py expects:

    <out>/audio/<clip>.wav          mono PCM, <sr> Hz, <clip_seconds> long
    <out>/frames/<clip>/000001.jpg  a moving-shape stack (non-trivial motion)
    <out>/first_frame/<clip>.jpg    the first frame
    <out>/train.csv , <out>/val.csv  SoM index (audio_path, frames_dir,
        trajectory_path, first_frame_path, category, video_id, shot_id,
        clip_start_sec)

Each category gets a distinct fundamental frequency so mix-and-separate is a
meaningful (learnable) problem -- with a few clips the loss should drop fast.

Usage (defaults match configs/som_smoke.yaml):
    python scripts/make_smoke_data.py --out datasets/smoke --n 6 --val-n 4
    python train.py --config configs/som_smoke.yaml

Requires numpy + opencv (already needed by the dataset). trajectory_path is left
empty so motion_mode=auto feeds raw frames and the flow backbone is exercised;
if you lack cupy/GPU for PWC-Net, set flow_impl: builtin in the config.
"""
from __future__ import annotations

import argparse
import csv
import os
import wave

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

CATEGORIES = ["violin", "flute", "cello", "trumpet", "guitar", "clarinet"]
_SCHEMA = ["audio_path", "frames_dir", "trajectory_path", "first_frame_path",
           "category", "video_id", "shot_id", "clip_start_sec"]


def _write_wav(path: str, sr: int, seconds: float, freq: float, rng) -> None:
    n = int(round(sr * seconds))
    t = np.arange(n) / float(sr)
    x = (0.6 * np.sin(2 * np.pi * freq * t)
         + 0.2 * np.sin(2 * np.pi * 2 * freq * t)
         + 0.1 * np.sin(2 * np.pi * 3 * freq * t))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t))     # slow amplitude wobble
    x = x + 0.01 * rng.standard_normal(n)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _write_frames(frames_dir: str, num_frames: int, size: int, phase: float,
                  color) -> str:
    os.makedirs(frames_dir, exist_ok=True)
    first = None
    for k in range(num_frames):
        img = np.full((size, size, 3), 12, np.uint8)
        ang = 2 * np.pi * (k / max(1, num_frames)) + phase
        cx = int((0.5 + 0.35 * np.sin(ang)) * size)
        cy = int((0.5 + 0.35 * np.cos(ang)) * size)
        r = max(6, size // 8)
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
        cv2.circle(img, (size // 2, size // 2), size // 5, (60, 120, 200), 3)
        p = os.path.join(frames_dir, f"{k + 1:06d}.jpg")
        cv2.imwrite(p, img)
        if first is None:
            first = p
    return first


def _make_clip(idx: int, args, rng):
    clip = f"clip{idx:03d}"
    cat = CATEGORIES[idx % len(CATEGORIES)]
    freq = 110.0 * (1 + (idx % len(CATEGORIES)))          # distinct pitch/category
    color = (int(80 + 20 * (idx % 6)), 160, int(200 - 20 * (idx % 6)))

    audio_dir = os.path.join(args.out, "audio")
    ff_dir = os.path.join(args.out, "first_frame")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(ff_dir, exist_ok=True)

    audio_path = os.path.join(audio_dir, f"{clip}.wav")
    frames_dir = os.path.join(args.out, "frames", clip)
    _write_wav(audio_path, args.sr, args.clip_seconds, freq, rng)
    first = _write_frames(frames_dir, args.num_frames, args.frame_size,
                          phase=idx, color=color)
    ff_path = os.path.join(ff_dir, f"{clip}.jpg")
    cv2.imwrite(ff_path, cv2.imread(first))
    return {
        "audio_path": audio_path,
        "frames_dir": frames_dir,
        "trajectory_path": "",
        "first_frame_path": ff_path,
        "category": cat,
        "video_id": f"vid{idx:03d}",
        "shot_id": "0",
        "clip_start_sec": "0",
    }


def _write_index(path: str, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SCHEMA)
        w.writeheader()
        w.writerows(rows)
    cats = sorted({r["category"] for r in rows})
    print(f"wrote {path}: {len(rows)} clips across {len(cats)} categories {cats}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/smoke")
    ap.add_argument("--n", type=int, default=6, help="train clips")
    ap.add_argument("--val-n", type=int, default=4, help="val clips")
    ap.add_argument("--sr", type=int, default=11025)
    ap.add_argument("--clip-seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num-frames", type=int, default=24)
    ap.add_argument("--frame-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if cv2 is None:
        raise SystemExit("opencv is required: pip install opencv-python")
    if args.n < 2:
        print("NOTE: --n < 2 leaves only one category; hetero mixing needs >=2.")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total = args.n + args.val_n
    rows = [_make_clip(i, args, rng) for i in range(total)]
    _write_index(os.path.join(args.out, "train.csv"), rows[:args.n])
    _write_index(os.path.join(args.out, "val.csv"), rows[args.n:total])
    print(f"\nSynthetic SoM smoke set ready under {args.out}/")
    print("Run:  python train.py --config configs/som_smoke.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
