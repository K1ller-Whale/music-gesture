"""Build train/val index CSVs linking audio, pose, and context frames.

Expected preprocessed layout:
    datasets/processed/audio/<clip>.wav
    datasets/processed/pose/<clip>.npy
    datasets/processed/frames/<clip>.jpg
    datasets/processed/meta.csv   # clip,category

Usage:
    python scripts/prepare_data.py --root datasets/processed --val_ratio 0.1
"""
from __future__ import annotations

import argparse
import csv
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/processed")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    meta_path = os.path.join(args.root, "meta.csv")
    rows = []
    with open(meta_path, newline="") as f:
        for r in csv.DictReader(f):
            clip, category = r["clip"], r["category"]
            rows.append({
                "audio_path": os.path.join(args.root, "audio", f"{clip}.wav"),
                "pose_path": os.path.join(args.root, "pose", f"{clip}.npy"),
                "context_frame_path": os.path.join(args.root, "frames", f"{clip}.jpg"),
                "category": category,
            })

    random.Random(args.seed).shuffle(rows)
    n_val = int(len(rows) * args.val_ratio)
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    fields = ["audio_path", "pose_path", "context_frame_path", "category"]
    for split, data in splits.items():
        out = os.path.join(args.root, f"{split}.csv")
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        print(f"wrote {out}: {len(data)} clips")


if __name__ == "__main__":
    main()
