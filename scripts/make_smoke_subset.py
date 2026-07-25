#!/usr/bin/env python3
"""Carve a tiny, category-balanced subset out of a prepared SoM index CSV.

The full curriculum trains on the whole MUSIC-21 index; for a fast end-to-end
sanity run (does the data -> STFT -> mix-and-separate -> backbones -> loss ->
step loop actually work with the real pretrained weights?) you only need a
handful of clips. This reads an existing index (produced by your SoM data prep,
e.g. datasets/processed/train.csv) and writes a small subset with the SAME
columns, spread across as many categories as possible so the 'hetero'
(different-instrument) mix policy has something to mix.

Schema-agnostic: it preserves whatever header the source CSV has (it does not
assume the SoM columns), so it also works on Music-Gesture indices.

Usage (defaults produce a 6-clip train + 4-clip val smoke set):
    python scripts/make_smoke_subset.py \
        --train-index datasets/processed/train.csv \
        --val-index   datasets/processed/val.csv \
        --n 6 --val-n 4 --seed 0
    # (defaults write to datasets/smoke/train.csv & val.csv, which
    #  configs/som_smoke.yaml reads)

Then:
    python train.py --config configs/som_smoke.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import OrderedDict, defaultdict
from typing import List


def _read(path: str):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or (list(rows[0].keys()) if rows else [])
    return rows, fields


def _balanced_pick(rows: List[dict], n: int, seed: int) -> List[dict]:
    """Pick up to n rows, round-robin across the 'category' column when present
    so the subset spans multiple instruments (needed for hetero mixing)."""
    if n <= 0 or not rows:
        return []
    rng = random.Random(seed)
    if "category" not in rows[0]:
        pool = rows[:]
        rng.shuffle(pool)
        return pool[:n]
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get("category", "")].append(r)
    for cat in buckets:
        rng.shuffle(buckets[cat])
    cats = list(buckets.keys())
    rng.shuffle(cats)
    picked, exhausted = [], set()
    while len(picked) < n and len(exhausted) < len(cats):
        for cat in cats:
            if len(picked) >= n:
                break
            if buckets[cat]:
                picked.append(buckets[cat].pop())
            else:
                exhausted.add(cat)
    return picked


def _verify_paths(rows: List[dict]) -> None:
    """Warn about missing referenced files so a smoke run fails loudly, early."""
    path_cols = ["audio_path", "frames_dir", "trajectory_path",
                 "first_frame_path", "pose_path", "context_frame_path"]
    missing = 0
    for r in rows:
        for c in path_cols:
            v = r.get(c, "") or ""
            if v and not os.path.exists(v):
                print(f"  WARNING: missing {c}: {v}")
                missing += 1
    if missing:
        print(f"  ({missing} referenced paths do not exist -- fix your prep or paths)")


def _write(path: str, rows: List[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    cats = OrderedDict()
    for r in rows:
        cats[r.get("category", "?")] = cats.get(r.get("category", "?"), 0) + 1
    print(f"wrote {path}: {len(rows)} clips across {len(cats)} categories "
          f"({dict(cats)})")


def _make(index: str, out: str, n: int, seed: int) -> int:
    if not index or not os.path.isfile(index):
        print(f"skip: index not found: {index}")
        return 0
    rows, fields = _read(index)
    picked = _balanced_pick(rows, n, seed)
    if "category" in (rows[0] if rows else {}) and len({r.get("category") for r in picked}) < 2:
        print("  NOTE: subset has <2 categories -- 'hetero' mixing will fall back "
              "to whatever is available; consider a larger --n or more prep data.")
    _verify_paths(picked)
    _write(out, picked, fields)
    return len(picked)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-index", default="datasets/processed/train.csv")
    ap.add_argument("--val-index", default="datasets/processed/val.csv")
    ap.add_argument("--train-out", default="datasets/smoke/train.csv")
    ap.add_argument("--val-out", default="datasets/smoke/val.csv")
    ap.add_argument("--n", type=int, default=6, help="train clips")
    ap.add_argument("--val-n", type=int, default=4, help="val clips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ntr = _make(args.train_index, args.train_out, args.n, args.seed)
    _make(args.val_index, args.val_out, args.val_n, args.seed + 1)
    if ntr == 0:
        print("\nNo train rows written. Run your SoM data prep first so that "
              f"{args.train_index} exists with the som_dataset schema "
              "(audio_path, frames_dir, trajectory_path, first_frame_path, "
              "category, video_id, shot_id, clip_start_sec).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
