#!/usr/bin/env python3
"""
test_data.py -- Validate the SoM MUSIC-21 preprocessed dataset produced by
scripts/prepare_music21_som.py, against exactly how datasets/som_dataset.py
consumes it (so a clean run here means training/eval won't hit a surprise).

Checks:
  1. CSV integrity: schema (meta_som.csv / train.csv / val.csv), duplicate
     clip ids, train+val row accounting against meta, video-level train/val
     leakage (split_by_video is supposed to guarantee none), and per-row
     field validity (clip_start_sec/shot_id/category/video_id).
  2. Filesystem existence for every row's audio_path / frames_dir /
     first_frame_path / trajectory_path (cheap, runs on ALL rows).
  3. Deep per-file content checks on a sample (or --full dataset):
       - audio: sample rate, duration vs training clip_seconds, mono-ness,
         NaN/Inf, near-silence (via soundfile + numpy)
       - frames: frame count vs model.num_frames, corrupt/unreadable images,
         undersized frames that will be upsampled (via opencv)
       - trajectory: shape [2, T-1, H, W], NaN/Inf, length vs num_frames-1
         (via numpy), only for rows with trajectory_path set
     This mirrors datasets/som_dataset.py's _read_wav / _load_frame_stack /
     _load_trajectory / _load_first_frame exactly.
  4. Dataset composition & mix_policy pairing feasibility: clips/video,
     shots/video, per-category clip/video counts and imbalance, plus
     same_video / homo / hetero pairing feasibility given the pairing rules
     in datasets/som_dataset.py::_sample_others (post [FIX #15]: same_video
     only needs >=2 clips total per video, not >=2 shots).

Expected audio/video parameters (sr, clip_seconds, fps, frame_size,
num_frames, num_mix) are read from the actual training config
(--config, default configs/som_paper_faithful.yaml) so this script checks
against reality, not guessed constants. It falls back to hardcoded defaults
only if PyYAML or the config file is unavailable.

Usage:
    python test_data.py
    python test_data.py --sample 500
    python test_data.py --full --report report.json
    python test_data.py --processed_root datasets/processed --config configs/som_paper_faithful.yaml

Requires (all already in requirements.txt): PyYAML, numpy, soundfile,
opencv-python. Missing packages degrade gracefully -- the corresponding deep
checks are skipped with a warning instead of crashing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import statistics
import sys

try:
    import yaml
except Exception:
    yaml = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import cv2
except Exception:
    cv2 = None


# Exact CSV schemas from scripts/prepare_music21_som.py -- keep in sync if
# that script's schema ever changes.
META_FIELDS = ["clip", "category", "video_id", "shot_id", "clip_start_sec",
               "audio_path", "frames_dir", "first_frame_path", "trajectory_path"]
SOM_FIELDS = ["audio_path", "frames_dir", "trajectory_path", "first_frame_path",
              "category", "video_id", "shot_id", "clip_start_sec"]


class Issues:
    """Collects hard errors and soft warnings without stopping the run."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


# ---------------------------------------------------------------- loading
def read_csv_rows(path, expected_fields, issues, label):
    if not os.path.exists(path):
        issues.error(f"{label}: file not found at {path}")
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        got_fields = reader.fieldnames or []
        if got_fields != expected_fields:
            issues.error(
                f"{label}: header mismatch.\n    expected: {expected_fields}\n    got:      {got_fields}")
        rows = list(reader)
    if not rows:
        issues.error(f"{label}: no rows found")
    return rows


def load_config(config_path, issues):
    """Best-effort load of sr/clip_seconds/fps/frame_size/num_frames/num_mix
    from the real training config (same keys datasets/som_dataset.py reads),
    so expectations match reality instead of guessed constants."""
    defaults = {"sr": 11025, "clip_seconds": 3.0, "fps": 8, "frame_size": 224,
                "num_frames": 24, "num_mix": 2}
    if yaml is None:
        issues.warn(f"PyYAML not available -- using hardcoded fallback expectations: {defaults}")
        return defaults, None
    if not os.path.exists(config_path):
        issues.warn(f"config not found at {config_path} -- using hardcoded fallback: {defaults}")
        return defaults, None
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    try:
        expect = {
            "sr": int(cfg["audio"]["sample_rate"]),
            "clip_seconds": float(cfg["audio"]["clip_seconds"]),
            "fps": int(cfg["video"]["fps"]),
            "frame_size": int(cfg["video"]["frame_size"]),
            "num_frames": int(cfg["video"]["num_frames"]),
            "num_mix": int(cfg["data"]["num_mix"]),
        }
    except Exception as e:
        issues.warn(f"could not read expected keys from {config_path} "
                    f"({type(e).__name__}: {e}) -- using hardcoded fallback: {defaults}")
        return defaults, cfg
    return expect, cfg


# ---------------------------------------------------------------- 1. CSV integrity
def check_csv_integrity(meta_rows, train_rows, val_rows, issues):
    print("\n=== 1. CSV integrity ===")
    clip_ids = [r["clip"] for r in meta_rows]
    dupes = [c for c, n in collections.Counter(clip_ids).items() if n > 1]
    if dupes:
        issues.error(f"{len(dupes)} duplicate clip id(s) in meta_som.csv, e.g. {dupes[:5]}")
    else:
        print(f"no duplicate clip ids ({len(clip_ids)} total)")

    n_meta, n_train, n_val = len(meta_rows), len(train_rows), len(val_rows)
    if n_train + n_val != n_meta:
        issues.error(
            f"train.csv + val.csv ({n_train}+{n_val}={n_train + n_val}) != meta_som.csv ({n_meta})")
    else:
        print(f"row accounting OK: train {n_train} + val {n_val} = meta {n_meta}")

    train_vids = {r["video_id"] for r in train_rows}
    val_vids = {r["video_id"] for r in val_rows}
    leaked = train_vids & val_vids
    if leaked:
        issues.error(
            f"{len(leaked)} video_id(s) leak across train/val (breaks the no-leakage guarantee "
            f"split_by_video is supposed to provide), e.g. {list(leaked)[:5]}")
    else:
        print(f"no train/val video_id leakage ({len(train_vids)} train videos, "
              f"{len(val_vids)} val videos)")

    for r in meta_rows:
        clip = r.get("clip", "?")
        try:
            float(r["clip_start_sec"])
        except Exception:
            issues.error(f"{clip}: clip_start_sec not a float: {r.get('clip_start_sec')!r}")
        try:
            int(r["shot_id"])
        except Exception:
            issues.error(f"{clip}: shot_id not an int: {r.get('shot_id')!r}")
        if not r.get("category"):
            issues.error(f"{clip}: empty category")
        if not r.get("video_id"):
            issues.error(f"{clip}: empty video_id")


# ---------------------------------------------------------------- 2. existence checks
def check_file_existence(rows, issues):
    print("\n=== 2. File existence (all rows) ===")
    missing_audio = missing_frames_dir = empty_frames_dir = missing_first_frame = 0
    missing_traj = has_traj = 0
    for r in rows:
        clip = r.get("clip", "?")
        apath = r.get("audio_path", "")
        if not apath or not os.path.exists(apath):
            missing_audio += 1
            issues.error(f"{clip}: audio missing at {apath}")
        elif os.path.getsize(apath) == 0:
            missing_audio += 1
            issues.error(f"{clip}: audio file is empty (0 bytes) at {apath}")

        fdir = r.get("frames_dir", "")
        if not fdir or not os.path.isdir(fdir):
            missing_frames_dir += 1
            issues.error(f"{clip}: frames_dir missing at {fdir}")
        else:
            n_imgs = sum(1 for fn in os.listdir(fdir) if fn.lower().endswith((".jpg", ".jpeg", ".png")))
            if n_imgs == 0:
                empty_frames_dir += 1
                issues.error(
                    f"{clip}: frames_dir has 0 images at {fdir} -- "
                    f"datasets/som_dataset.py._load_frame_stack silently returns an all-zero "
                    f"motion tensor for this case, no exception raised")

        ffp = r.get("first_frame_path", "")
        if ffp and not os.path.exists(ffp):
            missing_first_frame += 1
            issues.error(f"{clip}: first_frame missing at {ffp}")

        tpath = r.get("trajectory_path", "") or ""
        if tpath:
            has_traj += 1
            if not os.path.exists(tpath):
                missing_traj += 1
                issues.error(f"{clip}: trajectory_path set but file missing at {tpath}")

    print(f"checked {len(rows)} rows: {missing_audio} missing/empty audio, "
          f"{missing_frames_dir} missing frame dirs, {empty_frames_dir} empty frame dirs, "
          f"{missing_first_frame} missing first frames")
    print(f"trajectory_path set on {has_traj}/{len(rows)} rows "
          f"({'--cache_trajectories was used' if has_traj else 'on-the-fly flow only'}), "
          f"{missing_traj} of those missing on disk")


# ---------------------------------------------------------------- 3. deep content checks
def check_audio_deep(row, expect, issues):
    path = row.get("audio_path", "")
    if sf is None or not path or not os.path.exists(path):
        return
    clip = row.get("clip", "?")
    try:
        wav, sr = sf.read(path, dtype="float32")
    except Exception as e:
        issues.error(f"{clip}: soundfile failed to read {path} ({type(e).__name__}: {e})")
        return
    if sr != expect["sr"]:
        issues.error(f"{clip}: sample rate {sr} != expected {expect['sr']} ({path})")
    if wav.ndim > 1:
        issues.warn(f"{clip}: audio is not mono (shape {wav.shape}) -- _read_wav will average channels")
    mono = wav if wav.ndim == 1 else wav.mean(axis=1)
    dur = len(mono) / float(sr) if sr else 0.0
    if dur + 1e-3 < expect["clip_seconds"]:
        issues.warn(
            f"{clip}: audio duration {dur:.2f}s < training clip_seconds {expect['clip_seconds']}s "
            f"-- _read_wav will zero-pad this clip every time it's sampled")
    if np is not None and len(mono):
        if not np.isfinite(mono).all():
            issues.error(f"{clip}: audio contains NaN/Inf ({path})")
        rms = float(np.sqrt(np.mean(mono.astype("float64") ** 2)))
        if rms < 1e-5:
            issues.warn(f"{clip}: audio RMS {rms:.2e} looks silent/near-zero ({path})")


def check_frames_deep(row, expect, issues):
    fdir = row.get("frames_dir", "")
    if not fdir or not os.path.isdir(fdir):
        return
    clip = row.get("clip", "?")
    files = sorted(fn for fn in os.listdir(fdir) if fn.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        return
    if len(files) < expect["num_frames"]:
        issues.warn(
            f"{clip}: only {len(files)} extracted frame(s), fewer than model.num_frames="
            f"{expect['num_frames']} -- _load_frame_stack will clamp to the last frame to fill "
            f"the window")
    if cv2 is None:
        return
    for fn in dict.fromkeys([files[0], files[-1]]):  # dedupe if only 1 file
        img = cv2.imread(os.path.join(fdir, fn))
        if img is None:
            issues.error(f"{clip}: unreadable/corrupt frame {fn} in {fdir}")
        elif min(img.shape[0], img.shape[1]) < expect["frame_size"]:
            issues.warn(
                f"{clip}: frame {fn} is {img.shape[1]}x{img.shape[0]}, smaller than "
                f"model.frame_size={expect['frame_size']} -- will be upsampled at load time")


def check_first_frame_deep(row, issues):
    path = row.get("first_frame_path", "")
    if not path or not os.path.exists(path) or cv2 is None:
        return
    img = cv2.imread(path)
    if img is None:
        issues.error(f"{row.get('clip', '?')}: unreadable/corrupt first_frame at {path}")


def check_trajectory_deep(row, expect, issues):
    path = row.get("trajectory_path", "") or ""
    if not path or not os.path.exists(path) or np is None:
        return
    clip = row.get("clip", "?")
    try:
        traj = np.load(path)
    except Exception as e:
        issues.error(f"{clip}: np.load failed on {path} ({type(e).__name__}: {e})")
        return
    if traj.ndim != 4 or traj.shape[0] != 2:
        issues.error(f"{clip}: trajectory shape {traj.shape} != [2, T-1, H, W] ({path})")
        return
    if not np.isfinite(traj).all():
        issues.error(f"{clip}: trajectory contains NaN/Inf ({path})")
    want = expect["num_frames"] - 1
    if traj.shape[1] < want:
        issues.warn(
            f"{clip}: trajectory has {traj.shape[1]} frame(s), fewer than num_frames-1={want} "
            f"-- _load_trajectory will pad with edge-repeat")


def check_deep_sample(rows, expect, issues, sample_size, seed, full):
    label = "all rows (--full)" if full else f"a random sample of {min(sample_size, len(rows))}"
    print(f"\n=== 3. Deep per-file checks ({label}) ===")
    missing_libs = [name for name, mod in
                    (("soundfile", sf), ("opencv-python", cv2), ("numpy", np)) if mod is None]
    if missing_libs:
        issues.warn(f"skipping some deep checks -- missing packages: {', '.join(missing_libs)} "
                    f"(pip install -r requirements.txt to enable full checks)")
    sampled = rows if full else random.Random(seed).sample(rows, min(sample_size, len(rows)))
    for i, row in enumerate(sampled):
        check_audio_deep(row, expect, issues)
        check_frames_deep(row, expect, issues)
        check_first_frame_deep(row, issues)
        check_trajectory_deep(row, expect, issues)
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(sampled)} checked")
    print(f"deep-checked {len(sampled)}/{len(rows)} row(s)")


# ---------------------------------------------------------------- 4. composition stats
def check_composition(meta_rows, expect, issues):
    print("\n=== 4. Dataset composition & pairing feasibility ===")
    by_video_clips = collections.defaultdict(list)
    by_video_shots = collections.defaultdict(set)
    by_category = collections.defaultdict(list)
    video_to_cat = {}
    for r in meta_rows:
        by_video_clips[r["video_id"]].append(r["clip"])
        by_video_shots[r["video_id"]].add(r["shot_id"])
        by_category[r["category"]].append(r["clip"])
        video_to_cat[r["video_id"]] = r["category"]

    clip_counts = [len(v) for v in by_video_clips.values()]
    shot_counts = [len(v) for v in by_video_shots.values()]
    n_videos = len(by_video_clips)
    print(f"videos: {n_videos}, clips: {len(meta_rows)}, categories: {len(by_category)}")
    if expect.get("paper_categories") and len(by_category) < expect["paper_categories"]:
        issues.warn(f"only {len(by_category)}/{expect['paper_categories']} MUSIC-21 categories present")
    print(f"avg clips/video: {statistics.mean(clip_counts):.2f} "
          f"(min {min(clip_counts)}, max {max(clip_counts)})")
    print(f"avg distinct shots/video: {statistics.mean(shot_counts):.2f}")
    single_shot = sum(1 for c in shot_counts if c == 1)
    print(f"videos with only 1 detected shot: {single_shot}/{n_videos} "
          f"({100.0 * single_shot / n_videos:.1f}%)")

    # same_video pairing feasibility (post [FIX #15]: needs >= num_mix clips
    # total per video, any shot -- shot_id no longer required to differ).
    num_mix = expect["num_mix"]
    starved = [vid for vid, clips in by_video_clips.items() if len(clips) < num_mix]
    if starved:
        issues.warn(
            f"{len(starved)}/{n_videos} video(s) have fewer than num_mix={num_mix} clip(s) total -- "
            f"same_video mix_policy will still fall back to global-random pairing for these "
            f"(the residual case [FIX #15] does not solve), e.g. {starved[:5]}")
    else:
        print(f"same_video pairing OK for all {n_videos} videos (each has >= {num_mix} clips)")

    # homo/hetero pairing feasibility
    starved_cats = [c for c, clips in by_category.items() if len(clips) < num_mix]
    if starved_cats:
        issues.warn(
            f"{len(starved_cats)} categor(y/ies) have fewer than num_mix={num_mix} clip(s) -- "
            f"homo mix_policy can't find a same-category partner for these: {starved_cats}")
    if len(by_category) < num_mix:
        issues.error(f"only {len(by_category)} categor(y/ies) total -- hetero mix_policy needs "
                     f"at least {num_mix} distinct categories to ever find a partner")

    print("\nper-category clip counts:")
    vids_per_cat = collections.defaultdict(set)
    for vid, cat in video_to_cat.items():
        vids_per_cat[cat].add(vid)
    for cat, clips in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cat:<20s} {len(clips):>5d} clips  ({len(vids_per_cat[cat])} videos)")
    if by_category:
        counts = [len(v) for v in by_category.values()]
        imbalance = max(counts) / max(1, min(counts))
        if imbalance >= 3:
            issues.warn(f"category imbalance ratio {imbalance:.1f}x (largest/smallest) -- consider "
                        f"--max_videos_per_category or category-weighted sampling")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed_root", default="datasets/processed",
                    help="Folder produced by scripts/prepare_music21_som.py.")
    p.add_argument("--config", default="configs/som_paper_faithful.yaml",
                    help="Training config to read expected sr/clip_seconds/fps/frame_size/"
                         "num_frames/num_mix/train_index/val_index from. Falls back to "
                         "hardcoded defaults and --processed_root paths if missing.")
    p.add_argument("--sample", type=int, default=300,
                    help="Number of rows to deep-check (audio/frame/trajectory content). "
                         "Cheap existence checks (section 2) always run on every row.")
    p.add_argument("--full", action="store_true",
                    help="Deep-check every row instead of --sample (slow on large datasets).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--report", default=None, help="Optional path to write a JSON report.")
    args = p.parse_args()

    issues = Issues()
    expect, cfg = load_config(args.config, issues)
    expect["paper_categories"] = 21
    data_cfg = (cfg or {}).get("data", {}) if isinstance(cfg, dict) else {}

    meta_path = os.path.join(args.processed_root, "meta_som.csv")
    train_path = data_cfg.get("train_index") or os.path.join(args.processed_root, "train.csv")
    val_path = data_cfg.get("val_index") or os.path.join(args.processed_root, "val.csv")

    print(f"processed_root: {args.processed_root}")
    print(f"train_index: {train_path}")
    print(f"val_index:   {val_path}")
    print(f"expected audio/video params: {expect}")

    meta_rows = read_csv_rows(meta_path, META_FIELDS, issues, "meta_som.csv")
    train_rows = read_csv_rows(train_path, SOM_FIELDS, issues, "train.csv")
    val_rows = read_csv_rows(val_path, SOM_FIELDS, issues, "val.csv")

    if meta_rows:
        check_csv_integrity(meta_rows, train_rows, val_rows, issues)
        check_file_existence(meta_rows, issues)
        check_deep_sample(meta_rows, expect, issues, args.sample, args.seed, args.full)
        check_composition(meta_rows, expect, issues)
    else:
        issues.error("no rows to check -- did you point --processed_root at the right folder?")

    print("\n=== summary ===")
    print(f"{len(issues.errors)} error(s), {len(issues.warnings)} warning(s)")
    if issues.warnings:
        print("\nwarnings (expected-degradation findings, not necessarily bugs):")
        for w in issues.warnings[:50]:
            print(f"  [warn] {w}")
        if len(issues.warnings) > 50:
            print(f"  ...and {len(issues.warnings) - 50} more (see --report for the full list)")
    if issues.errors:
        print("\nerrors (should be fixed / investigated):")
        for e in issues.errors[:50]:
            print(f"  [ERROR] {e}")
        if len(issues.errors) > 50:
            print(f"  ...and {len(issues.errors) - 50} more (see --report for the full list)")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"errors": issues.errors, "warnings": issues.warnings}, f, indent=2)
        print(f"\nfull report written to {args.report}")

    sys.exit(1 if issues.errors else 0)


if __name__ == "__main__":
    main()
