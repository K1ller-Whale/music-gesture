#!/usr/bin/env python3
"""Build the Sound-of-Motions (SoM) index for MUSIC-21 from real videos.

Consumes solo instrument-performance videos laid out as:
    <videos_root>/<category>/<video_id>.<ext>
(exactly what scripts/prepare_music21.py --download produces, and what a
manually-downloaded MUSIC-21 tree looks like: one folder per instrument
category, videos inside.)

For every video this script:
  1. Probes duration (ffprobe).
  2. Detects visual SHOT cuts (ffmpeg scene-change filter), unless
     --no_shot_detect. The paper explicitly shot-detects untrimmed videos and
     tracks only within a shot to avoid trajectory drift (SOM_PAPER_SPEC.md
     Sec 1.2.1, "Step ii"). Clip windows never straddle a detected cut.
  3. Detects SILENCE (ffmpeg silencedetect) to skip near-silent windows -- the
     same principle as scripts/prepare_music21.py::select_windows (spread
     clips across voiced regions, not just the intro), but computed entirely
     via ffmpeg so this script needs no numpy/soundfile/cv2 for indexing.
  4. Tiles up to --max_clips_per_video non-overlapping --clip_seconds windows
     across the voiced regions of each shot, evenly spaced.
  5. For every kept window, extracts:
       <out>/audio/<clip>.wav           mono PCM at --sr, exactly clip_seconds
       <out>/frames/<clip>/000001.jpg... RGB frames at --fps spanning the window
       <out>/first_frame/<clip>.jpg      copy of that window's first frame
       <out>/trajectory/<clip>.npy       ONLY with --cache_trajectories
  6. Writes <out>/meta_som.csv (full bookkeeping) and <out>/train.csv +
     <out>/val.csv directly in the schema datasets/som_dataset.py expects:
       audio_path,frames_dir,trajectory_path,first_frame_path,category,
       video_id,shot_id,clip_start_sec
     split at the VIDEO level (seeded shuffle) so no video's clips leak across
     train/val.

Requires ffmpeg + ffprobe on PATH (same requirement as prepare_music21.py).
--cache_trajectories additionally needs torch + cv2 + a flow backend on a GPU
box; it reuses models/ddt.py + som_backends/pwc.py -- the exact modules
train.py uses -- so cached trajectories are bit-identical to on-the-fly flow.

Usage:
    python scripts/prepare_music21_som.py \\
        --videos_root D:/development/python/ai/download_music21_videos/music21_videos \\
        --out datasets/processed --sr 11025 --clip_seconds 6.0 --fps 8 \\
        --max_clips_per_video 6 --val_ratio 0.1

Then train directly -- configs/som_paper_faithful.yaml already points
train_index/val_index at datasets/processed/{train,val}.csv.

To cache trajectories too (GPU box, official PWC-Net + cupy already working):
    python scripts/prepare_music21_som.py --videos_root ... --out datasets/processed \\
        --cache_trajectories --flow_weights weights/network-default.pytorch
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

VID_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".m4v", ".mov")

SCENE_CUT_RE = re.compile(r"pts_time:([\-0-9.]+)")
SIL_START_RE = re.compile(r"silence_start:\s*([\-0-9.]+)")
SIL_END_RE = re.compile(r"silence_end:\s*([\-0-9.]+)")


# ---------------------------------------------------------------- subprocess
def sh(cmd: list) -> str:
    """Run a subprocess, returning combined stdout+stderr text. Never raises
    on a non-zero exit code -- callers decide whether that's fatal, since a
    single bad video should not abort a multi-hour batch job."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
        return p.stdout or ""
    except Exception as e:  # ffmpeg missing, timeout, etc.
        return f"__SH_ERROR__ {type(e).__name__}: {e}"


def check_ffmpeg() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise SystemExit(
                f"'{exe}' not found on PATH. This script (like prepare_music21.py) "
                f"requires ffmpeg + ffprobe to extract audio/frames from videos.")


def ffprobe_duration(video: str) -> float:
    out = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=noprint_wrappers=1:nokey=1", video])
    try:
        return float(out.strip().splitlines()[-1])
    except Exception:
        return 0.0


# ---------------------------------------------------------------- detection
def detect_shot_cuts(video: str, threshold: float = 0.4) -> list:
    """Return sorted cut timestamps (seconds) via ffmpeg's scene-change score.
    Empty list (no detected cuts, or ffmpeg/filter unavailable) means "treat
    the whole video as one shot" -- callers must handle that gracefully."""
    out = sh(["ffmpeg", "-i", video, "-vf",
              f"select='gt(scene,{threshold})',showinfo",
              "-an", "-f", "null", "-"])
    cuts = sorted(set(round(float(m), 3) for m in SCENE_CUT_RE.findall(out)))
    return cuts


def detect_silence_intervals(video: str, noise_db: float = -35.0,
                             min_dur: float = 0.3, duration: float = 0.0) -> list:
    """Return [(start,end), ...] silence intervals via ffmpeg's silencedetect.
    An unmatched trailing silence_start (silence runs to EOF) is closed at
    ``duration``."""
    out = sh(["ffmpeg", "-i", video, "-af",
              f"silencedetect=noise={noise_db}dB:d={min_dur}",
              "-vn", "-f", "null", "-"])
    starts = [float(x) for x in SIL_START_RE.findall(out)]
    ends = [float(x) for x in SIL_END_RE.findall(out)]
    intervals = list(zip(starts, ends))
    if len(starts) > len(ends) and duration > 0:
        intervals.append((starts[-1], duration))
    return intervals


def silent_overlap(a0: float, a1: float, intervals: list) -> float:
    """Total seconds of [a0,a1) covered by any silence interval."""
    total = 0.0
    for s0, s1 in intervals:
        lo, hi = max(a0, s0), min(a1, s1)
        if hi > lo:
            total += hi - lo
    return total


def shots_from_cuts(duration: float, cuts: list, min_shot_seconds: float) -> list:
    """Turn cut timestamps into [(shot_id, start, end), ...], dropping shots
    shorter than min_shot_seconds (too short to hold even one clip window)."""
    bounds = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    shots = []
    sid = 0
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        if s1 - s0 >= min_shot_seconds:
            shots.append((sid, s0, s1))
            sid += 1
    return shots


# ---------------------------------------------------------------- windowing
def select_clip_windows(shots: list, silence: list, clip_seconds: float,
                        max_clips: int, min_voiced_frac: float) -> list:
    """Tile non-overlapping clip_seconds windows across every shot, then keep
    up to max_clips_per_video, spread evenly across the (voiced, if any)
    windows -- mirrors prepare_music21.py::select_windows, generalised across
    shots. Returns [(shot_id, start_sec), ...] sorted by start_sec."""
    cands = []  # (shot_id, start, voiced_frac)
    for shot_id, s0, s1 in shots:
        t = s0
        while t + clip_seconds <= s1 + 1e-6:
            sil = silent_overlap(t, t + clip_seconds, silence)
            voiced_frac = 1.0 - min(1.0, sil / clip_seconds)
            cands.append((shot_id, round(t, 3), voiced_frac))
            t += clip_seconds
    if not cands:
        return []
    voiced = [c for c in cands if c[2] >= min_voiced_frac]
    pool = voiced if voiced else cands  # whole video near-silent -> keep it anyway
    if max_clips <= 0 or len(pool) <= max_clips:
        chosen = pool
    else:
        import numpy as _np  # only used here; falls back below if unavailable
        idxs = sorted(set(int(round(i)) for i in
                          _np.linspace(0, len(pool) - 1, max_clips)))
        chosen = [pool[i] for i in idxs]
    return sorted((c[0], c[1]) for c in chosen)


# ---------------------------------------------------------------- extraction
def extract_clip_audio(video: str, start: float, dur: float, sr: int, out_wav: str) -> bool:
    if os.path.exists(out_wav):
        return True
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    tmp = out_wav + ".tmp.wav"
    out = sh(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video, "-t", f"{dur:.3f}",
              "-ac", "1", "-ar", str(sr), "-vn", tmp])
    if not os.path.exists(tmp):
        return False
    os.replace(tmp, out_wav)
    return True


def extract_clip_frames(video: str, start: float, dur: float, fps: int, size: int,
                        out_dir: str) -> int:
    existing = glob.glob(os.path.join(out_dir, "*.jpg"))
    if existing:
        return len(existing)
    os.makedirs(out_dir, exist_ok=True)
    sh(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video, "-t", f"{dur:.3f}",
        "-vf", f"fps={fps},scale=-2:{size}", "-an",
        os.path.join(out_dir, "%06d.jpg")])
    return len(glob.glob(os.path.join(out_dir, "*.jpg")))


def make_first_frame(frames_dir: str, out_path: str) -> bool:
    if os.path.exists(out_path):
        return True
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not files:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copyfile(files[0], out_path)
    return True


# ---------------------------------------------------------------- optional: cached trajectories
def build_flow_backbone(args):
    """Lazy: only imports torch/cv2/model code when --cache_trajectories is set,
    so the rest of this script has zero heavy dependencies."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import torch
    from models.ddt import DDTConfig, DDTMotionNet
    net = DDTMotionNet(DDTConfig(freeze_flow=True, freeze_i3d=True))
    if args.flow_impl == "external":
        import importlib
        mod_name, _, fn_name = args.flow_factory.partition(":")
        ctor = getattr(importlib.import_module(mod_name), fn_name)
        net.flow_net = ctor(args.flow_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return net.to(device).eval(), device


def cache_trajectory(net, device, frames_dir: str, out_npy: str, frame_size: int) -> bool:
    if os.path.exists(out_npy):
        return True
    import cv2
    import numpy as np
    import torch
    from models.ddt import compute_trajectories
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if len(files) < 2:
        return False
    imgs = []
    for fp in files:
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (frame_size, frame_size))
        imgs.append(img.astype("float32") / 255.0)
    arr = np.stack(imgs).transpose(0, 3, 1, 2)  # [T,3,H,W]
    frames = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)
    with torch.no_grad():
        flows = net.flow_from_frames(frames)         # [1,T-1,2,H,W]
        traj = compute_trajectories(flows)            # [1,2,T-1,H,W]
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, traj.squeeze(0).detach().cpu().numpy().astype("float32"))
    return True


# ---------------------------------------------------------------- video discovery
def find_videos(videos_root: str, categories=None) -> list:
    """Return [(category, [video_path, ...]), ...] sorted by category then path."""
    wanted = set(categories) if categories else None
    out = []
    for cat in sorted(os.listdir(videos_root)):
        cdir = os.path.join(videos_root, cat)
        if not os.path.isdir(cdir):
            continue
        if wanted is not None and cat not in wanted:
            continue
        vids = sorted(f for f in os.listdir(cdir)
                      if os.path.splitext(f)[1].lower() in VID_EXTS)
        if vids:
            out.append((cat, [os.path.join(cdir, v) for v in vids]))
    return out


def safe_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


@dataclass
class Row:
    clip: str
    category: str
    video_id: str
    shot_id: int
    clip_start_sec: float
    audio_path: str
    frames_dir: str
    first_frame_path: str
    trajectory_path: str = ""


SOM_FIELDS = ["audio_path", "frames_dir", "trajectory_path", "first_frame_path",
              "category", "video_id", "shot_id", "clip_start_sec"]
META_FIELDS = ["clip", "category", "video_id", "shot_id", "clip_start_sec",
              "audio_path", "frames_dir", "first_frame_path", "trajectory_path"]


def process_video(cat: str, video: str, args, flow_ctx) -> list:
    video_id = safe_id(os.path.splitext(os.path.basename(video))[0])
    duration = ffprobe_duration(video)
    if duration <= 0:
        print(f"  [skip] {cat}/{video_id}: could not read duration (corrupt/unsupported file?)")
        return []
    if args.shot_detect:
        cuts = detect_shot_cuts(video, args.scene_threshold)
    else:
        cuts = []
    min_shot = args.min_shot_seconds or args.clip_seconds
    shots = shots_from_cuts(duration, cuts, min_shot)
    if not shots:
        print(f"  [skip] {cat}/{video_id}: no shot >= {min_shot:.1f}s "
              f"(duration {duration:.1f}s)")
        return []
    silence = detect_silence_intervals(video, args.silence_noise_db,
                                       args.silence_min_dur, duration)
    windows = select_clip_windows(shots, silence, args.clip_seconds,
                                  args.max_clips_per_video, args.min_voiced_frac)
    if not windows:
        print(f"  [skip] {cat}/{video_id}: no clip window fit ({len(shots)} shot(s))")
        return []

    rows = []
    for shot_id, start in windows:
        clip = f"{safe_id(cat)}__{video_id}__s{shot_id:02d}__{int(round(start * 1000)):08d}"
        wav_path = os.path.join(args.out, "audio", f"{clip}.wav")
        frames_dir = os.path.join(args.out, "frames", clip)
        first_frame_path = os.path.join(args.out, "first_frame", f"{clip}.jpg")
        traj_path = os.path.join(args.out, "trajectory", f"{clip}.npy")

        ok_audio = extract_clip_audio(video, start, args.clip_seconds, args.sr, wav_path)
        n_frames = extract_clip_frames(video, start, args.clip_seconds, args.fps,
                                       args.frame_size, frames_dir)
        ok_first = make_first_frame(frames_dir, first_frame_path)
        if not (ok_audio and n_frames > 0 and ok_first):
            print(f"  [warn] {clip}: incomplete extraction (audio={ok_audio}, "
                  f"frames={n_frames}), skipping row")
            continue

        traj_rel = ""
        if flow_ctx is not None:
            net, device = flow_ctx
            if cache_trajectory(net, device, frames_dir, traj_path, args.frame_size):
                traj_rel = traj_path

        rows.append(Row(clip, cat, video_id, shot_id, start, wav_path, frames_dir,
                        first_frame_path, traj_rel))
    print(f"  [ok] {cat}/{video_id}: {len(shots)} shot(s), {len(rows)} clip(s) "
          f"(duration {duration:.1f}s)")
    return rows


def write_csv(path: str, rows: list, fields: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def split_by_video(rows: list, val_ratio: float, seed: int):
    by_video = {}
    for r in rows:
        by_video.setdefault(r.video_id, []).append(r)
    video_ids = list(by_video.keys())
    random.Random(seed).shuffle(video_ids)
    total = len(rows)
    target_val = total * val_ratio
    val_rows, train_rows = [], []
    val_count = 0
    for vid in video_ids:
        clips = by_video[vid]
        if val_count < target_val:
            val_rows.extend(clips)
            val_count += len(clips)
        else:
            train_rows.extend(clips)
    return train_rows, val_rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_root", required=True,
                   help="Directory with <category>/<video_id>.<ext> layout.")
    p.add_argument("--out", default="datasets/processed")
    p.add_argument("--categories", nargs="*", default=None,
                   help="Optional subset of category folder names.")
    p.add_argument("--sr", type=int, default=11025)
    p.add_argument("--clip_seconds", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--frame_size", type=int, default=256,
                   help="Extracted frame side length (model resizes to video.frame_size at load).")
    p.add_argument("--max_clips_per_video", type=int, default=6)
    p.add_argument("--max_videos_per_category", type=int, default=0,
                   help="0 = no cap. Use a small number for a quick partial run.")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_shot_detect", action="store_true",
                   help="Skip ffmpeg scene-cut detection; treat each video as one shot.")
    p.add_argument("--scene_threshold", type=float, default=0.4)
    p.add_argument("--min_shot_seconds", type=float, default=0.0,
                   help="0 = default to --clip_seconds (a shot must fit >=1 clip window).")
    p.add_argument("--silence_noise_db", type=float, default=-35.0)
    p.add_argument("--silence_min_dur", type=float, default=0.3)
    p.add_argument("--min_voiced_frac", type=float, default=0.3,
                   help="Min non-silent fraction of a window to prefer it over silent windows.")
    p.add_argument("--cache_trajectories", action="store_true",
                   help="GPU-only: precompute DDT trajectories with models/ddt.py + "
                        "the configured flow backend (reuses train.py's own code).")
    p.add_argument("--flow_impl", choices=["external", "builtin"], default="external")
    p.add_argument("--flow_factory", default="som_backends.pwc:build_official_pwc")
    p.add_argument("--flow_weights", default="weights/network-default.pytorch")
    args = p.parse_args()
    args.shot_detect = not args.no_shot_detect

    check_ffmpeg()
    videos = find_videos(args.videos_root, args.categories)
    if not videos:
        raise SystemExit(f"no <category>/<video> layout found under {args.videos_root}")

    flow_ctx = None
    if args.cache_trajectories:
        print("[flow] building trajectory-caching backbone (this loads torch/cv2)...")
        flow_ctx = build_flow_backbone(args)

    all_rows = []
    n_videos = 0
    for cat, vids in videos:
        if args.max_videos_per_category:
            vids = vids[: args.max_videos_per_category]
        print(f"[{cat}] {len(vids)} video(s)")
        for video in vids:
            all_rows.extend(process_video(cat, video, args, flow_ctx))
            n_videos += 1

    if not all_rows:
        raise SystemExit("no clips extracted -- check ffmpeg output above")

    write_csv(os.path.join(args.out, "meta_som.csv"), all_rows, META_FIELDS)
    train_rows, val_rows = split_by_video(all_rows, args.val_ratio, args.seed)
    write_csv(os.path.join(args.out, "train.csv"), train_rows, SOM_FIELDS)
    write_csv(os.path.join(args.out, "val.csv"), val_rows, SOM_FIELDS)

    cats = sorted(set(r.category for r in all_rows))
    n_traj = sum(1 for r in all_rows if r.trajectory_path)
    print(f"\nprocessed {n_videos} video(s) across {len(cats)} categories {cats}")
    print(f"wrote {os.path.join(args.out, 'meta_som.csv')}: {len(all_rows)} clips")
    print(f"wrote {os.path.join(args.out, 'train.csv')}: {len(train_rows)} clips")
    print(f"wrote {os.path.join(args.out, 'val.csv')}: {len(val_rows)} clips")
    if args.cache_trajectories:
        print(f"cached trajectories: {n_traj}/{len(all_rows)} clips")
    print("\nPoint configs/som_paper_faithful.yaml's data.train_index/val_index at the "
          "train.csv/val.csv above (already the default for --out datasets/processed), then:\n"
          "    python train.py --config configs/som_paper_faithful.yaml")


if __name__ == "__main__":
    main()
