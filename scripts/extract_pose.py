"""Extract body + hand keypoints per frame from videos.

This is a thin wrapper describing the expected preprocessing. Plug in your pose
estimator of choice (OpenPose / AlphaPose / MMPose). The output per clip is a
numpy array of shape [T, V, 3] where V = body(18) + hand(21) + hand(21) = 60 and
the last channel is the detection confidence.

Usage:
    python scripts/extract_pose.py --videos_dir data/videos --out_dir datasets/processed/pose
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

BODY = 18
HAND = 21
V = BODY + 2 * HAND


def estimate_pose_for_video(path: str, fps: int, num_frames: int) -> np.ndarray:
    """Replace this stub with a real pose estimator.

    Returns [T, V, 3]. The stub returns zeros so the pipeline is runnable
    end-to-end for smoke testing without a pose model installed.
    """
    # TODO: integrate OpenPose/AlphaPose/MMPose here.
    return np.zeros((num_frames, V, 3), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=48)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(args.videos_dir, "*.mp4")))
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        kp = estimate_pose_for_video(v, args.fps, args.num_frames)
        np.save(os.path.join(args.out_dir, f"{name}.npy"), kp)
        print(f"saved pose for {name}: {kp.shape}")


if __name__ == "__main__":
    main()
