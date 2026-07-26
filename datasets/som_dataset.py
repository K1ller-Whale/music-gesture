"""Mix-and-Separate dataset for Sound of Motions (MUSIC-21 / URMP).

Same self-supervised recipe as datasets/music_dataset.py, but the per-source
visual conditioning is SoM's motion + appearance instead of pose keypoints:

  * motion:     a short video clip -- either a cached dense-trajectory volume
                [2, T-1, H, W] (the fast path, produced by
                scripts/prepare_music21_som.py) or a raw frame stack
                [T, 3, H, W] (flow computed on the fly by the model);
  * appearance: a single first frame [3, H, W].

Index CSV schema (one preprocessed solo clip per row):
    audio_path, frames_dir, trajectory_path, first_frame_path,
    category, video_id, shot_id, clip_start_sec
``trajectory_path`` may be empty (then frames_dir is used). ``video_id`` drives
the curriculum's same-video sampling policy (any other clip from the same
video_id; see [FIX #15] below -- ``shot_id`` is used only for DDT trajectory
tracking, not for same-video pairing).

The audio clip and the visual clip are cropped from one shared start time so the
motion the fusion module sees corresponds to the sound it must separate (the P1
alignment fix inherited from the sibling reproduction).
"""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import audio as A

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MusicMixSoMDataset(Dataset):
    def __init__(self, index_file: str, cfg: dict, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.num_mix = cfg["data"]["num_mix"]
        self.sr = cfg["audio"]["sample_rate"]
        self.clip_seconds = float(cfg["audio"]["clip_seconds"])
        self.clip_len = int(self.clip_seconds * self.sr)
        v = cfg["video"]
        self.frame_size = int(v["frame_size"])
        self.fps = int(v["fps"])
        self.num_frames = int(v["num_frames"])
        # 'raw' feeds frame stacks (model computes flow); 'trajectory' feeds the
        # cached DDT volume (fast). 'auto' uses the cache when the row has one.
        self.motion_mode = v.get("motion_mode", "auto")
        self.samples = self._read_index(index_file)
        self.mix_policy = cfg["data"].get("mix_policy", "random")
        self.by_category: Dict[str, List[int]] = {}
        self.by_video: Dict[str, List[int]] = {}
        for i, s in enumerate(self.samples):
            self.by_category.setdefault(s.get("category", ""), []).append(i)
            self.by_video.setdefault(s.get("video_id", ""), []).append(i)
        self.log_freq = cfg["audio"].get("log_freq", False)
        self.warp = None
        if self.log_freq:
            mat = A.build_log_freq_matrix(
                cfg["audio"]["n_freq"], cfg["audio"]["n_log_freq"], self.sr)
            self.warp = torch.from_numpy(mat)

    @staticmethod
    def _read_index(path: str) -> List[Dict[str, str]]:
        with open(path, newline="") as f:
            return [row for row in csv.DictReader(f)]

    def __len__(self) -> int:
        return len(self.samples)

    # ---- audio ----------------------------------------------------------
    def _read_wav(self, path: str) -> np.ndarray:
        if sf is None:
            raise ImportError("soundfile is required to load audio")
        wav, _ = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) < self.clip_len:
            wav = np.pad(wav, (0, self.clip_len - len(wav)))
        return wav

    # ---- visual ---------------------------------------------------------
    def _load_first_frame(self, path: str) -> torch.Tensor:
        if cv2 is None:
            raise ImportError("opencv is required to load frames")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.frame_size, self.frame_size))
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.from_numpy(img.transpose(2, 0, 1))

    def _load_frame_stack(self, frames_dir: str, f0: int) -> torch.Tensor:
        """Load ``num_frames`` RGB frames starting at index f0 -> [T,3,H,W]."""
        if cv2 is None:
            raise ImportError("opencv is required to load frames")
        files = sorted(
            fn for fn in os.listdir(frames_dir)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not files:
            return torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
        idxs = [min(f0 + k, len(files) - 1) for k in range(self.num_frames)]
        frames = []
        for k in idxs:
            img = cv2.imread(os.path.join(frames_dir, files[k]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.frame_size, self.frame_size))
            frames.append(img.astype(np.float32) / 255.0)
        arr = np.stack(frames).transpose(0, 3, 1, 2)  # [T,3,H,W], in [0,1]
        return torch.from_numpy(np.ascontiguousarray(arr))

    def _load_trajectory(self, path: str, f0: int) -> torch.Tensor:
        """Load a cached trajectory volume [2, T-1, H, W], cropped to num_frames-1."""
        traj = np.load(path)  # [2, Ttot-1, H, W]
        if traj.ndim != 4 or traj.shape[0] != 2:
            raise ValueError(f"bad trajectory shape {traj.shape} in {path}")
        want = self.num_frames - 1
        seg = traj[:, f0:f0 + want]
        if seg.shape[1] < want:
            pad = want - seg.shape[1]
            if seg.shape[1] == 0:
                seg = np.zeros((2, want, traj.shape[2], traj.shape[3]), np.float32)
            else:
                seg = np.pad(seg, ((0, 0), (0, pad), (0, 0), (0, 0)), mode="edge")
        return torch.from_numpy(np.ascontiguousarray(seg)).float()

    def _load_source(self, row: Dict[str, str]):
        """Return (wav [clip_len], motion tensor, first_frame [3,H,W])."""
        wav = self._read_wav(row["audio_path"])
        audio_dur = len(wav) / float(self.sr)
        max_start = max(0.0, audio_dur - self.clip_seconds)
        if self.split == "train" and max_start > 0.0:
            start_sec = random.uniform(0.0, max_start)
        else:
            start_sec = 0.0
        a0 = int(round(start_sec * self.sr))
        a0 = max(0, min(a0, len(wav) - self.clip_len))
        wav = wav[a0:a0 + self.clip_len]
        f0 = int(round(start_sec * self.fps))

        traj_path = row.get("trajectory_path", "") or ""
        use_traj = (self.motion_mode == "trajectory" or
                    (self.motion_mode == "auto" and traj_path and os.path.exists(traj_path)))
        if use_traj:
            motion = self._load_trajectory(traj_path, f0)
        else:
            motion = self._load_frame_stack(row["frames_dir"], f0)

        first_frame = self._load_first_frame(
            row.get("first_frame_path") or row["frames_dir"] + "/000001.jpg"
        ) if (row.get("first_frame_path") or row.get("frames_dir")) else \
            torch.zeros(3, self.frame_size, self.frame_size)
        return torch.from_numpy(np.ascontiguousarray(wav)), motion, first_frame

    # ---- mixing policy --------------------------------------------------
    def _sample_others(self, idx: int) -> List[int]:
        anchor = self.samples[idx]
        cat = anchor.get("category", "")
        vid = anchor.get("video_id", "")
        if self.mix_policy == "homo":
            pool = [j for j in self.by_category.get(cat, []) if j != idx]
        elif self.mix_policy == "hetero":
            pool = [j for j in range(len(self.samples))
                    if self.samples[j].get("category", "") != cat]
        elif self.mix_policy == "same_video":
            # [FIX #15] Only require a DIFFERENT CLIP from the same video, not
            # a different shot. Clip windows are already tiled non-overlapping
            # in time (prepare_music21_som.py::select_clip_windows), so any
            # other same-video clip already has distinct motion content --
            # shot_id is a DDT-tracking concept (avoid trajectory drift across
            # a cut), not a pairing-diversity requirement. Requiring a
            # different shot_id left the pool empty for every single-shot
            # video, silently falling back to fully global random pairing
            # (any video, any category) for 694/809 (85.8%) of a real
            # MUSIC-21 corpus -- defeating this stage's entire purpose of
            # forcing motion-based separation on same-appearance pairs.
            pool = [j for j in self.by_video.get(vid, []) if j != idx]
        else:
            pool = [j for j in range(len(self.samples)) if j != idx]
        if len(pool) < self.num_mix - 1:
            pool = [j for j in range(len(self.samples)) if j != idx]
        return random.sample(pool, self.num_mix - 1)

    def _spec(self, wav: torch.Tensor) -> torch.Tensor:
        c = self.cfg["audio"]
        return A.stft(wav, c["n_fft"], c["hop_length"], c["win_length"])

    def __getitem__(self, idx: int) -> Dict[str, object]:
        chosen = [self.samples[idx]] + [self.samples[o] for o in self._sample_others(idx)]
        loaded = [self._load_source(s) for s in chosen]
        waveforms = [t[0] for t in loaded]
        motions = [t[1] for t in loaded]
        first_frames = [t[2] for t in loaded]

        mixture, sources = A.mix_and_separate(waveforms)
        mix_mag = self._spec(mixture).abs()
        src_mags = [self._spec(s).abs() for s in sources]
        if self.warp is not None:
            mix_mag = A.warp_freq(mix_mag, self.warp)
            src_mags = [A.warp_freq(s, self.warp) for s in src_mags]
        net_input = A.log_magnitude(mix_mag).unsqueeze(0)
        mix_mag = mix_mag.unsqueeze(0)
        src_mags = [s.unsqueeze(0) for s in src_mags]

        return {
            "net_input": net_input,
            "mixture_mag": mix_mag,
            "mixture_wav": mixture,
            "source_mags": src_mags,
            "source_wavs": sources,
            "motion": motions,
            "first_frames": first_frames,
            "categories": [s.get("category", "") for s in chosen],
        }


def collate_som(batch: List[Dict[str, object]]) -> Dict[str, object]:
    num_mix = len(batch[0]["motion"])
    return {
        "net_input": torch.stack([b["net_input"] for b in batch]),
        "mixture_mag": torch.stack([b["mixture_mag"] for b in batch]),
        "mixture_wav": torch.stack([b["mixture_wav"] for b in batch]),
        "source_mags": [torch.stack([b["source_mags"][i] for b in batch]) for i in range(num_mix)],
        "source_wavs": [torch.stack([b["source_wavs"][i] for b in batch]) for i in range(num_mix)],
        "motion": [torch.stack([b["motion"][i] for b in batch]) for i in range(num_mix)],
        "first_frames": [torch.stack([b["first_frames"][i] for b in batch]) for i in range(num_mix)],
        "categories": [[b["categories"][i] for b in batch] for i in range(num_mix)],
    }
