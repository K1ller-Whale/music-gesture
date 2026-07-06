"""Mix-and-Separate dataset for MUSIC-21.

Each index file row points to a preprocessed solo clip:
    audio_path, pose_path, context_frame_path, category

``__getitem__`` samples ``num_mix`` solos, mixes their audio, and returns the
mixture spectrogram plus per-source keypoints/context/target so the model learns
to separate a source conditioned on its gestures.
"""
from __future__ import annotations

import csv
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import audio as A
from utils.pose import normalize_keypoints

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


class MusicMixDataset(Dataset):
    def __init__(self, index_file: str, cfg: dict, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.num_mix = cfg["data"]["num_mix"]
        self.sr = cfg["audio"]["sample_rate"]
        self.clip_len = int(cfg["audio"]["clip_seconds"] * self.sr)
        self.frame_size = cfg["video"]["frame_size"]
        self.samples = self._read_index(index_file)

    @staticmethod
    def _read_index(path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, path: str) -> torch.Tensor:
        if sf is None:
            raise ImportError("soundfile is required to load audio")
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) < self.clip_len:
            wav = np.pad(wav, (0, self.clip_len - len(wav)))
        start = random.randint(0, len(wav) - self.clip_len) if self.split == "train" else 0
        return torch.from_numpy(wav[start:start + self.clip_len])

    def _load_pose(self, path: str) -> torch.Tensor:
        kp = np.load(path)  # [T, V, 3]
        kp = normalize_keypoints(kp, self.frame_size, self.frame_size)
        return torch.from_numpy(kp).float()

    def _load_context(self, path: str) -> torch.Tensor:
        if cv2 is None:
            raise ImportError("opencv-python is required to load frames")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.frame_size, self.frame_size))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1))

    def _spec(self, wav: torch.Tensor) -> torch.Tensor:
        c = self.cfg["audio"]
        spec = A.stft(wav, c["n_fft"], c["hop_length"], c["win_length"])
        return spec

    def __getitem__(self, idx: int) -> Dict[str, object]:
        chosen = [self.samples[idx]]
        others = random.sample(range(len(self.samples)), self.num_mix - 1)
        chosen += [self.samples[o] for o in others]

        waveforms = [self._load_audio(s["audio_path"]) for s in chosen]
        mixture, sources = A.mix_and_separate(waveforms)

        c = self.cfg["audio"]
        mix_spec = self._spec(mixture)
        mix_mag = mix_spec.abs().unsqueeze(0)
        src_mags = [self._spec(s).abs().unsqueeze(0) for s in sources]

        keypoints = [self._load_pose(s["pose_path"]) for s in chosen]
        contexts = [self._load_context(s["context_frame_path"]) for s in chosen]

        return {
            "mixture_mag": mix_mag,
            "mixture_wav": mixture,
            "source_mags": src_mags,
            "source_wavs": sources,
            "keypoints": keypoints,
            "contexts": contexts,
            "categories": [s["category"] for s in chosen],
        }


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack a batch of variable-source samples (fixed num_mix)."""
    num_mix = len(batch[0]["keypoints"])
    out: Dict[str, object] = {
        "mixture_mag": torch.stack([b["mixture_mag"] for b in batch]),
        "mixture_wav": torch.stack([b["mixture_wav"] for b in batch]),
        "source_mags": [torch.stack([b["source_mags"][i] for b in batch]) for i in range(num_mix)],
        "source_wavs": [torch.stack([b["source_wavs"][i] for b in batch]) for i in range(num_mix)],
        "keypoints": [torch.stack([b["keypoints"][i] for b in batch]) for i in range(num_mix)],
        "contexts": [torch.stack([b["contexts"][i] for b in batch]) for i in range(num_mix)],
    }
    return out
