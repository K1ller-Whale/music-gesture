from .music_dataset import MusicMixDataset, collate
from .som_dataset import MusicMixSoMDataset, collate_som

__all__ = [
    "MusicMixDataset",
    "collate",
    "MusicMixSoMDataset",
    "collate_som",
    "build_dataset",
]


def build_dataset(cfg: dict, index_file: str, split: str):
    """Return (dataset, collate_fn) for the model named by cfg['model'].type.

    'som' -> motion + appearance SoM dataset; anything else -> the pose-based
    Music Gesture dataset. Keeps train.py / eval agnostic to the architecture.
    """
    kind = cfg.get("model", {}).get("type", "music_gesture").lower()
    if kind in ("som", "sound_of_motions", "soundofmotions"):
        return MusicMixSoMDataset(index_file, cfg, split=split), collate_som
    return MusicMixDataset(index_file, cfg, split=split), collate
