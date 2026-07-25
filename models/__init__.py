from .audio_net import AudioUNet
from .context_net import ContextNet
from .pose_net import ContextAwareGraphCNN
from .fusion import AudioVisualFusion
from .synthesizer import MaskHead, apply_mask
from .music_gesture import MusicGesture
from .som import SoundOfMotions

__all__ = [
    "AudioUNet",
    "ContextNet",
    "ContextAwareGraphCNN",
    "AudioVisualFusion",
    "MaskHead",
    "apply_mask",
    "MusicGesture",
    "SoundOfMotions",
    "build_model",
]


def build_model(cfg: dict):
    """Instantiate the model named by ``cfg['model'].get('type')``.

    'music_gesture' (default) -> the ST-GCN pose model; 'som' -> the Sound of
    Motions DDT + appearance model. Keeping both behind one registry lets
    train.py / test.py / eval switch architectures from the config alone.
    """
    kind = cfg.get("model", {}).get("type", "music_gesture").lower()
    if kind in ("som", "sound_of_motions", "soundofmotions"):
        return SoundOfMotions(cfg)
    if kind in ("music_gesture", "musicgesture", "mg"):
        return MusicGesture(cfg)
    raise ValueError(f"unknown model.type {kind!r}")
