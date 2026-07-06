# Architecture notes

This document maps each paper component to the code that implements it.

| Paper component | Module | File |
| --- | --- | --- |
| Audio analysis/synthesis U-Net | `AudioUNet` | `models/audio_net.py` |
| Semantic context (ResNet-50) | `ContextNet` | `models/context_net.py` |
| Context-aware Graph CNN (CT-GCN) over body+hand keypoints | `ContextAwareGraphCNN` | `models/pose_net.py` |
| Audio-visual self-attention fusion | `AudioVisualFusion` | `models/fusion.py` |
| Mask prediction + masking | `MaskHead`, `apply_mask` | `models/synthesizer.py` |
| Full pipeline | `MusicGesture` | `models/music_gesture.py` |
| Mix-and-Separate data | `MusicMixDataset` | `datasets/music_dataset.py` |
| Skeleton graph (body 18 + 2x hand 21) | `build_skeleton_adjacency` | `utils/pose.py` |

## Data flow

1. **Audio branch.** The mixture waveform is converted to a (log-)magnitude
   spectrogram and encoded by the U-Net encoder into a grid of bottleneck
   tokens.
2. **Visual branch.** For each musician, per-frame body + hand keypoints form a
   spatio-temporal graph. `ContextAwareGraphCNN` runs ST-GCN blocks modulated by
   the ResNet-50 semantic context of that musician (FiLM), producing gesture
   tokens.
3. **Fusion.** `AudioVisualFusion` concatenates audio tokens with the gesture
   tokens of the target source and applies Transformer self-attention so the
   audio representation is conditioned on that source's motion.
4. **Synthesis.** The U-Net decoder maps fused tokens back to a mask; the mask
   is applied to the mixture magnitude and inverted with the mixture phase.

## Why gestures solve same-category separation

Appearance features cannot distinguish two same-type instruments. Fine-grained
finger/body motion is discriminative even when appearance is identical, so
conditioning separation on per-musician gestures resolves the homo-musical case
that *The Sound of Pixels* fails on, while *The Sound of Motions* addresses it
with dense motion instead of sparse keypoints.

## Differences from the (unreleased) original

The paper leaves several details unspecified. Documented choices here:

- 3-subset ST-GCN spatial partitioning for the skeleton graph.
- FiLM used to inject the semantic context into every graph block.
- Transformer-encoder fusion (self-attention over concatenated tokens).
- Ratio mask + L1 loss by default (binary mask + BCE is selectable in config).
