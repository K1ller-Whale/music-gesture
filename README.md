# Music Gesture (Reimplementation)

A clean-room PyTorch reimplementation of

> **Music Gesture for Visual Sound Separation**
> Chuang Gan, Deng Huang, Hang Zhao, Joshua B. Tenenbaum, Antonio Torralba.
> CVPR 2020. [Project page](http://music-gesture.csail.mit.edu) · [arXiv:2004.09476](https://arxiv.org/abs/2004.09476)

> **Disclaimer.** The original authors never released official source code. This
> repository is an independent, from-the-paper reconstruction of the described
> architecture and training procedure. It is **not** a copy of any official code
> and may differ in low-level details the paper leaves unspecified.

## Key idea

Appearance-based visual sound separation (e.g. *The Sound of Pixels*) fails to
separate instruments of the **same category** (two violins) because they look
alike. *Music Gesture* replaces dense appearance features with **structured body
and hand keypoints** that capture how each musician *moves*, and fuses them with
audio through a self-attention audio-visual module. Motion/gesture is
discriminative even when appearance is not, which solves the homo-musical
(same-category) separation problem.

## Architecture overview

```
            video frames                          mixture waveform
                 |                                       |
        pose estimator (body 18 + hands 21x2)        STFT (log-mag)
                 |                                       |
   +----------------------------+                +----------------+
   | Context-aware Graph CNN    |   ResNet-50    |   Audio U-Net  |
   | (CT-GCN over keypoints)    |<--context------|   (encoder)    |
   +----------------------------+   features     +----------------+
                 |  visual tokens                        | audio tokens
                 +-------------------+   +---------------+
                                     v   v
                    Audio-Visual Self-Attention Fusion
                                     |
                             Audio U-Net (decoder)
                                     |
                        per-source spectrogram mask
                                     |
                       apply mask + iSTFT -> waveform
```

Training is **Mix-and-Separate**: artificially mix solo clips, then ask the model
to recover each source conditioned on that source's gestures.

## Repository layout

```
music-gesture/
  configs/default.yaml        # all hyperparameters
  datasets/music_dataset.py   # Mix-and-Separate dataset (MUSIC-21)
  models/
    audio_net.py              # audio U-Net
    context_net.py            # ResNet-50 semantic context extractor
    pose_net.py               # context-aware ST/graph CNN (CT-GCN)
    fusion.py                 # audio-visual self-attention fusion
    synthesizer.py            # mask head + spectrogram masking
    music_gesture.py          # full model
  utils/
    audio.py                  # STFT / iSTFT / masks / mix-and-separate
    pose.py                   # skeleton graph (body + hands)
    metrics.py                # SDR / SIR / SAR
  scripts/
    extract_pose.py           # run pose estimation over videos
    prepare_data.py           # build train/val index files
  train.py
  test.py
  separate.py                 # inference on a single mixture video
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Uses the **MUSIC-21** dataset (21 instrument categories) released with
*The Sound of Pixels*: https://github.com/roudimit/MUSIC_dataset

1. Download the YouTube clips listed in `MUSIC21_solo_videos.json`.
2. Extract per-frame pose with `python scripts/extract_pose.py`.
3. Build index files with `python scripts/prepare_data.py`.

## Training

```bash
python train.py --config configs/default.yaml
```

## Evaluation

```bash
python test.py --config configs/default.yaml --checkpoint runs/best.pth
```

## Inference on your own video

```bash
python separate.py --checkpoint runs/best.pth --video mix.mp4 --out out/
```

## Citation

```bibtex
@inproceedings{gan2020music,
  title     = {Music Gesture for Visual Sound Separation},
  author    = {Gan, Chuang and Huang, Deng and Zhao, Hang and Tenenbaum, Joshua B. and Torralba, Antonio},
  booktitle = {CVPR},
  year      = {2020}
}
```

## License

MIT (this reimplementation). See `LICENSE`.
