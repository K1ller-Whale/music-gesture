# Sound of Motions — Implementation Plan & Reuse Map

Clean-room, paper-faithful reimplementation of **The Sound of Motions**
(Zhao, Gan, Ma, Torralba; ICCV 2019, arXiv:1904.05979), built *on top of* the
existing Music Gesture (CVPR 2020) reproduction in this repo. Trained and
evaluated on MUSIC / MUSIC-21 (and URMP) with Mix-and-Separate self-supervision.

The full paper spec (every equation/value + ambiguities & assumptions with
citations) lives in `SoM_REIMPLEMENTATION_SPEC_AND_PLAN.md` (project root, one
level above the repo). This file is the engineering plan: what was reused, what
was replaced, the new file inventory, key design decisions, and how to run it.

---

## 1. Reuse vs. Replace

| Component | Decision | Where |
|---|---|---|
| Dataset download / clip extraction / silence-aware sampling | **Reuse** (extend) | `scripts/prepare_music21*.py` + new `scripts/prepare_music21_som.py` |
| Audio pipeline: 11025 Hz, STFT, log-freq warp, `mix_and_separate` | **Reuse** as-is | `utils/audio.py` |
| Audio analysis/synthesis U-Net (encode→bottleneck tokens→decode) | **Reuse** as-is | `models/audio_net.py` |
| Binary dominant mask (Eq.1) + energy-weighted BCE + curriculum scaffolding | **Reuse** as-is | `train.py` |
| Checkpointing (atomic last/best), stage chaining | **Reuse** (extend: epoch-level intra-stage resume) | `train.py` |
| SDR/SIR/SAR eval harness + ablation diagnostics | **Reuse** (new SoM-specific driver) | `utils/metrics.py`, new `scripts/eval_som.py` |
| Config format | **Reuse** (new config) | new `configs/som_paper_faithful.yaml` |
| **Pose branch (ST-GCN + fusion transformer)** | **Replace** | new `models/ddt.py`, `models/som_fusion.py` |
| Appearance branch | **Add** | new `models/som_appearance.py` |
| Motion branch (PWC-Net flow + dense trajectories + I3D) | **Add** | new `models/pwcnet.py`, `models/i3d.py`, `models/ddt.py` |
| FiLM bottleneck conditioning (Eq.2) | **Add** | `models/som.py` (`FiLMConditioner`) |
| Preprocessing: pose `.npy` → frame stacks + first frame + cached trajectories | **Replace/extend** | new `datasets/som_dataset.py`, `scripts/prepare_music21_som.py` |

Both architectures now live behind registries so `train.py` / eval stay
architecture-agnostic: `models.build_model(cfg)` and
`datasets.build_dataset(cfg, index, split)` switch on `cfg["model"]["type"]`
(`music_gesture` default, `som` for this work). The Music Gesture path is
untouched.

---

## 2. New file inventory

```
models/pwcnet.py         PWC-Net optical flow (pure PyTorch, swappable backend)
models/i3d.py            InceptionI3d, configurable in_channels, feature-map out
models/ddt.py            Deep Dense Trajectory: flow -> trajectory volume -> I3D
models/som_appearance.py ResNet-18 over the first frame (spatial map + vector)
models/som_fusion.py     appearance spatial-attention gate over motion features
models/som.py            top-level SoundOfMotions (audio U-Net + DDT + appearance
                         + fusion + FiLM bottleneck + vision-gated head)
datasets/som_dataset.py  Mix-and-Separate with motion + appearance conditioning
configs/som_paper_faithful.yaml
scripts/prepare_music21_som.py   frames + first_frame + shots + cached trajectories
scripts/eval_som.py      SDR/SIR/SAR, N-sweep, zero_motion/zero_appearance, per-instrument
```

Edited: `models/__init__.py`, `datasets/__init__.py`, `train.py`.

---

## 3. Data flow (one source)

```
video clip ─┬─ frames [T,3,H,W] ─ PWC-Net ─ flows [T-1,2,H,W] ─ trajectory
            │                    tracking G_{t+1}=G_t+grid_sample(ω_t,G_t)
            │                                     └─ traj volume [2,T-1,H,W] ─ I3D ─ motion_feat [Cm,T',h,w]
            └─ first frame [3,H,W] ─ ResNet-18 ─ appearance map [Ca,ha,wa]

SoM fusion:  attn = σ(conv(appearance)) → gate motion → concat appearance →
             conv3d → spatial max-pool → f_v_time [T',Kv], f_v_vec [Kv]

audio:  mix log-mag ─ U-Net.encode ─ bottleneck tokens [B,h*w,D]
        FiLM(tokens, f_v_vec) = (1+γ(f_v))·tokens + β(f_v)          (Eq.2)
        U-Net.decode ─ featmap [B,K,F,T]
        vision-gated head: logits = einsum("bkft,bk->bft", featmap, W(f_v)) + b(f_v)
        MaskHead(sigmoid) → mask; binary mask thresholded at inference
```

---

## 4. Key design decisions (and why)

1. **FiLM is clip-level** (γ,β from the temporally-pooled visual vector), matching
   Eq.2's single f_v per source. Time-varying FiLM is a documented future knob;
   the visual temporal features are already computed and available.
2. **Vision-gated output head is preserved** from the Music Gesture repro — it is
   the head that fixed representation collapse and made URMP reproduce. It gives
   the visual feature a direct per-pixel path to the mask alongside FiLM.
3. **PWC-Net and I3D are faithful, runnable pure-PyTorch backends** structured as
   swappable modules. Exact Sintel/ImageNet fidelity requires loading the
   official reference weights (`PWCNet.load_pretrained`, `InceptionI3d.
   load_pretrained`); the DDT/appearance/fusion/audio code depends only on the
   `[B,2,H,W]` flow contract and the `[B,C,T',H',W']` feature contract, so the
   backend can be swapped without touching the rest.
4. **Trajectories are cached** (`scripts/prepare_music21_som.py --cache-
   trajectories`) because flow + tracking is the expensive, deterministic part.
   The dataset uses the cache when present (`motion_mode: auto`) and otherwise
   feeds raw frames so the model computes flow on the fly.
5. **3-stage curriculum** (different-instrument → same-instrument → same-video)
   with epoch-level intra-stage resume, since the full MUSIC-21 run is long on a
   single GPU and the sibling project was bitten by mid-stage crashes restarting
   from epoch 0.

---

## 5. Ambiguities / assumptions (see full spec for citations)

- Clip length: SoM text ≈ 6 s; repo audio config uses 6 s / STFT settings that
  the U-Net depends on → kept 6 s (`clip_seconds: 6.0`). Paper also mentions
  24 frames @ 8 FPS (3 s) for the visual clip; `video.num_frames`/`fps` are
  exposed so the visual clip can be set independently of the audio clip.
- `frame_size = 224` (standard ResNet/I3D input) — not pinned by the paper.
- Log-frequency warp is inherited from the sibling audio pipeline; not in the
  SoM text. Toggle with `audio.log_freq`.
- Feature dims (K_m/K_a/K_v), batch size, epoch counts, LR schedule, weight
  decay, augmentation: repo defaults / SoM's SGD(momentum 0.9, lr 1e-3 sep+
  fusion, 1e-4 motion+appearance). All in the config.

---

## 6. How to run

```bash
# 0. clean isolated venv (see HANDOFF.md §9); single OpenMP/BLAS runtime.
# 1. preprocess MUSIC-21 into the SoM layout (frames + first frame + shots)
python scripts/prepare_music21_som.py --videos_root <videos> \
    --out datasets/processed --solo_json <solo.json> --fps 8 \
    --clip_seconds 6.0 --shot-detect            # add --cache-trajectories on the GPU box
# 2. make the train/val split CSVs (reuse the existing splitter over meta_som.csv)
python scripts/prepare_data.py --meta datasets/processed/meta_som.csv \
    --out datasets/processed                     # -> train.csv / val.csv
# 3. (optional) fetch official backbone weights, then point the loaders at them
#    PWCNet.load_pretrained(...), InceptionI3d.load_pretrained('rgb_imagenet.pt')
# 4. train the 3-stage curriculum (idempotent / resumable)
python train.py --config configs/som_paper_faithful.yaml
# 5. evaluate: SDR/SIR/SAR + ablations + N sweep + per-instrument
python scripts/eval_som.py --config configs/som_paper_faithful.yaml \
    --checkpoint runs/som_music21/last.pth --n-sources 2 3 4 --per-instrument
```

---

## 7. Verification status

- **Static:** every new/edited module passes `python -m py_compile`. The SoM
  model's use of the reused `AudioUNet` (`encode`→`(tokens,(h,w),skips)`,
  `decode`, `num_downs`, `bottleneck_dim`) and `MaskHead`/`apply_mask` was
  checked against source.
- **Runtime:** NOT run here — the authoring sandbox has no torch/torchvision/
  numpy and no network. A CPU smoke test (tiny subset, `num_workers=0`) and the
  full curriculum must run on the GPU box. See `SOM_RESULTS.md` for the smoke
  checklist and the target tables to fill in.
