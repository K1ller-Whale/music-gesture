# Sound of Motions (SoM) — Reproduction Handoff

> **Purpose:** hand this to another AI agent (or engineer) so they can continue
> this project with zero prior chat history. It captures the goal, the current
> state, the codebase map, the exact environment, every bug already fixed, and
> the concrete next actions.
>
> **Sibling doc:** `HANDOFF.md` at the repo root covers the *other* project in
> this repo (Music Gesture, Gan et al. CVPR 2020). This file is specifically
> about the **Sound of Motions** reproduction built on top of that codebase.
>
> **Last updated:** 2026-07-25

---

## 0. TL;DR — read this first

- **Project:** clean-room, paper-faithful reimplementation of *The Sound of
  Motions* (Zhao, Gan, Ma, Torralba, **ICCV 2019**, arXiv:1904.05979), built
  inside an existing Music Gesture reproduction repo (it reuses that repo's
  audio pipeline and training scaffolding).
- **Scope agreed with the user:** an EXACT replica — full MUSIC-21 replica of
  all paper tables.
- **Status: TRAINING IS RUNNING AND HEALTHY.** The user is partway through
  epoch 0 of stage 0 of the 3-stage curriculum on real MUSIC-21 data. Loss is
  decreasing smoothly (1.4381 → ~0.688 within the first ~1260 steps).
- **All known blockers are resolved.** The last bug (an eval-time state_dict
  key mismatch) was fixed on 2026-07-25.
- **Next action:** let the curriculum run; periodically sanity-check a
  checkpoint with `scripts/eval_som.py` (Section 6), then fill in
  `docs/SOM_RESULTS.md`.

---

## 1. The paper (targets)

- *The Sound of Motions*, Zhao, Gan, Ma, Torralba, ICCV 2019.
- **Headline target — Table 1, N=2, RGB + Trajectory:**
  **SDR 8.31 / SIR 14.82 / SAR 13.11**
- **MUSIC-21 dataset:** 21 categories, 1365 videos (1065 train / 300 test),
  5861 shots.
- **Curriculum (3 stages):** different-instrument → same-instrument →
  same-video (different shots).
- **Optimizer:** SGD, momentum 0.9. LR 1e-3 for separation net + fusion;
  LR 1e-4 for motion + appearance backbones.
- **Video sampling:** 8 FPS, 24 frames per clip (⇒ 23 flow fields), ~6 s clips.

See `docs/SOM_PAPER_SPEC.md` for the full spec, including which values are
pinned by the paper text and which are marked `[assumption]`.

---

## 2. Architecture as implemented

```
mixture spectrogram ──> audio U-Net encoder ──┐
                                              ├─> FiLM (SoM Eq. 2) ─> decoder ─> vision-gated head ─> mask
  frames ─> PWC-Net flow ─> dense trajectories ─> I3D ─┐             │
                                                       ├─> SoM fusion ─> f_v
  first frame ─> ResNet-18 appearance ─────────────────┘
```

- **Motion branch (DDT)** — `models/ddt.py`: PWC-Net optical flow between
  consecutive frames → track a pixel grid through the flow to get a dense
  trajectory volume `[B, 2, T-1, H, W]` → I3D → `[B, 1024, T', H', W']`.
- **Appearance branch** — `models/som_appearance.py`: ResNet-18 over the first
  frame.
- **SoM fusion** — `models/som_fusion.py`: an appearance-derived spatial
  attention map `a(x) = sigmoid(conv(appearance))` **gates** the motion
  features; fused and pooled into a per-source visual feature `f_v`.
- **FiLM conditioning** — `models/som.py::FiLMConditioner`: modulates the audio
  bottleneck by `f_v`.
- **Vision-gated output head** — `models/synthesizer.py::MaskHead`, a
  Sound-of-Pixels-style synthesizer giving `f_v` a direct per-pixel path to the
  mask. **This is inherited from the sibling Music Gesture reproduction, where
  it was the fix that prevented mask collapse. Keep it.**

### Motion tensor contract (auto-detected by shape in `SoundOfMotions._motion_features`)
- frames: `[B, T, 3, H, W]` → flow + tracking computed on the fly
- trajectories: `[B, 2, T-1, H, W]` → cached, the fast training path

---

## 3. Codebase map (SoM-relevant files only)

| Path | Role |
|---|---|
| `models/som.py` | `SoundOfMotions` main model, FiLM, backbone swap/load logic |
| `models/ddt.py` | `DDTConfig`, `DDTMotionNet`, `flow_from_frames`, `compute_trajectories` |
| `models/som_fusion.py` | `SoMFusion` — appearance-gated fusion + attention heatmap |
| `models/som_appearance.py` | `AppearanceNet` (ResNet-18) |
| `models/pwcnet.py` | **Built-in clean-room** PWC-Net (export-friendly, lower fidelity) |
| `models/i3d.py` | **Built-in clean-room** I3D |
| `som_backends/pwc.py` | **External official** PWC-Net wrapper (sniklaus, needs cupy) |
| `som_backends/i3d.py` | **External official** I3D wrapper (piergiaj) |
| `datasets/som_dataset.py` | SoM dataset / collate |
| `configs/som_paper_faithful.yaml` | **The main config.** 3-stage curriculum |
| `configs/som_smoke.yaml` | Tiny smoke-test config |
| `train.py` | Training + curriculum driver (shared with Music Gesture) |
| `scripts/prepare_music21_som.py` | **SoM data prep** (shots, silence, frames, audio, optional cached trajectories) |
| `scripts/eval_som.py` | SDR/SIR/SAR + ablations + per-instrument |
| `scripts/smoke_test_som.py` | Fast end-to-end smoke test |
| `scripts/check_weights.py` | Reports how much of an official checkpoint the built-in loaders absorb |
| `docs/SOM_PAPER_SPEC.md` | Full paper spec + `[assumption]` annotations |
| `docs/SOM_IMPLEMENTATION_PLAN.md` | Implementation plan (⚠️ its "How to run" section is stale) |
| `docs/SOM_RESULTS.md` | **Results table — still to be filled in** |

---

## 4. Environment (user's machine — Windows)

| Item | Value |
|---|---|
| GPU | **NVIDIA GeForce GTX 1660 Ti, 6144 MiB VRAM** ← the binding constraint |
| Driver / CUDA | 566.14 / CUDA 12.7 driver-reported; **CUDA Toolkit v11.2 installed** |
| Python | 3.12 at `C:\Users\H\AppData\Local\Programs\Python\Python312` |
| Key packages | `cupy-cuda11x` 13.6.0, `numpy` 1.26.4 |
| Repo path | `D:\development\python\ai\music-gesture` |
| MUSIC-21 raw videos | `D:\development\python\ai\download_music21_videos\music21_videos` (21 category folders + `download_manifest.csv`) |

**6 GB VRAM is the dominant practical constraint** and is why `batch_size: 2`.
SoM's motion branch (per-frame PWC-Net flow + I3D 3D-conv, run once **per
source**, so ×2 at `num_mix=2`) is far heavier than Music Gesture's
ST-GCN-over-keypoints branch. This is expected architectural behavior, not a bug.

**Do not run training and evaluation concurrently — it will OOM.**

---

## 5. Data preparation

`scripts/prepare_music21_som.py` is written, tested end-to-end, and working. It
does shot detection, silence detection/exclusion, frame extraction, audio
extraction, first-frame extraction, and optionally caches trajectories.

**CSV schema** (`meta_som.csv`, `train.csv`, `val.csv`) — must match
`datasets/som_dataset.py`:

```
audio_path,frames_dir,trajectory_path,first_frame_path,category,video_id,shot_id,clip_start_sec
```

**`--cache_trajectories` is the single biggest training speedup available.** It
precomputes flow + trajectories to disk, removing PWC-Net from the training loop
entirely. It needs a GPU. If training is too slow or VRAM-bound, do this first.

---

## 6. How to run

### Train (full 3-stage curriculum)
```
python train.py --config configs/som_paper_faithful.yaml
```
Runs are **idempotent and resumable at epoch granularity** — re-invoking skips
finished stages and resumes an interrupted stage from its own `last.pth`.

**Epochs per stage** (total 100):

| Stage | dir | mix_policy | Epochs | LR drops | lr_scale |
|---|---|---|---|---|---|
| `s1_different_instrument` | `stage0_s1_different_instrument/` | hetero | 50 | 25, 40 | — |
| `s2_same_instrument` | `stage1_s2_same_instrument/` | homo | 30 | 15, 25 | 0.1 |
| `s3_same_video` | `stage2_s3_same_video/` | same_video | 20 | 10, 16 | 0.01 |

Checkpoints: `runs/som_music21/stage{i}_{name}/last.pth` (rolling) and
`best.pth`. `ckpt_interval: 1`, so one save per epoch.

### Evaluate
Quick collapse sanity-check on an in-progress checkpoint:
```
python scripts/eval_som.py --config configs/som_paper_faithful.yaml \
    --checkpoint runs/som_music21/stage0_s1_different_instrument/last.pth --n 20 --soft
```
Compare the `full` condition against the `mask_0.5` row — `mask_0.5` is the
collapse floor. If `full` isn't clearly beating it, masks are near-constant.

Full paper-table reproduction after all 3 stages:
```
python scripts/eval_som.py --config configs/som_paper_faithful.yaml \
    --checkpoint runs/som_music21/stage2_s3_same_video/last.pth \
    --n-sources 2 3 4 --per-instrument
```

**eval_som.py flags:** `--config`, `--checkpoint` (required), `--n` (default
300 samples per N), `--n-sources` (default `2`), `--per-instrument`, `--soft`
(disable mask thresholding), `--min-ref-rms` (default 1e-4).

**Ablation conditions:** `full`, `zero_motion`, `zero_appearance`, `mask_0.5`.

---

## 7. Pretrained backbones — IMPORTANT

There are **two independent concerns**, and confusing them causes bugs:

1. **`setup_backbones(cfg)` — STRUCTURAL.** If `motion.flow_impl` or
   `motion.i3d_impl == 'external'`, this *replaces the module objects* with the
   official ones from the configured factory. **It must run for every
   stage/script, BEFORE loading any checkpoint**, so state_dict keys match.
2. **`load_pretrained_backbones(cfg)` — VALUES.** Loads Sintel / rgb_imagenet
   weights into the modules. **Only on a truly fresh start.** Later curriculum
   stages inherit fine-tuned weights via `init_from`/`resume`, and eval inherits
   them from the checkpoint.

**The config uses the external official backbones** (`i3d_impl: external`,
`flow_impl: external`) because the clean-room built-in loaders only partially
absorb the official checkpoints — measured by `scripts/check_weights.py` at
**I3D 17%, PWC 38%**. External is required for a faithful reproduction.

**Key-name signature (useful for diagnosing mismatches):**

| | flow | I3D |
|---|---|---|
| built-in clean-room | `motion_net.flow_net.pyramid.*` | `motion_net.i3d.mixed_*` |
| external official | `motion_net.flow_net.net.netExtractor.*` | `motion_net.i3d.net.Mixed_*` |

Weights needed: `weights/rgb_imagenet.pt` (from `piergiaj/pytorch-i3d`) and
`weights/network-default.pytorch` (from `sniklaus/pytorch-pwc`).

**The external PWC-Net requires a GPU + cupy** (no CPU support). Fallback if
cupy is unavailable: `flow_impl: builtin`, `flow_weights: null`,
`freeze_flow: false` — dependency-free but **not paper-faithful**.

---

## 8. Bugs already found and fixed (do not re-introduce)

1. **Gradient leak in the external PWC wrapper** (`som_backends/pwc.py`) —
   caused a backward crash. Fixed; validated by a successful 5-epoch smoke run
   (loss 1.1296 → 0.7035).
2. **CuPy import crash / PWC correlation crash** — fixed and validated.
3. **Stray garbage line in `scripts/prepare_music21_som.py`** — a malformed line
   referencing undefined `METMOD`/`MOD_META`. Removed; file now compiles and is
   validated end-to-end against synthetic ffmpeg test videos (shot splits,
   silence exclusion, CSV schema, frame counts, audio durations all verified).
4. **VRAM saturation on the 6 GB card** — `configs/som_paper_faithful.yaml`
   lowered to `batch_size: 2`, `num_workers: 2`, with an explanatory comment.
   Not a code bug; inherent to SoM's motion branch.
5. **Sequential per-frame-pair optical flow** (`models/ddt.py::flow_from_frames`)
   — was one PWC-Net call per frame pair in a Python loop (23 sequential kernel
   launches per source per clip). Now **batches frame-pairs in chunks** via the
   new `DDTConfig.flow_chunk_size` (default 8, exposed as
   `model.motion.flow_chunk_size` in YAML). Set to 1 to restore the old
   lowest-memory behavior; raise toward T-1 for max throughput.
6. **Redundant audio encoding per source** (`models/som.py`) — `forward()` ran
   `audio_net.encode(spec)` once *per source* even though all sources in a
   mixture share the identical mixture spectrogram. Hoisted into
   `_encode_mixture()` so it runs **once per mixture**. `separate_one()`'s
   signature changed accordingly (it now takes `audio_tokens, hw, skips, Fdim,
   Tdim, ...`); it is only called internally from `forward()`.
7. **`eval_som.py` state_dict key mismatch** (fixed 2026-07-25) — the script
   built the model but never called `setup_backbones(cfg)` before
   `load_state_dict`, so it constructed clean-room backbones and tried to load a
   checkpoint containing external official ones. Fixed by calling
   `setup_backbones(cfg)` before the load (and deliberately **not** calling
   `load_pretrained_backbones`).

---

## 9. Real-time inference / heatmaps (design work, partially done)

The user wants to eventually use the trained model as a **real-time sound
separator + heatmap generator**. Items 5 and 6 above were done partly for this.

**Heatmap signal:** `SoMFusion.forward(..., return_attn=True)` now returns
`attn_raw` `[B, 1, Ha, Wa]` — the raw-resolution sigmoid attention over the
appearance map, i.e. the "where is the sounding object" localization map. It is
plumbed through as `SoundOfMotions.forward(..., return_heatmaps=True)`, which
returns `(masks, heatmaps)`. **Default behavior is unchanged** — with
`return_heatmaps=False` it still returns just `List[Tensor]` of masks, so
`train.py` and `eval_som.py` are unaffected.

Note the heatmap depends only on the appearance frame (not audio), so it can be
computed far more cheaply than a full separation pass.

**NOT yet built — the streaming wrapper.** Proposed design, if asked to build it:
- Maintain a rolling buffer of the last T frames; on each new frame compute only
  the ONE new flow pair instead of recomputing the whole window.
- Keep persistent grid state `G` across calls — `compute_trajectories`' grid
  tracking is inherently sequential, which actually suits streaming well.
- Re-run I3D only every K frames rather than every frame.
- Audio: replace the clip-based 6 s STFT/mask/iSTFT with a sliding-window STFT +
  overlap-add/cross-fade.

**Deployment blocker to flag:** the official external PWC-Net depends on a custom
**cupy correlation kernel** that is **not TorchScript/ONNX/TensorRT
exportable**. For a deployed build, switch to the built-in pure-PyTorch
`models/pwcnet.py` (export-friendly), possibly distilled/fine-tuned from the
official Sintel flow to recover accuracy.

**Other latency levers:** fp16/autocast for flow_net + I3D, lower
`video.frame_size`, raise `motion.traj_grid_stride`, and warm up the cupy JIT
with one dummy inference at startup.

---

## 10. Current state (as of 2026-07-25)

**Training is live and healthy.** Observed:

```
=== curriculum stage 0: s1_different_instrument (mix_policy=hetero, epochs=50) ===
[backbone] flow: external via som_backends.pwc:build_official_pwc
[backbone] official I3D: missing 0, unexpected 0
[backbone] i3d: external via som_backends.i3d:build_official_i3d
[optim] SGD(momentum=0.9) audio+fusion: 40 params @ lr 0.001; gcn+appearance: 233 params @ lr 0.0001
epoch 0 step 0/2152 loss 1.4381
...
epoch 0 step 1260/2152 loss 0.6877
```

**Assessment:** healthy. External backbones loaded correctly (`missing 0,
unexpected 0`); loss decreasing smoothly and monotonically with the expected
fast-then-slowing shape; no NaNs, spikes, or plateaus. 2152 steps/epoch at
`batch_size: 2`.

**Caveat:** loss is not separation quality. Nothing is proven until
`eval_som.py` reports SDR/SIR/SAR.

---

## 11. Next actions (in order)

1. **Let the curriculum finish** — 100 epochs across 3 stages. It's resumable,
   so interruptions are safe.
2. **Sanity-check a checkpoint** once epoch 0 completes, using the `--n 20
   --soft` command in Section 6. Confirm `full` beats `mask_0.5`. Pause training
   first (6 GB VRAM).
3. **Full evaluation** after stage 2 finishes: `--n-sources 2 3 4
   --per-instrument`.
4. **Fill in `docs/SOM_RESULTS.md`** and compare against the N=2 target
   (SDR 8.31 / SIR 14.82 / SAR 13.11).
5. *(Optional, on request)* Build the streaming/real-time wrapper from Section 9.
6. *(Low priority)* Update the stale "How to run" section in
   `docs/SOM_IMPLEMENTATION_PLAN.md` to match the shipped
   `prepare_music21_som.py` CLI.

---

## 12. Working agreements with the user

- The user wants an **exact, paper-faithful replica** — prefer fidelity over
  convenience. When deviating, mark it clearly (the config uses `[assumption]`
  and `[hardware]` annotation tags).
- The user runs everything on their own Windows machine; the assistant's sandbox
  has **no torch/torchvision/cupy/soundfile**, so model code can only be
  `py_compile`-verified there, never runtime-verified. ffmpeg/ffprobe **are**
  available, which is how the data-prep script was tested end-to-end.
- Deliverables are shipped as a zip of the whole repo.
