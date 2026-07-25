# The Sound of Motions (ICCV 2019) — Clean-room reimplementation spec and plan

## Scope note
This document is based on the provided paper text for Zhao, Gan, Ma, and Torralba, **The Sound of Motions**, arXiv:1904.05979v1 / ICCV 2019, and the sibling-project `HANDOFF.md` for the Music Gesture reproduction. The actual Music Gesture repository was **not** attached in this thread, so this is a faithful implementation blueprint and patch map rather than a verified code patch against a real checkout.

---

## 1. Paper spec

### 1.1 Task and core claim
SoM performs **vision-guided sound source separation**. Given a mixed audio waveform and visual input for one source, the model predicts the source-specific spectrogram mask and reconstructs the separated waveform. The paper’s novelty is replacing mostly static appearance cues with explicit **motion cues**, especially for same-instrument duets where appearance alone is ambiguous.

The headline same-category claim is: with motion and a curriculum, the system can separate sounds from **duets of the same instrument category**, e.g. violin/violin, where prior appearance-based approaches fail substantially.

### 1.2 Network components and connections
The full architecture has four components (paper §3.3, Fig. 2):

1. **Motion network = Deep Dense Trajectory (DDT)**
2. **Appearance network**
3. **Fusion module**
4. **Sound separation network**

Connection summary:

```text
RGB frame sequence ──> DDT motion network ──> trajectory features ┐
                                                                  ├─> fusion module ──> visual features ─┐
first RGB frame ─────> appearance ResNet ─────> appearance feats ─┘                                      │
                                                                                                          ▼
mixed waveform ─> STFT magnitude/log-mag ─> audio U-Net, FiLM-conditioned by visual features ─> mask ─> iSTFT
```

#### 1.2.1 Motion network: Deep Dense Trajectory (DDT)
Paper source: §3.2 and §3.3 “Motion Network”.

Input:
- A video clip represented as a sequence of RGB frames.
- Paper experiment config: 3-second clips, 8 FPS, so **24 RGB frames** (§4.2.1).

DDT steps:

**Step i — Dense optical-flow estimation.**
- Use **PWC-Net** (Sun et al. 2018) as the learnable flow estimator (§3.3).
- For 24 input frames, estimate **23 dense optical flow fields** (§4.2.1).
- PWC-Net is initialized from pretrained weights on MPI Sintel (§4.2.1).

**Step ii — Dense trajectory estimation.**
Notation (§3.2):
- Dense optical flow at time `t`: `ω_t = (u_t, v_t)`.
- Tracked pixel position: `P_t = (x_t, y_t)`.
- Adjacent-frame association:

```text
P_{t+1} = (x_{t+1}, y_{t+1}) = (x_t, y_t) + ω_t |_(x_t,y_t)
```

- Position-invariant displacement trajectory representation:

```text
T = (ΔP_t, ΔP_{t+1}, ΔP_{t+2}, ...)
where ΔP_t = (x_{t+1} - x_t, y_{t+1} - y_t)
```

Neural implementation (§3.3):
- Initialize a regular 2D grid `G_0` for frame 0.
- Iteratively sample flow and update the grid:

```text
G_{t+1} = G_t + grid_sample(ω_t, G_t)
```

- Dense trajectories:

```text
T = (ΔP_0, ..., ΔP_t, ...)
  = (grid_sample(ω_0, G_0), ..., grid_sample(ω_t, G_t), ...)
```

- Trajectory tensor dimension is:

```text
T × H × W × 2
```

where the last dimension is `(x, y)` displacement (§3.3).

**Step iii — Dense trajectory feature extraction.**
- Apply a CNN to dense trajectories.
- Paper uses **I3D** (Carreira & Zisserman 2017), initialized from ImageNet-pretrained inflated 2D filters (§3.3, §4.2.1).
- Output feature maps are described as:

```text
T × H × W × K_m
```

in §4.2.1 and as trajectory features `T × H × W × K_T` in Figs. 2–3.

Important implementation notes:
- Classical dense trajectories often subsample, smooth, and normalize trajectories. SoM explicitly **does not** do those operations, assuming the learning system can handle dense/noisy signals (§3.2).
- To avoid tracking drift, the paper performs **shot detection** on untrimmed videos and tracks only within each video shot (§3.2).

#### 1.2.2 Appearance network
Paper source: §3.3 “Appearance Network”.

Input:
- Only the **first frame** of the clip.
- Rationale: keep trajectory feature maps registered with appearance feature maps (§3.3).

Architecture:
- **ResNet-18**.
- Remove layers after spatial average pooling; in practice, use the spatial convolutional feature map before global average pooling so output is spatial:

```text
1 × H × W × K_a
```

as described in §4.2.1 and Fig. 2/Fig. 3.

Output:
- Appearance features `H × W × K_A` / `1 × H × W × K_a`.

#### 1.2.3 Fusion module
Paper source: §3.3 “Attention based Fusion Module” and Fig. 3.

Goal:
- Fuse appearance features and trajectory features into final visual features.

Paper’s attention-based fusion:
1. Predict a **single-channel spatial attention map** from appearance/RGB features:

```text
A ∈ H × W × 1
A = sigmoid(conv(appearance_features))
```

2. Inflate this attention map over time and channel dimensions.
3. Multiply/gate trajectory features with the inflated attention map.
4. Inflate appearance features over time.
5. Concatenate inflated appearance features and gated trajectory features.
6. Apply a few convolution layers.
7. Spatial max-pool to obtain final visual features:

```text
visual_features ∈ T × K_V
```

Paper Fig. 3 also labels a concatenation-only alternative, but the text specifies attention-based fusion as the used module.

Implementation implication vs. Music Gesture:
- Music Gesture’s fusion transformer / sound-query attention should **not** be reused for SoM fusion.
- SoM fusion is a **spatial attention gate + convolution + spatial pooling** module, not a transformer.

#### 1.2.4 Sound separation network
Paper source: §3.3 “Sound Separation Network”, Eq. (2), Fig. 2.

Input:
- Spectrogram of the mixed sound, a 2D time-frequency representation.

Architecture:
- **U-Net** (Ronneberger et al. 2015) with equal-size output mask.
- Experiment config says **6 convolution and 6 deconvolution layers** (§4.2.1).
- Visual conditioning is inserted in the middle/bottleneck of the U-Net.

Visual conditioning:
1. Temporally align visual features with sound features.
2. Apply **Feature-wise Linear Modulation (FiLM)** to audio features.

Paper Eq. (2):

```text
FiLM(f_s) = γ(f_v) · f_s + β(f_v)
```

where:
- `f_s` = sound/audio features,
- `f_v` = visual features,
- `γ(·)` and `β(·)` are single linear layers outputting feature-wise scale and bias.

Output:
- A spectrogram mask after sigmoid activation.
- During reconstruction, the mask is thresholded and multiplied with the input spectrogram, then inverse STFT recovers waveform (§3.3).

### 1.3 Audio front end, mask target, and loss
Paper sources: §3.1, §4.2.1.

Audio preprocessing:
- Sample rate: **11 kHz** in paper text; sibling handoff uses exact **11025 Hz**. Use **11025 Hz** for implementation because the handoff confirms the shared pipeline and exact value.
- Clip length for SoM experiments: **3 seconds** (§4.2.1).
- STFT frame size: **1022** (§4.2.1).
- STFT hop size: **172** (§4.2.1).
- Window length: paper says frame size 1022; assume `win_length = n_fft = 1022` unless repository evidence says otherwise.
- Spectrogram is the model input.

Mask target (§3.1, Eq. (1)):
- During training, randomly select `N` video clips with paired frames/audio `{V_n, S_n}`.
- Mix audios:

```text
S_mix = Σ_{n=1..N} S_n
```

- Given one video clip `V_n`, model predicts target source:

```text
Ŝ_n = f(S_mix, V_n)
```

- The direct output is a **binary mask** applied to the input mixture spectrogram.
- Ground-truth binary mask for source `n`:

```text
M_n(u, v) = 𝟙[S_n(u, v) ≥ S_m(u, v)],  ∀m = 1, ..., N
```

where `(u, v)` are time-frequency coordinates.

Loss:
- **Per-pixel binary cross-entropy loss** between predicted mask and target binary mask (§3.1).

Ambiguity:
- The paper does not mention log-frequency warping. The Music Gesture handoff says to reuse the shared audio pipeline including log-frequency warp. For strict SoM, leave log-frequency configurable and default to the sibling pipeline if matching the existing codebase/eval; mark this as a deviation/assumption.

### 1.4 Mix-and-Separate self-supervised training
Paper source: §3.1.

Procedure:
1. Sample `N` video clips with paired video/audio.
2. Sum their audio tracks to make a synthetic mixture.
3. For each source, condition the separation model on that source’s visual input.
4. Predict the binary source mask.
5. Train with BCE against the ideal dominant-source binary mask.

Self-supervision:
- The network is supervised by synthetic mixtures but requires no semantic labels for audio separation targets (§3.1).

Values:
- Main 2-source experiments use `N = 2` (§4.2.2, Table 1).
- Additional experiments use `N = 3, 4` (§4.2.2, Table 2).

### 1.5 Curriculum learning
Paper source: §3.4.

Used for the same-instrument separation task due to difficulty.

Three-stage curriculum:

1. **Different-instrument mixtures**
   - Randomly sample two video shots from the whole training set.
   - Mix their sounds.
   - Train separation on mixtures of different instruments.

2. **Same-kind instrument mixtures**
   - Initialize from stage 1.
   - Train only with mixtures from the same instrument class, e.g. two cello videos.

3. **Same-video mixtures**
   - Initialize from stage 2.
   - Sample two different video shots from the same long video.
   - Mix them.
   - This is the hardest stage because semantic/context cues can be identical and motion is the main useful cue.

Implementation implication:
- Existing Music Gesture two-stage hetero → homo scaffolding can be reused for stages 1 and 2.
- Need to add stage 3 sampling: same `video_id` / same source long video but distinct shot/clip IDs.

### 1.6 Dataset, splits, clip/frame details, preprocessing
Paper source: §4.1, §4.2.1.

Datasets:
- **MUSIC** (Zhao et al. 2018): unlabeled YouTube videos of instrument solos/duets.
- **URMP** (Li et al. 2019): small high-quality studio multi-instrument video dataset.
- SoM enlarges MUSIC to **MUSIC-21**.

MUSIC-21 categories:
- Original 11 MUSIC categories:
  - accordion, acoustic guitar, cello, clarinet, erhu, flute, saxophone, trumpet, tuba, violin, xylophone
- Additional 10:
  - bagpipe, banjo, bassoon, congas, drum, electric bass, guzheng, piano, pipa, ukulele

MUSIC-21 size/split:
- **1365 untrimmed videos** total.
- **1065 train videos**.
- **300 test videos**.
- After shot detection: **5861 video shots** (§4.1).

Video collection:
- Query YouTube with instrument name plus “cover” (§4.1).

Shot detection:
- Densely sample frames.
- Compute adjacent-frame color histogram changes over time.
- Use a double-thresholding approach based on Canny-style thresholding (§4.1).
- Purpose: prevent DDT tracking across shot boundaries.

Clip details:
- Training clip length: **3 seconds** (§4.2.1).
- RGB frames sampled at **8 FPS**, giving **24 frames** (§4.2.1).
- Audio sampled at **11 kHz / 11025 Hz** (§4.2.1 + handoff exact value).

Frame size:
- Not specified in SoM paper text. Assume **224×224** because PWC-Net/I3D/ResNet pipelines commonly use resized inputs and the Music Gesture handoff uses 224. Flag as ambiguity.

### 1.7 Optimizer and hyperparameters
Paper source: §4.2.1.

Optimizer:
- **SGD** with momentum **0.9**.

Learning rates:
- Sound Separation Network + fusion module: **1e-3**.
- Motion Network + Appearance Network: **1e-4**.
- Rationale: motion/appearance use pretrained modules.

Pretraining:
- ResNet/I3D initialized from ImageNet-pretrained models (§4.2.1).
- PWC-Net initialized from MPI Sintel (§4.2.1).

Not specified / ambiguous:
- Batch size.
- Number of epochs.
- LR schedule.
- Weight decay.
- Data augmentation.
- Exact feature dimensions `K_m`, `K_a`, `K_v`.
- Exact U-Net channel widths beyond 6 conv/6 deconv.

Assumption:
- Reuse Music Gesture training loop/checkpointing/config conventions and set unspecified values to the proven sibling defaults, but annotate every such value as repo-assumed, not paper-specified.

### 1.8 Evaluation protocol and target quantitative results
Metrics:
- `mir_eval` metrics: SDR, SIR, SAR in dB (§4.2.2).

Validation:
- Different-instrument Table 1 uses a validation set with **256 pairs of sound mixtures** (§4.2.2).
- Models are trained/tested with 3-second audios.
- Vision-dependent models take 24 frames (§4.2.2).

#### Table 1 — Different-instrument, N = 2 mixture

| Method | SDR | SIR | SAR |
|---|---:|---:|---:|
| NMF | 2.78 | 6.70 | 9.21 |
| Deep Separation | 4.75 | 7.00 | 10.82 |
| MIML | 4.25 | 6.23 | 11.10 |
| Sound of Pixels | 7.52 | 13.01 | 11.53 |
| SoM RGB single frame | 7.04 | 12.10 | 11.05 |
| SoM RGB multi-frame | 7.67 | 14.81 | 11.24 |
| SoM RGB+Flow | 8.05 | 14.73 | 12.65 |
| **SoM RGB+Trajectory** | **8.31** | **14.82** | **13.11** |

#### Table 2 — Different-instrument, larger mixtures

| N | Method | SDR | SIR | SAR |
|---:|---|---:|---:|---:|
| 3 | NMF | 2.01 | 2.08 | 9.36 |
| 3 | Sound of Pixels | 3.65 | 8.77 | 8.48 |
| 3 | **SoM RGB+Trajectory** | **4.87** | **9.48** | **9.24** |
| 4 | NMF | 0.93 | -1.01 | 9.01 |
| 4 | Sound of Pixels | 1.21 | 6.58 | 4.19 |
| 4 | **SoM RGB+Trajectory** | **3.05** | **8.50** | **7.45** |

#### Table 3 — Curriculum for same-instrument separation

| Schedule | SDR | SIR | SAR |
|---|---:|---:|---:|
| Single Stage | 1.91 | 5.73 | 8.83 |
| Curriculum Stage 1 | 3.14 | 7.52 | 13.06 |
| Curriculum Stage 2 | 5.72 | 13.89 | 11.92 |
| Curriculum Stage 3 | 5.93 | 14.41 | 12.08 |

#### Table 4 — Same-instrument SDR by instrument

| Instrument | Sound of Pixels | SoM |
|---|---:|---:|
| violin | 1.95 | 6.33 |
| cello | 2.62 | 5.48 |
| congas | 2.90 | 5.21 |
| erhu | 1.67 | 6.13 |
| xylophone | 3.56 | 6.50 |

#### Table 5 — Human evaluation, same-instrument mixtures

| Instrument | Sound of Pixels | SoM |
|---|---:|---:|
| violin | 38.75% | 61.25% |
| cello | 39.21% | 60.79% |
| congas | 35.42% | 64.58% |
| erhu | 44.59% | 55.41% |
| xylophone | 35.56% | 64.44% |

Human evaluation protocol:
- 100 testing videos per instrument.
- Compare SoP vs SoM separated results with ground-truth references.
- 3 AMT workers per job.
- Question: “Which sound separation result is closer to the ground truth?” (§4.3.3).

---

## 2. Reuse plan from Music Gesture handoff

### 2.1 Reuse verbatim or with minimal changes
From `HANDOFF.md`:

- Dataset download for MUSIC-21 (`scripts/download_music21.py`).
- Clip extraction and RMS silence-aware sampling (`scripts/prepare_music21.py`) but adapt output to frame stacks / shot IDs.
- Train/val splitting (`scripts/prepare_data.py`), with extension to preserve original `video_id` and `shot_id`.
- Audio pipeline:
  - sample rate 11025 Hz,
  - STFT utilities,
  - iSTFT utilities,
  - binary masks,
  - Mix-and-Separate mixture construction,
  - SDR/SIR/SAR evaluation harness.
- `train.py` staged curriculum scaffolding, checkpointing, resume, logging, deterministic seeds.
- `eval_diag.py` structure for ablations, modified to `zero_motion` and `zero_appearance`.
- Config format and run directory conventions.
- Environment rules: clean venv, one OpenMP/BLAS runtime, native libs single-threaded in workers, spawn start method, scan/clean pass.

### 2.2 Replace
Replace the Music Gesture visual branch:

- Remove/disable pose `.npy` dependency.
- Replace ST-GCN / CT-GCN over keypoints with DDT:
  - PWC-Net flow network,
  - differentiable trajectory tracker,
  - I3D trajectory feature extractor.
- Replace Music Gesture fusion transformer with SoM’s attention-based spatial fusion.
- Use ResNet-18 appearance branch instead of the Music Gesture context ResNet-50 unless keeping ResNet-50 only as a non-paper ablation.

### 2.3 Extend preprocessing and CSV schema
Current sibling CSV:

```text
audio_path,pose_path,context_frame_path,category
```

Proposed SoM CSV:

```text
audio_path,frames_path,first_frame_path,category,video_id,shot_id,clip_start_sec,clip_seconds
```

Where:
- `frames_path` points to cached RGB frame stack, e.g. `.npy` uint8 `[24,H,W,3]`, or a directory of JPEGs.
- `first_frame_path` can be derived from `frames_path` but is kept for compatibility and faster loading.
- `video_id` and `shot_id` are required for curriculum stage 3 same-video/different-shot sampling.
- `clip_start_sec` helps reproducibility/debugging.

Optional heavy-cache fields:

```text
flow_path,trajectory_path
```

Recommendation:
- Start paper-faithful with online PWC-Net/DDT in the model.
- Add optional precomputed `flow_path` for throughput, but mark it as an engineering cache. If flow is precomputed with a frozen PWC-Net, that deviates from end-to-end DDT unless finetuning is disabled by design.

---

## 3. Concrete implementation plan and file map

Assuming the Music Gesture repo layout from `HANDOFF.md`.

### 3.1 Documentation

Create:
- `docs/SOM_PAPER_SPEC.md` — this paper spec.
- `docs/SOM_IMPLEMENTATION_PLAN.md` — file-level plan and deviations.
- `docs/SOM_RESULTS.md` — metrics vs paper tables after training.

Update:
- `HANDOFF.md` — SoM-specific status, exact commands, environment, current blockers.

### 3.2 Preprocessing

Modify/create:

- `scripts/prepare_music21_som.py`
  - Reuse `prepare_music21.py` download/audio/RMS logic.
  - Add shot detection matching paper: frame histogram differences + double thresholding.
  - Extract 3-second clips.
  - Save audio mono 11025 Hz, 3 s.
  - Save RGB stacks at 8 FPS = 24 frames.
  - Save `video_id`, `shot_id`, `clip_start_sec` in metadata.

- `scripts/prepare_urmp_som.py`
  - Same schema for URMP.

- `scripts/scan_som_samples.py`
  - Validate wav decode, frame-stack decode, frame count, shape, non-silence.

- `datasets/som_dataset.py`
  - Class: `SoMMixDataset`.
  - Loads `frames_path` instead of pose.
  - Returns for each mixture component:
    - waveform/source magnitude,
    - RGB frame sequence `[T,3,H,W]`,
    - first frame `[3,H,W]`,
    - category,
    - video_id, shot_id.
  - Sampling policies:
    - `hetero`: different category.
    - `homo`: same category, preferably different source video unless stage says otherwise.
    - `same_video`: same `video_id`, different `shot_id`/clip.
    - `random` for Table 1-style mixed baseline if needed.

### 3.3 Models

Create:

- `models/som.py`
  - Top-level `SoundOfMotions` module.
  - Forward signature compatible with existing training loop.
  - For each source visual input, compute visual features and condition audio U-Net.

- `models/ddt.py`
  - `PWCFlowBackbone` wrapper.
  - `DenseTrajectoryLayer` implementing iterative `grid_sample` tracking.
  - `TrajectoryI3D` feature extractor.
  - `DDTMotionNet` = flow → trajectories → I3D features.

- `models/pwcnet.py`
  - PWC-Net implementation or vendored clean-room equivalent.
  - Load MPI Sintel weights from a user-provided/checkpoint path.
  - If weights are unavailable, fail clearly or allow config `pretrained: false` for smoke tests only.

- `models/i3d.py`
  - I3D implementation for 2-channel trajectory input, initialized from inflated ImageNet 2D weights where possible.

- `models/som_appearance.py`
  - ResNet-18 spatial feature extractor.
  - Output pre-pool spatial map.

- `models/som_fusion.py`
  - `AppearanceMotionFusion`:
    - conv/sigmoid attention from appearance map,
    - inflate over time/channel,
    - gate trajectory features,
    - inflate appearance over time,
    - concatenate,
    - conv stack,
    - spatial max pool → `[B,T,Kv]`.

Modify:

- `models/audio_net.py`
  - Reuse U-Net if it already supports FiLM conditioning.
  - If not, add bottleneck FiLM:
    - temporal alignment of `[B,Tv,Kv]` visual to audio bottleneck time dimension,
    - linear γ/β projection,
    - apply to bottleneck feature maps.

- `models/__init__.py`
  - Register `som` model type.

### 3.4 Training

Modify:

- `train.py`
  - Add model selection for `model.name: som`.
  - Add dataset selection for `dataset.name: som_mix`.
  - Add automatic intra-stage resume: if stage output `last.pth` exists, resume epoch/optimizer unless `--no-auto-resume`.
  - Keep spawn and worker-init safety.

- `utils/audio.py`
  - Keep existing binary dominant mask.
  - Add SoM STFT config `n_fft=1022`, `hop=172`, `win=1022`, `clip_seconds=3.0`.

- `eval_diag.py`
  - Add ablations:
    - `zero_motion`: zero trajectory features after DDT or bypass DDT with zeros.
    - `zero_appearance`: zero appearance map/features before fusion.
    - optionally `zero_visual`: zeros all visual conditioning.
  - Report SDR/SIR/SAR for N=2/3/4.
  - Report same-instrument instruments: violin, cello, congas, erhu, xylophone.

### 3.5 Config

Create:
- `configs/som_paper_faithful.yaml`

Recommended values:

```yaml
seed: 1234

data:
  dataset: som_mix
  train_index: datasets/som_processed/train.csv
  val_index: datasets/som_processed/val.csv
  num_mix: 2
  categories:
    - accordion
    - acoustic_guitar
    - cello
    - clarinet
    - erhu
    - flute
    - saxophone
    - trumpet
    - tuba
    - violin
    - xylophone
    - bagpipe
    - banjo
    - bassoon
    - congas
    - drum
    - electric_bass
    - guzheng
    - piano
    - pipa
    - ukulele
  mix_policy: hetero

audio:
  sample_rate: 11025      # paper: 11 kHz; handoff exact value
  clip_seconds: 3.0       # paper §4.2.1
  n_fft: 1022             # paper §4.2.1
  win_length: 1022        # assumed equal to frame size
  hop_length: 172         # paper §4.2.1
  mask_type: binary       # paper Eq. (1)
  mask_target: dominant   # paper Eq. (1)
  loss: bce               # paper §3.1
  log_freq: true          # inherited from sibling pipeline; paper ambiguous
  n_log_freq: 256         # repo assumption if log_freq=true

video:
  fps: 8                  # paper §4.2.1
  clip_frames: 24         # 3 sec × 8 FPS
  frame_size: 224         # paper ambiguous; repo assumption
  shot_detection: true

model:
  name: som
  motion:
    name: ddt
    flow: pwcnet
    pwc_pretrained: checkpoints/pwcnet_sintel.pth
    trajectory_input_channels: 2
    feature_extractor: i3d
  appearance:
    backbone: resnet18
    pretrained: true
    output: spatial_pre_pool
  fusion:
    type: appearance_spatial_attention
    attention_activation: sigmoid
    spatial_pool: max
  audio_unet:
    conv_layers: 6
    deconv_layers: 6
    conditioning: film

train:
  optimizer: sgd
  momentum: 0.9
  lr_audio_fusion: 1.0e-3       # paper §4.2.1
  lr_motion_appearance: 1.0e-4  # paper §4.2.1
  batch_size: 8                 # not paper-specified; repo assumption
  epochs: 100                   # not paper-specified; repo assumption
  num_workers: 4
  deterministic: true
  stages:
    - name: stage1_hetero_different_instruments
      mix_policy: hetero
      epochs: 60                # repo assumption; paper gives stage definition, not epochs
    - name: stage2_homo_same_instrument
      mix_policy: homo
      epochs: 40                # repo assumption
      init_from_previous: true
    - name: stage3_same_video
      mix_policy: same_video
      epochs: 20                # assumption; paper gives stage definition, not epochs
      init_from_previous: true
```

### 3.6 Verification milestones

1. **Static validation**
   - Load one batch.
   - Confirm shapes:
     - frames `[B,N,24,3,224,224]`,
     - first frame `[B,N,3,224,224]`,
     - audio mixture spectrogram shape matches U-Net output.

2. **DDT unit test**
   - Input frames → 23 flow fields.
   - Flow → trajectory tensor `[B,23,H,W,2]`.
   - Trajectory I3D → `[B,T',H',W',Km]`.

3. **Fusion unit test**
   - Appearance map and trajectory map spatial sizes align or are resized.
   - Output visual features `[B,Tv,Kv]`.

4. **Forward/backward smoke test**
   - Tiny dataset, batch 1–2, 5–20 steps.
   - BCE decreases / gradients finite.

5. **Ablation smoke tests**
   - Full vs zero-motion vs zero-appearance produce valid metrics and non-identical masks.

6. **Paper table reproduction runs**
   - Table 1: N=2 different-instrument.
   - Table 2: N=3/N=4 different-instrument.
   - Table 3: same-instrument curriculum stages.
   - Table 4: same-instrument SDR by violin/cello/congas/erhu/xylophone.

---

## 4. Results report template

Fill after training:

### 4.1 Environment
- Python:
- Torch/CUDA:
- OpenCV:
- GPU:
- Dataset root:
- Commit hash:
- Config path:

### 4.2 Smoke test
- Batch shapes:
- Initial loss:
- Final smoke loss:
- Max GPU memory:

### 4.3 Different-instrument N=2

| Method/config | SDR | SIR | SAR | Gap vs paper RGB+Trajectory |
|---|---:|---:|---:|---:|
| Paper SoM | 8.31 | 14.82 | 13.11 | 0 |
| This run | TBD | TBD | TBD | TBD |
| zero-motion | TBD | TBD | TBD | TBD |
| zero-appearance | TBD | TBD | TBD | TBD |

### 4.4 N=3/N=4

| N | Paper SDR | This SDR | Paper SIR | This SIR | Paper SAR | This SAR |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 4.87 | TBD | 9.48 | TBD | 9.24 | TBD |
| 4 | 3.05 | TBD | 8.50 | TBD | 7.45 | TBD |

### 4.5 Same-instrument curriculum

| Stage | Paper SDR | This SDR | Paper SIR | This SIR | Paper SAR | This SAR |
|---|---:|---:|---:|---:|---:|---:|
| Single stage | 1.91 | TBD | 5.73 | TBD | 8.83 | TBD |
| Curriculum 1 | 3.14 | TBD | 7.52 | TBD | 13.06 | TBD |
| Curriculum 2 | 5.72 | TBD | 13.89 | TBD | 11.92 | TBD |
| Curriculum 3 | 5.93 | TBD | 14.41 | TBD | 12.08 | TBD |

### 4.6 Known gaps / likely causes
- Dataset availability and exact YouTube videos may differ.
- Paper omits batch size, epochs, LR schedule, weight decay, augmentation.
- PWC-Net/I3D pretrained checkpoint differences can affect DDT.
- Log-frequency warp is inherited from sibling implementation but not explicit in SoM paper.
- Frame size unspecified.

---

## 5. Updated handoff for next agent

### Current completed work
- Paper read end-to-end from provided text.
- SoM paper spec extracted with equations, values, tables, and ambiguities.
- Music Gesture `HANDOFF.md` reviewed.
- Reuse/replace plan written.

### Still needed
- Attach or clone the actual Music Gesture repo.
- Implement files listed in §3.
- Add `configs/som_paper_faithful.yaml`.
- Run smoke tests and full training.
- Populate results report.

### Critical blockers to avoid
- Do not run in mixed conda+pip interpreter.
- Use clean venv with one OpenMP runtime.
- Cache DDT/flow carefully; it is compute-heavy.
- Add same-video sampling metadata before preprocessing, or stage 3 cannot be faithful.

