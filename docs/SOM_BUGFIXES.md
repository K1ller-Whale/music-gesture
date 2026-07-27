# SoM reproduction — bug fix log

All fixes applied in one pass, in response to the stage-0 ablation result that
showed the motion branch contributing nothing:

| run | mix_policy | n | full | zero_motion | zero_appearance | mask_0.5 | Δmotion | Δapp |
|-----|-----------|---|------|-------------|-----------------|----------|---------|------|
| 1 | random | 20 | 1.084 | 1.060 | 0.344 | 0.458 | +0.025 | +0.740 |
| 2 | random | 20 | 1.413 | 1.367 | 0.455 | 0.439 | +0.046 | +0.957 |
| 3 | **homo** | 60 | 0.150 | 0.175 | 0.274 | 0.313 | **−0.025** | **−0.123** |

All three were the same checkpoint (stage 0, 2 of 50 epochs, `--soft`).

Every change is tagged in the source with `[FIX #n]` matching the numbers below,
so `grep -rn "\[FIX #" .` shows all of them in context.

> **Verification status:** every file was checked with `python -m compileall`
> and the config was parsed and asserted with PyYAML. **Nothing was executed at
> runtime** — the sandbox this was fixed in has no torch, CUDA, or cupy. The
> numerical claims below are diagnoses from reading the code, not measured
> results. See "How to verify" at the end.

---

## P0 — must be fixed before any further training

### #1 Config: stage 0 mix policy
`train.stages[0].mix_policy` is **`hetero`** in this package.

The local copy had been edited to `homo`, which makes stages 0 and 1 identical
and collapses the 3-stage curriculum to 2. **Use the config in this zip and
discard the locally edited one.** The knob that changes evaluation is the
top-level `data.mix_policy` (eval does not apply stage overrides); it is left at
`random` and is now printed at eval startup so the two can never be confused
again.

### #2 Trajectory volume was fed to I3D completely unnormalized
**`models/ddt.py`, `models/som.py`, config.** `compute_trajectories` returns raw
pixel displacements (plausibly ±10–50 px at 224×224), and `DDTMotionNet.forward`
passed them straight into an I3D stem inflated from `rgb_imagenet.pt`, which was
pretrained on inputs in ~[−1, 1]. Inputs 10–50× out of distribution saturate the
backbone and produce meaningless motion features — consistent with the measured
`full − zero_motion ≈ 0`.

Added `DDTConfig.traj_norm_scale` (default 20.0) and `traj_clamp`, applied in a
new `normalize_trajectories()` called on **both** the on-the-fly and cached
paths so training and eval always see the same scale.

### #3 `f_v_time` was computed and thrown away
**`models/som.py`.** `SoMFusion` returns `f_v_time [B, T', Kv]`, but
`separate_one` used only `f_v_vec = f_v_time.mean(dim=1)`. A repo-wide grep
confirmed `f_v_time` was never consumed anywhere. All visual conditioning was a
single global 512-d vector broadcast identically across every bottleneck token —
so *when* each source moved, the core SoM cue for separating two same-instrument
sources, was averaged away before it could affect the mask.

`FiLMConditioner.forward` now accepts `f_v_time` and `hw`. The bottleneck token
grid is `[B, h, w, D]` with **h over frequency and w over time** (verified in
`audio_net.encode`: a `[B, D, h, w]` map is flattened, and the spectrogram is
`[B, 1, F, T]`). The visual feature is resampled from `T'` onto those `w` time
steps and a separate (γ, β) is emitted per time step, broadcast over frequency.
Controlled by `model.fusion.temporal_film` (default true); setting it false
restores the old global behavior. No new parameters, so state_dict shape is
unchanged.

### #4 PWC-Net flow rescale was a silent no-op
**`som_backends/pwc.py`.** The rescale ran *after* interpolating to `(H, W)`, so
`flow.shape[-1] == W` and the ratio was exactly 1.0. Official `run.py` scales by
original ÷ padded. At `frame_size: 224` → `Hp = Wp = 256`, so flow magnitudes
were ~14% too large. Now scaled by `(W/Wp, H/Hp)` out-of-place (autograd-safe).
The docstring claimed "pad … and crop" while the code resizes — corrected.

---

## P1 — correctness bugs that silently corrupt results

### #5 LR scheduler was not checkpointed
**`train.py`.** Saved state had no scheduler. On resume `MultiStepLR` was rebuilt
with `last_epoch = 0` while the optimizer restored its already-decayed LR, so
milestones re-fired *relative to the resume point*. Interrupting stage 0 at
epoch 30 added extra decays at absolute epochs 55 and 70, ending the stage at
1e-6 instead of 1e-5. Now saved and restored; legacy checkpoints align
`last_epoch` without double-decaying and print a warning.

### #6 / #10 No temporal augmentation; half of every clip unused
**config `audio.clip_seconds: 6.0 → 3.0`.** `prepare_music21_som.py` writes clips
of exactly `clip_seconds`, so `max_start = audio_dur − clip_seconds = 0` and the
random crop in `som_dataset._load_source` could never move: `start_sec` was
always 0.0 even in train. Consequences: zero temporal augmentation across the
whole 100-epoch curriculum (identical clips every epoch — a real overfitting
risk on 1065 videos), and `f0 = 0` always, so only frames **0–23 of 48** were
ever read.

The visual window is `num_frames / fps = 24 / 8 = 3.0 s`, so a 6 s audio clip
also asked the model to separate 3 s of audio it never saw (#10). Setting
`clip_seconds` to the visual window length fixes both at once: audio and video
now cover the same 3 s, and with 6 s files already on disk the crop roams over
[0, 3] s so all 48 frames get used. **No re-prep required.**

Verified safe: `_load_frame_stack` clamps (`min(f0 + k, len(files) - 1)`) and
`_load_trajectory` edge-pads short segments. At `f0 ≤ 24`, `traj[:, 24:47]` is
exactly the 23 steps wanted.

> **[deviation]** SoM's text mentions ~6 s clips but also 8 FPS × 24 frames =
> 3 s. Those are inconsistent; this matches the explicit frame spec. To go the
> other way instead, re-prep with longer clips and raise `num_frames` to 48.

---

## P2 — evaluation correctness

### #7 Evaluation was unseeded
**`scripts/eval_som.py`.** No seed was set, and the val dataset draws each
mixture's partner sources with `random.sample` on every `__getitem__` (only
`start_sec` was deterministic), so every run scored a *different* set of
mixtures. The two `random` runs above differ by **0.33 dB** on `full` — 7–13×
the motion delta being measured. Added `--seed`, defaulting to
`experiment.seed`; the seed and active `mix_policy` are printed at startup.

### #8 Per-instrument SDR was misattributed
**`scripts/eval_som.py`, `utils/metrics.py`.** `compute_sdr` returns one
mixture-level mean, and that single number was appended to *every* category in
the mixture — so a violin mixed with a tuba credited both identically, and the
per-instrument table could only ever reproduce the overall mean. Added
`compute_sdr_per_source`. `mir_eval` selects the best permutation `popt` and
returns `sdr[popt, arange(nsrc)]`, so the arrays are indexed by **reference**
index and element `i` is genuinely source `i`'s score.

### #9 `zero_appearance` was not a true ablation
**`models/som_fusion.py`, `models/som.py`, `scripts/eval_som.py`.** Zeroing the
input frame still yields non-zero ResNet features (conv biases + BN shift), and
`sigmoid(conv(·))` becomes a constant ≈0.5 map rather than off — so the branch
kept contributing and the delta understated its real value. Ablations are now
requested via `model(..., ablate="motion"|"appearance")` and applied to the
**features**: appearance features → 0 and the gate → 1 (uniform pass-through).

---

## P3 — fidelity deviations

### #13 I3D stem inflation was not scale-preserving
**`som_backends/i3d.py`.** `w.mean(dim=1).repeat(1, in_channels, …)` multiplies
the layer's total response by `in_channels / old_in` — with 3→2 channels the
stem output was 2/3 of the pretrained scale, shifting every downstream BN off
its pretrained statistics. Now rescaled by `old_in / in_channels`.

### #14 `freeze_flow: false` was silently ignored
**`som_backends/pwc.py`, `models/som.py`.** `OfficialPWC.forward` hardcoded
`@torch.no_grad()`, so flow could never be fine-tuned on the external path and
no error was raised. The grad context now follows a `freeze` flag that
`setup_backbones` passes from config (with a `TypeError` fallback for older
factories).

### #12 BatchNorm at batch_size 2
**config `model.freeze_bn: true`, `models/som.py`, `train.py`.** At batch_size 2
(forced by the 6 GB card) BN estimates statistics from 2 samples per step, which
destroys the pretrained ImageNet/Kinetics running stats. Backbone BN layers are
now held in eval mode via `freeze_bn_stats()`, re-applied after every
`model.train()`. Affine weights still train. Set false if you move to a GPU that
allows batch_size ≳ 16.

### #11 Loss / mask-target deviations — documented, NOT changed
`loss_mode: bce_energy_weighted` and `mask_target: dominant` still deviate from
the paper's plain BCE / literal mask. These were deliberate anti-collapse
choices inherited from the sibling reproduction; changing them at the same time
as everything else would risk mask collapse and confound the diagnosis. The
knobs (`bce_plain`, `mask_type`) exist if you want to test paper-literal
behavior later.

---

## P1 (found later, during stage-0 training)

### #15 `same_video` pairing required a different shot
Stage 2's `same_video` pool required the partner clip to come from a *different*
`shot_id`. Measured on the real preprocessed data, **694 of 809 videos (85.8%)
are single-shot**, so that requirement left most videos with an empty pool and
silently fell back to another policy — stage 2 would barely have trained on its
intended distribution. The pool now only excludes the clip itself:

```python
pool = [j for j in self.by_video.get(vid, []) if j != idx]
```

Two clips from the same single-shot video are still a valid same-video pair (they
are different time offsets of the same recording), which is what the stage is
meant to exercise.

### #16 RNG state was not checkpointed; no mid-epoch saving
Two related problems in `train.py`, both observed live during stage 0:

**(a) The random stream restarted on every resume.** `set_seed()` runs once per
process, and the checkpoint stored only model/optimizer/scheduler/epoch. Because
mix-and-separate *synthesises* its training data, the first epoch after a resume
replayed the draws the run's very first epoch had consumed — different mixture
pairings and different clip crops than an uninterrupted run would have used. The
observed symptom: a resumed "epoch 11" reported `avg_loss 0.4113` while the
interrupted attempt at the same epoch, from identical weights, tracked ~0.439.
Epoch-to-epoch loss comparison was silently invalid across any interruption.

**(b) Checkpoints were written only at epoch end,** so a kill mid-epoch discarded
up to ~48 min of work at `batch_size: 2` on a 6 GB card.

Fixes:

- `capture_rng_state()` / `restore_rng_state()` snapshot and restore the python,
  numpy, torch and CUDA RNGs. `torch.load(map_location=device)` moves the RNG
  ByteTensors onto the GPU, so they are forced back to CPU before restoring.
- `ResumableRandomSampler` replaces `DataLoader(shuffle=True)`. The permutation is
  derived from `(seed, epoch)`, so **an epoch's data is now a pure function of the
  config** rather than of how far the RNG stream had advanced. It also supports a
  prefix skip, letting a mid-epoch resume drop already-consumed items *by index*
  — fast-forwarding through the DataLoader instead would recompute trajectories
  on the fly and cost nearly as much as training the steps.
- `WorkerInit` keys each worker's python/numpy RNG to
  `(seed, epoch, worker_id)`. The dataset draws its clip offset
  (`random.uniform`) and mixing partner (`random.sample`) from python's global RNG
  *inside the worker*, and PyTorch seeds workers off the global torch RNG, so
  those draws were stream-position dependent too. `seed_epoch_rng()` covers the
  `num_workers: 0` case, where no worker exists.
- The checkpoint gains `step`, `running` and `rng`. `step: -1` marks a completed
  epoch; `step >= 0` marks a mid-epoch save, and `running` carries the epoch's
  partial loss sum so the printed running mean stays continuous across a resume.
- The curriculum stage-completion test in `main()` now requires
  `epoch >= epochs-1` **and** `step < 0`, so a mid-epoch checkpoint on the final
  epoch is no longer mistaken for a finished stage.
- New knob `train.save_every_steps` (500 in the paper-faithful config; 0 disables).

**Backward compatible:** checkpoints without a `step` key are treated as
epoch-complete, so existing `last.pth` / `best.pth` resume exactly as before —
they just print a warning that their RNG state is unavailable.

### #17 `worker_init_fn` must be picklable (Windows/macOS `spawn`)
A regression introduced by #16, caught on the first run: the worker seeder was a
closure returned by a factory function. On Linux the DataLoader forks and never
serialises it, but **Windows and macOS use the `spawn` start method, which
pickles `worker_init_fn` into each child process**, and a local function cannot
be pickled by name:

```
AttributeError: Can't pickle local object 'make_worker_init_fn.<locals>._init'
```

(The accompanying `EOFError: Ran out of input` in a second traceback is a
secondary artifact — the half-spawned child reading a stream the parent aborted
mid-dump — not an independent bug.)

The closure is now a module-level callable class, `WorkerInit`, holding only the
base seed and a reference to the sampler (plain ints), so it pickles cleanly.
Reading `self.sampler.epoch` inside `__call__` remains correct under `spawn`
because the sampler is pickled when the iterator is created — that is, after
`set_epoch()` has already run for that epoch.

This is also a reminder that anything passed to a DataLoader on this platform
must be a module-level, picklable object: no closures, no lambdas, no local
classes.

**Deliberate deviation, flagged:** per-epoch deterministic shuffling is not
specified by the paper. It does not reduce randomness (every epoch still draws
different pairings and crops); it only makes the draw reproducible. That is
strictly better for a reproduction study, but it does mean this fork's epoch
ordering will not match an upstream run seeded the same way.

---

## Impact on the existing checkpoint

**The 2-epoch stage-0 checkpoint is invalid and must be discarded.** #2, #3, and
#4 all change what the network computes. Restart the curriculum:

```
rm -rf runs/som_music21
python train.py --config configs/som_paper_faithful.yaml
```

This costs 2 epochs now instead of 50 after stage 0 completes.

## How to verify the fixes actually worked

The diagnosis for #2/#3/#4 is strongly evidenced but **unproven** until measured.
The decisive check, after a few epochs:

```
python scripts/eval_som.py --config configs/som_paper_faithful.yaml \
    --checkpoint runs/som_music21/stage0_s1_different_instrument/last.pth \
    --n 60 --soft
```

`full − zero_motion` should move clearly off zero. Because eval is now seeded,
re-running gives identical numbers, so any change is real rather than the
~0.33 dB run-to-run noise seen before. Train and eval still must not run at the
same time on the 6 GB card.
