# Music Gesture — Reproduction & Debugging Handoff

> Purpose: hand this to another AI agent (or engineer) so they can continue the
> project without the prior chat history. It captures the goal, the dataset
> preprocessing pipeline, the codebase map, the exact environment, everything
> already fixed, and — most importantly — the **active blocker** and how to
> resolve it.

---

## 0. TL;DR — read this first

- **Project:** clean-room reproduction of *Music Gesture for Visual Sound
  Separation* (Gan et al., CVPR 2020). Mix-and-Separate self-supervision;
  audio U-Net + ResNet-50 context + ST-GCN pose + fusion transformer, with a
  **vision-gated output head** (the fix that made it reproduce).
- **URMP path: WORKING.** With AlphaPose (Halpe-136) pose, Stage 0 (hetero, 30
  epochs) and Stage 1 (homo, 20 epochs) reproduce cleanly.
- **MUSIC-21 path: BLOCKED.** Dataset is downloaded and fully preprocessed with
  **MediaPipe** pose, but training dies with **intermittent memory corruption**
  (three different "impossible" errors at the same correct line, plus segfaults).
- **The corruption is NOT in the code.** The dataset code, the utils, the data,
  and the config have all been audited/proven correct. It is an **environment**
  problem: the training runs in the conda \`tf_gpu\` interpreter while torch etc.
  come from pip \`~/.local\` — a duplicate OpenMP/MKL runtime that corrupts the
  heap over time. Possible secondary suspect: failing server RAM.
- **Next action:** build a clean, isolated virtualenv (commands in Section 9) and
  run training there. If it still corrupts inside a clean venv, run \`memtester\`
  — suspect hardware.

---

## 1. The paper (targets)

- *Music Gesture for Visual Sound Separation*, Gan, Huang, Zhao, Tenenbaum,
  Torralba, CVPR 2020.
- Headline (Table 1, 2-mix MUSIC-21, hetero, no curriculum): **10.12 SDR /
  15.81 SIR**.
- Mask threshold in the paper: **0.7** (our config currently uses 0.5 — see
  Section 7 note).
- Two-stage curriculum: hetero-musical pretrain -> homo-musical finetune
  (Tables 4/5; homo curriculum highlighted on body-related instruments).
- Architecture: audio separation U-Net; ResNet-50 appearance/context features
  (2048-d, no projection); an 11-layer spatial-temporal Graph CNN (CT-GCN) over
  the keypoints; a fusion transformer where **sound queries attend to visual
  features**; vision-conditioned mask prediction.

---

## 2. Current status

| Dataset | Pose | Stage 0 (hetero) | Stage 1 (homo) | State |
|---|---|---|---|---|
| URMP | AlphaPose (Halpe-136) | reproduces (SDR ~4.795) | done (SDR ~3.236) | WORKING |
| MUSIC-21 | MediaPipe | blocked | blocked | training crashes (env) |

**IMPORTANT — do not compare MUSIC-21 numbers to the URMP numbers.** The ~4.8 dB
Stage-0 URMP result was on **AlphaPose** data. MUSIC-21 uses **MediaPipe** pose,
so train==inference must be re-baselined on its own terms (see Section 12).

URMP eval reference numbers (AlphaPose):
- Stage-0: SDR 4.795 / zero_pose 3.608 / zero_ctx 0.339 / mask@0.5 0.236 (std 0.3097)
- Stage-1: SDR 3.236 / zero_pose 1.958 / zero_ctx 0.559 / mask@0.5 0.209 (std 0.2506)

---

## 3. ACTIVE BLOCKER — intermittent memory corruption during MUSIC-21 training

### Symptom
Training dies non-deterministically. Over several runs it produced **three
different, mutually-impossible errors at the exact same line**
(\`datasets/music_dataset.py\`, in \`_sample_others\`, the hetero list
comprehension \`self.samples[j].get("category", "")\`):
1. \`KeyError: 550\`
2. \`TypeError: descriptor 'get' for 'dict' objects doesn't apply to a 'method_descriptor' object\`
3. \`TypeError: list indices must be integers or slices, not str\`

Earlier it also produced hard **segmentation faults** in DataLoader workers
(\`rebuild_storage_fd\` / \`ConnectionResetError: [Errno 104]\` / "worker killed
by signal: Segmentation fault").

### Why this is corruption, not a bug (already proven)
- \`j\` comes from \`range(len(self.samples))\` so it is always an int; a
  diagnostic confirmed \`self.samples\` is a \`list\` and every element
  (\`[0], [1], [550], [4391]\`) is a real \`dict\` with a working \`.get\`.
- An earlier full scan called \`.get("audio_path")\` on all 4,392 rows: 0 errors.
- The crash appeared only **after ~39 epochs** (~750M successful identical calls)
  and the **error type changes every run** — the fingerprint of random heap
  corruption, not a logic bug.
- It happens even with \`num_workers: 0\` (single process), so it is not purely a
  multiprocessing/shared-memory issue.

### What has been ruled out
- **Code:** \`music_dataset.py\` is correct (see Section 6). \`utils/audio.py\`
  and \`utils/pose.py\` audited end-to-end — only standard numpy/torch ops, no
  raw buffers / \`frombuffer\` / \`ctypes\` / stride hacks.
- **Data:** \`scan_samples.py\` scanned all 4,392 samples (audio decode, pose
  load, frame decode) with 0 recoverable-bad and no segfault.
- **OpenCV:** was a pre-release \`5.0.0\`; downgraded to stable 4.x. Helped
  (got further) but did not fix it.
- **DataLoader fd/shm:** \`ulimit -n 65535\` set; \`/dev/shm\` 44 GB, 1% used.
- **spawn start method:** added to \`train.py\`; each worker is a fresh
  interpreter. Still crashes with workers>0 -> points at the packages
  themselves, not fork-inherited state.

### Leading root cause
**Duplicate OpenMP/BLAS runtime.** Training runs under the conda \`tf_gpu\`
interpreter (TensorFlow -> Intel MKL + \`libiomp5\`) while torch/opencv/etc. are
pip \`~/.local\` (\`libgomp\`). Two OpenMP runtimes in one process is a
well-known cause of random heap corruption and segfaults. \`num_workers: 0\`
makes it rarer (got to epoch 39) but not gone.

### Definitive fix (do this next) — clean venv, see Section 9.
If a clean venv **also** corrupts -> suspect **failing RAM**; run
\`sudo memtester 4G 1\` or a memtest86 pass, or try a different machine.

### Immediate stopgap that works
\`num_workers: 0\` reaches epoch 39+ (slower, GPU underfed, but correct). Use it
to make progress while the environment is rebuilt. NOTE: a crash mid-stage
currently restarts that stage from epoch 0 (no intra-stage resume — Section 10).

---

## 4. Environment (uni GPU server)

- Host: \`jaafer.mahfoud@ubuntu-gpu\`, Ubuntu, **16 CPUs, RTX 4090 24 GB**, Python 3.10.
- Repo: \`/home/jaafer.mahfoud/music-gesture/\`. Dataset already uploaded here.
- **Interpreter (problematic mix):** \`/opt/anaconda3/envs/tf_gpu/bin/python\`
  running pip packages from \`/home/jaafer.mahfoud/.local/lib/python3.10/...\`.
- Versions: torch \`2.2.2+cu121\`, numpy \`1.26.4\`, cv2 downgraded to 4.x
  (was 5.0.0), soundfile \`0.12.1\`.
- \`ulimit -n 65535\` set; \`/dev/shm\` 44 GB.
- Laptop (secondary, for local video download only): Intel i7-9750H (6c/12t,
  16 GB), Windows/PowerShell.

---

## 5. Dataset & preprocessing pipeline (MUSIC-21)

### 5.1 Categories (21)
accordion, acoustic_guitar, cello, clarinet, erhu, flute, saxophone, trumpet,
tuba, violin, xylophone, bagpipe, banjo, bassoon, congas, drum, electric_bass,
guzheng, piano, pipa, ukulele.

### 5.2 End-to-end pipeline
1. **Download** (\`scripts/download_music21.py\`): downloads the MUSIC-21
   YouTube videos per category. Streaming/disk-safe; handles unavailable videos.
   (A local laptop variant was also produced.)
2. **Clip extraction + silence-aware sampling** (\`scripts/prepare_music21.py\`):
   for each source video, compute audio RMS **across the whole video** and keep
   up to \`--max_clips_per_video\` non-silent windows whose RMS >= \`--min_rms\`.
   This across-video RMS silence-skipping avoids the "first-6-clips are silence"
   failure and is applied on **both** the AlphaPose and the MediaPipe paths.
   Each kept clip is \`clip_seconds\` (6 s).
3. **Pose extraction** per clip -> \`pose/<clip>.npy\` shaped **[48, 60, 3]**
   (num_frames=48, V=60 joints, 3 = x, y, confidence). 60 = 18 body + 21 right
   hand + 21 left hand.
   - **AlphaPose (URMP):** Halpe-136 -> COCO-18 body + hands via \`HALPE2COCO\`,
     \`HALPE_RHAND0=115\`, \`HALPE_LHAND0=94\`. Persistent worker
     (\`scripts/alphapose_worker.py\`) to avoid per-clip startup cost.
   - **MediaPipe (MUSIC-21):** Holistic pose+hands mapped to the same 60-joint
     layout.
4. **Audio** per clip -> \`audio/<clip>.wav\`, mono, sample_rate 11025, 6 s.
5. **Context frame** per clip -> \`frames/<clip>.jpg\` (a representative RGB
   frame; resized to 224 at load, ImageNet-normalized).
6. **Index** \`meta.csv\` (one row per clip: \`clip,category\`). Sharded runs emit
   \`meta.part<idx>.csv\`, merged by \`scripts/merge_meta.sh\`.
7. **Train/val split** (\`scripts/prepare_data.py --root datasets/processed
   --val_ratio 0.1 --seed 1234\`) -> \`train.csv\` / \`val.csv\`.

### 5.3 Sharded preprocessing (multi-core)
\`scripts/run_shards.sh\` launches N shards on one machine and merges the
per-shard \`meta.part*.csv\` automatically at the end.
\`\`\`
bash scripts/run_shards.sh -n 6 --out datasets/processed [--split] -- \\
  --videos_root <path> --pose mediapipe --max_clips_per_video 6 \\
  --min_rms 0.01 --pose_stride 3
\`\`\`
Do NOT put \`--shard/--tmp/--out\` after \`--\` (those are launcher args). Shards
tile clips by index; they do not overlap. Merge is header-once + append.

### 5.4 On-disk layout (\`datasets/processed/\`)
\`\`\`
audio/<clip>.wav          # mono, 11025 Hz, 6 s
pose/<clip>.npy           # [48, 60, 3]
frames/<clip>.jpg         # context frame
meta.csv                  # clip,category   (shards -> meta.part<idx>.csv)
train.csv / val.csv       # audio_path,pose_path,context_frame_path,category
\`\`\`
- Split CSV columns: \`audio_path,pose_path,context_frame_path,category\`.
- \`meta.csv\` columns: \`clip,category\`.
- Constants: \`BODY=18, HAND=21, VJ=60\`; \`MIN_OK_BYTES=200*1024\`.
- Current train set size: **4,392 clips** -> 549 steps/epoch at batch 8, drop_last.

---

## 6. Codebase map

### 6.1 Dataset — \`datasets/music_dataset.py\` (CONFIRMED CORRECT)
- \`MusicMixDataset(index_file, cfg, split)\`: reads CSV rows into a
  \`list[dict]\` (\`_read_index\` via \`csv.DictReader\`). \`__init__\`
  normalizes samples to a list, builds \`by_category: {category: [indices]}\`,
  sets up pose augmentation and the optional log-freq warp matrix.
- \`__getitem__\`: **Mix-and-Separate**. Picks \`num_mix\` solos (anchor +
  \`_sample_others\` partners chosen by \`mix_policy\`), sums waveforms into a
  mixture, STFTs to magnitude, optional log-frequency warp; returns net_input
  (log-magnitude of mixture), per-source magnitudes, per-source keypoints and
  context frames, and categories.
- **P1 temporal alignment:** one shared \`start_sec\` crops BOTH audio (samples)
  and pose (frames) to the same 6 s window; pose forced to exactly 48 frames.
- \`_sample_others(idx)\`: policy pool — homo (same category via by_category),
  hetero (different category), random; falls back to the unrestricted pool if
  too small.
- \`collate\`: stacks the fixed-num_mix batch (module-level fn; picklable).
- \`cv2.setNumThreads(0)\` at import; \`_load_context\` raises a clear
  \`RuntimeError\` on an unreadable frame instead of feeding None to cvtColor.

### 6.2 Training — \`train.py\`
- CLI: \`--config\` (default \`configs/default.yaml\`), \`--resume\`.
- \`main()\` sets multiprocessing start method to **spawn** (worker-stability
  mitigation) then runs. If \`cfg.train.stages\` exists it runs the curriculum:
  each stage \`train_model(stage_cfg, ..., init_from=prev_ckpt)\` (loads model
  weights only; fresh optimizer). Non-staged path honors \`--resume\`.
- \`train_model\` builds the DataLoader (\`num_workers\` from cfg,
  \`persistent_workers\`/\`worker_init_fn\` only when workers>0, \`pin_memory\`,
  \`drop_last\`, \`file_system\` sharing strategy), the model, SGD/Adam optimizer,
  MultiStepLR scheduler, and BCE/L1 criterion.
- Checkpointing: atomic \`last.pth.tmp\` -> \`last.pth\` each epoch, copy to
  \`best.pth\` on improvement. Keys: \`model, optimizer, epoch, best_loss, cfg\`.
- Optimizer groups (SGD): names starting \`audio_net.\`/\`fusion.\` -> higher LR
  (0.01); everything else (GCN pose + context + mask head) -> lower LR (0.001);
  momentum 0.9.
- Loss: \`bce_energy_weighted\` (default; weights bins by energy above a floor so
  silent bins don't dominate), \`bce_plain\`, or L1 for ratio masks. Targets:
  dominant-source ideal binary mask (or \`literal_mix\`, or ratio).

### 6.3 Models — \`models/\`
\`music_gesture.py\` (top-level module), \`audio_net.py\` (U-Net, ngf 64, 4
downs), \`context_net.py\` (ResNet-50, 2048-d), \`pose_net.py\` (ST/CT-GCN),
\`fusion.py\` (transformer, sound-queries-attend-to-visual), \`synthesizer.py\`
(\`apply_mask\`, vision-gated output head).

### 6.4 Utils — \`utils/\` (AUDITED CLEAN)
\`audio.py\` (STFT/iSTFT, log-freq warp matrices, masks, mix_and_separate),
\`pose.py\` (skeleton adjacency, normalize/augment keypoints), \`metrics.py\`
(SDR/SIR/SAR).

### 6.5 Scripts — \`scripts/\`
- \`download_music21.py\` — download MUSIC-21 YouTube videos.
- \`prepare_music21.py\` — clip extraction + RMS sampling + pose/audio/frame +
  meta. Supports \`--pose mediapipe|alphapose\`, \`--max_clips_per_video\`,
  \`--min_rms\`, \`--pose_stride\`. **Latent bug: see Section 10.**
- \`run_shards.sh\` — N-shard preprocessing + auto-merge (Section 5.3).
- \`merge_meta.sh\` — standalone bash merge of \`meta.part*.csv\` -> \`meta.csv\`.
  Usage: \`bash scripts/merge_meta.sh [DIR=datasets/processed] [OUTPUT]\`.
- \`prepare_data.py\` — train/val split.
- \`alphapose_worker.py\` — persistent AlphaPose worker.
- \`extract_pose.py\` — pose extraction helper.
- \`clean_dataset.py\` — scan every clip in the CSVs; validate audio (full
  \`sf.read\`), pose (\`np.load\`), context frame; \`--fix\` rewrites CSVs
  dropping bad rows (backs up to \`*.bak\`); \`--delete-bad-files\` optional.
  Usage: \`python scripts/clean_dataset.py --root datasets/processed [--fix] [--delete-bad-files]\`.
- \`scan_samples.py\` — faulthandler-armed main-process scanner; prints
  \`[i/n] <path>\` before decoding so the last line before a crash is the
  culprit. Flags: \`--config\`, \`--split\`, \`--limit\`, \`--only audio|pose|frame|all\`.
  Usage: \`python scripts/scan_samples.py --config configs/music21_paper_faithful.yaml\`.
- \`eval_diag.py\` — evaluation/diagnostics (SDR + ablations: zero_pose,
  zero_ctx, mask@thr).
- \`prepare_urmp.py\`, \`prepare_atinpiano.py\` — other datasets.

### 6.6 Other
\`test.py\`, \`separate.py\` (inference; currently a stub), \`docs/ARCHITECTURE.md\`,
\`configs/\` (see Section 7), \`requirements.txt\`.

---

## 7. Config — \`configs/music21_paper_faithful.yaml\`

Key values (paper-faithful on MUSIC-21):
- data: \`num_mix 2\`, \`categories 21\`, \`train_index datasets/processed/train.csv\`,
  \`val_index datasets/processed/val.csv\`, \`mix_policy random\` (overridden per stage).
- audio: \`sample_rate 11025\`, \`clip_seconds 6.0\`, \`n_fft 1022\`, \`hop 256\`,
  \`win 1022\`, \`n_freq 512\`, \`log_freq true\`, \`n_log_freq 256\`,
  \`mask_type binary\`, \`mask_target dominant\`, \`loss_mode bce_energy_weighted\`,
  \`mask_threshold 0.5\` (**paper uses 0.7** — revisit for eval), \`clip_grad 5.0\`,
  \`loss_energy_floor 0.1\`.
- video: \`fps 8\`, \`num_frames 48\`, \`frame_size 224\`, \`body 18\`, \`hand 21\`,
  \`pose_augment true\` (translate 0.05 / scale 0.1 / rotate 10 deg).
- model.audio: \`ngf 64\`, \`num_downs 4\`, \`output_nc 32\`.
  model.context: \`resnet50\`, \`feat_dim 2048\`, pretrained.
  model.pose: graph \`[64,64,64,128,128,128,256,256,256,256,256]\` (11 layers),
  \`stride_layers [3,6]\`, \`temporal_kernel 9\`, \`embed_dim 512\`,
  \`context_inject_after 1\`, \`context_proj_dim 128\`.
  model.fusion: \`dim 512\`, \`depth 3\`, \`heads 8\`, \`mode paper_attn\`.
- train: \`batch_size 8\`, \`num_workers 4\`, \`epochs 100\`, \`optimizer sgd\`,
  \`momentum 0.9\`, \`lr_steps [40,80]\`, \`lr_gamma 0.1\`,
  \`param_groups {audio_fusion 0.01, gcn_appearance 0.001}\`, \`ckpt_interval 1\`.
- **Curriculum stages:** stage 0 \`pretrain_hetero\` (hetero, 60 epochs,
  lr_steps [30,50]); stage 1 \`finetune_homo\` (homo, 40 epochs, lr_steps
  [20,35], lr_scale 0.1). Output under \`runs/stage0_pretrain_hetero/\` and
  \`runs/stage1_finetune_homo/\`.

Other configs: \`music21_paper_faithful_multigpu.yaml\` (adds
\`data_parallel: true\`, \`gpu_ids: null\` for Kaggle 2xT4),
\`urmp_paper_faithful.yaml\`, \`urmp.yaml\`, \`paper_faithful.yaml\`,
\`curriculum.yaml\`, \`atinpiano.yaml\`, \`default.yaml\`.

---

## 8. Fixes already applied (chronological, condensed)

1. **Vision-gated output head** — the key change that made URMP reproduce.
2. Multi-GPU (DataParallel) support + \`*_multigpu.yaml\` (Kaggle 2xT4);
   1-GPU reproducibility preserved.
3. \`resume_stage1.py\`-style stage0->stage1 chaining via \`init_from\`.
4. MUSIC-21 pipeline: \`download_music21.py\`, \`prepare_music21.py\`, notebook,
   configs.
5. Corrupt-video handling in ffmpeg extraction (skip non-zero exit clips).
6. Streaming / disk-overflow-safe download (don't keep all raw videos).
7. **Across-video RMS silence-skipping** clip sampling on both AlphaPose and
   MediaPipe paths (\`--max_clips_per_video\`, \`--min_rms\`).
8. AlphaPose persistent worker; \`DataWriter.final_result\` fix.
9. **NumPy 1.x/2.x ABI fix:** \`pip install --user "numpy<2"\` (-> 1.26.4).
10. **LibsndfileError (bad wav) handling:** \`clean_dataset.py\` scan-and-clean.
11. **KeyError:550 fix:** normalize \`self.samples\` dict->list in
    \`music_dataset.__init__\`.
12. DataLoader worker-stability attempts: \`file_system\` sharing, \`_worker_init\`
    (1 thread + \`cv2.setNumThreads(0)\`), \`pin_memory\`, \`persistent_workers\`,
    \`worker_init_fn\`, None-guard in \`_load_context\`.
13. \`scan_samples.py\` faulthandler diagnostic (+ sys.path repo-root fix).
14. **OpenCV 5.0.0 (pre-release) -> 4.x** downgrade.
15. **spawn** multiprocessing start method in \`train.py main()\`.

(Items 11-15 chased the corruption; the real fix is the environment — Section 9.)

---

## 9. NEXT STEPS (do in order)

### Step 1 — Rebuild a clean, isolated environment (the real fix)
\`\`\`bash
conda deactivate                  # exit tf_gpu (repeat until no env prefix)
/usr/bin/python3 -m venv ~/mgenv  # SYSTEM python, NOT conda's
source ~/mgenv/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4 torch==2.2.2 torchvision==0.17.2 \\
    --index-url https://download.pytorch.org/whl/cu121
pip install soundfile "opencv-python-headless<4.11" pyyaml scipy mediapipe
python -c "import torch,cv2,soundfile,numpy; print('ok', torch.__version__, cv2.__version__, 'cuda', torch.cuda.is_available())"
cd ~/music-gesture
python train.py --config configs/music21_paper_faithful.yaml   # num_workers: 4
\`\`\`
One OpenMP runtime in the env should eliminate BOTH the segfaults and the
phantom \`TypeError\`s, and restore multi-worker loading.

### Step 2 — Cheap check before/instead of rebuilding
\`\`\`bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python train.py --config configs/music21_paper_faithful.yaml
\`\`\`
(Avoid \`KMP_DUPLICATE_LIB_OK=TRUE\` — it silences the abort but lets the
corruption happen silently.)

### Step 3 — If a clean venv ALSO corrupts -> suspect hardware
\`sudo memtester 4G 1\` (or memtest86), or try another machine/GPU node.

### Step 4 — Train-now stopgap
Set \`num_workers: 0\` in the config. Confirmed to reach epoch 39+. Slower but
correct.

### Step 5 — After training completes
- Re-baseline MUSIC-21 + MediaPipe with \`eval_diag.py\` (train==inference;
  do NOT compare to URMP AlphaPose numbers).
- Consider raising \`mask_threshold\` to 0.7 to match the paper for eval.
- Verify the stage0->stage1 chaining produced sane \`runs/stage*/best.pth\`.

---

## 10. Known issues / gotchas / latent bugs

- **No intra-stage resume.** The curriculum path chains stages via
  \`init_from\` but ignores \`--resume\` within a stage, so a crash restarts the
  current stage from epoch 0. Recommended: add epoch-level auto-resume
  (detect latest \`runs/stage<i>_*/last.pth\` and continue from its epoch).
- **Latent URL-brace bug** in \`prepare_music21.py\` \`download_videos()\`
  (~line 121): an f-string of the form \`f"{{...{vid}}}"\` produces literal
  braces in the YouTube URL. The same bug was already fixed in
  \`download_music21.py\`. Fix before using \`prepare_music21.py --download\`.
- **Server file drift.** The server's \`music_dataset.py\` had at times diverged
  from the reference copy (GitHub clone vs uploaded zip). It is now confirmed to
  match the current reference; if line numbers in a traceback don't match,
  re-diff before patching.
- \`separate.py\` is a stub (inference path not finished).
- Cosmetic warnings (torchvision \`pretrained\` deprecation / \`weights\` enum)
  are safe to ignore.

---

## 11. Roadmap (from the plan page)

- **Phase A:** seeded \`eval_diag\`, mask threshold 0.7, hetero/homo split reporting.
- **Phase B:** AlphaPose coverage for the 44 URMP pieces.
- **Phase C:** homo mixing, MUSIC-21 training.
- **Phase D:** report hetero and homo results separately.
- Optional: near-realtime inference tier; note AlphaPose is the runtime
  bottleneck (MediaPipe / on-device keypoint extraction is the mobile path).

---

## 12. Interpreting the numbers (important framing)

- URMP Stage-0 ~4.8 dB SDR was on **AlphaPose** pose data. The paper's headline
  (10.12 SDR) is MUSIC-21, 2-mix, hetero, no curriculum, mask threshold 0.7.
- MUSIC-21 here uses **MediaPipe** pose, so it is a different pose distribution;
  evaluate train==inference on MUSIC-21 itself and do not directly compare to
  the URMP AlphaPose numbers or assume the paper's absolute SDR.
- Ablations to report per run: full model vs \`zero_pose\` vs \`zero_ctx\`, and
  the mask@threshold sanity metric, so the pose/context contributions are
  visible.

---

*End of handoff.*


---

# 13. Sound of Motions (ICCV 2019) branch — added on top of this repo

> A second architecture now lives in this repo: a clean-room, paper-faithful
> reimplementation of **The Sound of Motions** (Zhao, Gan, Ma, Torralba, ICCV
> 2019, arXiv:1904.05979). It reuses this repo's audio pipeline, training loop,
> checkpointing, curriculum scaffolding, eval harness, and config format, and
> **replaces the pose branch** with SoM's Deep Dense Trajectory (DDT) motion
> branch + appearance branch + appearance-gated fusion + FiLM conditioning.
> The Music Gesture path above is unchanged.

## 13.1 Where to read
- Full paper spec (equations, values, ambiguities+assumptions w/ citations):
  `SoM_REIMPLEMENTATION_SPEC_AND_PLAN.md` (project root).
- Engineering plan + reuse/replace map + file inventory + how-to-run:
  `docs/SOM_IMPLEMENTATION_PLAN.md`.
- Results template + smoke checklist + paper target tables: `docs/SOM_RESULTS.md`.

## 13.2 Architecture switch (registries)
- `models.build_model(cfg)` and `datasets.build_dataset(cfg, index, split)`
  switch on `cfg["model"]["type"]` (`music_gesture` default | `som`).
- `train.py` is architecture-agnostic: it calls `build_model` / `build_dataset`
  and `visual_inputs(batch)` (returns pose+context for MG, motion+first_frame
  for SoM). SGD param groups route audio_net/fusion/som_fusion/film/vis_gate to
  the higher LR and pose/motion/appearance backbones to the lower LR.

## 13.3 New files
```
models/pwcnet.py  models/i3d.py  models/ddt.py
models/som_appearance.py  models/som_fusion.py  models/som.py
datasets/som_dataset.py
configs/som_paper_faithful.yaml
scripts/prepare_music21_som.py   scripts/eval_som.py
```
Edited: `models/__init__.py`, `datasets/__init__.py`, `train.py`.

## 13.4 Model (models/som.py)
audio U-Net (reused) + DDT (PWC-Net flow → `G_{t+1}=G_t+grid_sample(ω_t,G_t)`
trajectory volume `[2,T-1,H,W]` → I3D) + ResNet-18 appearance over the first
frame + SoM fusion (σ spatial attention from appearance gates motion, concat,
conv3d, spatial max-pool → f_v) + **FiLM bottleneck** `(1+γ(f_v))·tokens+β(f_v)`
(Eq.2, clip-level) + the **preserved vision-gated output head** (the collapse
fix). Per-source motion tensor auto-detected by shape: frames `[B,T,3,H,W]` or
cached trajectories `[B,2,T-1,H,W]`.

## 13.5 Data (datasets/som_dataset.py, scripts/prepare_music21_som.py)
- CSV schema: `audio_path, frames_dir, trajectory_path, first_frame_path,
  category, video_id, shot_id, clip_start_sec`.
- Preprocessing writes frames + first frame + shots (histogram-diff shot
  detection) + optional cached dense trajectories (`--cache-trajectories`, GPU).
- 3-stage curriculum sampling: `hetero` (different instrument) → `homo` (same
  instrument) → `same_video` (same video_id, different shot_id).
- Audio clip and visual clip cropped from one shared start time (the P1
  alignment fix carried over).

## 13.6 Curriculum resume (train.py)
Staged runs are now **idempotent and epoch-level resumable**: re-invoking skips
finished stages and resumes an interrupted stage from its own `last.pth`. This
fixes the sibling project's "crash mid-stage restarts from epoch 0" pain
(Section 3 stopgap note).

## 13.7 Eval (scripts/eval_som.py)
SDR/SIR/SAR with ablations `full` / `zero_motion` / `zero_appearance` /
`mask_0.5` (SoM analogues of MG's zero_pose/zero_ctx), N-source sweep
(`--n-sources 2 3 4`), per-instrument breakdown (`--per-instrument`),
paper-faithful thresholded binary inference (`--soft` to disable).
SoM Table 1 target (N=2, RGB+Trajectory): **SDR 8.31 / SIR 14.82 / SAR 13.11**.

## 13.8 Verification status & what still needs the GPU box
- **Static:** all new/edited files pass `python -m py_compile`; SoM model
  checked against the reused `AudioUNet`/`MaskHead` source.
- **NOT run here:** authoring sandbox has no torch/torchvision/numpy and no
  network. TODO on the GPU box: (1) CPU smoke test on a tiny subset
  (`docs/SOM_RESULTS.md` §1); (2) load official PWC-Net (Sintel) + I3D
  (`rgb_imagenet.pt`) weights via `PWCNet.load_pretrained` /
  `InceptionI3d.load_pretrained` for faithful numbers; (3) full 3-stage
  curriculum; (4) fill in `docs/SOM_RESULTS.md` tables.
- Same env guards as MG apply: clean venv / single OpenMP, `cv2.setNumThreads(0)`,
  `torch.set_num_threads(1)`, spawn, `num_workers=0` fallback, fixed seeds.
