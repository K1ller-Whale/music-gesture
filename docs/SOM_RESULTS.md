# Sound of Motions — Results (template + smoke checklist)

Fill these tables in as runs complete on the GPU box. Numbers cannot be produced
in the authoring sandbox (no torch/GPU/network there).

---

## 1. Smoke test (do this FIRST, before the full run)

Clean isolated venv, single OpenMP/BLAS runtime. Then:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# tiny subset: ~50 clips, batch 2, num_workers 0, a couple of epochs
python train.py --config configs/som_paper_faithful.yaml   # point train_index at a tiny CSV
```

Expect / verify:
- [ ] one forward+backward step completes (no shape errors through DDT → fusion → FiLM → U-Net).
- [ ] loss is finite and **decreases** over a few hundred steps on the tiny set.
- [ ] `zero_motion` and `zero_appearance` in `eval_som.py` both *reduce* SDR vs `full`
      (sanity that both modalities are actually used).
- [ ] `mask_0.5` floor is well below `full` (no representation collapse).
- [ ] cached-trajectory path (`motion_mode: auto`) and raw-frame path give the
      same shapes.

Env guards (carried from the sibling project): `cv2.setNumThreads(0)`,
`torch.set_num_threads(1)`, spawn start method, `num_workers=0` fallback, fixed
seeds, guard corrupt media.

---

## 2. MUSIC-21, 2-mix — main table (paper targets in parentheses)

| Method | SDR | SIR | SAR |
|---|---|---|---|
| SoM RGB + Trajectory (full) | ___ (8.31) | ___ (14.82) | ___ (13.11) |
| − zero_motion (appearance only) | ___ | ___ | ___ |
| − zero_appearance (motion only) | ___ | ___ | ___ |
| mask=0.5 floor | ___ | ___ | ___ |

(Target row = SoM Table 1, N=2, RGB+Trajectory.)

---

## 3. N-source sweep (self-supervised mixtures)

| N | SDR | SIR | SAR |
|---|---|---|---|
| 2 | ___ | ___ | ___ |
| 3 | ___ | ___ | ___ |
| 4 | ___ | ___ | ___ |

---

## 4. Per-instrument SDR (full, N=2)

| Instrument | SDR | n |
|---|---|---|
| violin | ___ | ___ |
| cello | ___ | ___ |
| congas | ___ | ___ |
| erhu | ___ | ___ |
| xylophone | ___ | ___ |
| … | ___ | ___ |

---

## 5. Curriculum ablation

| Stages run | SDR (N=2) |
|---|---|
| s1 different-instrument only | ___ |
| + s2 same-instrument | ___ |
| + s3 same-video (full curriculum) | ___ |

---

## 6. Notes / deviations from the paper

- Record the exact backbone weights used (official PWC-Net Sintel? official I3D
  `rgb_imagenet.pt`? or trained-from-scratch backends) — this materially affects
  the numbers.
- Record inference masking (thresholded binary @ `mask_threshold` vs `--soft`).
- Record clip length (6 s audio vs SoM's ~3 s / 24-frame visual) and any
  log-freq-warp toggle.
