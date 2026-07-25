# Pretrained backbones for Sound of Motions

The smoke test runs with random weights. To reproduce the paper you need three
pretrained backbones. This doc lists exactly what to clone/download and how the
code loads each one.

| Backbone | Used for | How to get it |
|---|---|---|
| ResNet-18 | appearance branch | torchvision (automatic) |
| I3D `rgb_imagenet.pt` | motion features | `git clone https://github.com/piergiaj/pytorch-i3d` |
| PWC-Net (Sintel) | optical flow | `git clone https://github.com/sniklaus/pytorch-pwc` |

## 0. Where to put weights
```
mkdir -p weights
```

## 1. ResNet-18 (appearance) -- nothing to clone
The config sets `model.appearance.pretrained: true`, so torchvision downloads
ImageNet weights automatically the first time (needs network once; cached to
`~/.cache/torch/hub/checkpoints/`). To pre-cache for an offline box, run this
once on a networked machine and copy the cached file over:
```python
import torchvision; torchvision.models.resnet18(weights="IMAGENET1K_V1")
```

## 2. I3D -- rgb_imagenet.pt
```
git clone https://github.com/piergiaj/pytorch-i3d
cp pytorch-i3d/models/rgb_imagenet.pt weights/rgb_imagenet.pt   # ~50 MB, ships in repo
```

## 3. PWC-Net -- Sintel weights
**The sniklaus repo does NOT ship a weights file** (so `cp pytorch-pwc/network-
default.pytorch` will fail -- that's expected). Its `Network` self-downloads the
weights via torch.hub the first time it's constructed. Two ways:

- **Let it auto-download** (easiest): use the external adapter with
  `flow_weights: null`. On first construction it fetches + caches the weights.
  Needs internet once.
- **Pre-cache the file** (offline boxes):
  ```
  git clone https://github.com/sniklaus/pytorch-pwc
  curl -L http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch \
       -o weights/network-default.pytorch
  ```
  PowerShell equivalent:
  ```
  Invoke-WebRequest http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch -OutFile weights\network-default.pytorch
  ```

**Heads-up:** sniklaus PWC-Net uses a custom **cupy** CUDA correlation kernel --
it needs a GPU and `pip install cupy-cuda12x` (match your CUDA) and does **not**
run on CPU. If you want CPU-capable flow, keep the built-in `models/pwcnet.py`.
(Alternative weights source: NVlabs/PWC-Net for the official Caffe->PyTorch.)

---

# Two ways to load them

## Option A -- built-in loader (quick)
Point the config at the files. On a fresh training start the model calls each
module's `load_pretrained()` and prints how many tensors were populated.
```yaml
model:
  motion:
    i3d_weights:  weights/rgb_imagenet.pt
    flow_weights: weights/network-default.pytorch
```
Expected log:
```
[backbone] I3D (rgb_imagenet): loaded X/Y tensors (missing .., unexpected ..)
[backbone] PWC-Net (Sintel):   loaded X/Y tensors (missing .., unexpected ..)
```
**Watch the numbers.** Because `models/i3d.py` and `models/pwcnet.py` are
clean-room reimplementations, the loader name-matches first and then falls back
to positional shape-matching. **Measured on the official checkpoints this transfer
is poor -- I3D 57/342 (17%), PWC-Net 48/126 (38%)** -- so Option A is NOT paper-
faithful for these files. Use it only for a dependency-free smoke run; for real
reproduction use Option B. Verify anytime with:
```
python scripts/check_weights.py --config configs/som_paper_faithful.yaml \
    --i3d weights/rgb_imagenet.pt --flow weights/network-default.pytorch
```

## Option B -- external official backbone (recommended for exact fidelity)
Keep the official module as the backbone; the DDT only needs the flow/feature
contracts. Adapter templates live in `som_backends/`.
```yaml
model:
  motion:
    i3d_impl: external
    i3d_factory: som_backends.i3d:build_official_i3d
    i3d_weights: weights/rgb_imagenet.pt
    flow_impl: external
    flow_factory: som_backends.pwc:build_official_pwc
    flow_weights: weights/network-default.pytorch
```
Make sure the cloned repos are importable (add them to `PYTHONPATH`, or copy
`pytorch-pwc/run.py`'s `Network` and `pytorch-i3d/pytorch_i3d.py` into your
project). The adapters in `som_backends/` are TEMPLATES -- open them and confirm
the import paths/class names match the commit you cloned; they document the two
non-obvious gotchas (PWC input must be divisible by 64; official I3D stem is
3-channel and must be inflated to the 2-channel trajectory volume, and the
final spatial avg-pool must be skipped to keep a spatial feature map).

## Sanity check after loading
Run a real-data mini training (a handful of clips) and confirm the loss drops,
then `scripts/eval_som.py` -- the `zero_motion` and `zero_appearance` ablations
should both score below `full` if both modalities are contributing.
