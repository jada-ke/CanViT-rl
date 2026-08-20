# CanViT-rl

Reinforcement-learning experiments for active-vision glimpse selection on top of a frozen CanViT backbone.

## Repository Tour

- `canvit_rl/environment/`: image/model-dependent CanViT environment
  integration.
- `canvit_rl/policies/` and `canvit_rl/state/`: image-independent policy,
  action-conversion, and sequence-state utilities.
- `canvit_rl/canvas/`: shared Canvas-state models, SAC/PPO helpers,
  checkpointing, and visualization.
- `canvit_rl/ade20k/`: ADE20K-specific labels, datasets, rewards, greedy
  segmentation baselines, and mIoU evaluation.
- `canvit_rl/in21k/`: dense ImageNet-21k feature-distillation loaders,
  rewards, checkpoints, and `canvit-pretrain` import shims.
- `canvit_rl/vision/`: CanViT runtime utilities such as precision handling.
- `scripts/`: runnable entry points grouped by workflow, including
  `training/`, `evaluation/`, `analysis/`, `sweeps/`, `diagnostics/`, and
  `baselines/`, then by `ade20k/` or `in21k/` when domain-specific. See
  `scripts/README.md` before adding a new script.
- `scripts/synthetic_dataset/` and `scripts/saliency/`: dataset generation and
  saliency-baseline tooling.
- `data/`: sample datasets for local development (see `data/README.md`).
- `tests/`: focused unit tests grouped by package subsystem.

## Setup

```bash
uv sync
cp .envrc.example .envrc
```

Edit `.envrc` for your machine, then load it:

```bash
source .envrc
```

The template keeps local dataset/checkpoint paths out of Git. If
`CANVIT_CHECKPOINT` is empty, code that accepts a model repo will use its
built-in default or download the configured checkpoint on first use.

After changing CanViT Git dependencies, refresh the lockfile:

```bash
uv lock --upgrade-package canvit-pytorch --upgrade-package canvit-eval --upgrade-package canvit-specialize
uv sync
```


## Apptainer

Build the container from the repo root on a machine with Apptainer and network
access:

```bash
apptainer build apptainer/canvit_rl.sif apptainer/canvit_rl.def
```

Smoke test imports and GPU visibility:

```bash
apptainer exec --nv apptainer/canvit_rl.sif \
  python -c "import torch; import canvit_rl; import canvit_eval; print(torch.__version__, torch.cuda.is_available())"
```
