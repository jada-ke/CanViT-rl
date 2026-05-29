# CanViT-rl

Reinforcement-learning experiments for active-vision glimpse selection on top of a frozen CanViT backbone.

## Setup

```bash
uv sync
cp .envrc.example .envrc
source .envrc
```

Fill in `.envrc` with local dataset/checkpoint paths as needed. If
`CANVIT_CHECKPOINT` is empty, the code will try to download the configured
checkpoint on first use.

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