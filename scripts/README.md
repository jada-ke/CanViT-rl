# scripts/

Runnable entry points, organized by **what kind of task** they run, then by
**dataset/domain**.

```text
scripts/
training/{ade20k,in21k}/     training loops (PPO, SAC, critic pretraining, BC)
evaluation/{ade20k,in21k}/   mIoU / metric evaluation of trained checkpoints
analysis/{ade20k,in21k}/     post-hoc analysis and plotting of results
sweeps/{ade20k,in21k}/       Optuna hyperparameter sweeps
diagnostics/in21k/           one-off correctness/debug checks
baselines/ade20k/            non-learned baseline runners (greedy, entropy c2f)
synthetic_dataset/           generate/inspect the synthetic segmentation dataset
saliency/                    saliency-baseline precomputation and evaluation
```

## Adding a new script

1. Pick the category that matches the task's *purpose* (training vs.
   evaluation vs. analysis, etc.), not the model or paper it came from.
2. Put it under `<category>/<domain>/`, where `<domain>` is `ade20k` or
   `in21k` (add a new domain folder only if the script doesn't fit either).
3. Scripts should be thin: parse args, load data/checkpoints, call into
   `canvit_rl/`. Real logic belongs in the library package, not here -- if a
   script is growing past ~100 lines of non-argparse code, that's a sign
   something should move into `canvit_rl/<subpackage>/` and be imported
   instead.
4. Every script needs a runnable `--help` and a one-line module docstring
   describing what it does and what it expects as input.

## What NOT to put here

- Reusable models, losses, env logic, or data loaders -- those go in
  `canvit_rl/`.
- Sample/generated data -- goes in `data/`.
- Anything imported by more than one script -- extract it into `canvit_rl/`
  first.
