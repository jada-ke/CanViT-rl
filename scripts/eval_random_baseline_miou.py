"""
Evaluate a random-viewpoint baseline with full-scene t=0 and random glimpses.

Usage:
    uv run python scripts/eval_random_baseline_miou.py
    uv run python scripts/eval_random_baseline_miou.py --t 5 --miou-mode mean
    uv run python scripts/eval_random_baseline_miou.py --t 5 --miou-mode accumulator
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_pytorch.policies import random_viewpoints
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    NUM_CLASSES,
    ADE20kDataset,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import miou_from_state


def _update_miou(
    acc: mIoUAccumulator,
    probe: torch.nn.Module,
    features: Tensor,
    masks: Tensor,
) -> None:
    """Update one timestep's dataset-level mIoU accumulator."""
    with torch.autocast(device_type=features.device.type, enabled=False):
        logits = probe(features.float())
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    acc.update(logits.argmax(dim=1), masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t",
        type=int,
        default=5,
        help="Number of random glimpses after the fixed full-scene t=0",
    )
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="validation",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/random_baseline_miou.pt"),
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--miou-mode",
        choices=["accumulator", "mean"],
        default="accumulator",
        help=(
            "accumulator computes dataset-level mIoU in batches; mean forces "
            "batch_size=1 and averages per-image mIoU values"
        ),
    )
    args = parser.parse_args()

    if args.t < 0:
        raise ValueError("--t must be non-negative.")
    if args.min_scale <= 0 or args.max_scale > 1 or args.min_scale > args.max_scale:
        raise ValueError("Require 0 < --min-scale <= --max-scale <= 1.")

    torch.manual_seed(args.seed)
    cfg = CanViTEnvConfig()
    device = get_device()
    amp = not args.no_amp
    amp_dtype = torch.bfloat16 if amp else torch.float32
    print(f"Device: {device}")

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    effective_batch_size = 1 if args.miou_mode == "mean" else args.batch_size
    if args.miou_mode == "mean" and args.batch_size != 1:
        print(
            "Using effective batch_size=1 for --miou-mode mean; "
            f"requested batch_size={args.batch_size}"
        )
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} {args.split} images")

    probe_repo = args.probe_repo or resolve_canvit_repo(
        f"probe-ade20k-40k-s512-c{cfg.canvas_grid_size}-in21k"
    )
    print(f"Loading CanViT segmentation model with probe: {probe_repo}")
    seg = (
        CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo=cfg.checkpoint,
            probe_repo=probe_repo,
        )
        .eval()
        .to(device)
    )
    model = seg.canvit
    probe = seg.head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in probe.parameters():
        p.requires_grad_(False)

    n_steps = args.t + 1
    accs = (
        [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(n_steps)]
        if args.miou_mode == "accumulator"
        else None
    )
    miou_sums = [0.0 for _ in range(n_steps)]
    scale_sums = [0.0 for _ in range(n_steps)]
    count_sums = [0 for _ in range(n_steps)]
    n_images = 0
    t_start = time.monotonic()

    with torch.inference_mode():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.shape[0]
            n_images += batch_size
            state = model.init_state(
                batch_size=batch_size,
                canvas_grid_size=cfg.canvas_grid_size,
            )

            for step_idx in range(n_steps):
                if step_idx == 0:
                    vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
                else:
                    vp = random_viewpoints(
                        batch_size=batch_size,
                        device=device,
                        n_viewpoints=1,
                        min_scale=args.min_scale,
                        max_scale=args.max_scale,
                        start_with_full_scene=False,
                    ).pop()

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp,
                ):
                    glimpse = sample_at_viewpoint(
                        spatial=images,
                        viewpoint=vp,
                        glimpse_size_px=cfg.glimpse_size_px,
                    )
                    out = model(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state

                if args.miou_mode == "accumulator":
                    assert accs is not None
                    spatial = model.get_spatial(state.canvas).view(
                        batch_size,
                        cfg.canvas_grid_size,
                        cfg.canvas_grid_size,
                        -1,
                    )
                    _update_miou(accs[step_idx], probe, spatial, masks)
                else:
                    miou_sums[step_idx] += (
                        miou_from_state(
                            model=model,
                            state=state,
                            probe=probe,
                            mask=masks,
                            canvas_grid_size=cfg.canvas_grid_size,
                        )
                        * batch_size
                    )
                scale_sums[step_idx] += float(vp.scales.detach().sum().item())
                count_sums[step_idx] += batch_size

    if args.miou_mode == "accumulator":
        assert accs is not None
        mious = {f"t{t}": acc.compute() for t, acc in enumerate(accs)}
    else:
        mious = {f"t{t}": miou_sums[t] / count_sums[t] for t in range(n_steps)}
    mean_scales = {
        f"t{t}": scale_sums[t] / count_sums[t] for t in range(n_steps)
    }
    wall_time = time.monotonic() - t_start

    title = (
        "Dataset-Level Random Baseline Metrics"
        if args.miou_mode == "accumulator"
        else "Mean Random Baseline Metrics"
    )
    print(f"\n--- {title} ---")
    for t in range(n_steps):
        key = f"t{t}"
        label = "full_scene" if t == 0 else "random"
        print(
            f"  t={t} ({label}): "
            f"scale={mean_scales[key]:.3f}  "
            f"miou={mious[key]:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mious": mious,
            "mean_scales": mean_scales,
            "metadata": {
                "policy": "random_viewpoints_after_full_scene",
                "dataset": args.dataset,
                "split": args.split,
                "n_images": n_images,
                "canvas_grid_size": cfg.canvas_grid_size,
                "glimpse_size_px": cfg.glimpse_size_px,
                "scene_size_px": cfg.scene_size_px,
                "n_random_glimpses": args.t,
                "n_logged_steps": n_steps,
                "min_scale": args.min_scale,
                "max_scale": args.max_scale,
                "requested_batch_size": args.batch_size,
                "effective_batch_size": effective_batch_size,
                "probe_repo": probe_repo,
                "model_repo": cfg.checkpoint,
                "seed": args.seed,
                "amp": amp,
                "miou_mode": args.miou_mode,
                "wall_time_seconds": wall_time,
            },
        },
        args.output,
    )
    print(f"\nSaved {args.output} after {wall_time:.1f}s")


if __name__ == "__main__":
    main()
