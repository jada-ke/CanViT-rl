"""
scripts/run_greedy.py

Run a greedy episode and print per-step diagnostics.

Usage:
    python scripts/run_greedy.py
    python scripts/run_greedy.py \
        --episodes 10 --seed 42 --dataset datasets/ADE20k --verbose
    python scripts/run_greedy.py \
        --miou --episodes 5 --t 5 --k 10 --dataset datasets/ADE20k
    python scripts/run_greedy.py --miou-mode accumulator --all --split validation --t 5 --k 10
    python scripts/run_greedy.py \
        --batch-size 8 --miou-mode accumulator --episodes 64 --t 5 --k 50
    python scripts/run_greedy.py \
        --batch-size 8 --miou-mode accumulator --episodes 16 --k 10 \
        --experiment-name greedy-k10-bs8
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from comet_ml import Experiment

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    resolve_canvit_repo,
)
from canvit_rl.greedy import run_greedy_batch
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    NUM_CLASSES,
    ADE20kDataset,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator

def _make_comet_experiment(args: argparse.Namespace):
    """Create a Comet experiment unless explicitly disabled."""
    if args.no_comet:
        return None

    if Experiment is None:
        raise RuntimeError(
            "Comet logging is enabled by default, but comet_ml is not installed. "
            "Install comet-ml or run with --no-comet."
        )

    experiment_name = args.experiment_name or args.comet_experiment_name
    comet_kwargs = {
        "project_name": args.comet_project,
        "auto_param_logging": True,
        "auto_metric_logging": True,
    }
    if args.comet_workspace:
        comet_kwargs["workspace"] = args.comet_workspace
    experiment = Experiment(**comet_kwargs)
    if experiment_name:
        experiment.set_name(experiment_name)
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    return experiment


def _sync_for_timing(device: torch.device) -> None:
    """Synchronize CUDA work so GPU throughput timing reflects completed kernels."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _update_miou(
    acc: mIoUAccumulator,
    model,
    probe: torch.nn.Module,
    state,
    masks: Tensor,
    canvas_grid_size: int,
) -> None:
    """Update one timestep's dataset-level mIoU accumulator."""
    batch_size = masks.shape[0]
    spatial = model.get_spatial(state.canvas).view(
        batch_size,
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float())
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    acc.update(logits.argmax(dim=1), masks)


def _mean_metrics(
    *,
    step_count: int,
    scale_sums: list[float],
    score_sums: list[float],
    loss_reduction_sums: list[float],
    count_sums: list[int],
) -> dict[str, float]:
    """Build flat per-timestep metric names for progress logging."""
    metrics = {}
    for step in range(step_count):
        if count_sums[step] == 0:
            continue
        prefix = f"step_{step + 1}"
        metrics[f"{prefix}/scale"] = scale_sums[step] / count_sums[step]
        metrics[f"{prefix}/seg_ce"] = score_sums[step] / count_sums[step]
        metrics[f"{prefix}/loss_reduction"] = (
            loss_reduction_sums[step] / count_sums[step]
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every image in the selected split instead of sampling --episodes",
    )
    parser.add_argument("--t", type=int, default=5, help="Timesteps per episode")
    parser.add_argument("--k", type=int, default=5, help="Candidates per step")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--miou",
        action="store_true",
        help="Alias for --miou-mode mean",
    )
    parser.add_argument(
        "--miou-mode",
        choices=["none", "mean", "accumulator"],
        default="none",
        help=(
            "none disables mIoU; mean averages per-image mIoU; accumulator "
            "uses dataset-level class intersections/unions"
        ),
    )
    parser.add_argument(
        "--probe-repo",
        type=str,
        default=None,
        help="ADE20K probe repo for segmentation cross-entropy scoring",
    )
    parser.add_argument(
        "--no-full-scene-start",
        action="store_true",
        help=(
            "Use greedy random-candidate search at t=0 instead of a "
            "full-scene glimpse"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ADE20k",
        help="Path to ADE20K root directory",
    )
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="validation",
        help="ADE20K split to sample or run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode step metrics; default is quiet for --all",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Disable Comet logging for this run.",
    )
    parser.add_argument("--comet-project", type=str, default="canvit-rl")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the Comet experiment.",
    )
    parser.add_argument(
        "--comet-experiment-name",
        type=str,
        default=None,
        help="Deprecated alias for --experiment-name.",
    )
    parser.add_argument(
        "--comet-tags",
        type=str,
        default="",
        help="Comma-separated Comet tags.",
    )
    parser.add_argument(
        "--comet-log-interval",
        type=int,
        default=1,
        help="Log Comet progress metrics every N batches.",
    )
    args = parser.parse_args()
    if args.miou and args.miou_mode == "none":
        args.miou_mode = "mean"
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.comet_log_interval < 1:
        raise ValueError("--comet-log-interval must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = CanViTEnvConfig()
    device = get_device()
    print(f"Device: {device}")
    comet_exp = _make_comet_experiment(args)

    # Dataset — use squish mode and ImageNet normalization
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )

    if args.all:
        indices = list(range(len(dataset)))
    else:
        indices = random.sample(range(len(dataset)), min(args.episodes, len(dataset)))
    eval_dataset = Subset(dataset, indices)
    print(f"Dataset: {len(dataset)} {args.split} images, evaluating {len(indices)}")
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

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

    accs = (
        [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(args.t)]
        if args.miou_mode == "accumulator"
        else None
    )
    scale_sums = [0.0 for _ in range(args.t)]
    score_sums = [0.0 for _ in range(args.t)]
    loss_reduction_sums = [0.0 for _ in range(args.t)]
    miou_sums = [0.0 for _ in range(args.t)]
    count_sums = [0 for _ in range(args.t)]
    show_episode_logs = args.verbose and args.batch_size == 1
    n_images_seen = 0
    elapsed_eval_seconds = 0.0
    committed_glimpses_seen = 0
    candidate_glimpses_seen = 0
    candidates_per_image = (
        1 + (args.t - 1) * args.k
        if not args.no_full_scene_start
        else args.t * args.k
    )

    progress = tqdm(loader, desc="Evaluating greedy", unit="batch")
    for batch_idx, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]
        n_images_seen += batch_size

        init_state = model.init_state(
            batch_size=batch_size,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        _sync_for_timing(device)
        batch_start_time = time.perf_counter()
        result = run_greedy_batch(
            model=model,
            images=images,
            init_state=init_state,
            t=args.t,
            k=args.k,
            device=device,
            seed=args.seed + batch_idx,
            masks=masks,
            probe=probe,
            canvas_grid_size=cfg.canvas_grid_size,
            start_with_full_scene=not args.no_full_scene_start,
            compute_miou=args.miou_mode == "mean",
            keep_states=args.miou_mode == "accumulator",
        )
        _sync_for_timing(device)
        batch_elapsed = time.perf_counter() - batch_start_time
        elapsed_eval_seconds += batch_elapsed
        committed_glimpses_seen += batch_size * args.t
        candidate_glimpses_seen += batch_size * candidates_per_image
        batch_committed_gps = (batch_size * args.t) / max(batch_elapsed, 1e-12)
        batch_candidate_gps = (batch_size * candidates_per_image) / max(
            batch_elapsed,
            1e-12,
        )
        progress.set_postfix(
            {
                "glimpses/s": f"{batch_committed_gps:.1f}",
                "cand/s": f"{batch_candidate_gps:.1f}",
            }
        )

        for step in range(args.t):
            scores = result["scores"][step]
            rewards = result["rewards"][step]
            scales = result["scales"][step]
            centers = result["centers"][step]
            scale_sums[step] += float(scales.sum().item())
            score_sums[step] += float(scores.sum().item())
            loss_reduction_sums[step] += float(rewards.sum().item())
            count_sums[step] += batch_size
            if args.miou_mode == "accumulator":
                assert accs is not None
                _update_miou(
                    acc=accs[step],
                    model=model,
                    probe=probe,
                    state=result["states"][step],
                    masks=masks,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
            elif args.miou_mode == "mean":
                miou_sums[step] += float(result["mious"][step].sum().item())
            if show_episode_logs:
                miou_text = ""
                if args.miou_mode == "mean":
                    miou_text = f"  miou={float(result['mious'][step][0].item()):.4f}"
                print(
                    f"  step {step + 1}: "
                    f"scale={float(scales[0].item()):.3f}  "
                    f"center=({float(centers[0, 0].item()):+.3f}, "
                    f"{float(centers[0, 1].item()):+.3f})  "
                    f"seg_ce={float(scores[0].item()):.4f}  "
                    f"loss_reduction={float(rewards[0].item()):+.4f}"
                    f"{miou_text}"
                )

        if comet_exp is not None and (batch_idx + 1) % args.comet_log_interval == 0:
            progress_metrics = _mean_metrics(
                step_count=args.t,
                scale_sums=scale_sums,
                score_sums=score_sums,
                loss_reduction_sums=loss_reduction_sums,
                count_sums=count_sums,
            )
            progress_metrics["progress/images_seen"] = n_images_seen
            progress_metrics["progress/batches_seen"] = batch_idx + 1
            # Fixed by Codex on 2026-06-10
            # Problem: greedy sweeps reported accuracy/loss but not how fast
            # the GPU was processing committed or candidate glimpse forwards.
            # Solution: time each completed rollout batch and log both
            # committed glimpses/sec and k-greedy candidate glimpses/sec.
            progress_metrics["throughput/batch_committed_glimpses_per_sec"] = (
                batch_committed_gps
            )
            progress_metrics["throughput/batch_candidate_glimpses_per_sec"] = (
                batch_candidate_gps
            )
            progress_metrics["throughput/mean_committed_glimpses_per_sec"] = (
                committed_glimpses_seen / max(elapsed_eval_seconds, 1e-12)
            )
            progress_metrics["throughput/mean_candidate_glimpses_per_sec"] = (
                candidate_glimpses_seen / max(elapsed_eval_seconds, 1e-12)
            )
            comet_exp.log_metrics(progress_metrics, step=batch_idx + 1)

    print("\n--- Mean metrics per timestep ---")
    mean_committed_gps = committed_glimpses_seen / max(elapsed_eval_seconds, 1e-12)
    mean_candidate_gps = candidate_glimpses_seen / max(elapsed_eval_seconds, 1e-12)
    print(
        "Throughput: "
        f"{mean_committed_gps:.2f} committed glimpses/s, "
        f"{mean_candidate_gps:.2f} candidate glimpses/s"
    )
    final_metrics = _mean_metrics(
        step_count=args.t,
        scale_sums=scale_sums,
        score_sums=score_sums,
        loss_reduction_sums=loss_reduction_sums,
        count_sums=count_sums,
    )
    for step in range(args.t):
        mean_scale = scale_sums[step] / count_sums[step]
        mean_score = score_sums[step] / count_sums[step]
        mean_loss_reduction = loss_reduction_sums[step] / count_sums[step]
        bar = "█" * int(mean_scale * 20)
        score_text = (
            f"seg_ce={mean_score:.4f}  "
            f"loss_reduction={mean_loss_reduction:+.4f}"
        )
        miou_text = ""
        if args.miou_mode == "accumulator":
            assert accs is not None
            miou_value = accs[step].compute()
            miou_text = f"  miou={miou_value:.4f}"
            final_metrics[f"step_{step + 1}/miou"] = miou_value
        elif args.miou_mode == "mean":
            miou_value = miou_sums[step] / count_sums[step]
            miou_text = f"  miou={miou_value:.4f}"
            final_metrics[f"step_{step + 1}/miou"] = miou_value
        print(
            f"  step {step + 1}: "
            f"scale={mean_scale:.3f} {bar}  "
            f"{score_text}"
            f"{miou_text}"
        )
    if comet_exp is not None:
        final_metrics["final/images_seen"] = n_images_seen
        final_metrics["throughput/mean_committed_glimpses_per_sec"] = (
            mean_committed_gps
        )
        final_metrics["throughput/mean_candidate_glimpses_per_sec"] = (
            mean_candidate_gps
        )
        final_metrics["throughput/eval_seconds"] = elapsed_eval_seconds
        final_metrics["throughput/committed_glimpses"] = committed_glimpses_seen
        final_metrics["throughput/candidate_glimpses"] = candidate_glimpses_seen
        comet_exp.log_metrics(final_metrics)
        comet_exp.end()


if __name__ == "__main__":
    main()
