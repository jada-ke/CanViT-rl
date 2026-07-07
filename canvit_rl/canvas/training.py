"""Training-facing helpers for Canvas SAC scripts."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from canvit_specialize.datasets.ade20k import make_val_transforms
from torch.utils.data import DataLoader, RandomSampler

from canvit_rl.canvas.eval import dataloader_kwargs
from canvit_rl.env import CanViTEnvConfig
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic
from canvit_rl.synthetic_data import build_segmentation_dataset
from canvit_rl.viewpoint_policy import randomize_actor_mean_viewpoint_prior


@dataclass
class CanvasSacData:
    """Datasets and loaders needed by a Canvas SAC training run."""

    train_dataset: object
    train_eval_dataset: object
    eval_dataset: object
    train_loader: DataLoader
    eval_loader: DataLoader
    train_eval_loader: DataLoader


def sync_for_timing(device: torch.device) -> None:
    """Synchronize CUDA kernels before reading throughput timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def limit_dataset(dataset, max_samples: int | None, *, offset: int = 0):
    """Restrict datasets while preserving deterministic validation subsets."""
    if max_samples is None:
        return dataset
    start = min(offset, len(dataset))
    stop = min(start + max_samples, len(dataset))
    return torch.utils.data.Subset(dataset, range(start, stop))


def build_canvas_sac_dataset(
    *,
    args: argparse.Namespace,
    cfg: CanViTEnvConfig,
    split: str,
    img_tf,
    mask_tf,
):
    """Build ADE20K or synthetic segmentation data from Canvas SAC args."""
    # Problem: script-local wrappers hid that all Canvas trainers use the same
    # dataset contract. Solution: keep argparse-to-dataset plumbing in this
    # package-level helper and delegate actual loading to synthetic_data.
    return build_segmentation_dataset(
        root=Path(args.dataset),
        split=split,
        scene_size_px=cfg.scene_size_px,
        img_transform=img_tf,
        mask_transform=mask_tf,
        dataset_format=args.dataset_format,
        synthetic_image_dir=(
            Path(args.synthetic_image_dir) if args.synthetic_image_dir else None
        ),
        synthetic_mask_dir=(
            Path(args.synthetic_mask_dir) if args.synthetic_mask_dir else None
        ),
    )


def build_canvas_sac_data(
    *,
    args: argparse.Namespace,
    cfg: CanViTEnvConfig,
    device: torch.device,
) -> CanvasSacData:
    """Build Canvas SAC datasets and DataLoaders."""
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    train_dataset = build_canvas_sac_dataset(
        args=args,
        cfg=cfg,
        split=args.split,
        img_tf=img_tf,
        mask_tf=mask_tf,
    )
    train_dataset = limit_dataset(train_dataset, args.max_samples)
    train_eval_dataset = limit_dataset(train_dataset, args.eval_images)
    eval_dataset = build_canvas_sac_dataset(
        args=args,
        cfg=cfg,
        split=args.eval_split,
        img_tf=img_tf,
        mask_tf=mask_tf,
    )
    eval_dataset = limit_dataset(eval_dataset, args.eval_images)
    if len(train_dataset) == 0 or len(train_eval_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("Train and validation datasets must be non-empty.")

    loader_kwargs = dataloader_kwargs(args, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=True),
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return CanvasSacData(
        train_dataset=train_dataset,
        train_eval_dataset=train_eval_dataset,
        eval_dataset=eval_dataset,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_eval_loader=train_eval_loader,
    )


def build_canvas_sac_networks(
    *,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    device: torch.device,
) -> tuple[
    CanvasStateActor,
    CanvasStateCritic,
    CanvasStateCritic,
    CanvasStateCritic,
    CanvasStateCritic,
]:
    """Construct current-canvas actor, twin critics, and target critics."""
    kwargs = dict(
        canvas_feature_dim=canvas_feature_dim,
        d_model=args.d_model,
        rff_dim=args.rff_dim,
        rff_seed=args.rff_seed,
    )
    actor = CanvasStateActor(**kwargs).to(device)
    if (
        getattr(args, "randomize_actor_init", False)
        and getattr(args, "resume", None) is None
        and getattr(args, "init_actor_checkpoint", None) is None
    ):
        prior = randomize_actor_mean_viewpoint_prior(
            actor,
            min_scale=args.min_scale,
            center_radius=args.actor_init_center_radius,
        )
        print(
            "Randomized canvas SAC actor init: "
            f"center=({prior['center_y']:+.3f}, {prior['center_x']:+.3f}) "
            f"scale={prior['scale']:.3f}"
        )
    critic_kwargs = dict(
        kwargs,
        use_action_location_features=getattr(
            args,
            "critic_local_action_features",
            False,
        ),
    )
    # Problem: baseline Canvas critics only saw pooled, action-agnostic canvas
    # features. Solution: pass the opt-in local-feature flag only to critics.
    # Result: actor checkpoints stay comparable while Q networks can be A/B
    # tested with action-location feature sampling.
    q1 = CanvasStateCritic(**critic_kwargs).to(device)
    q2 = CanvasStateCritic(**critic_kwargs).to(device)
    target_q1 = copy.deepcopy(q1).to(device)
    target_q2 = copy.deepcopy(q2).to(device)
    return actor, q1, q2, target_q1, target_q2


def split_eval_metrics(metrics: dict[str, float], split: str) -> dict[str, float]:
    """Return explicit split-prefixed aliases for eval metrics."""
    prefix = f"eval/{split}/"
    return {
        f"{prefix}{name.removeprefix('eval/')}": value
        for name, value in metrics.items()
        if name.startswith("eval/")
    }


def combine_eval_metrics(
    *,
    selected_metrics: dict[str, float],
    train_metrics: dict[str, float],
    selected_split: str,
    train_split: str,
) -> dict[str, float]:
    """Combine eval metrics with clear split names while preserving old aliases."""
    combined = dict(selected_metrics)
    combined.update(split_eval_metrics(selected_metrics, selected_split))
    combined.update(split_eval_metrics(train_metrics, train_split))
    return combined
