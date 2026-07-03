"""Centralized Canvas SAC validation helpers used by training scripts."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.policies import random_viewpoints
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    NUM_CLASSES,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_rl.canvas.state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    empty_viewpoint_history,
)
from canvit_rl.env import CanViTEnvConfig
from canvit_rl.greedy import _segmentation_cross_entropy_losses
from canvit_rl.sac_models import CanvasStateActor
from canvit_rl.synthetic_data import build_segmentation_dataset
from canvit_rl.viewpoint_policy import action_to_viewpoint

EVAL_REPO = Path(__file__).resolve().parents[1] / "CanViT-eval"
if EVAL_REPO.is_dir() and str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))

DATALOADER_PREFETCH_FACTOR = 4


def viewpoint_entropy(values: list[np.ndarray], *, bins: int) -> float:
    """Entropy of visited (y, x, scale) bins, normalized to [0, 1]."""
    if not values:
        return 0.0
    points = np.concatenate(values, axis=0)
    if points.shape[0] <= 1:
        return 0.0
    hist, _ = np.histogramdd(
        points,
        bins=bins,
        range=[[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
    )
    probs = hist.reshape(-1).astype(np.float64)
    probs = probs[probs > 0]
    probs = probs / probs.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / max(float(np.log(hist.size)), 1e-12)


def dataloader_kwargs(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Build DataLoader settings that overlap input loading with CUDA work."""
    kwargs: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR
    return kwargs


def segmentation_metrics(
    *,
    model,
    probe: torch.nn.Module,
    state,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    acc: mIoUAccumulator | None = None,
) -> tuple[torch.Tensor, float | None]:
    """Return per-image CE and optionally update a dataset-level mIoU acc."""
    losses = _segmentation_cross_entropy_losses(
        model=model,
        state=state,
        probe=probe,
        canvas_grid_size=cfg.canvas_grid_size,
        mask=masks,
        batch_size=masks.shape[0],
    )
    miou = None
    if acc is not None:
        update_miou_accumulator(
            model=model,
            probe=probe,
            state=state,
            masks=masks,
            cfg=cfg,
            acc=acc,
        )
        miou = float(acc.compute())
    return losses, miou


def update_miou_accumulator(
    *,
    model,
    probe: torch.nn.Module,
    state,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    acc: mIoUAccumulator,
) -> None:
    """Update dataset-level mIoU for one CanViT canvas state without CE."""
    spatial = model.get_spatial(state.canvas).reshape(
        masks.shape[0],
        cfg.canvas_grid_size,
        cfg.canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float()).float()
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    acc.update(logits.argmax(dim=1), masks)


def eval_random_batch(
    *,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvit_dtype: torch.dtype,
    acc: mIoUAccumulator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out random learned viewpoints after a full-scene warm-up."""
    device = images.device
    state = model.init_state(batch_size=images.shape[0], canvas_grid_size=cfg.canvas_grid_size)
    full_vp = Viewpoint.full_scene(batch_size=images.shape[0], device=device)
    with torch.inference_mode():
        full_glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        ).to(dtype=canvit_dtype)
        full_out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
        state = full_out.state
        initial_ce, _ = segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg
        )
        for _ in range(args.t):
            vp = random_viewpoints(
                batch_size=images.shape[0],
                device=device,
                n_viewpoints=1,
                min_scale=args.min_scale,
                max_scale=1.0,
                start_with_full_scene=False,
            )[0]
            glimpse = sample_at_viewpoint(
                spatial=images,
                viewpoint=vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state
        final_ce, _ = segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg, acc=acc
        )
    return initial_ce, final_ce


def eval_egc2f_batch(
    *,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvit_dtype: torch.dtype,
    acc: mIoUAccumulator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate canvit-eval entropy-guided coarse-to-fine on the same horizon."""
    try:
        from canvit_eval.episode import run_episode
        from canvit_eval.policies import make_policy
    except ImportError as exc:
        raise RuntimeError(
            "EG-C2F evaluation requires canvit-eval import support. "
            "Place CanViT-eval next to this repo or install it."
        ) from exc
    if args.t + 1 > 21:
        raise ValueError("EG-C2F has 21 built-in timesteps; require --t <= 20.")
    batch_size = images.shape[0]
    policy = make_policy(
        "entropy_coarse_to_fine",
        batch_size=batch_size,
        device=images.device,
        n_viewpoints=args.t + 1,
        canvas_grid=cfg.canvas_grid_size,
        probe=probe,
        get_spatial_fn=model.get_spatial,
    )
    with torch.inference_mode():
        steps = run_episode(
            model=model,
            images=images.to(dtype=canvit_dtype),
            policy=policy,
            n_timesteps=args.t + 1,
            canvas_grid=cfg.canvas_grid_size,
            glimpse_px=cfg.glimpse_size_px,
        )
        initial_ce, _ = segmentation_metrics(
            model=model, probe=probe, state=steps[0].state, masks=masks, cfg=cfg
        )
        final_ce, _ = segmentation_metrics(
            model=model,
            probe=probe,
            state=steps[-1].state,
            masks=masks,
            cfg=cfg,
            acc=acc,
        )
    return initial_ce, final_ce


def eval_canvas_sac_batch(
    *,
    actor: CanvasStateActor,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    canvit_dtype: torch.dtype,
    acc: mIoUAccumulator,
    scale_sums: list[float],
    scale_counts: list[int],
    entropy_points: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out deterministic image-dependent SAC over a validation batch."""
    del canvas_feature_dim
    device = images.device
    batch_size = images.shape[0]
    state = model.init_state(batch_size=batch_size, canvas_grid_size=cfg.canvas_grid_size)
    coords, lengths = empty_viewpoint_history(
        batch_size=batch_size,
        max_steps=args.max_history,
        device=device,
    )
    full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
    with torch.inference_mode():
        full_glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        ).to(dtype=canvit_dtype)
        full_out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
        state = full_out.state
        initial_ce, _ = segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg
        )
        canvas_summary = canvas_layernorm_spatial(
            model=model,
            state=state,
            canvas_grid_size=cfg.canvas_grid_size,
        )
    coords, lengths = append_viewpoint_history(
        coords=coords,
        lengths=lengths,
        viewpoint=full_vp,
        step=0,
    )
    for step_idx in range(args.t):
        obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
        with torch.no_grad():
            action = actor.deterministic_action(obs)
        vp = action_to_viewpoint(action, min_scale=args.min_scale)
        entropy_points.append(
            torch.cat([vp.centers, vp.scales[:, None]], dim=1).detach().cpu().numpy()
        )
        scale_sums[step_idx] += float(vp.scales.detach().sum().item())
        scale_counts[step_idx] += batch_size
        with torch.inference_mode():
            glimpse = sample_at_viewpoint(
                spatial=images,
                viewpoint=vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=cfg.canvas_grid_size,
            )
        coords, lengths = append_viewpoint_history(
            coords=coords,
            lengths=lengths,
            viewpoint=vp,
            step=step_idx + 1,
        )
    with torch.inference_mode():
        final_ce, _ = segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg, acc=acc
        )
    return initial_ce, final_ce


def evaluate_canvas_sac(
    *,
    actor: CanvasStateActor,
    eval_loader: DataLoader,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    canvit_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate Random, EG-C2F, and canvas SAC on a fixed validation subset."""
    eval_random = not args.skip_eval_random
    eval_egc2f = not args.skip_eval_egc2f
    random_acc = (
        mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) if eval_random else None
    )
    egc2f_acc = (
        mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) if eval_egc2f else None
    )
    sac_acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    ce_sums = defaultdict(float)
    n_images = 0
    sac_scale_sums = [0.0 for _ in range(args.t)]
    sac_scale_counts = [0 for _ in range(args.t)]
    sac_entropy_points: list[np.ndarray] = []

    for images, masks in tqdm(eval_loader, desc="Evaluating canvas SAC", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]
        n_images += batch_size

        if eval_random:
            assert random_acc is not None
            rand_initial, rand_final = eval_random_batch(
                model=model,
                probe=probe,
                images=images,
                masks=masks,
                cfg=cfg,
                args=args,
                canvit_dtype=canvit_dtype,
                acc=random_acc,
            )
            ce_sums["random_initial"] += float(rand_initial.sum().item())
            ce_sums["random_final"] += float(rand_final.sum().item())
        if eval_egc2f:
            assert egc2f_acc is not None
            eg_initial, eg_final = eval_egc2f_batch(
                model=model,
                probe=probe,
                images=images,
                masks=masks,
                cfg=cfg,
                args=args,
                canvit_dtype=canvit_dtype,
                acc=egc2f_acc,
            )
            ce_sums["egc2f_initial"] += float(eg_initial.sum().item())
            ce_sums["egc2f_final"] += float(eg_final.sum().item())
        sac_initial, sac_final = eval_canvas_sac_batch(
            actor=actor,
            model=model,
            probe=probe,
            images=images,
            masks=masks,
            cfg=cfg,
            args=args,
            canvas_feature_dim=canvas_feature_dim,
            canvit_dtype=canvit_dtype,
            acc=sac_acc,
            scale_sums=sac_scale_sums,
            scale_counts=sac_scale_counts,
            entropy_points=sac_entropy_points,
        )
        ce_sums["sac_initial"] += float(sac_initial.sum().item())
        ce_sums["sac_final"] += float(sac_final.sum().item())

    random_miou = float(random_acc.compute()) if random_acc is not None else math.nan
    egc2f_miou = float(egc2f_acc.compute()) if egc2f_acc is not None else math.nan
    sac_miou = float(sac_acc.compute())
    random_ce = ce_sums["random_final"] / max(n_images, 1) if eval_random else math.nan
    egc2f_ce = ce_sums["egc2f_final"] / max(n_images, 1) if eval_egc2f else math.nan
    sac_initial_ce = ce_sums["sac_initial"] / max(n_images, 1)
    sac_ce = ce_sums["sac_final"] / max(n_images, 1)
    ce_gain = sac_initial_ce - sac_ce
    eval_reward = ce_gain / max(sac_initial_ce, 1e-12)
    metrics = {
        "eval/random_miou": random_miou,
        "eval/egc2f_miou": egc2f_miou,
        "eval/sac_miou": sac_miou,
        "eval/random_final_ce": random_ce,
        "eval/egc2f_final_ce": egc2f_ce,
        "eval/sac_final_ce": sac_ce,
        "eval/final_miou": sac_miou,
        "eval/final_ce": sac_ce,
        "eval/miou_gain": sac_miou - random_miou,
        "eval/ce_gain": ce_gain,
        "eval/reward": eval_reward,
        "eval/relative_ce_gain": eval_reward,
        "eval/random_ce_gain": (
            ce_sums["random_initial"] / max(n_images, 1) - random_ce
            if eval_random
            else math.nan
        ),
        "eval/egc2f_ce_gain": (
            ce_sums["egc2f_initial"] / max(n_images, 1) - egc2f_ce
            if eval_egc2f
            else math.nan
        ),
        "eval/sac_vs_random": sac_miou - random_miou,
        "eval/sac_vs_egc2f": sac_miou - egc2f_miou,
        "eval/sac_viewpoint_entropy": viewpoint_entropy(
            sac_entropy_points,
            bins=args.viewpoint_entropy_bins,
        ),
    }
    metrics.update(
        {
            "final_miou": metrics["eval/final_miou"],
            "final_ce": metrics["eval/final_ce"],
            "miou_gain": metrics["eval/miou_gain"],
            "ce_gain": metrics["eval/ce_gain"],
            "reward": metrics["eval/reward"],
            "relative_ce_gain": metrics["eval/relative_ce_gain"],
            "sac_vs_random": metrics["eval/sac_vs_random"],
            "sac_vs_egc2f": metrics["eval/sac_vs_egc2f"],
            "viewpoint_entropy": metrics["eval/sac_viewpoint_entropy"],
        }
    )
    for step_idx in range(args.t):
        metrics[f"eval/sac_mean_scale_by_t{step_idx + 1}"] = (
            sac_scale_sums[step_idx] / max(sac_scale_counts[step_idx], 1)
        )
    return metrics


def should_run_final_full_validation_miou(args: argparse.Namespace) -> bool:
    """Return whether this run should finish with a full validation mIoU pass."""
    if getattr(args, "skip_final_full_validation_miou", False):
        return False
    return getattr(args, "optuna_trial", None) is None


def evaluate_best_full_validation_miou(
    *,
    actor: CanvasStateActor,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvit_dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[float]]:
    """Evaluate best.pt per timestep on validation with mIoUAccumulator."""
    best_path = args.checkpoint_dir / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Cannot run final full mIoU; missing {best_path}")

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    full_validation_dataset = build_segmentation_dataset(
        root=Path(args.dataset),
        split="validation",
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
    if len(full_validation_dataset) == 0:
        raise ValueError("Full validation dataset must be non-empty.")

    full_validation_loader = DataLoader(
        full_validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **dataloader_kwargs(args, device),
    )

    accs = [
        mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
        for _ in range(args.t + 1)
    ]
    n_images = 0

    for images, masks in tqdm(
        full_validation_loader,
        desc="Full validation timestep mIoU for best Canvas SAC",
        leave=False,
    ):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]
        n_images += batch_size
        state = model.init_state(
            batch_size=batch_size,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        coords, lengths = empty_viewpoint_history(
            batch_size=batch_size,
            max_steps=args.max_history,
            device=device,
        )
        full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
        with torch.inference_mode():
            full_glimpse = sample_at_viewpoint(
                spatial=images,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
            state = out.state
            update_miou_accumulator(
                model=model,
                probe=probe,
                state=state,
                masks=masks,
                cfg=cfg,
                acc=accs[0],
            )
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=cfg.canvas_grid_size,
            )
        coords, lengths = append_viewpoint_history(
            coords=coords,
            lengths=lengths,
            viewpoint=full_vp,
            step=0,
        )
        for step_idx in range(args.t):
            obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
            with torch.no_grad():
                action = actor.deterministic_action(obs)
            vp = action_to_viewpoint(action, min_scale=args.min_scale)
            with torch.inference_mode():
                glimpse = sample_at_viewpoint(
                    spatial=images,
                    viewpoint=vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                ).to(dtype=canvit_dtype)
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state
                update_miou_accumulator(
                    model=model,
                    probe=probe,
                    state=state,
                    masks=masks,
                    cfg=cfg,
                    acc=accs[step_idx + 1],
                )
                canvas_summary = canvas_layernorm_spatial(
                    model=model,
                    state=state,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
            coords, lengths = append_viewpoint_history(
                coords=coords,
                lengths=lengths,
                viewpoint=vp,
                step=step_idx + 1,
            )

    metrics = {
        f"final_full_validation/miou_t{step_idx}": float(acc.compute())
        for step_idx, acc in enumerate(accs)
    }
    timesteps = list(range(args.t + 1))
    miou_values = [
        metrics[f"final_full_validation/miou_t{step_idx}"]
        for step_idx in timesteps
    ]

    print(
        "Best checkpoint full validation mIoU by timestep "
        f"(accumulator, {n_images} images):"
    )
    for step_idx, miou in zip(timesteps, miou_values, strict=True):
        print(f"  t{step_idx}: {miou:.4f}")
    return metrics, timesteps, miou_values
