"""Probe Stage 0 end-to-end differentiability on dense IN21k shards.

This script intentionally has no policy and no stochastic rollout. It learns
nothing; it only asks whether one deterministic Viewpoint's center/scale tensors
receive gradient from the dense CanViT distillation objective.

Example:
    uv run python scripts/probe_i21k_stage0_differentiability.py \
        --feature-base-dir datasets/imagenet_ood/features \
        --feature-image-root datasets/imagenet_ood/images \
        --subset-size 1 \
        --output results/i21k_stage0_diff/probe.png
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub

from canvit_rl.canvit_precision import resolve_canvit_dtype
from canvit_rl.env import get_device
from canvit_rl.pretrain_IN21k.dense_train_batch import (
    FixedDenseSubsetLoader,
    apply_dense_feature_config,
    dense_glimpse_images,
    init_normalizer_stats_from_shard,
    load_dense_train_batch,
)
from canvit_rl.pretrain_IN21k.pretrain_modules import load_pretrain_modules
from canvit_rl.pretrain_IN21k.reward import dense_distillation_metrics

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def parse_args() -> argparse.Namespace:
    """Parse the minimal dense-feature inputs needed for the Stage 0 probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-base-dir", type=Path, required=True)
    parser.add_argument("--feature-image-root", type=Path, default=None)
    parser.add_argument("--tar-dir", type=Path, default=None)
    parser.add_argument(
        "--model-repo",
        type=str,
        default=(
            "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-"
            "2026-02-02"
        ),
    )
    parser.add_argument("--teacher-name", type=str, default="dinov3_vitb16")
    parser.add_argument("--scene-resolution", type=int, default=512)
    parser.add_argument("--glimpse-grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--subset-shards", type=int, default=1)
    parser.add_argument("--normalizer-max-samples", type=int, default=0)
    parser.add_argument("--reset-normalizer", action="store_true")
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--scene-reward-weight", type=float, default=1.0)
    parser.add_argument("--cls-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--canvit-dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/i21k_stage0_diff/probe.png"))
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--surface-radius",
        type=float,
        default=0.20,
        help="Center-coordinate radius used for the optional local loss surface.",
    )
    parser.add_argument("--surface-grid-size", type=int, default=9)
    parser.add_argument(
        "--gradient-grid",
        action="store_true",
        help="Overlay a quiver field of per-grid-point negative autograd gradients.",
    )
    parser.add_argument(
        "--gradient-grid-scale",
        type=float,
        default=0.35,
        help="Relative arrow length for the optional gradient-grid quiver overlay.",
    )
    args = parser.parse_args()
    if (args.feature_image_root is None) == (args.tar_dir is None):
        raise ValueError("Exactly one of --feature-image-root or --tar-dir is required.")
    if args.batch_size != 1:
        raise ValueError("Stage 0 probe currently visualizes exactly one sample.")
    if args.scale <= 0.0 or args.scale > 1.0:
        raise ValueError("--scale must be in (0, 1].")
    if args.surface_grid_size < 2:
        raise ValueError("--surface-grid-size must be at least 2.")
    return args


def build_pretrain_config(args: argparse.Namespace):
    """Create the same dense-feature config shape used by IN21k SAC."""
    modules = load_pretrain_modules()
    cfg = modules.Config()
    apply_dense_feature_config(
        cfg,
        feature_base_dir=args.feature_base_dir,
        feature_image_root=args.feature_image_root,
        tar_dir=args.tar_dir,
    )
    cfg.teacher_name = args.teacher_name
    cfg.scene_resolution = args.scene_resolution
    cfg.glimpse_grid_size = args.glimpse_grid_size
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.device = get_device()
    cfg.normalizer_max_samples = args.normalizer_max_samples
    cfg.reset_normalizer = args.reset_normalizer
    return cfg, modules


def build_dense_loader(args: argparse.Namespace, cfg, modules):
    """Build a one-sample loader while reusing the training shard contract."""
    shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
    if args.subset_size > 0:
        return FixedDenseSubsetLoader(
            shards_dir=shards_dir,
            image_size=cfg.scene_resolution,
            batch_size=args.batch_size,
            subset_size=max(args.subset_size, args.batch_size),
            subset_seed=args.subset_seed,
            subset_shards=args.subset_shards,
            image_root=cfg.feature_image_root,
            tar_dir=cfg.tar_dir,
        )
    return modules.ShardedFeatureLoader(
        shards_dir=shards_dir,
        image_size=cfg.scene_resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        start_step=0,
        image_root=cfg.feature_image_root,
        tar_dir=cfg.tar_dir,
        steps_per_job=1,
    )


def load_frozen_model(args: argparse.Namespace, cfg):
    """Load a frozen CanViT while leaving input/viewpoint autograd enabled."""
    model = CanViTForPretrainingHFHub.from_pretrained(args.model_repo).to(cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch_size_px = model.backbone.patch_size_px
    glimpse_size_px = cfg.glimpse_grid_size * patch_size_px
    cfg.model = model.cfg
    cfg.canvas_patch_grid_size = model.canvas_patch_grid_sizes[0]
    return model, glimpse_size_px


def make_viewpoint(center: torch.Tensor, scale: torch.Tensor) -> Viewpoint:
    """Wrap differentiable action tensors in CanViT's Viewpoint interface."""
    return Viewpoint(centers=center.unsqueeze(0), scales=scale.reshape(1))


def dense_loss_for_viewpoint(
    *,
    model,
    batch,
    scene_norm,
    cls_norm,
    canvas_grid_size: int,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    viewpoint: Viewpoint,
    scene_weight: float,
    cls_weight: float,
) -> tuple[torch.Tensor, object, torch.Tensor]:
    """Run one differentiable glimpse and return scalar loss, state, and glimpse."""
    # Problem: normal Gym-style environments detach actions/rewards for RL.
    # Solution: this probe keeps the single Viewpoint path under normal autograd.
    # Result: loss.backward() can expose whether sampling/model/loss are
    # differentiable with respect to raw center and scale tensors.
    state = model.init_state(batch_size=batch.images.shape[0], canvas_grid_size=canvas_grid_size)
    glimpse = sample_at_viewpoint(
        spatial=dense_glimpse_images(batch),
        viewpoint=viewpoint,
        glimpse_size_px=glimpse_size_px,
    ).to(dtype=canvit_dtype)
    out = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
    metrics = dense_distillation_metrics(
        model=model,
        state=out.state,
        batch=batch,
        scene_denorm=scene_norm.destandardize,
        cls_denorm=cls_norm.destandardize,
        scene_weight=scene_weight,
        cls_weight=cls_weight,
    )
    return metrics.loss_norm.mean(), out.state, glimpse


def denormalized_uint8_image(image: torch.Tensor) -> np.ndarray:
    """Convert one ImageNet-normalized tensor into an RGB uint8 image."""
    restored = image.detach().cpu().float() * IMAGENET_STD + IMAGENET_MEAN
    return (restored.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def viewpoint_box(
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Convert CanViT [y, x] center in [-1, 1] and scale to pixel bounds."""
    cy, cx = center.detach().cpu().float().tolist()
    scale_value = float(scale.detach().cpu().float().item())
    center_x = (cx + 1.0) * 0.5 * (width - 1)
    center_y = (cy + 1.0) * 0.5 * (height - 1)
    box_w = scale_value * (width - 1)
    box_h = scale_value * (height - 1)
    left = int(round(max(0.0, center_x - box_w * 0.5)))
    top = int(round(max(0.0, center_y - box_h * 0.5)))
    right = int(round(min(width - 1.0, center_x + box_w * 0.5)))
    bottom = int(round(min(height - 1.0, center_y + box_h * 0.5)))
    return left, top, right, bottom


def finite_difference_surface(
    *,
    args: argparse.Namespace,
    model,
    batch,
    scene_norm,
    cls_norm,
    canvas_grid_size: int,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    scale: float,
    center: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate a small local loss surface, optionally with point gradients."""
    radius = float(args.surface_radius)
    grid_size = int(args.surface_grid_size)
    y0, x0 = center.detach().cpu().float().tolist()
    y_values = torch.linspace(y0 - radius, y0 + radius, grid_size).clamp(-1.0, 1.0)
    x_values = torch.linspace(x0 - radius, x0 + radius, grid_size).clamp(-1.0, 1.0)
    losses = torch.empty(grid_size, grid_size)
    gradient_grid = (
        torch.empty(grid_size, grid_size, 2) if args.gradient_grid else None
    )
    for row, y in enumerate(y_values):
        for col, x in enumerate(x_values):
            if args.gradient_grid:
                # Problem: the loss heatmap alone shows sampled values, not the
                # local autograd direction at each sampled center. Solution:
                # make each grid center differentiable and backprop just that
                # scalar probe. Result: the optional quiver field shows the
                # negative gradient vector across the local action neighborhood.
                probe_center = torch.tensor(
                    [float(y), float(x)],
                    device=batch.images.device,
                    requires_grad=True,
                )
                probe_scale = torch.tensor(scale, device=batch.images.device)
                vp = make_viewpoint(probe_center, probe_scale)
                loss, _, _ = dense_loss_for_viewpoint(
                    model=model,
                    batch=batch,
                    scene_norm=scene_norm,
                    cls_norm=cls_norm,
                    canvas_grid_size=canvas_grid_size,
                    glimpse_size_px=glimpse_size_px,
                    canvit_dtype=canvit_dtype,
                    viewpoint=vp,
                    scene_weight=args.scene_reward_weight,
                    cls_weight=args.cls_reward_weight,
                )
                loss.backward()
                assert probe_center.grad is not None
                losses[row, col] = loss.detach().cpu()
                gradient_grid[row, col] = probe_center.grad.detach().cpu()
            else:
                with torch.no_grad():
                    vp = make_viewpoint(
                        torch.tensor([float(y), float(x)], device=batch.images.device),
                        torch.tensor(scale, device=batch.images.device),
                    )
                    loss, _, _ = dense_loss_for_viewpoint(
                        model=model,
                        batch=batch,
                        scene_norm=scene_norm,
                        cls_norm=cls_norm,
                        canvas_grid_size=canvas_grid_size,
                        glimpse_size_px=glimpse_size_px,
                        canvit_dtype=canvit_dtype,
                        viewpoint=vp,
                        scene_weight=args.scene_reward_weight,
                        cls_weight=args.cls_reward_weight,
                    )
                    losses[row, col] = loss.detach().cpu()
    return (
        y_values.numpy(),
        x_values.numpy(),
        losses.numpy(),
        None if gradient_grid is None else gradient_grid.numpy(),
    )


def save_visualization(
    *,
    args: argparse.Namespace,
    batch,
    center: torch.Tensor,
    scale: torch.Tensor,
    center_grad: torch.Tensor,
    scale_grad: torch.Tensor,
    loss: torch.Tensor,
    surface: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
) -> Path:
    """Save a compact visual explanation of the differentiable Stage 0 path."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except ImportError as exc:
        raise RuntimeError("Install matplotlib or pass --no-viz.") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_np = denormalized_uint8_image(dense_glimpse_images(batch)[0])
    height, width = image_np.shape[:2]
    left, top, right, bottom = viewpoint_box(center, scale, height=height, width=width)
    y_values, x_values, loss_surface, gradient_grid = surface

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.5, 5.4),
        dpi=170,
        gridspec_kw={"width_ratios": [1.1, 1.25, 1.05]},
    )
    axes[0].imshow(image_np)
    axes[0].add_patch(
        patches.Rectangle(
            (left, top),
            max(right - left, 1),
            max(bottom - top, 1),
            linewidth=2.5,
            edgecolor="lime",
            facecolor="none",
        )
    )
    axes[0].set_title("input + Viewpoint")
    axes[0].axis("off")

    x_pixels = (x_values + 1.0) * 0.5 * (width - 1)
    y_pixels = (y_values + 1.0) * 0.5 * (height - 1)
    center_x_px = float((center[1].detach().cpu() + 1.0) * 0.5 * (width - 1))
    center_y_px = float((center[0].detach().cpu() + 1.0) * 0.5 * (height - 1))
    axes[1].imshow(image_np)
    # Problem: a standalone center-coordinate loss surface is hard to relate
    # back to the scene. Solution: remap the probed center grid to image pixels
    # and overlay it on a larger input panel. Result: low/high loss regions and
    # gradient directions are visible in the same frame as the sampled objects.
    im = axes[1].imshow(
        loss_surface,
        extent=[x_pixels[0], x_pixels[-1], y_pixels[-1], y_pixels[0]],
        cmap="viridis",
        alpha=0.62,
        origin="upper",
        aspect="auto",
    )
    axes[1].add_patch(
        patches.Rectangle(
            (left, top),
            max(right - left, 1),
            max(bottom - top, 1),
            linewidth=2.0,
            edgecolor="white",
            facecolor="none",
        )
    )
    axes[1].scatter(
        [center_x_px],
        [center_y_px],
        c="white",
        edgecolors="black",
        s=45,
        zorder=3,
    )
    if gradient_grid is not None:
        xx, yy = np.meshgrid(x_pixels, y_pixels)
        neg_grad = -gradient_grid
        norms = np.linalg.norm(neg_grad, axis=-1, keepdims=True)
        unit = neg_grad / np.maximum(norms, 1e-12)
        step_px = min(
            max(float(np.diff(x_pixels).mean()), 1.0),
            max(float(np.diff(y_pixels).mean()), 1.0),
        )
        arrow_scale = step_px * float(args.gradient_grid_scale)
        axes[1].quiver(
            xx,
            yy,
            unit[..., 1] * arrow_scale,
            unit[..., 0] * arrow_scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="white",
            width=0.006,
            headwidth=4.0,
            headlength=5.0,
            alpha=0.9,
            zorder=4,
        )
    grad = center_grad.detach().cpu().float()
    norm = float(grad.norm().item())
    if norm > 0.0 and math.isfinite(norm):
        # Problem: autograd gradients point uphill for loss. Solution: draw the
        # negative gradient as the immediate direction that would reduce loss.
        # Result: the heatmap gives a spatial intuition for differentiability.
        arrow = (-grad / norm * args.surface_radius * 0.45).numpy()
        axes[1].arrow(
            center_x_px,
            center_y_px,
            float(arrow[1] * 0.5 * (width - 1)),
            float(arrow[0] * 0.5 * (height - 1)),
            color="white",
            width=1.8,
            head_width=12.0,
            length_includes_head=True,
        )
    title = "loss overlay + gradient grid" if gradient_grid is not None else "loss surface overlay + -grad"
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].axis("off")
    lines = [
        "Autograd path",
        "center, scale require_grad",
        "  -> Viewpoint",
        "  -> sample_at_viewpoint",
        "  -> frozen CanViT forward",
        "  -> dense distillation loss",
        "  -> loss.backward()",
        "",
        f"loss_norm: {float(loss.detach().cpu()):.6f}",
        f"center [y,x]: {center.detach().cpu().tolist()}",
        f"center.grad: {center_grad.detach().cpu().tolist()}",
        f"scale: {float(scale.detach().cpu()):.6f}",
        f"scale.grad: {float(scale_grad.detach().cpu()):.6e}",
        f"gradient_grid: {bool(args.gradient_grid)}",
    ]
    axes[2].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    return args.output


def main() -> None:
    """Run the Stage 0 proof and optional visualization."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg, modules = build_pretrain_config(args)
    loader = build_dense_loader(args, cfg, modules)
    model, glimpse_size_px = load_frozen_model(args, cfg)
    canvit_dtype = resolve_canvit_dtype(args.canvit_dtype, cfg.device)
    model.to(device=cfg.device, dtype=canvit_dtype)
    for module in model.modules():
        if module.__class__.__name__ == "VPEEncoder":
            module.to(device=cfg.device, dtype=torch.float32)

    shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
    G = cfg.canvas_patch_grid_size
    cls_norm, scene_norm = model.standardizers(G)
    if args.reset_normalizer or not scene_norm.initialized:
        init_normalizer_stats_from_shard(
            shards_dir=shards_dir,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            device=cfg.device,
            max_samples=args.normalizer_max_samples,
        )

    batch = load_dense_train_batch(
        train_loader=loader,
        device=cfg.device,
        scene_norm=scene_norm,
        cls_norm=cls_norm,
        non_blocking=True,
    )
    center = torch.tensor(
        [args.center_y, args.center_x],
        dtype=torch.float32,
        device=cfg.device,
        requires_grad=True,
    )
    scale = torch.tensor(args.scale, dtype=torch.float32, device=cfg.device, requires_grad=True)
    vp = make_viewpoint(center, scale)
    loss, _, glimpse = dense_loss_for_viewpoint(
        model=model,
        batch=batch,
        scene_norm=scene_norm,
        cls_norm=cls_norm,
        canvas_grid_size=G,
        glimpse_size_px=glimpse_size_px,
        canvit_dtype=canvit_dtype,
        viewpoint=vp,
        scene_weight=args.scene_reward_weight,
        cls_weight=args.cls_reward_weight,
    )
    loss.backward()
    if center.grad is None or scale.grad is None:
        raise RuntimeError("Gradient did not reach center and scale tensors.")

    center_grad = center.grad.detach().clone()
    scale_grad = scale.grad.detach().clone()
    grad_norm = float(torch.cat([center_grad.reshape(-1), scale_grad.reshape(-1)]).norm().item())
    print(f"loss_norm={float(loss.detach().cpu()):.8f}")
    print(f"center.grad={center_grad.detach().cpu().tolist()}")
    print(f"scale.grad={float(scale_grad.detach().cpu()):.8e}")
    print(f"action_grad_norm={grad_norm:.8e}")
    if grad_norm == 0.0 or not math.isfinite(grad_norm):
        raise RuntimeError("Action gradient is zero or non-finite.")

    if not args.no_viz:
        surface = finite_difference_surface(
            args=args,
            model=model,
            batch=batch,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            canvas_grid_size=G,
            glimpse_size_px=glimpse_size_px,
            canvit_dtype=canvit_dtype,
            scale=float(scale.detach().cpu()),
            center=center,
        )
        output = save_visualization(
            args=args,
            batch=batch,
            center=center,
            scale=scale,
            center_grad=center_grad,
            scale_grad=scale_grad,
            loss=loss,
            surface=surface,
        )
        print(f"saved_visualization={output}")


if __name__ == "__main__":
    main()
