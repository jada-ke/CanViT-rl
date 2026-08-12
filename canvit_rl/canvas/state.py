"""Canvas-state helpers for image-dependent SAC policies."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from canvit_pytorch import Viewpoint


def canvas_layernorm_spatial(*, model, state, canvas_grid_size: int) -> torch.Tensor:
    """Return the current normalized spatial canvas map as [B, D, G, G]."""
    canvas = state.canvas.float()
    normed = F.layer_norm(canvas, (canvas.shape[-1],))
    spatial = model.get_spatial(normed).reshape(
        canvas.shape[0],
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    return spatial.permute(0, 3, 1, 2).contiguous()


def canvas_segmentation_entropy(
    *,
    model,
    probe: torch.nn.Module,
    state,
    canvas_grid_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return normalized probe entropy over the current canvas as [B, 1, G, G]."""
    spatial = model.get_spatial(state.canvas).reshape(
        state.canvas.shape[0],
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float()).float()
    probs = logits.softmax(dim=1)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    entropy = entropy / math.log(logits.shape[1])
    if entropy.shape[-2:] != (canvas_grid_size, canvas_grid_size):
        entropy = F.interpolate(
            entropy,
            size=(canvas_grid_size, canvas_grid_size),
            mode="bilinear",
            align_corners=False,
        )
    return entropy.contiguous()


def canvas_dinov3_reconstruction_norm(
    *,
    model,
    state,
    canvas_grid_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return normalized reconstructed DINOv3 feature norms as [B, 1, G, G]."""
    scene_pred = model.predict_teacher_scene(state.canvas).float()
    norm_map = scene_pred.norm(dim=-1).reshape(
        scene_pred.shape[0],
        1,
        canvas_grid_size,
        canvas_grid_size,
    )
    flat = norm_map.flatten(1)
    min_val = flat.min(dim=1).values[:, None, None, None]
    max_val = flat.max(dim=1).values[:, None, None, None]
    return ((norm_map - min_val) / (max_val - min_val).clamp_min(eps)).contiguous()


def canvas_teacher_reconstruction_error(
    *,
    model,
    state,
    scene_target: torch.Tensor,
    canvas_grid_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return normalized teacher-feature reconstruction MSE as [B, 1, G, G]."""
    scene_pred = model.predict_teacher_scene(state.canvas).float()
    per_patch_error = (scene_pred - scene_target.float()).pow(2).mean(dim=-1)
    # Problem: direct dense distillation error is useful as an aux state, but it
    # requires teacher targets unlike reconstruction norm. Solution: keep this
    # target-based map in its own helper and flag. Result: experiments can
    # choose supervised error explicitly without changing the target-free path.
    error_map = per_patch_error.reshape(
        per_patch_error.shape[0],
        1,
        canvas_grid_size,
        canvas_grid_size,
    )
    flat = error_map.flatten(1)
    min_val = flat.min(dim=1).values[:, None, None, None]
    max_val = flat.max(dim=1).values[:, None, None, None]
    return ((error_map - min_val) / (max_val - min_val).clamp_min(eps)).contiguous()


def scale_aware_detail_debt(
    *,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    canvas_grid_size: int,
    min_scale: float,
) -> torch.Tensor:
    """Return viewpoint-history detail debt as [B, 1, G, G]."""
    if min_scale <= 0.0 or min_scale > 1.0:
        raise ValueError("Require 0 < min_scale <= 1 for detail-debt state.")
    batch_size = coords.shape[0]
    device = coords.device
    grid = torch.linspace(-1.0, 1.0, canvas_grid_size, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    coverage = torch.zeros(
        batch_size,
        canvas_grid_size,
        canvas_grid_size,
        device=device,
        dtype=torch.float32,
    )
    for step_idx in range(coords.shape[1]):
        valid = step_idx < lengths
        if not bool(valid.any()):
            continue
        center_y = coords[:, step_idx, 0].float()[:, None, None]
        center_x = coords[:, step_idx, 1].float()[:, None, None]
        scale = coords[:, step_idx, 2].float().clamp(min=min_scale, max=1.0)
        half_extent = scale[:, None, None]
        in_footprint = (
            valid[:, None, None]
            & ((yy[None] - center_y).abs() <= half_extent)
            & ((xx[None] - center_x).abs() <= half_extent)
        )
        detail = (min_scale / scale).view(batch_size, 1, 1)
        coverage = torch.maximum(coverage, torch.where(in_footprint, detail, coverage))
    return (1.0 - coverage.clamp(0.0, 1.0))[:, None].contiguous()


def canvas_cosine_dissimilarity(
    *,
    current: torch.Tensor,
    previous: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-cell cosine dissimilarity between two canvas feature maps."""
    if current.shape != previous.shape:
        raise ValueError(
            f"Canvas maps must have matching shapes, got {current.shape} and {previous.shape}."
        )
    # Problem: detail debt says where resolution is still owed, but not whether
    # the last glimpse changed the representation. Solution: add a label-free
    # per-cell feature-change map from current-vs-previous cosine distance.
    # Result: policies can separately learn planned coverage and observed novelty.
    return (1.0 - F.cosine_similarity(current.float(), previous.float(), dim=1, eps=eps))[
        :, None
    ].contiguous()


def empty_viewpoint_history(
    *,
    batch_size: int,
    max_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate fixed-slot viewpoint history on device."""
    coords = torch.zeros(batch_size, max_steps, 3, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    return coords, lengths


def append_viewpoint_history(
    *,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    viewpoint: Viewpoint,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one batched Viewpoint timestep without mutating prior aliases."""
    if step >= coords.shape[1]:
        raise ValueError(
            f"History slot {step} is out of range for max_steps={coords.shape[1]}."
        )
    next_coords = coords.clone()
    next_coords[:, step, :2] = viewpoint.centers.detach().float()
    next_coords[:, step, 2] = viewpoint.scales.detach().float()
    return next_coords, lengths + 1
