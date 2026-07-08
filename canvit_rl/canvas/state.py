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
