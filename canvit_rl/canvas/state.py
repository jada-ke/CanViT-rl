"""Canvas-state helpers for image-dependent SAC policies."""

from __future__ import annotations

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
