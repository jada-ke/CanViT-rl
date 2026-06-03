"""State-sequence utilities for continuous CanViT SAC scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from canvit_pytorch import Viewpoint


def extract_local_patches(out: Any) -> torch.Tensor:
    """Extract CanViT local patch tokens from a forward output."""
    for name in ("local_patches", "patches", "local_tokens"):
        value = getattr(out, name, None)
        if isinstance(value, torch.Tensor):
            patches = value
            break
    else:
        raise AttributeError(
            "CanViT output does not expose local patch tokens. Expected one of "
            "`out.local_patches`, `out.patches`, or `out.local_tokens`."
        )
    if patches.ndim == 2:
        patches = patches.unsqueeze(0)
    if patches.ndim != 3:
        raise ValueError(f"Expected local patches [B, N, D], got {patches.shape}.")
    return patches.detach().float()


def append_glimpse(
    *,
    seq: dict[str, torch.Tensor],
    patches: torch.Tensor,
    viewpoint: Viewpoint,
) -> dict[str, torch.Tensor]:
    """Append one detached glimpse tuple to a CPU sequence state."""
    coords = torch.cat(
        [
            viewpoint.centers.detach().float(),
            viewpoint.scales[:, None].detach().float(),
        ],
        dim=-1,
    )
    return {
        "patches": torch.cat([seq["patches"], patches.cpu()], dim=0),
        "coords": torch.cat([seq["coords"], coords.cpu()], dim=0),
    }


def empty_sequence(*, n_patches: int, patch_dim: int) -> dict[str, torch.Tensor]:
    """Create an empty CanViT sequence state."""
    return {
        "patches": torch.zeros(0, n_patches, patch_dim),
        "coords": torch.zeros(0, 3),
    }


def sequence_to_arrays(
    seq: dict[str, torch.Tensor],
    *,
    max_steps: int,
    n_patches: int,
    patch_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad a variable-length sequence for replay or batch storage."""
    length = min(int(seq["patches"].shape[0]), max_steps)
    patches = np.zeros((max_steps, n_patches, patch_dim), dtype=np.float32)
    coords = np.zeros((max_steps, 3), dtype=np.float32)
    if length:
        patches[:length] = seq["patches"][:length].numpy()
        coords[:length] = seq["coords"][:length].numpy()
    return patches, coords, length


def batch_from_sequence(
    seq: dict[str, torch.Tensor],
    *,
    max_steps: int,
    n_patches: int,
    patch_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert one CPU sequence to a padded one-item torch batch."""
    patches, coords, length = sequence_to_arrays(
        seq,
        max_steps=max_steps,
        n_patches=n_patches,
        patch_dim=patch_dim,
    )
    return {
        "patches": torch.as_tensor(patches[None], device=device),
        "coords": torch.as_tensor(coords[None], device=device),
        "lengths": torch.as_tensor([length], device=device),
    }
