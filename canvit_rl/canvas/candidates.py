"""Candidate Viewpoint tensor samplers for critic diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CandidateDistribution = Literal[
    "random",
    "fixed_scale",
    "position_perturb",
    "scale_perturb",
    "history_perturb",
]


@dataclass(frozen=True)
class CandidateViewpoints:
    """Plain tensor representation of chunk-ordered candidate Viewpoints."""

    centers: torch.Tensor
    scales: torch.Tensor


def random_candidate_viewpoints(
    *,
    batch_size: int,
    k: int,
    min_scale: float,
    device: torch.device,
) -> CandidateViewpoints:
    """Sample random in-bounds candidate Viewpoints in chunk order [k, batch]."""
    scales = torch.rand(k, batch_size, device=device) * (1.0 - min_scale) + min_scale
    bounds = (1.0 - scales).clamp_min(0.0)
    centers = (
        (torch.rand(k, batch_size, 2, device=device) * 2.0 - 1.0)
        * bounds[..., None]
    )
    return CandidateViewpoints(
        centers=centers.reshape(k * batch_size, 2),
        scales=scales.reshape(-1),
    )


def last_history_viewpoint(
    *,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    fallback_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the most recent committed center/scale for candidate anchoring."""
    batch_size, seq_len, _ = coords.shape
    last_step = lengths.clamp_min(1).sub(1).clamp_max(seq_len - 1)
    batch_ids = torch.arange(batch_size, device=coords.device)
    centers = coords[batch_ids, last_step, :2].float()
    scales = coords[batch_ids, last_step, 2].float()
    fallback = torch.full_like(scales, float(fallback_scale))
    # Problem: the t0 history contains a full-scene viewpoint, whose scale=1
    # leaves no legal center motion. Solution: use the requested diagnostic
    # fallback scale only for full-scene or unset anchors. Result: local
    # candidate distributions can still probe useful center neighborhoods
    # immediately after reset without overwriting later zoom history.
    return centers, torch.where((scales >= 0.999) | (scales <= 0.0), fallback, scales)


def clamp_centers_to_scale(centers: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Clamp normalized centers so sampled Viewpoints stay in bounds."""
    bounds = (1.0 - scales).clamp_min(0.0)[..., None]
    centers = centers.clamp(min=-1.0, max=1.0)
    return torch.maximum(torch.minimum(centers, bounds), -bounds)


def fixed_scale_candidate_viewpoints(
    *,
    batch_size: int,
    k: int,
    scale: float,
    device: torch.device,
) -> CandidateViewpoints:
    """Sample random centers at a fixed scale in chunk order [k, batch]."""
    scales = torch.full((k, batch_size), float(scale), device=device)
    bounds = (1.0 - scales).clamp_min(0.0)
    centers = (
        (torch.rand(k, batch_size, 2, device=device) * 2.0 - 1.0)
        * bounds[..., None]
    )
    return CandidateViewpoints(
        centers=centers.reshape(k * batch_size, 2),
        scales=scales.reshape(-1),
    )


def perturbed_candidate_viewpoints(
    *,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    min_scale: float,
    fallback_scale: float,
    position_std: float,
    scale_std: float,
    mode: CandidateDistribution,
) -> CandidateViewpoints:
    """Sample local center/scale perturbation candidates around history anchors."""
    batch_size = coords.shape[0]
    centers, scales = last_history_viewpoint(
        coords=coords,
        lengths=lengths,
        fallback_scale=fallback_scale,
    )
    centers = centers[None, :, :].expand(k, -1, -1).clone()
    scales = scales[None, :].expand(k, -1).clone()
    if mode in {"position_perturb", "history_perturb"}:
        centers = centers + torch.randn_like(centers) * float(position_std)
    else:
        centers = clamp_centers_to_scale(centers, scales)
    if mode in {"scale_perturb", "history_perturb"}:
        scales = (scales + torch.randn_like(scales) * float(scale_std)).clamp(
            min=float(min_scale),
            max=1.0,
        )
    elif mode == "position_perturb":
        scales = torch.full_like(scales, float(fallback_scale))
    centers = clamp_centers_to_scale(centers, scales)
    return CandidateViewpoints(
        centers=centers.reshape(k * batch_size, 2),
        scales=scales.reshape(-1),
    )


def sample_candidate_viewpoints(
    *,
    distribution: CandidateDistribution,
    batch_size: int,
    k: int,
    min_scale: float,
    fixed_scale: float,
    position_std: float,
    scale_std: float,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> CandidateViewpoints:
    """Dispatch candidate sampling without changing the critic target contract."""
    if distribution == "random":
        return random_candidate_viewpoints(
            batch_size=batch_size,
            k=k,
            min_scale=min_scale,
            device=device,
        )
    if distribution == "fixed_scale":
        return fixed_scale_candidate_viewpoints(
            batch_size=batch_size,
            k=k,
            scale=fixed_scale,
            device=device,
        )
    return perturbed_candidate_viewpoints(
        coords=coords,
        lengths=lengths,
        k=k,
        min_scale=min_scale,
        fallback_scale=fixed_scale,
        position_std=position_std,
        scale_std=scale_std,
        mode=distribution,
    )
