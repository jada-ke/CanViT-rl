from types import SimpleNamespace

import torch

from canvit_rl.canvas.candidates import (
    fixed_scale_candidate_viewpoints,
    last_history_viewpoint,
    sample_candidate_viewpoints,
)


def _history(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.zeros(batch_size, 4, 3)
    coords[:, 0, 2] = 1.0
    coords[:, 1, :2] = torch.tensor([0.2, -0.2])
    coords[:, 1, 2] = 0.5
    lengths = torch.full((batch_size,), 2, dtype=torch.long)
    return coords, lengths


def test_fixed_scale_candidates_stay_in_bounds():
    torch.manual_seed(0)
    vp = fixed_scale_candidate_viewpoints(
        batch_size=4,
        k=5,
        scale=0.6,
        device=torch.device("cpu"),
    )

    assert vp.centers.shape == (20, 2)
    assert vp.scales.shape == (20,)
    assert torch.allclose(vp.scales, torch.full((20,), 0.6))
    assert torch.all(vp.centers.abs() <= 0.4 + 1e-6)


def test_full_scene_history_uses_candidate_fallback_scale():
    coords = torch.zeros(2, 3, 3)
    coords[:, 0, 2] = 1.0
    lengths = torch.ones(2, dtype=torch.long)

    centers, scales = last_history_viewpoint(
        coords=coords,
        lengths=lengths,
        fallback_scale=0.6,
    )

    assert torch.allclose(centers, torch.zeros(2, 2))
    assert torch.allclose(scales, torch.full((2,), 0.6))


def test_position_perturb_candidates_keep_requested_scale_and_bounds():
    torch.manual_seed(1)
    coords, lengths = _history(batch_size=2)
    args = SimpleNamespace(
        candidate_distribution="position_perturb",
        candidate_fixed_scale=0.6,
        candidate_position_std=0.2,
        candidate_scale_std=0.1,
        k=6,
        min_scale=0.25,
    )

    vp = sample_candidate_viewpoints(
        distribution=args.candidate_distribution,
        batch_size=2,
        k=args.k,
        min_scale=args.min_scale,
        fixed_scale=args.candidate_fixed_scale,
        position_std=args.candidate_position_std,
        scale_std=args.candidate_scale_std,
        coords=coords,
        lengths=lengths,
        device=torch.device("cpu"),
    )

    assert vp.centers.shape == (12, 2)
    assert torch.allclose(vp.scales, torch.full((12,), 0.6))
    assert torch.all(vp.centers.abs() <= 0.4 + 1e-6)


def test_history_perturb_candidates_vary_scale_within_bounds():
    torch.manual_seed(2)
    coords, lengths = _history(batch_size=2)
    args = SimpleNamespace(
        candidate_distribution="history_perturb",
        candidate_fixed_scale=0.6,
        candidate_position_std=0.2,
        candidate_scale_std=0.2,
        k=8,
        min_scale=0.25,
    )

    vp = sample_candidate_viewpoints(
        distribution=args.candidate_distribution,
        batch_size=2,
        k=args.k,
        min_scale=args.min_scale,
        fixed_scale=args.candidate_fixed_scale,
        position_std=args.candidate_position_std,
        scale_std=args.candidate_scale_std,
        coords=coords,
        lengths=lengths,
        device=torch.device("cpu"),
    )

    bounds = (1.0 - vp.scales).clamp_min(0.0)[:, None]
    assert torch.all(vp.scales >= 0.25)
    assert torch.all(vp.scales <= 1.0)
    assert torch.all(vp.centers <= bounds + 1e-6)
    assert torch.all(vp.centers >= -bounds - 1e-6)
    assert float(vp.scales.std(unbiased=False)) > 0.0
