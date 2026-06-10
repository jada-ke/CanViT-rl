"""
canvit_rl/greedy.py

Greedy image-independent policy for CanViT active-vision episodes.

At each of T timesteps, samples K candidate viewpoints uniformly at random,
runs a CanViT forward pass for each (without committing state), and selects
the candidate with the lowest segmentation cross-entropy loss.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import torch
import torch.nn.functional as F

from canvit_pytorch import (
    CanViTForPretrainingHFHub,
    Viewpoint,
    sample_at_viewpoint,
)
from canvit_pytorch.model import RecurrentState
from canvit_pytorch.policies import random_viewpoints
from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES

if TYPE_CHECKING:
    from canvit_pytorch import CanViT
else:
    CanViT = Any

GreedyModel: TypeAlias = CanViT | CanViTForPretrainingHFHub


def _replace_tensor_fields(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply a batch-tensor transform while preserving the upstream container."""
    if hasattr(obj, "_replace"):
        values = {
            name: fn(value) if torch.is_tensor(value) else value
            for name, value in zip(obj._fields, obj)
        }
        return obj._replace(**values)
    if is_dataclass(obj):
        values = {
            field.name: fn(value) if torch.is_tensor(value) else value
            for field in fields(obj)
            for value in [getattr(obj, field.name)]
        }
        return type(obj)(**values)
    values = {
        key: fn(value) if torch.is_tensor(value) else value
        for key, value in vars(obj).items()
    }
    return type(obj)(**values)


def _repeat_state_chunks(state: RecurrentState, chunks: int) -> RecurrentState:
    """Repeat a batched recurrent state in chunk order: [cand0 batch, cand1 batch]."""
    return _replace_tensor_fields(
        state,
        lambda tensor: tensor
        if tensor.ndim == 0
        else tensor.repeat((chunks,) + (1,) * (tensor.ndim - 1)),
    )


def _index_state_batch(state: RecurrentState, index: torch.Tensor) -> RecurrentState:
    """Select batch rows from a recurrent state after batched candidate scoring."""
    return _replace_tensor_fields(
        state,
        lambda tensor: tensor if tensor.ndim == 0 else tensor.index_select(0, index),
    )


def _make_viewpoint_like(template: Viewpoint, centers: torch.Tensor, scales: torch.Tensor) -> Viewpoint:
    """Build a Viewpoint without depending on the exact upstream container type."""
    if hasattr(template, "_replace"):
        return template._replace(centers=centers, scales=scales)
    if is_dataclass(template):
        values = {
            field.name: getattr(template, field.name)
            for field in fields(template)
        }
        values.update({"centers": centers, "scales": scales})
        return type(template)(**values)
    try:
        return type(template)(centers=centers, scales=scales)
    except TypeError:
        return Viewpoint(centers=centers, scales=scales)


def _seg_logits_from_state(
    model: GreedyModel,
    state: RecurrentState,
    probe: torch.nn.Module,
    canvas_grid_size: int,
    batch_size: int,
) -> torch.Tensor:
    """Decode ADE20K segmentation logits from a CanViT canvas state."""
    spatial = model.get_spatial(state.canvas).view(
        batch_size,
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        return probe(spatial.float()).float()


def _segmentation_cross_entropy_losses(
    model: GreedyModel,
    state: RecurrentState,
    probe: torch.nn.Module,
    canvas_grid_size: int,
    mask: torch.Tensor,
    batch_size: int,
    ignore_label: int = IGNORE_LABEL,
) -> torch.Tensor:
    """
    Evaluate per-image ADE20K segmentation CE from a CanViT canvas state.
    """
    logits = _seg_logits_from_state(
        model=model,
        state=state,
        probe=probe,
        canvas_grid_size=canvas_grid_size,
        batch_size=batch_size,
    )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.long()
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    # Fixed by Codex on 2026-06-09
    # Problem: Greedy selection was scalar and serial, so K candidates could not
    # share one GPU forward pass. Solution: keep unreduced CE and reduce per
    # image, allowing candidate losses to be compared after a [B*K] forward.
    pixel_loss = F.cross_entropy(
        logits,
        mask,
        ignore_index=ignore_label,
        reduction="none",
    )
    valid = mask != ignore_label
    loss_sum = pixel_loss.flatten(1).sum(dim=1)
    denom = valid.flatten(1).sum(dim=1).clamp_min(1)
    return loss_sum / denom


def _mean_iou_from_prediction(
    pred: torch.Tensor,
    mask: torch.Tensor,
    n_classes: int = NUM_CLASSES,
    ignore_label: int = IGNORE_LABEL,
) -> float:
    """
    Compute single-batch mIoU for a predicted ADE20K mask.

    """
    pred = pred.long()
    mask = mask.long()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)

    valid = mask != ignore_label
    ious = []
    for cls_idx in range(n_classes):
        pred_cls = (pred == cls_idx) & valid
        mask_cls = (mask == cls_idx) & valid
        union = pred_cls | mask_cls
        union_count = union.sum()
        if union_count == 0:
            continue
        inter_count = (pred_cls & mask_cls).sum()
        ious.append(inter_count.float() / union_count.float())

    if not ious:
        return 0.0
    return float(torch.stack(ious).mean().item())


def miou_from_state(
    model: GreedyModel,
    state: RecurrentState,
    probe: torch.nn.Module,
    mask: torch.Tensor,
    canvas_grid_size: int,
    n_classes: int = NUM_CLASSES,
    ignore_label: int = IGNORE_LABEL,
) -> float:
    """
    Evaluate ADE20K mIoU from the current CanViT canvas state.
    """
    with torch.inference_mode():
        batch_size = mask.shape[0] if mask.ndim == 3 else 1
        logits = _seg_logits_from_state(
            model=model,
            state=state,
            probe=probe,
            canvas_grid_size=canvas_grid_size,
            batch_size=batch_size,
        )
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pred = logits.argmax(dim=1)
        return _mean_iou_from_prediction(
            pred=pred,
            mask=mask,
            n_classes=n_classes,
            ignore_label=ignore_label,
        )


def greedy_step(
    model: GreedyModel,
    image: torch.Tensor,
    state: RecurrentState,
    k: int,
    glimpse_size_px: int,
    device: torch.device,
    min_scale: float = 0.10,
    max_scale: float = 1.0,
    sample_seed: int | None = None,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[Viewpoint, RecurrentState, float]:
    """
    Evaluate K candidate viewpoints and commit the lowest-loss one.

    Candidates are sampled using canvit_pytorch's random_viewpoints, which
    uses a safe-box-area scale distribution p(s) ~ (1-s) and constrains
    centers to stay within scene bounds.

    Args:
        model:          Frozen CanViT model.
        image:          Current scene tensor, shape [1, 3, H, W].
        state:          Current RecurrentState (not mutated).
        k:              Number of candidates to evaluate.
        glimpse_size_px: Glimpse resolution in pixels.
        device:         Torch device.
        min_scale:      Minimum viewpoint scale.
        max_scale:      Maximum viewpoint scale.
        sample_seed: Optional seed for deterministic candidate sampling.
        probe: ADE20K probe for segmentation scoring.
        canvas_grid_size: Canvas grid size for segmentation scoring.
        mask: ADE20K target mask for cross-entropy scoring.

    Returns:
        best_vp:        The winning Viewpoint.
        best_state:     The RecurrentState after committing the winning viewpoint.
        best_score:     Segmentation cross-entropy loss used for selection.
    """
    if probe is None or canvas_grid_size is None or mask is None:
        raise ValueError("greedy_step requires probe, canvas_grid_size, and mask.")

    if sample_seed is None:
        candidates = random_viewpoints(
            batch_size=1,
            device=device,
            n_viewpoints=k,
            min_scale=min_scale,
            max_scale=max_scale,
            start_with_full_scene=False,
        )
    else:
        with torch.random.fork_rng():
            torch.manual_seed(sample_seed)
            candidates = random_viewpoints(
                batch_size=1,
                device=device,
                n_viewpoints=k,
                min_scale=min_scale,
                max_scale=max_scale,
                start_with_full_scene=False,
            )

    best_vp = None
    best_state = None
    best_score = float("inf")

    with torch.inference_mode():
        for vp in candidates:
            glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=vp,
                glimpse_size_px=glimpse_size_px,
            )
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            score = _segmentation_cross_entropy_losses(
                model=model,
                state=out.state,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                mask=mask,
                batch_size=image.shape[0],
            )[0].item()

            if score < best_score:
                best_score = score
                best_vp = vp
                best_state = out.state

    assert best_vp is not None and best_state is not None
    return best_vp, best_state, best_score


def full_scene_step(
    model: GreedyModel,
    image: torch.Tensor,
    state: RecurrentState,
    glimpse_size_px: int,
    device: torch.device,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[Viewpoint, RecurrentState, float]:
    """Commit one full-scene glimpse and return the updated state/loss."""
    if probe is None or canvas_grid_size is None or mask is None:
        raise ValueError("full_scene_step requires probe, canvas_grid_size, and mask.")

    with torch.inference_mode():
        vp = Viewpoint.full_scene(batch_size=image.shape[0], device=device)
        glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=vp,
            glimpse_size_px=glimpse_size_px,
        )
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        score = _segmentation_cross_entropy_losses(
            model=model,
            state=out.state,
            probe=probe,
            canvas_grid_size=canvas_grid_size,
            mask=mask,
            batch_size=image.shape[0],
        ).mean().item()
    return vp, out.state, score


def greedy_step_batch(
    model: GreedyModel,
    images: torch.Tensor,
    state: RecurrentState,
    k: int,
    glimpse_size_px: int,
    device: torch.device,
    min_scale: float = 0.10,
    max_scale: float = 1.0,
    sample_seed: int | None = None,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[Viewpoint, RecurrentState, torch.Tensor]:
    """
    Evaluate B*K candidates in one forward and commit one low-loss candidate per image.
    """
    if probe is None or canvas_grid_size is None or masks is None:
        raise ValueError("greedy_step_batch requires probe, canvas_grid_size, and masks.")
    batch_size = images.shape[0]
    if sample_seed is None:
        candidates = random_viewpoints(
            batch_size=batch_size,
            device=device,
            n_viewpoints=k,
            min_scale=min_scale,
            max_scale=max_scale,
            start_with_full_scene=False,
        )
    else:
        with torch.random.fork_rng():
            torch.manual_seed(sample_seed)
            candidates = random_viewpoints(
                batch_size=batch_size,
                device=device,
                n_viewpoints=k,
                min_scale=min_scale,
                max_scale=max_scale,
                start_with_full_scene=False,
            )

    centers = torch.cat([vp.centers for vp in candidates], dim=0)
    scales = torch.cat([vp.scales for vp in candidates], dim=0)
    candidate_vp = _make_viewpoint_like(candidates[0], centers=centers, scales=scales)
    candidate_images = images.repeat((k,) + (1,) * (images.ndim - 1))
    candidate_masks = masks.repeat((k,) + (1,) * (masks.ndim - 1))
    candidate_state = _repeat_state_chunks(state, k)

    with torch.inference_mode():
        glimpse = sample_at_viewpoint(
            spatial=candidate_images,
            viewpoint=candidate_vp,
            glimpse_size_px=glimpse_size_px,
        )
        out = model(glimpse=glimpse, state=candidate_state, viewpoint=candidate_vp)
        losses = _segmentation_cross_entropy_losses(
            model=model,
            state=out.state,
            probe=probe,
            canvas_grid_size=canvas_grid_size,
            mask=candidate_masks,
            batch_size=batch_size * k,
        ).view(k, batch_size)

    best_candidate = losses.argmin(dim=0)
    batch_index = torch.arange(batch_size, device=device)
    flat_index = best_candidate * batch_size + batch_index
    best_state = _index_state_batch(out.state, flat_index)
    best_vp = _make_viewpoint_like(
        candidates[0],
        centers=centers.index_select(0, flat_index),
        scales=scales.index_select(0, flat_index),
    )
    best_losses = losses[best_candidate, batch_index]
    return best_vp, best_state, best_losses


def full_scene_step_batch(
    model: GreedyModel,
    images: torch.Tensor,
    state: RecurrentState,
    glimpse_size_px: int,
    device: torch.device,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[Viewpoint, RecurrentState, torch.Tensor]:
    """Commit a full-scene glimpse for a batch and return per-image losses."""
    if probe is None or canvas_grid_size is None or masks is None:
        raise ValueError("full_scene_step_batch requires probe, canvas_grid_size, and masks.")
    with torch.inference_mode():
        vp = Viewpoint.full_scene(batch_size=images.shape[0], device=device)
        glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=vp,
            glimpse_size_px=glimpse_size_px,
        )
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        losses = _segmentation_cross_entropy_losses(
            model=model,
            state=out.state,
            probe=probe,
            canvas_grid_size=canvas_grid_size,
            mask=masks,
            batch_size=images.shape[0],
        )
    return vp, out.state, losses


def run_greedy_episode(
    model: GreedyModel,
    image: torch.Tensor,
    init_state: RecurrentState,
    t: int = 5,
    k: int = 3,
    glimpse_size_px: int = 128,
    device: torch.device | None = None,
    seed: int | None = None,
    mask: torch.Tensor | None = None,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    start_with_full_scene: bool = True,
    compute_miou: bool = False,
    keep_states: bool = False,
) -> dict:
    """
    Run a full greedy episode of T steps with K candidates per step.

    Args:
        model:          Frozen CanViT model.
        image:          Scene tensor, shape [1, 3, H, W].
        init_state:     Initial RecurrentState (from model.init_state).
        t:              Number of timesteps.
        k:              Number of candidates per step.
        glimpse_size_px: Glimpse resolution in pixels.
        device:         Torch device.
        seed:           Optional random seed for reproducibility.
        mask:           ADE20K target mask, shape [1, H, W] or [H, W].
        probe:          Segmentation probe for CE scoring and mIoU diagnostics.
        canvas_grid_size: Canvas grid size used to reshape the state features.
        start_with_full_scene: If True, timestep 0 is a committed full-scene
            glimpse; later timesteps use greedy K-candidate search.
        compute_miou:   If True, include per-step mIoU diagnostics.
        keep_states:    If True, return committed CanViT recurrent states for
            downstream inspection/visualization.
    Returns:
        Dictionary with per-step diagnostics:
            - viewpoints: list of chosen Viewpoint objects
            - scores:     list of segmentation CE losses used for selection
            - rewards:    list of loss reductions (positive means lower loss)
            - scales:     list of chosen scales (for coarse-to-fine analysis)
            - centers:    list of chosen centers
            - states:     optional list of committed recurrent states
            - mious:      optional list of mIoU after each committed step
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    if probe is None or canvas_grid_size is None or mask is None:
        raise ValueError(
            "Greedy segmentation CE selection requires probe, "
            "canvas_grid_size, and mask."
        )

    state = init_state
    prev_score: float | None = None

    viewpoints, scores, rewards, scales, centers, states = [], [], [], [], [], []
    mious = [] if compute_miou else None
    mask_dev = mask.to(device) if mask is not None else None

    for step_idx in range(t):
        if step_idx == 0 and start_with_full_scene:
            vp, state, score = full_scene_step(
                model=model,
                image=image,
                state=state,
                glimpse_size_px=glimpse_size_px,
                device=device,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                mask=mask_dev,
            )
        else:
            vp, state, score = greedy_step(
                model=model,
                image=image,
                state=state,
                k=k,
                glimpse_size_px=glimpse_size_px,
                device=device,
                sample_seed=None if seed is None else seed + step_idx,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                mask=mask_dev,
            )
        reward = 0.0 if prev_score is None else prev_score - score
        prev_score = score

        viewpoints.append(vp)
        scores.append(score)
        rewards.append(reward)
        scales.append(float(vp.scales[0].item()))
        centers.append(vp.centers[0].cpu().tolist())
        # Fixed by Codex on 2026-06-09
        # Problem: the k-greedy visualizer could show where the policy looked,
        # but not what segmentation canvas that sequence produced.
        # Solution: optionally retain the committed recurrent state after each
        # step so visualization tools can decode per-timestep predictions.
        if keep_states:
            states.append(state)
        if compute_miou:
            assert mious is not None
            assert mask_dev is not None and canvas_grid_size is not None
            mious.append(
                miou_from_state(
                    model=model,
                    state=state,
                    probe=probe,
                    mask=mask_dev,
                    canvas_grid_size=canvas_grid_size,
                )
            )

    result = {
        "viewpoints": viewpoints,
        "scores": scores,
        "rewards": rewards,
        "scales": scales,
        "centers": centers,
    }
    if keep_states:
        result["states"] = states
    if mious is not None:
        result["mious"] = mious
    return result


def run_greedy_batch(
    model: GreedyModel,
    images: torch.Tensor,
    init_state: RecurrentState,
    t: int = 5,
    k: int = 3,
    glimpse_size_px: int = 128,
    device: torch.device | None = None,
    seed: int | None = None,
    masks: torch.Tensor | None = None,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    start_with_full_scene: bool = True,
    compute_miou: bool = False,
    keep_states: bool = False,
) -> dict:
    """
    Run batched greedy episodes, evaluating each step's B*K candidates together.
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    if probe is None or canvas_grid_size is None or masks is None:
        raise ValueError(
            "Greedy segmentation CE selection requires probe, "
            "canvas_grid_size, and masks."
        )

    state = init_state
    prev_scores: torch.Tensor | None = None
    masks_dev = masks.to(device)
    batch_size = images.shape[0]

    viewpoints, scores, rewards, scales, centers, states = [], [], [], [], [], []
    mious = [] if compute_miou else None

    for step_idx in range(t):
        if step_idx == 0 and start_with_full_scene:
            vp, state, step_scores = full_scene_step_batch(
                model=model,
                images=images,
                state=state,
                glimpse_size_px=glimpse_size_px,
                device=device,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                masks=masks_dev,
            )
        else:
            vp, state, step_scores = greedy_step_batch(
                model=model,
                images=images,
                state=state,
                k=k,
                glimpse_size_px=glimpse_size_px,
                device=device,
                sample_seed=None if seed is None else seed + step_idx,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                masks=masks_dev,
            )
        step_rewards = torch.zeros_like(step_scores) if prev_scores is None else prev_scores - step_scores
        prev_scores = step_scores

        viewpoints.append(vp)
        scores.append(step_scores.detach())
        rewards.append(step_rewards.detach())
        scales.append(vp.scales.detach())
        centers.append(vp.centers.detach())
        if keep_states:
            states.append(state)
        if compute_miou:
            assert mious is not None
            image_mious = []
            for image_idx in range(batch_size):
                one_state = _index_state_batch(
                    state,
                    torch.tensor([image_idx], device=device),
                )
                image_mious.append(
                    miou_from_state(
                        model=model,
                        state=one_state,
                        probe=probe,
                        mask=masks_dev[image_idx : image_idx + 1],
                        canvas_grid_size=canvas_grid_size,
                    )
                )
            mious.append(torch.tensor(image_mious, device=device))

    result = {
        "viewpoints": viewpoints,
        "scores": scores,
        "rewards": rewards,
        "scales": scales,
        "centers": centers,
    }
    if keep_states:
        result["states"] = states
    if mious is not None:
        result["mious"] = mious
    return result
