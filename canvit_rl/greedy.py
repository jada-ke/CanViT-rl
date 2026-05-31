"""
canvit_rl/greedy.py

Greedy image-independent policy for CanViT active-vision episodes.

At each of T timesteps, samples K candidate viewpoints uniformly at random,
runs a CanViT forward pass for each (without committing state), and selects
the candidate with the highest cosine similarity to the teacher CLS.

Key property: the policy is image-independent — candidates are sampled
uniformly over the scene regardless of image content. A correct implementation
should exhibit emergent coarse-to-fine structure: early steps favor wide
glimpses (large scale, fast reward gain) and later steps favor finer ones
(small scale, marginal gains). If this doesn't emerge, something is wrong
with the reward or the forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

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

from canvit_rl.reward import reconstruction_reward

if TYPE_CHECKING:
    from canvit_pytorch import CanViT
else:
    CanViT = Any

GreedyObjective = str
GreedyModel: TypeAlias = CanViT | CanViTForPretrainingHFHub


def _score_canvas_cls(
    canvas_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
    objective: GreedyObjective,
    kl_temperature: float,
) -> float:
    """
    Score a candidate canvas state for greedy selection.

    Cosine is the default baseline. KL treats teacher/canvas CLS vectors as
    temperature-softmax distributions and maximizes negative KL(teacher||canvas).
    """
    if objective == "cosine":
        return reconstruction_reward(canvas_cls, teacher_cls)
    if objective == "kl":
        teacher_prob = F.softmax(teacher_cls.float() / kl_temperature, dim=-1)
        canvas_log_prob = F.log_softmax(canvas_cls.float() / kl_temperature, dim=-1)
        kl = F.kl_div(canvas_log_prob, teacher_prob, reduction="batchmean")
        return -float(kl.item())
    raise ValueError(f"Unknown greedy objective: {objective!r}")


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


def _score_segmentation_kl(
    model: GreedyModel,
    state: RecurrentState,
    probe: torch.nn.Module,
    canvas_grid_size: int,
    target_prob: torch.Tensor,
    kl_temperature: float,
    batch_size: int,
) -> float:
    """
    Score a canvas state by negative KL to a full-scene segmentation target.
    """
    logits = _seg_logits_from_state(
        model=model,
        state=state,
        probe=probe,
        canvas_grid_size=canvas_grid_size,
        batch_size=batch_size,
    )
    if logits.shape[-2:] != target_prob.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target_prob.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    log_prob = F.log_softmax(logits / kl_temperature, dim=1)
    kl = F.kl_div(log_prob, target_prob, reduction="batchmean")
    return -float(kl.item())


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
    teacher_cls: torch.Tensor,
    k: int,
    glimpse_size_px: int,
    device: torch.device,
    min_scale: float = 0.10,
    max_scale: float = 1.0,
    objective: GreedyObjective = "cosine",
    kl_temperature: float = 1.0,
    sample_seed: int | None = None,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    seg_kl_target_prob: torch.Tensor | None = None,
) -> tuple[Viewpoint, RecurrentState, float, float]:
    """
    Evaluate K candidate viewpoints and commit the best one.

    Candidates are sampled using canvit_pytorch's random_viewpoints, which
    uses a safe-box-area scale distribution p(s) ~ (1-s) and constrains
    centers to stay within scene bounds.

    Args:
        model:          Frozen CanViT model.
        image:          Current scene tensor, shape [1, 3, H, W].
        state:          Current RecurrentState (not mutated).
        teacher_cls:    Frozen teacher CLS, shape [1, 768].
        k:              Number of candidates to evaluate.
        glimpse_size_px: Glimpse resolution in pixels.
        device:         Torch device.
        min_scale:      Minimum viewpoint scale.
        max_scale:      Maximum viewpoint scale.
        objective:      Candidate score: "cosine", "kl", or "seg-kl".
        kl_temperature: Temperature for KL softmax distributions.
        sample_seed: Optional seed for deterministic candidate sampling.
        probe: Optional ADE20K probe for segmentation-KL scoring.
        canvas_grid_size: Canvas grid size for segmentation-KL scoring.
        seg_kl_target_prob: Full-scene segmentation target distribution.

    Returns:
        best_vp:        The winning Viewpoint.
        best_state:     The RecurrentState after committing the winning viewpoint.
        best_sim:       Cosine similarity achieved by the winning viewpoint.
        best_score:     Objective score used to select the winning viewpoint.
    """
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
        # Fixed by Codex on 2026-05-29
        # Problem: Candidate samples changed between reruns, making a given
        # image/timestep hard to compare even with the same greedy settings.
        # Solution: Seed only the candidate draw inside a forked RNG context,
        # using a timestep-specific seed supplied by the episode runner.
        # Result: Different timesteps still get different candidates, while
        # rerunning the same seeded episode reproduces each timestep's pool.
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
    best_score = -float("inf")
    best_sim = -float("inf")

    with torch.inference_mode():
        for vp in candidates:
            glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=vp,
                glimpse_size_px=glimpse_size_px,
            )
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            canvas_cls = out.state.recurrent_cls.squeeze(1).float()  # [1, 768]
            sim = reconstruction_reward(canvas_cls, teacher_cls)
            if objective == "seg-kl":
                assert probe is not None
                assert canvas_grid_size is not None
                assert seg_kl_target_prob is not None
                score = _score_segmentation_kl(
                    model=model,
                    state=out.state,
                    probe=probe,
                    canvas_grid_size=canvas_grid_size,
                    target_prob=seg_kl_target_prob,
                    kl_temperature=kl_temperature,
                    batch_size=image.shape[0],
                )
            else:
                score = _score_canvas_cls(
                    canvas_cls=canvas_cls,
                    teacher_cls=teacher_cls,
                    objective=objective,
                    kl_temperature=kl_temperature,
                )

            if score > best_score:
                best_score = score
                best_sim = sim
                best_vp = vp
                best_state = out.state

    assert best_vp is not None and best_state is not None
    return best_vp, best_state, best_sim, best_score


def full_scene_step(
    model: GreedyModel,
    image: torch.Tensor,
    state: RecurrentState,
    teacher_cls: torch.Tensor,
    glimpse_size_px: int,
    device: torch.device,
    objective: GreedyObjective = "cosine",
    kl_temperature: float = 1.0,
    probe: torch.nn.Module | None = None,
    canvas_grid_size: int | None = None,
    seg_kl_target_prob: torch.Tensor | None = None,
) -> tuple[Viewpoint, RecurrentState, float, float]:
    """Commit one full-scene glimpse and return the updated state/similarity."""
    with torch.inference_mode():
        vp = Viewpoint.full_scene(batch_size=image.shape[0], device=device)
        glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=vp,
            glimpse_size_px=glimpse_size_px,
        )
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        canvas_cls = out.state.recurrent_cls.squeeze(1).float()
        sim = reconstruction_reward(canvas_cls, teacher_cls)
        if objective == "seg-kl":
            assert probe is not None
            assert canvas_grid_size is not None
            assert seg_kl_target_prob is not None
            score = _score_segmentation_kl(
                model=model,
                state=out.state,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                target_prob=seg_kl_target_prob,
                kl_temperature=kl_temperature,
                batch_size=image.shape[0],
            )
        else:
            score = _score_canvas_cls(
                canvas_cls=canvas_cls,
                teacher_cls=teacher_cls,
                objective=objective,
                kl_temperature=kl_temperature,
            )
    return vp, out.state, sim, score


def run_greedy_episode(
    model: GreedyModel,
    image: torch.Tensor,
    teacher_cls: torch.Tensor,
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
    objective: GreedyObjective = "cosine",
    kl_temperature: float = 1.0,
) -> dict:
    """
    Run a full greedy episode of T steps with K candidates per step.

    Args:
        model:          Frozen CanViT model.
        image:          Scene tensor, shape [1, 3, H, W].
        teacher_cls:    Frozen teacher CLS, shape [1, 768].
        init_state:     Initial RecurrentState (from model.init_state).
        t:              Number of timesteps.
        k:              Number of candidates per step.
        glimpse_size_px: Glimpse resolution in pixels.
        device:         Torch device.
        seed:           Optional random seed for reproducibility.
        mask:           Optional ADE20K target mask, shape [1, H, W] or [H, W].
        probe:          Optional segmentation probe for per-timestep mIoU.
        canvas_grid_size: Canvas grid size used to reshape the state features.
        start_with_full_scene: If True, timestep 0 is a committed full-scene
            glimpse; later timesteps use greedy K-candidate search.
        objective: Greedy search objective: "cosine", "kl", or "seg-kl".
        kl_temperature: Softmax temperature used by the KL objective.

    Returns:
        Dictionary with per-step diagnostics:
            - viewpoints: list of chosen Viewpoint objects
            - sims:       list of cosine similarities after each step
            - scores:     list of objective scores used for selection
            - rewards:    list of delta rewards (gain per step)
            - scales:     list of chosen scales (for coarse-to-fine analysis)
            - centers:    list of chosen centers
            - mious:      optional list of mIoU after each committed step
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    if objective not in {"cosine", "kl", "seg-kl"}:
        raise ValueError("objective must be 'cosine', 'kl', or 'seg-kl'.")
    if kl_temperature <= 0:
        raise ValueError("kl_temperature must be positive.")
    if (mask is None) != (probe is None):
        raise ValueError("Pass both mask and probe to compute per-timestep mIoU.")
    if probe is not None and canvas_grid_size is None:
        raise ValueError("Pass canvas_grid_size when computing per-timestep mIoU.")
    if objective == "seg-kl" and (probe is None or canvas_grid_size is None):
        raise ValueError("seg-kl objective requires --miou/probe and canvas_grid_size.")

    state = init_state
    prev_sim = 0.0

    viewpoints, sims, scores, rewards, scales, centers = [], [], [], [], [], []
    mious = [] if probe is not None else None
    mask_dev = mask.to(device) if mask is not None else None
    seg_kl_target_prob = None
    if objective == "seg-kl":
        assert probe is not None and canvas_grid_size is not None
        # Fixed by Codex on 2026-05-29
        # Problem: KL over CLS embedding coordinates is not an AdaGlimpse-like
        # task loss and can leave candidate rankings unchanged by temperature.
        # Solution: Build a full-scene ADE20K probe distribution once per
        # episode and score candidate states by negative segmentation KL.
        # Result: The k-armed greedy runner can use a meaningful KL objective
        # whenever the --miou segmentation probe path is active.
        with torch.inference_mode():
            teacher_vp = Viewpoint.full_scene(batch_size=image.shape[0], device=device)
            teacher_glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=teacher_vp,
                glimpse_size_px=glimpse_size_px,
            )
            teacher_out = model(
                glimpse=teacher_glimpse,
                state=init_state,
                viewpoint=teacher_vp,
            )
            teacher_logits = _seg_logits_from_state(
                model=model,
                state=teacher_out.state,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                batch_size=image.shape[0],
            )
            seg_kl_target_prob = F.softmax(
                teacher_logits / kl_temperature,
                dim=1,
            ).detach()

    for step_idx in range(t):
        if step_idx == 0 and start_with_full_scene:
            vp, state, sim, score = full_scene_step(
                model=model,
                image=image,
                state=state,
                teacher_cls=teacher_cls,
                glimpse_size_px=glimpse_size_px,
                device=device,
                objective=objective,
                kl_temperature=kl_temperature,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                seg_kl_target_prob=seg_kl_target_prob,
            )
        else:
            vp, state, sim, score = greedy_step(
                model=model,
                image=image,
                state=state,
                teacher_cls=teacher_cls,
                k=k,
                glimpse_size_px=glimpse_size_px,
                device=device,
                objective=objective,
                kl_temperature=kl_temperature,
                sample_seed=None if seed is None else seed + step_idx,
                probe=probe,
                canvas_grid_size=canvas_grid_size,
                seg_kl_target_prob=seg_kl_target_prob,
            )
        reward = sim - prev_sim
        prev_sim = sim

        viewpoints.append(vp)
        sims.append(sim)
        scores.append(score)
        rewards.append(reward)
        scales.append(float(vp.scales[0].item()))
        centers.append(vp.centers[0].cpu().tolist()) 
        if probe is not None:
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
        "sims": sims,
        "scores": scores,
        "rewards": rewards,
        "scales": scales,
        "centers": centers,
    }
    if mious is not None:
        result["mious"] = mious
    return result
