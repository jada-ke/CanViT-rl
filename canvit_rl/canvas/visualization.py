"""Live visualization hooks for Canvas SAC training."""

from __future__ import annotations

import argparse

import torch

from canvit_rl.env import CanViTEnvConfig
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic

try:
    from visualize_sac_reward_maps import visualize_reward_maps_for_indices
except ImportError:
    from scripts.visualize_sac_reward_maps import visualize_reward_maps_for_indices
try:
    from visualize_policy_glimpses import visualize_canvas_policy_for_indices
except ImportError:
    from scripts.visualize_policy_glimpses import visualize_canvas_policy_for_indices


def parse_reward_map_scales(value: str) -> list[float]:
    """Parse comma-separated reward-map scales."""
    scales = [float(item) for item in value.split(",") if item.strip()]
    if not scales or any(scale <= 0 or scale > 1 for scale in scales):
        raise ValueError("--reward-map-scales must contain values in (0, 1].")
    return scales


def maybe_visualize_canvas_sac_reward_maps(
    *,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    eval_dataset,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    device: torch.device,
    canvit_dtype: torch.dtype,
    update_count: int,
    comet_exp,
) -> None:
    """Optionally save live canvas SAC reward/Q maps after validation."""
    if args.reward_map_images <= 0:
        return
    indices = list(range(min(args.reward_map_images, len(eval_dataset))))
    # Problem: live reward diagnostics require two visualization scripts, which
    # made the trainer import and coordinate plotting details directly.
    # Solution: keep that orchestration here so train_canvas_sac only decides
    # when visualization should run.
    paths = visualize_reward_maps_for_indices(
        actor=actor,
        q1=q1,
        q2=q2,
        dataset=eval_dataset,
        indices=indices,
        model=model,
        probe=probe,
        cfg=cfg,
        device=device,
        min_scale=args.min_scale,
        scales=parse_reward_map_scales(args.reward_map_scales),
        grid_size=args.reward_map_grid_size,
        chunk_size=args.reward_map_chunk_size,
        output_dir=args.reward_map_output_dir,
        split_label=args.eval_split,
        title_prefix=f"Canvas SAC validation reward map update={update_count}",
        policy_kind="canvas",
        max_history=args.max_history,
        output_name_suffix=f"update_{update_count:06d}",
    )
    paths.extend(
        visualize_canvas_policy_for_indices(
            actor=actor,
            dataset=eval_dataset,
            indices=indices,
            model=model,
            probe=probe,
            cfg=cfg,
            device=device,
            t=args.t,
            max_history=args.max_history,
            min_scale=args.min_scale,
            output_dir=args.reward_map_output_dir,
            split_label=args.eval_split,
            title_prefix=f"Canvas SAC validation policy update={update_count}",
            canvit_dtype=canvit_dtype,
            output_name_suffix=f"update_{update_count:06d}",
        )
    )
    if comet_exp is not None:
        for path in paths:
            comet_exp.log_image(str(path), name=path.name, step=update_count)
