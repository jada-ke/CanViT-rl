"""Checkpoint helpers for IN21k dense-feature SAC pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from canvit_rl.canvas.sac import CanvasSAC
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic


def save_dense_sac_checkpoint(
    *,
    path: Path,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
    agent: CanvasSAC,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    batch: int,
    updates: int,
    metrics: dict[str, float],
) -> None:
    """Save dense SAC state without assuming ADE/CE validation semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "target_q1": target_q1.state_dict(),
            "target_q2": target_q2.state_dict(),
            "actor_opt": agent.actor_opt.state_dict(),
            "q_opt": agent.q_opt.state_dict(),
            "alpha_opt": agent.alpha_opt.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "args": vars(args),
            "canvas_feature_dim": canvas_feature_dim,
            "batch": batch,
            "updates": updates,
            "metrics": metrics,
            "state_representation": "current_canvas_layernorm_with_viewpoint_history",
            "reward": getattr(args, "reward_mode", "raw_mse_reduction"),
        },
        path,
    )


def load_dense_sac_resume(
    *,
    path: Path | None,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
    agent: CanvasSAC,
) -> tuple[int, int]:
    """Resume dense SAC optimizer and network state from a checkpoint."""
    if path is None:
        return 1, 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint["q2"])
    target_q1.load_state_dict(checkpoint.get("target_q1", q1.state_dict()))
    target_q2.load_state_dict(checkpoint.get("target_q2", q2.state_dict()))
    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "q_opt" in checkpoint:
        agent.q_opt.load_state_dict(checkpoint["q_opt"])
    if "alpha_opt" in checkpoint:
        agent.alpha_opt.load_state_dict(checkpoint["alpha_opt"])
    if "log_alpha" in checkpoint:
        agent.log_alpha.data.copy_(checkpoint["log_alpha"])
    return int(checkpoint.get("batch", 0)) + 1, int(checkpoint.get("updates", 0))
