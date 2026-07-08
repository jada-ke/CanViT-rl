"""Checkpoint helpers for image-dependent Canvas SAC training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from canvit_rl.canvas.sac import CanvasSAC
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic


def save_canvas_sac_checkpoint(
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
    best_relative_ce_gain: float,
    eval_metrics: dict[str, float] | None,
) -> None:
    """Save canvas SAC state; best selection matches relative CE reward."""
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
            "best_relative_ce_gain": best_relative_ce_gain,
            "selection_metric": "eval/reward",
            "eval_metrics": eval_metrics or {},
            "state_representation": (
                "current_canvas_layernorm_entropy_with_viewpoint_history"
                if getattr(args, "canvas_entropy_state", False)
                else "current_canvas_layernorm_with_viewpoint_history"
            ),
        },
        path,
    )


def load_canvas_sac_resume(
    *,
    args: argparse.Namespace,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
    agent: CanvasSAC,
) -> tuple[int, int, float]:
    """Resume a canvas SAC checkpoint while keeping every network trainable."""
    if args.resume is None:
        return 1, 0, float("-inf")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
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
    return (
        int(checkpoint.get("batch", 0)) + 1,
        int(checkpoint.get("updates", 0)),
        float(
            checkpoint.get(
                "best_relative_ce_gain",
                checkpoint.get("best_ce_gain", float("-inf")),
            )
        ),
    )


def checkpoint_module_state(
    checkpoint: object,
    key: str,
    *,
    path: Path,
) -> dict[str, torch.Tensor]:
    """Extract either a named module state dict or a bare state dict."""
    if isinstance(checkpoint, dict) and key in checkpoint:
        state = checkpoint[key]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict for {key} in {path}.")
    return state


def load_canvas_sac_pretrained_initializers(
    *,
    args: argparse.Namespace,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
) -> None:
    """Initialize SAC modules from pretrained actor/critic checkpoints."""
    if args.resume is not None:
        if (
            args.init_actor_checkpoint is not None
            or args.init_critic_checkpoint is not None
        ):
            raise ValueError(
                "--resume cannot be combined with --init-actor-checkpoint or "
                "--init-critic-checkpoint; resume already restores all SAC state."
            )
        return

    if args.init_actor_checkpoint is not None:
        checkpoint = torch.load(
            args.init_actor_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        actor.load_state_dict(
            checkpoint_module_state(
                checkpoint,
                "actor",
                path=args.init_actor_checkpoint,
            )
        )
        print(f"Initialized canvas SAC actor from {args.init_actor_checkpoint}")

    if args.init_critic_checkpoint is not None:
        checkpoint = torch.load(
            args.init_critic_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "--init-critic-checkpoint expects a dict checkpoint with q1/q2 keys."
            )
        if "q1" not in checkpoint:
            raise ValueError(
                f"Expected q1 in critic checkpoint: {args.init_critic_checkpoint}"
            )

        q1.load_state_dict(checkpoint["q1"])
        q2.load_state_dict(checkpoint.get("q2", checkpoint["q1"]))
        target_q1.load_state_dict(q1.state_dict())
        target_q2.load_state_dict(q2.state_dict())
        print(f"Initialized canvas SAC critics from {args.init_critic_checkpoint}")
