"""Checkpoint helpers for image-dependent Canvas SAC training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from canvit_rl.policies.canvas_state.sac import CanvasSAC
from canvit_rl.policies.canvas_state.models import CanvasStateActor, CanvasStateCritic
from canvit_rl.policies.canvas_state.state import canvas_aux_state_kind


def canvas_state_representation(args: argparse.Namespace) -> str:
    """Return the saved label for Canvas SAC actor/critic state inputs."""
    history_suffix = (
        "_with_viewpoint_history"
        if not getattr(args, "disable_viewpoint_history_state", False)
        else ""
    )
    # Label the aux channel by its signal so ADE eval/visualization (and the
    # IN21k-compatible state_representation parser) rebuild the right map.
    aux_suffix = {
        None: "",
        "segmentation_entropy": "_entropy",
        "reconstruction_norm": "_reconstruction_norm",
    }[canvas_aux_state_kind(args)]
    return f"current_canvas_layernorm{aux_suffix}{history_suffix}"


# Saved-arg names that fix the CanvasStateActor/Critic parameter shapes. Both
# ADE Canvas SAC and IN21k dense SAC checkpoints store them under "args".
CANVAS_STATE_ARCH_ARGS = (
    "d_model",
    "rff_dim",
    "rff_seed",
    "disable_canvas_avg_pool",
    "disable_canvas_max_pool",
    "disable_viewpoint_history_state",
)
_STATE_REPRESENTATION_PREFIX = "current_canvas_layernorm"
_HISTORY_SUFFIX = "_with_viewpoint_history"
# state_representation aux token -> canvas_aux_state_kind. ADE writes
# "entropy" for probe entropy; IN21k dense SAC writes "reconstruction_norm"
# for both its --canvas-entropy-state and --reconstruction-norm-state.
_AUX_TOKEN_KINDS = {
    "": None,
    "entropy": "segmentation_entropy",
    "reconstruction_norm": "reconstruction_norm",
}


def checkpoint_aux_state_kind(checkpoint: object) -> str | None:
    """Return the aux kind encoded in a checkpoint's state_representation.

    Raises for aux states ADE Canvas SAC cannot rebuild (e.g. IN21k
    teacher_reconstruction_error/detail_debt/cos_prev or multi-channel combos).
    Returns None when the checkpoint has no state_representation metadata.
    """
    if not isinstance(checkpoint, dict):
        return None
    representation = checkpoint.get("state_representation")
    if not isinstance(representation, str):
        return None
    if not representation.startswith(_STATE_REPRESENTATION_PREFIX):
        raise ValueError(f"Unknown canvas state representation: {representation}")
    token = (
        representation.removeprefix(_STATE_REPRESENTATION_PREFIX)
        .removesuffix(_HISTORY_SUFFIX)
        .removeprefix("_")
    )
    if token not in _AUX_TOKEN_KINDS:
        raise ValueError(
            f"Checkpoint aux state '{token}' ({representation}) is not supported "
            "by ADE Canvas SAC; only probe entropy and reconstruction_norm are."
        )
    return _AUX_TOKEN_KINDS[token]


def inherit_canvas_state_args(args: argparse.Namespace) -> None:
    """Copy actor/critic state settings from the --resume/--init-actor checkpoint.

    Problem: fine-tuning an IN21k dense SAC actor on ADE required repeating its
    architecture flags by hand, and IN21k's --canvas-entropy-state means the
    reconstructed DINOv3 feature norm, not ADE probe entropy. A mismatched flag
    either failed the strict load or silently fed the wrong aux signal.
    Solution: read the checkpoint's saved args (CANVAS_STATE_ARCH_ARGS) and its
    state_representation aux token, and overwrite the matching run args.
    Result: --init-actor-checkpoint <in21k>/latest.pt rebuilds the same actor
    shape and aux map; --resume restores the run's own state settings.
    """
    path = args.resume if args.resume is not None else args.init_actor_checkpoint
    if path is None:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if not isinstance(saved_args, dict):
        # Bare actor_final.pt state dicts carry no metadata; keep CLI values.
        print(f"No saved args in {path}; using CLI actor/critic state settings.")
        return

    inherited: dict[str, object] = {}
    for name in CANVAS_STATE_ARCH_ARGS:
        if name in saved_args:
            inherited[name] = saved_args[name]
    aux_kind = checkpoint_aux_state_kind(checkpoint)
    if aux_kind is None and "state_representation" not in checkpoint:
        # Pre-label ADE checkpoints: the flag still meant probe entropy.
        aux_kind = (
            "segmentation_entropy"
            if saved_args.get("canvas_entropy_state", False)
            else None
        )
    inherited["canvas_entropy_state"] = aux_kind == "segmentation_entropy"
    inherited["reconstruction_norm_state"] = aux_kind == "reconstruction_norm"

    for name, value in inherited.items():
        current = getattr(args, name, None)
        if current != value:
            print(f"Inherited {name}={value!r} from {path} (CLI had {current!r}).")
        setattr(args, name, value)
    print(f"Canvas aux state from {path}: {aux_kind or 'none'}")


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
    best_eval_final_ce: float,
    eval_metrics: dict[str, float] | None,
) -> None:
    """Save canvas SAC state; best selection matches final validation CE."""
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
            "best_eval_final_ce": best_eval_final_ce,
            "comet_experiment_key": getattr(args, "comet_experiment_key", None),
            "selection_metric": "eval/final_ce",
            "selection_mode": "min",
            "eval_metrics": eval_metrics or {},
            # Problem: core-state ablations now alter actor/critic input shape.
            # Solution: persist a state label that records whether explicit
            # Viewpoint history was encoded. Result: eval/visualization can
            # rebuild the matching CanvasStateActor without CLI guesswork.
            "state_representation": canvas_state_representation(args),
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
        return 1, 0, float("inf")
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
    selection_metric = checkpoint.get("selection_metric")
    best_eval_final_ce = checkpoint.get("best_eval_final_ce")
    if best_eval_final_ce is None and selection_metric == "eval/final_ce":
        best_eval_final_ce = checkpoint.get("eval_metrics", {}).get("eval/final_ce")
    if best_eval_final_ce is None:
        best_eval_final_ce = float("inf")
    return (
        int(checkpoint.get("batch", 0)) + 1,
        int(checkpoint.get("updates", 0)),
        float(best_eval_final_ce),
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
