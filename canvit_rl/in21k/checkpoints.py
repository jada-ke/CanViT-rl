"""Checkpoint helpers for IN21k dense-feature SAC pretraining."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from canvit_rl.policies.canvas_state.sac import CanvasSAC
from canvit_rl.policies.canvas_state.models import CanvasStateActor, CanvasStateCritic


def _dense_state_representation(args: argparse.Namespace) -> str:
    """Return the Canvas state representation label stored in dense SAC checkpoints."""
    history_suffix = (
        "_with_viewpoint_history"
        if not getattr(args, "disable_viewpoint_history_state", False)
        else ""
    )
    aux_parts: list[str] = []
    if getattr(args, "reconstruction_norm_state", False) or getattr(
        args,
        "canvas_entropy_state",
        False,
    ):
        aux_parts.append("reconstruction_norm")
    if getattr(args, "teacher_reconstruction_error_state", False):
        aux_parts.append("teacher_reconstruction_error")
    if getattr(args, "detail_debt", False):
        aux_parts.append("detail_debt")
    if getattr(args, "cos_prev", False):
        aux_parts.append("cos_prev")
    # Problem: combined dense aux-state ablations need to be reloadable from
    # checkpoint metadata. Solution: encode all selected aux maps in the same
    # order used by canvas_aux_state_map. Result: eval scripts can reconstruct
    # both the channel count and semantic map order without CLI guesses.
    aux_suffix = "" if not aux_parts else "_" + "_".join(aux_parts)
    return f"current_canvas_layernorm{aux_suffix}{history_suffix}"


def _dense_aux_state_channels(args: argparse.Namespace) -> int:
    """Return the optional aux-state channel count for checkpoint reconstruction."""
    return (
        int(
            getattr(args, "canvas_entropy_state", False)
            or getattr(args, "reconstruction_norm_state", False)
        )
        + int(getattr(args, "teacher_reconstruction_error_state", False))
        + int(getattr(args, "detail_debt", False))
        + int(getattr(args, "cos_prev", False))
    )


class DenseSACResume(NamedTuple):
    """Counters and RNG state restored from a dense SAC checkpoint."""

    start_batch: int
    updates: int
    glimpses: int
    rng_state: dict[str, Any] | None


# Args that describe *this* launch rather than the training run, so a resume
# never inherits them from the checkpoint: the resume source itself, the
# target batch, the (already resolved) output folder, the fresh-run-only critic
# initializer, and the Comet key (resolved separately from the checkpoint).
RESUME_NON_INHERITED_ARGS = frozenset(
    {
        "resume",
        "batches",
        "checkpoint_dir",
        "init_critic_checkpoint",
        "comet_experiment_key",
    }
)


def inherit_resume_defaults(
    parser: argparse.ArgumentParser,
    saved_args: dict[str, Any],
    *,
    exclude: frozenset[str] = RESUME_NON_INHERITED_ARGS,
) -> set[str]:
    """Make a checkpoint's saved args the parser defaults; return inherited dests.

    Problem: resuming re-applied every CLI default, so omitting a flag from the
    original run (e.g. --t 4) silently changed hyperparameters. Solution: before
    parsing, set each known action's default to the checkpoint value and drop
    its ``required`` flag. Result: a resume reproduces the original run's
    hyperparameters, while flags passed explicitly on the CLI still override.
    """
    inherited: set[str] = set()
    for action in parser._actions:
        dest = action.dest
        if dest in exclude or dest not in saved_args:
            continue
        action.default = saved_args[dest]
        action.required = False
        inherited.add(dest)
    return inherited


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python, NumPy, and torch (CPU/CUDA/MPS) global RNG states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore global RNG states captured by ``capture_rng_state``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    # Device states are only restored on matching hardware; resuming on a
    # different device type keeps the CPU/Python/NumPy continuation.
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            torch.cuda.set_rng_state(cuda_states[0])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


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
    glimpses: int = 0,
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
            # Problem: the glimpse counter (Comet x-axis) restarted at 0 on
            # resume. Solution: persist it with the batch/update counters.
            # Result: resumed runs continue the same glimpse axis.
            "glimpses": glimpses,
            # Problem: resume re-seeded from --seed, replaying the original
            # run's exploration/replay sampling from the start. Solution: save
            # global RNG states at the batch boundary. Result: a resume
            # continues the random stream where the checkpoint left off.
            "rng_state": capture_rng_state(),
            "metrics": metrics,
            "comet_experiment_key": getattr(args, "comet_experiment_key", None),
            # Problem: dense SAC now has several aux-map ablations that share
            # the legacy replay key. Solution: persist both a descriptive state
            # label and the channel count. Result: ADE eval can rebuild the
            # actor shape and aux map order without relying on CLI guesses.
            "state_representation": _dense_state_representation(args),
            "aux_state_channels": _dense_aux_state_channels(args),
            "reward": getattr(args, "reward_mode", "raw_mse_l0_delta"),
        },
        path,
    )


def format_glimpse_count(glimpses: int) -> str:
    """Format a glimpse count as a short folder label, e.g. 2000000 -> "2M"."""
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if glimpses >= divisor:
            # Up to five decimals, trailing zeros trimmed: 2.5M, 1.23457M, 2M.
            value = f"{glimpses / divisor:.5f}".rstrip("0").rstrip(".")
            return f"{value}{suffix}"
    return str(glimpses)


def resolve_dense_checkpoint_dir(
    *,
    checkpoint_dir: Path | None,
    checkpoint_root: Path,
    experiment_name: str | None,
    start_batch: int,
    end_batch: int,
    batch_size: int,
    t: int,
    resume_glimpses: int,
) -> Path:
    """Return the dense SAC checkpoint directory for this run.

    An explicit ``checkpoint_dir`` always wins. Otherwise, with an experiment
    name, checkpoints go to ``<root>/<experiment_name>/<target glimpses>``,
    where the target is the glimpse count at ``end_batch`` (resumed count plus
    the glimpses this run will add). Without a name, ``checkpoint_root`` is
    used directly (the previous default behavior).
    """
    if checkpoint_dir is not None:
        return checkpoint_dir
    if not experiment_name:
        return checkpoint_root
    # Problem: resumed runs were manually filed under
    # <root>/<experiment>/<glimpses> to keep a checkpoint history. Solution:
    # derive the folder from the target glimpse count. Result: e.g. resuming a
    # 1M-glimpse checkpoint for another 1M glimpses saves under ".../2M".
    # The train loader uses drop_last=True, so every batch has batch_size.
    remaining_batches = max(end_batch - start_batch + 1, 0)
    target_glimpses = resume_glimpses + remaining_batches * batch_size * (t + 1)
    return checkpoint_root / experiment_name / format_glimpse_count(target_glimpses)


def load_dense_sac_resume(
    *,
    path: Path | None,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
    agent: CanvasSAC,
) -> DenseSACResume:
    """Resume dense SAC networks/optimizers and return counters + RNG state."""
    if path is None:
        return DenseSACResume(start_batch=1, updates=0, glimpses=0, rng_state=None)
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
    batch = int(checkpoint.get("batch", 0))
    return DenseSACResume(
        start_batch=batch + 1,
        updates=int(checkpoint.get("updates", 0)),
        glimpses=_checkpoint_glimpses(checkpoint, batch),
        rng_state=checkpoint.get("rng_state"),
    )


def _checkpoint_glimpses(checkpoint: dict, batch: int) -> int:
    """Return the glimpse count stored in a checkpoint, estimating it for old ones."""
    if "glimpses" in checkpoint:
        return int(checkpoint["glimpses"])
    # Problem: checkpoints saved before the "glimpses" key existed cannot
    # restore the exact count. Solution: estimate from the checkpoint's own
    # saved args (not the current CLI, which may change batch size or T), as
    # batch * batch_size * (t + 1). Result: legacy resumes continue close to
    # the original Comet glimpse axis instead of restarting at 0.
    saved_args = checkpoint.get("args", {}) or {}
    batch_size = int(saved_args.get("batch_size", 0))
    t = int(saved_args.get("t", 0))
    return batch * batch_size * (t + 1)


def load_dense_sac_critic_initializer(
    *,
    path: Path | None,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
) -> None:
    """Initialize dense SAC critics from a critic-only checkpoint."""
    if path is None:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "q1" not in checkpoint:
        raise ValueError(f"Expected critic checkpoint with a q1 key: {path}")
    # Problem: critic-only IN21k diagnostics save q1/q2 without actor,
    # optimizers, alpha, replay, or counters. Solution: load only critic
    # weights into a fresh SAC run and immediately sync target critics. Result:
    # a supervised top-tail critic can seed SAC without pretending to be a full
    # resume checkpoint.
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint.get("q2", checkpoint["q1"]))
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
