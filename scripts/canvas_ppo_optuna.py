"""Optuna sweep utilities for the canvas-state PPO trainer."""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
from typing import Any, Callable

import torch


def add_canvas_ppo_optuna_args(parser: argparse.ArgumentParser) -> None:
    """Register Optuna flags without coupling PPO to SAC search spaces."""
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-study-name", type=str, default="ade20k-ppo-t1")
    parser.add_argument("--optuna-storage", type=str, default=None)


def _trial_checkpoint_dir(base_dir: Path, trial_number: int) -> Path:
    """Return the isolated checkpoint directory for one Optuna trial."""
    return base_dir / f"trial_{trial_number}"


def _ensure_optuna_output_dirs(args: argparse.Namespace) -> None:
    """Create checkpoint and local SQLite storage parents before Optuna starts."""
    # Problem: Optuna opens SQLite storage before the first PPO trial creates
    # checkpoint output directories. Solution: mirror the SAC helper by making
    # both checkpoint and local sqlite parents up front. Result: sweeps can be
    # launched directly with sqlite:///.../optuna.db paths.
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.optuna_storage or not args.optuna_storage.startswith("sqlite:///"):
        return
    db_path = args.optuna_storage.removeprefix("sqlite:///")
    if db_path in {"", ":memory:"}:
        return
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def build_canvas_ppo_trial_args(
    args: argparse.Namespace,
    trial: Any,
) -> argparse.Namespace:
    """Apply a PPO-focused search space to a deep-copied argparse namespace."""
    trial_args = copy.deepcopy(args)
    trial_args.optuna_trial = trial.number
    trial_args.seed = args.seed + trial.number
    trial_args.checkpoint_dir = _trial_checkpoint_dir(args.checkpoint_dir, trial.number)
    if args.experiment_name:
        trial_args.experiment_name = f"{args.experiment_name}-trial-{trial.number}"
    else:
        trial_args.experiment_name = f"canvas-ppo-trial-{trial.number}"
    trial_args.comet_tags = ",".join(
        tag
        for tag in [
            args.comet_tags,
            "optuna",
            f"trial-{trial.number}",
        ]
        if tag
    )

    # Problem: the first PPO search space spent trials on secondary knobs that
    # are less likely to explain policy collapse. Solution: tune only the
    # high-impact optimization/stability/exploration controls and leave all
    # other PPO settings fixed from the CLI. Result: small sweeps converge
    # faster and are easier to interpret.
    trial_args.actor_lr = trial.suggest_float("actor_lr", 5e-5, 5e-4, log=True)
    trial_args.critic_lr = trial.suggest_float("critic_lr", 5e-5, 6e-4, log=True)
    trial_args.ppo_epochs = trial.suggest_categorical("ppo_epochs", [1, 2, 4])
    trial_args.ppo_entropy_coef = trial.suggest_float(
        "ppo_entropy_coef",
        0.01,
        0.1,
        log=True,
    )
    trial_args.ppo_target_kl = trial.suggest_float("ppo_target_kl", 0.01, 0.06)

    # Optional broader sweep knobs. Keep these commented for small, readable
    # searches; uncomment when the main stability controls are no longer enough.
    # trial_args.ppo_minibatch_size = trial.suggest_categorical(
    #     "ppo_minibatch_size",
    #     [value for value in [4, 8, 16, 32] if value <= args.batch_size * args.t],
    # )
    # trial_args.ppo_clip_coef = trial.suggest_categorical(
    #     "ppo_clip_coef",
    #     [0.1, 0.15, 0.2, 0.3],
    # )
    # trial_args.gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
    # trial_args.max_grad_norm = trial.suggest_categorical(
    #     "max_grad_norm",
    #     [0.3, 0.5, 1.0],
    # )
    return trial_args


def run_canvas_ppo_optuna(
    args: argparse.Namespace,
    train_once: Callable[[argparse.Namespace], float],
) -> None:
    """Run Optuna trials against train_canvas_ppo.train_once."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optuna or run without --optuna-trials.") from exc

    if args.resume is not None:
        raise ValueError("--resume cannot be used with --optuna-trials.")
    _ensure_optuna_output_dirs(args)

    def objective(trial: Any) -> float:
        trial_args = build_canvas_ppo_trial_args(args, trial)
        try:
            return train_once(trial_args)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    study = optuna.create_study(
        direction="minimize",
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage),
    )
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
