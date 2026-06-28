"""Optuna sweep utilities for the canvas-state SAC trainer."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Callable


def add_canvas_sac_optuna_args(parser: argparse.ArgumentParser) -> None:
    """Register Optuna flags without duplicating train_canvas_sac setup."""
    # Problem: Canvas SAC needs hyperparameter sweeps, but copying the full
    # training parser/setup into a second script would make future CLI changes
    # drift.
    # Solution: expose only sweep-control flags here and let train_canvas_sac.py
    # keep owning dataset/model/checkpoint arguments.
    # Result: One trainer configuration can run either a single SAC job or an
    # Optuna study over trial-specific hyperparameters.
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-study-name", type=str, default="canvas-sac")
    parser.add_argument("--optuna-storage", type=str, default=None)


def _trial_checkpoint_dir(base_dir: Path, trial_number: int) -> Path:
    """Return the isolated checkpoint directory for one Optuna trial."""
    return base_dir / f"trial_{trial_number}"


def _ensure_optuna_output_dirs(args: argparse.Namespace) -> None:
    """Create checkpoint and local SQLite storage parents before Optuna opens them."""
    # Problem: Optuna opens SQLite storage before the first trial calls
    # train_once(), so train_canvas_sac.py never gets a chance to create the
    # checkpoint directory that commonly holds optuna.db.
    # Solution: create the base checkpoint directory here, and when storage is
    # a local sqlite:/// path, create the database file's parent directory too.
    # Result: cluster runs no longer need a separate mkdir -p before launching
    # --optuna-storage sqlite:///.../optuna.db.
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.optuna_storage or not args.optuna_storage.startswith("sqlite:///"):
        return
    db_path = args.optuna_storage.removeprefix("sqlite:///")
    if db_path in {"", ":memory:"}:
        return
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def build_canvas_sac_trial_args(
    args: argparse.Namespace,
    trial: Any,
) -> argparse.Namespace:
    """Apply a Canvas SAC search space to a deep-copied argparse namespace."""
    trial_args = copy.deepcopy(args)

    # Problem: Sharing checkpoint/resume state across trials contaminates the
    # objective with weights from a different hyperparameter setting.
    # Solution: each trial writes to its own child directory and uses a unique
    # seed while still inheriting all fixed CLI setup from the base namespace.
    # Result: Optuna compares independent Canvas SAC runs without duplicating
    # the expensive CanViT/data configuration code.
    trial_args.optuna_trial = trial.number
    trial_args.seed = args.seed + trial.number
    trial_args.checkpoint_dir = _trial_checkpoint_dir(args.checkpoint_dir, trial.number)
    if args.experiment_name:
        trial_args.experiment_name = f"{args.experiment_name}-trial-{trial.number}"
    else:
        trial_args.experiment_name = f"canvas-sac-trial-{trial.number}"
    trial_args.comet_tags = ",".join(
        tag
        for tag in [
            args.comet_tags,
            "optuna",
            f"trial-{trial.number}",
        ]
        if tag
    )

    trial_args.actor_lr = trial.suggest_float("actor_lr", 1e-5, 3e-3, log=True)
    trial_args.critic_lr = trial.suggest_float("critic_lr", 1e-5, 3e-3, log=True)
    trial_args.alpha_lr = trial.suggest_float("alpha_lr", 1e-5, 3e-3, log=True)
    trial_args.init_alpha = trial.suggest_float("init_alpha", 0.01, 0.5, log=True)
    trial_args.target_entropy = trial.suggest_float("target_entropy", -6.0, -1.0)
    trial_args.tau = trial.suggest_float("tau", 0.001, 0.03, log=True)
    # trial_args.gamma = trial.suggest_categorical("gamma", [0.0, 0.25, 0.5, 0.9])
    trial_args.d_model = trial.suggest_categorical("d_model", [128, 256, 384])
    # trial_args.rff_dim = trial.suggest_categorical("rff_dim", [64, 128, 256])
    trial_args.replay_batch_size = trial.suggest_categorical(
        "replay_batch_size",
        [16, 32, 64, 128, 256],
    )
    trial_args.learning_starts = trial.suggest_categorical(
        "learning_starts",
        [1, 8, 16, 32],
    )
    trial_args.updates_per_batch = trial.suggest_categorical(
        "updates_per_batch",
        [1, 2, 4],
    )
    return trial_args


def run_canvas_sac_optuna(
    args: argparse.Namespace,
    train_once: Callable[[argparse.Namespace], float],
) -> None:
    """Run Optuna trials against train_canvas_sac.train_once."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optuna or run without --optuna-trials.") from exc

    if args.resume is not None:
        raise ValueError("--resume cannot be used with --optuna-trials.")
    _ensure_optuna_output_dirs(args)

    def objective(trial: Any) -> float:
        trial_args = build_canvas_sac_trial_args(args, trial)
        return train_once(trial_args)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage),
    )
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
