"""Optuna launcher for IN21k dense-feature Canvas SAC hyperparameter tuning.

This script wraps ``scripts/training/in21k/train_i21k_dense_sac.py`` in one process per trial
so the dense trainer can keep owning model/data initialization, checkpointing,
and Comet logging.

Example:
    uv run python scripts/sweeps/in21k/i21k_dense_sac_optuna.py \
        --optuna-trials 20 \
        --optuna-storage sqlite:///checkpoints/i21k_dense_sac/optuna/optuna.db \
        --optuna-checkpoint-dir checkpoints/i21k_dense_sac/optuna \
        -- \
        --feature-base-dir datasets/imagenet_ood/features \
        --feature-image-root datasets/imagenet_ood/images \
        --eval-images 16 \
        --batches 1000 \
        --batch-size 8 \
        --t 2 \
        --no-comet
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "training" / "in21k" / "train_i21k_dense_sac.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse Optuna-owned flags separately from dense SAC trainer flags."""
    raw_args = sys.argv[1:]
    if "--" in raw_args:
        separator = raw_args.index("--")
        optuna_argv = raw_args[:separator]
        train_argv = raw_args[separator + 1 :]
    else:
        # Problem: the wrapped trainer has many flags and changes often.
        # Solution: parse only known Optuna flags here and forward the rest.
        # Result: new dense SAC flags can be used without editing this launcher.
        optuna_argv = raw_args
        train_argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-study-name", type=str, default="i21k-dense-sac")
    parser.add_argument("--optuna-storage", type=str, default=None)
    parser.add_argument(
        "--optuna-checkpoint-dir",
        type=Path,
        default=Path("checkpoints/i21k_dense_sac/optuna"),
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default="eval/reward",
        help="Metric key read from each trial's final checkpoint.",
    )
    parser.add_argument(
        "--objective-direction",
        choices=["maximize", "minimize"],
        default="maximize",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--search-architecture",
        action="store_true",
        help=(
            "Currently disabled/no-op. Architecture dimensions are fixed while "
            "SAC dynamics are being swept."
        ),
    )
    parser.add_argument(
        "--search-replay",
        action="store_true",
        help="Also tune replay batch size, warmup, and update cadence.",
    )
    parser.add_argument(
        "--search-gamma",
        action="store_true",
        help="Also tune gamma. Leave off for immediate-reward dense experiments.",
    )
    parser.add_argument(
        "--optuna-comet-prefix",
        type=str,
        default="i21k-dense-sac-optuna",
        help="Experiment-name prefix passed to the wrapped trainer.",
    )
    parser.add_argument(
        "--optuna-comet-tags",
        type=str,
        default="optuna",
        help="Comma-separated Comet tags passed to the wrapped trainer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first trial command and exit without starting Optuna.",
    )
    parser.add_argument(
        "--trial-output",
        choices=["filtered", "all", "quiet"],
        default="filtered",
        help=(
            "Control wrapped trainer output. 'filtered' streams normal lines "
            "but drops tqdm carriage-return progress updates."
        ),
    )
    args, unknown = parser.parse_known_args(optuna_argv)
    if train_argv and unknown:
        raise ValueError(
            "Unknown Optuna flags before '--': "
            + " ".join(unknown)
            + ". Put dense trainer flags after '--'."
        )
    if not train_argv:
        train_argv = unknown
    if args.optuna_trials < 1:
        raise ValueError("--optuna-trials must be positive.")
    return args, train_argv


def _ensure_output_dirs(args: argparse.Namespace) -> None:
    """Create checkpoint and local SQLite storage parents before trials start."""
    args.optuna_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.optuna_storage or not args.optuna_storage.startswith("sqlite:///"):
        return
    db_path = args.optuna_storage.removeprefix("sqlite:///")
    if db_path in {"", ":memory:"}:
        return
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _suggest_trial_overrides(args: argparse.Namespace, trial: Any) -> list[str]:
    """Return CLI overrides for one conservative dense SAC search point."""
    overrides = [
        "--actor-lr",
        str(trial.suggest_float("actor_lr", 5e-5, 3e-4, log=True)),
        "--critic-lr",
        str(trial.suggest_float("critic_lr", 2e-4, 8e-4, log=True)),
        "--alpha-lr",
        str(trial.suggest_float("alpha_lr", 1e-4, 8e-4, log=True)),
        "--init-alpha",
        str(trial.suggest_float("init_alpha", 0.01, 0.10, log=True)),
        "--target-entropy",
        str(trial.suggest_float("target_entropy", -5.0, -2.0)),
    ]
    if args.search_replay:
        # Problem: the original replay search included tiny debug settings that
        # update SAC from little replay diversity. Solution: keep replay search
        # opt-in, but use warmups/batch sizes appropriate for small 80/20 image
        # subset sweeps. Result: Optuna spends trials on plausible SAC dynamics
        # instead of near-immediate updates from a handful of transitions.
        overrides.extend(
            [
                "--replay-batch-size",
                str(trial.suggest_categorical("replay_batch_size", [16, 32, 64])),
                "--learning-starts",
                str(trial.suggest_categorical("learning_starts", [64, 128, 256])),
                "--updates-per-batch",
                str(trial.suggest_categorical("updates_per_batch", [1, 2, 4])),
            ]
        )
    # Problem: architecture sweeps add many trials and recent experiments point
    # at SAC dynamics rather than model capacity. Solution: keep
    # --search-architecture accepted for command compatibility, but leave the
    # dimension search disabled until capacity is the active question again.
    # Result: Optuna focuses on LR/alpha/replay without silently
    # changing actor or critic size.
    # if args.search_architecture:
    #     overrides.extend(
    #         [
    #             "--d-model",
    #             str(trial.suggest_categorical("d_model", [128, 256, 384])),
    #             "--rff-dim",
    #             str(trial.suggest_categorical("rff_dim", [64, 128, 256])),
    #             "--critic-d-model",
    #             str(trial.suggest_categorical("critic_d_model", [256, 384, 512])),
    #             "--critic-rff-dim",
    #             str(trial.suggest_categorical("critic_rff_dim", [128, 256, 384])),
    #         ]
    #     )
    if args.search_gamma:
        overrides.extend(
            [
                "--gamma",
                str(trial.suggest_categorical("gamma", [0.0, 0.25, 0.5])),
            ]
        )
    return overrides


def _build_trial_command(
    *,
    args: argparse.Namespace,
    train_argv: list[str],
    trial: Any,
) -> tuple[list[str], Path]:
    """Build a trainer subprocess command with trial-local outputs."""
    checkpoint_dir = args.optuna_checkpoint_dir / f"trial_{trial.number}"
    tags = ",".join(
        tag
        for tag in [
            args.optuna_comet_tags,
            f"trial-{trial.number}",
        ]
        if tag
    )
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        *train_argv,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--seed",
        str(args.base_seed + trial.number),
        "--experiment-name",
        f"{args.optuna_comet_prefix}-trial-{trial.number}",
        "--comet-tags",
        tags,
        *_suggest_trial_overrides(args, trial),
    ]
    return command, checkpoint_dir


def _load_objective(checkpoint_dir: Path, metric_name: str) -> float:
    """Read the requested metric from a dense SAC final checkpoint."""
    checkpoint_path = checkpoint_dir / "final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trial checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {})
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(
            f"Metric {metric_name!r} not found in {checkpoint_path}. "
            f"Available metrics: {sorted(metrics)}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Metric {metric_name!r} is not finite: {value}")
    return value


def _looks_like_tqdm_progress(record: str) -> bool:
    """Detect transient tqdm progress records emitted with carriage returns."""
    stripped = record.strip()
    if not stripped:
        return True
    if "Training IN21k dense SAC:" in stripped:
        return True
    return (
        "%" in stripped
        and "|" in stripped
        and ("it/s" in stripped or "s/it" in stripped)
    )


def _run_filtered_trial_command(command: list[str]) -> None:
    """Stream trainer output while dropping progress-bar redraws."""
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    record = ""
    while True:
        char = process.stdout.read(1)
        if char == "":
            break
        if char in {"\n", "\r"}:
            if record and not _looks_like_tqdm_progress(record):
                print(record, flush=True)
            record = ""
            continue
        record += char
    if record and not _looks_like_tqdm_progress(record):
        print(record, flush=True)
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _run_trial_command(
    command: list[str],
    _checkpoint_dir: Path,
    output_mode: str,
) -> None:
    """Run one dense SAC trial without flooding scheduler stdout with tqdm bars."""
    if output_mode == "all":
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        return
    if output_mode == "quiet":
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    # Problem: tqdm writes frequent carriage-return redraws that clutter batch
    # output files. Solution: stream the child process through a small filter
    # and print only stable newline-style messages. Result: eval/status lines
    # remain visible without creating per-trial log files.
    _run_filtered_trial_command(command)


def main() -> None:
    """Run the Optuna study."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optuna before running this script.") from exc

    args, train_argv = parse_args()
    _ensure_output_dirs(args)
    if args.dry_run:
        # Problem: dry-run previously entered Optuna and printed one command per
        # default trial. Solution: build one lightweight fixed trial command and
        # return immediately. Result: dry-run is a quick path/executable smoke
        # check for the nested trainer location.
        class DryRunTrial:
            number = 0

            @staticmethod
            def suggest_float(_name, low, high, **_kwargs):
                return (low + high) / 2.0

            @staticmethod
            def suggest_categorical(_name, choices):
                return choices[0]

        command, _checkpoint_dir = _build_trial_command(
            args=args,
            train_argv=train_argv,
            trial=DryRunTrial(),
        )
        print(" ".join(command))
        return

    def objective(trial: Any) -> float:
        command, checkpoint_dir = _build_trial_command(
            args=args,
            train_argv=train_argv,
            trial=trial,
        )
        trial.set_user_attr("checkpoint_dir", str(checkpoint_dir))
        try:
            _run_trial_command(command, checkpoint_dir, args.trial_output)
            value = _load_objective(checkpoint_dir, args.objective_metric)
        except Exception as exc:
            # Problem: a single failed GPU trial should not kill a long sweep.
            # Solution: mark failed or metric-less runs as pruned. Result:
            # Optuna keeps the useful completed trials and continues searching.
            raise optuna.TrialPruned(str(exc)) from exc
        trial.set_user_attr(args.objective_metric, value)
        return value

    study = optuna.create_study(
        direction=args.objective_direction,
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage),
    )
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
