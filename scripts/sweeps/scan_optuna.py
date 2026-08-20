"""Print the best completed trials from an Optuna study.

Usage:
    uv run python scripts/sweeps/scan_optuna.py \
        --storage sqlite:///checkpoints/canvas_ppo/ade20k-optuna/optuna.db \
        --study-name ade20k-ppo
"""

from __future__ import annotations

import argparse

import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage",
        default="sqlite:///checkpoints/canvas_ppo/ade20k-optuna/optuna.db",
        help="Optuna storage URL.",
    )
    parser.add_argument(
        "--study-name",
        default="ade20k-ppo",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of completed trials to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    trials = sorted(
        [trial for trial in study.trials if trial.value is not None],
        key=lambda trial: trial.value,
    )

    # Problem: this file previously contained a shell heredoc, so it could not
    # be compiled or run as Python. Solution: keep the same default study but
    # expose it through argparse. Result: sweep inspection is a normal script.
    for trial in trials[: args.limit]:
        print(f"\ntrial={trial.number} value={trial.value:.6f}")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
