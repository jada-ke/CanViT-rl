"""Print the best completed trials from an Optuna study.

Usage:
    uv run python scripts/sweeps/scan_optuna.py \
        --storage sqlite:///checkpoints/canvas_ppo/ade20k-optuna/optuna.db \
        --study-name ade20k-ppo
"""

from __future__ import annotations

import argparse
from typing import Any

import optuna


def _param_to_cli_flag(name: str) -> str:
    """Convert Optuna's snake_case parameter names to trainer CLI flags."""
    return "--" + name.replace("_", "-")


def _format_cli_value(value: Any) -> str:
    """Return a shell-friendly scalar representation for an Optuna parameter."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _format_cli_flags(params: dict[str, Any], *, indent: str = "  ") -> list[str]:
    """Format one trial's params as backslash-continued trainer CLI overrides."""
    items = list(params.items())
    lines: list[str] = []
    for index, (key, value) in enumerate(items):
        suffix = " \\" if index < len(items) - 1 else ""
        lines.append(
            f"{indent}{_param_to_cli_flag(key)} {_format_cli_value(value)}{suffix}"
        )
    return lines


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
    parser.add_argument(
        "--output",
        choices=["summary", "cli"],
        default="summary",
        help="Print either key/value summaries or trainer-compatible CLI flags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    reverse = study.direction.name == "MAXIMIZE"
    trials = sorted(
        [trial for trial in study.trials if trial.value is not None],
        key=lambda trial: trial.value,
        reverse=reverse,
    )

    # Problem: copying best Optuna params by hand into long Apptainer training
    # commands is typo-prone. Solution: keep the summary view but add a CLI
    # output mode that maps snake_case study params to the trainer's kebab-case
    # argparse flags. Result: best trials can be pasted directly after the
    # fixed non-Optuna trainer args.
    for trial in trials[: args.limit]:
        print(f"\ntrial={trial.number} value={trial.value:.6f}")
        if args.output == "cli":
            for line in _format_cli_flags(trial.params):
                print(line)
            continue
        for key, value in trial.params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
