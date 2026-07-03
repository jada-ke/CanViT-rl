"""Optional Comet logging helpers for Canvas SAC-style training scripts."""

from __future__ import annotations

import argparse
from typing import Any

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None


def add_canvas_sac_comet_args(parser: argparse.ArgumentParser) -> None:
    """Register Comet options used by the Canvas SAC trainer."""
    parser.add_argument("--comet-log-interval", type=int, default=20)
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="canvas-sac")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="canvas-sac")


def make_comet_experiment(args: argparse.Namespace):
    """Create a Comet experiment unless disabled for local dry runs."""
    if args.no_comet:
        return None
    if Experiment is None:
        raise RuntimeError(
            "Comet logging is enabled by default, but comet_ml is not installed. "
            "Install comet-ml or run with --no-comet."
        )
    kwargs: dict[str, Any] = {
        "project_name": args.comet_project,
        "auto_param_logging": True,
        "auto_metric_logging": True,
    }
    if args.comet_workspace:
        kwargs["workspace"] = args.comet_workspace
    experiment = Experiment(**kwargs)
    experiment.set_name(args.experiment_name or "canvas-state-sac")
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    return experiment


def log_canvas_sac_final_metrics(
    *,
    comet_exp,
    metrics: dict[str, float],
    step: int,
) -> None:
    """Log the final Canvas SAC summary metrics when Comet is enabled."""
    if comet_exp is None:
        return
    comet_exp.log_metric("final/reward", metrics["eval/reward"], step=step)
    comet_exp.log_metric("final/ce_gain", metrics["eval/ce_gain"], step=step)
    comet_exp.log_metric("final/miou", metrics["eval/sac_miou"], step=step)


def log_final_full_validation_miou_curve(
    *,
    comet_exp,
    timesteps: list[int],
    miou_values: list[float],
    step: int,
) -> None:
    """Log final mIoU vs timestep as a Comet curve panel."""
    if comet_exp is None:
        return
    if hasattr(comet_exp, "log_curve"):
        comet_exp.log_curve(
            "final_full_validation_miou_by_timestep",
            x=timesteps,
            y=miou_values,
            step=step,
        )
        return
    for timestep, miou in zip(timesteps, miou_values, strict=True):
        comet_exp.log_metric(
            "final_full_validation/miou_by_timestep",
            miou,
            step=timestep,
        )
