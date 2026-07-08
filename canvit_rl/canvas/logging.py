"""Optional Comet logging helpers for Canvas SAC-style training scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
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
    initial_miou_values: list[float] | None = None,
    egc2f_miou_values: list[float] | None = None,
    comparison_output: Path | None = None,
) -> None:
    """Log final mIoU vs timestep and optionally save one multi-series overlay."""
    plot_path: Path | None = None
    plot_figure = None
    comparison_series = [
        ("initialized", initial_miou_values),
        ("EG-C2F", egc2f_miou_values),
    ]
    comparison_series = [
        (label, values) for label, values in comparison_series if values is not None
    ]
    if comparison_output is not None and comparison_series:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "Skipped full-validation mIoU overlay plot; matplotlib is "
                "unavailable."
            )
        else:
            # Problem: separate Comet curve/metric names become separate plots.
            # Solution: build one multi-line figure containing all requested
            # baselines plus final best. Result: Comet receives one comparable
            # plot instead of separate initialized/EG-C2F/final plots.
            comparison_output.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
            for label, values in comparison_series:
                ax.plot(timesteps, values, marker="o", label=label)
            ax.plot(timesteps, miou_values, marker="o", label="final best")
            ax.set_xlabel("timestep")
            ax.set_ylabel("mIoU")
            ax.set_title("Full validation mIoU by timestep")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(comparison_output)
            plot_figure = fig
            plot_path = comparison_output

    if comet_exp is None:
        if plot_figure is not None:
            import matplotlib.pyplot as plt

            plt.close(plot_figure)
        return
    # Log all comparison curves at a stable step so cross-run Comet panels do
    # not shift just because different jobs trained for different update counts.
    curve_step = 0
    if hasattr(comet_exp, "log_curve"):
        # Problem: Comet's native log_curve path is single-series here; replacing
        # it with a figure made the historical curve artifact disappear.
        # Solution: always keep the final-only native curve, and log the
        # multi-line comparison as a separate figure/image artifact.
        # Result: final_full_validation_miou_by_timestep remains available while
        # initialized/EG-C2F/final can still be compared in one plot.
        comet_exp.log_curve(
            "final_full_validation_miou_by_timestep",
            x=timesteps,
            y=miou_values,
            step=curve_step,
        )
        if initial_miou_values is not None:
            # Problem: the multi-line comparison figure is not shown in Comet's
            # native Curves section. Solution: log the initialized model as its
            # own native curve next to the final curve. Result: users can open
            # both full-validation curves from the same Comet Curves section.
            comet_exp.log_curve(
                "initial_full_validation_miou_by_timestep",
                x=timesteps,
                y=initial_miou_values,
                step=curve_step,
            )
    else:
        for timestep, miou in zip(timesteps, miou_values, strict=True):
            comet_exp.log_metric(
                "final_full_validation/miou_by_timestep",
                miou,
                step=timestep,
            )
        if initial_miou_values is not None:
            for timestep, miou in zip(timesteps, initial_miou_values, strict=True):
                comet_exp.log_metric(
                    "initial_full_validation/miou_by_timestep",
                    miou,
                    step=timestep,
                )

    if plot_figure is not None and hasattr(comet_exp, "log_figure"):
        comet_exp.log_figure(
            figure_name="final_full_validation_miou_by_timestep_comparison",
            figure=plot_figure,
            step=curve_step,
        )
    elif plot_path is not None and hasattr(comet_exp, "log_image"):
        comet_exp.log_image(
            str(plot_path),
            name="final_full_validation_miou_by_timestep_comparison",
            step=curve_step,
        )
    if plot_figure is not None:
        import matplotlib.pyplot as plt

        plt.close(plot_figure)
