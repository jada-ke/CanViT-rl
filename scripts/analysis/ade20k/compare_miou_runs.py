"""Plot SAC/PPO mIoU and reward comparisons.

The input files are Plotly-style JSON traces from ``results/json``. Source mIoU
values are stored as fractions; this script plots them as percentages. Reward
trace x values are logged in batches and converted to collected glimpses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, variance


# Problem: Matplotlib/fontconfig default to user-level cache paths that are not
# writable in some sandboxed runs. Solution: point both caches into the
# repo-local ignored results directory before importing pyplot. Result: PNG
# generation works without touching global configuration.
_cache_dir = Path("results/.cache").resolve()
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

@dataclass(frozen=True)
class SeriesStats:
    """Mean and run-to-run spread for one series at each x value."""

    x: int
    mean: float
    variance: float
    std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trained-sac",
        "--trained",
        dest="trained_sac",
        type=Path,
        default=Path("results/json/sac_t4.json"),
        help="JSON file containing trained SAC mIoU traces.",
    )
    parser.add_argument(
        "--ppo",
        type=Path,
        default=Path("results/json/train_ppo.json"),
        help="JSON file containing trained PPO mIoU traces.",
    )
    parser.add_argument(
        "--include-ppo",
        action="store_true",
        help="Include the PPO mIoU traces in the mIoU plot.",
    )
    parser.add_argument(
        "--untrained",
        type=Path,
        default=Path("results/json/untrained_t4.json"),
        help="JSON file containing untrained-model mIoU traces.",
    )
    parser.add_argument(
        "--egc2f",
        type=Path,
        default=Path("results/json/egc2f.json"),
        help="JSON file containing the EG-C2F mIoU baseline trace.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/json/miou_sac_t4_vs_untrained_t4.png"),
        help="Output PNG plot path.",
    )
    parser.add_argument(
        "--variance",
        choices=["sample", "population"],
        default="sample",
        help="Variance estimator for the five seed runs.",
    )
    parser.add_argument(
        "--val-reward-sac",
        type=Path,
        default=Path("results/json/val_reward_sac.json"),
        help="JSON file containing SAC validation reward traces.",
    )
    parser.add_argument(
        "--val-reward-ppo",
        type=Path,
        default=Path("results/json/val_reward_ppo.json"),
        help="JSON file containing PPO validation reward traces.",
    )
    parser.add_argument(
        "--online-reward-sac",
        type=Path,
        default=Path("results/json/online_reward_sac.json"),
        help="JSON file containing SAC online reward traces.",
    )
    parser.add_argument(
        "--online-reward-ppo",
        type=Path,
        default=Path("results/json/online_reward_ppo.json"),
        help="JSON file containing PPO online reward traces.",
    )
    parser.add_argument(
        "--val-reward-plot",
        type=Path,
        default=Path("results/json/val_reward_sac_vs_ppo.png"),
        help="Output PNG plot path for validation reward.",
    )
    parser.add_argument(
        "--online-reward-plot",
        type=Path,
        default=Path("results/json/online_reward_sac_vs_ppo.png"),
        help="Output PNG plot path for online reward.",
    )
    parser.add_argument(
        "--val-reward-batch-plot",
        type=Path,
        default=Path("results/json/val_reward_sac_vs_ppo_batches.png"),
        help="Output PNG plot path for validation reward over batches.",
    )
    parser.add_argument(
        "--online-reward-batch-plot",
        type=Path,
        default=Path("results/json/online_reward_sac_vs_ppo_batches.png"),
        help="Output PNG plot path for online reward over batches.",
    )
    parser.add_argument(
        "--sac-batch-size",
        type=int,
        default=16,
        help="Batch size used to convert SAC reward trace x values to glimpses.",
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=64,
        help="Batch size used to convert PPO reward trace x values to glimpses.",
    )
    parser.add_argument(
        "--reward-min-runs",
        type=int,
        default=2,
        help="Minimum number of runs required to plot a reward x value.",
    )
    return parser.parse_args()


def load_runs(path: Path, *, x_scale: int = 1) -> dict[int, list[float]]:
    """Load traces and group y values by scaled x coordinate."""
    traces = json.loads(path.read_text())
    grouped: dict[int, list[float]] = {}
    for trace_idx, trace in enumerate(traces):
        xs = trace.get("x")
        ys = trace.get("y")
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
            raise ValueError(f"{path} trace {trace_idx} must contain equal-length x/y lists.")
        for x_value, y_value in zip(xs, ys, strict=True):
            grouped.setdefault(int(x_value) * x_scale, []).append(float(y_value))
    return grouped


def load_single_trace(path: Path, *, x_scale: int = 1, y_scale: float = 1.0) -> list[SeriesStats]:
    """Load one Plotly-style baseline trace without run-to-run spread."""
    trace = json.loads(path.read_text())
    if isinstance(trace, list):
        if len(trace) != 1:
            raise ValueError(f"{path} must contain exactly one baseline trace.")
        trace = trace[0]
    xs = trace.get("x")
    ys = trace.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
        raise ValueError(f"{path} must contain equal-length x/y lists.")
    return [
        SeriesStats(
            x=int(x_value) * x_scale,
            mean=float(y_value) * y_scale,
            variance=0.0,
            std=0.0,
        )
        for x_value, y_value in zip(xs, ys, strict=True)
    ]


def summarize(
    grouped: dict[int, list[float]],
    variance_mode: str,
    *,
    y_scale: float = 1.0,
    min_runs: int = 1,
) -> list[SeriesStats]:
    """Summarize run-level values into mean/variance/std per x coordinate."""
    stats: list[SeriesStats] = []
    for x_value in sorted(grouped):
        if len(grouped[x_value]) < min_runs:
            continue
        values = [value * y_scale for value in grouped[x_value]]
        if len(values) < 2:
            spread = 0.0
        elif variance_mode == "population":
            avg = mean(values)
            spread = sum((value - avg) ** 2 for value in values) / len(values)
        else:
            spread = variance(values)
        stats.append(
            SeriesStats(
                x=x_value,
                mean=mean(values),
                variance=spread,
                std=math.sqrt(spread),
            )
        )
    return stats


def write_png_plot(
    path: Path,
    trained_sac: list[SeriesStats],
    ppo: list[SeriesStats] | None,
    untrained: list[SeriesStats],
    egc2f: list[SeriesStats],
) -> None:
    """Draw mIoU mean curves with filled standard-deviation bands."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Problem: the t4 comparison has no PPO run, and error bars made the mIoU
    # chart harder to read than filled variance bands.
    # Solution: make PPO optional and draw SAC/untrained run spread as
    # translucent mean +/- std bands, with EG-C2F as a single baseline curve.
    # Result: the default t4 figure compares trained SAC, untrained, and EG-C2F
    # without implying a nonexistent t4 PPO result.
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    _plot_filled_series(ax, trained_sac, label="SAC", color="#577D33")
    if ppo is not None:
        _plot_filled_series(ax, ppo, label="PPO", color="#D97706")
    _plot_filled_series(ax, untrained, label="Untrained", color="#467ed7")
    _plot_line_series(ax, egc2f, label="EG-C2F", color="#B75F95")

    all_stats = trained_sac + untrained + egc2f
    if ppo is not None:
        all_stats += ppo
    y_values = [item.mean + item.std for item in all_stats]
    y_values += [item.mean - item.std for item in all_stats]
    y_min = math.floor((min(y_values) - 0.35) * 2.0) / 2.0
    y_max = math.ceil((max(y_values) + 0.35) * 2.0) / 2.0

    timesteps = sorted({item.x for item in all_stats})
    ax.set_xlabel("timestep t")
    ax.set_ylabel("mIoU (%)")
    ax.set_xticks(timesteps)
    ax.set_xlim(min(timesteps) - 0.15, max(timesteps) + 0.15)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_line_series(axes, stats: list[SeriesStats], *, label: str, color: str) -> None:
    xs = [item.x for item in stats]
    means = [item.mean for item in stats]
    axes.plot(xs, means, color=color, linewidth=2.2, marker="o", markersize=4.5, label=label)


def write_reward_plot(
    path: Path,
    *,
    sac: list[SeriesStats],
    ppo: list[SeriesStats],
    title: str,
    xlabel: str = "glimpses",
) -> None:
    """Draw reward mean curves with run-to-run spread bands."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Problem: reward traces need both sample-efficiency and raw-training-step
    # views, while a few exported online points exist for only one seed and can
    # look like zero-variance spikes.
    # Solution: summarize either scaled glimpses or raw batches upstream, then
    # draw mean +/- std bands only for x values seen in enough runs.
    # Result: reward plots can show collected-sample efficiency or same-horizon
    # batch progress without single-run spike artifacts.
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    _plot_filled_series(ax, sac, label="SAC", color="#577D33")
    _plot_filled_series(ax, ppo, label="PPO", color="#D97706")

    all_stats = sac + ppo
    y_values = [item.mean + item.std for item in all_stats]
    y_values += [item.mean - item.std for item in all_stats]
    y_pad = max((max(y_values) - min(y_values)) * 0.08, 1e-4)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("reward")
    ax.set_ylim(min(y_values) - y_pad, max(y_values) + y_pad)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_filled_series(axes, stats: list[SeriesStats], *, label: str, color: str) -> None:
    xs = [item.x for item in stats]
    means = [item.mean for item in stats]
    lower = [item.mean - item.std for item in stats]
    upper = [item.mean + item.std for item in stats]
    axes.fill_between(xs, lower, upper, color=color, alpha=0.16, linewidth=0)
    axes.plot(
        xs,
        means,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label=label,
    )


def trim_to_shared_x(
    sac: list[SeriesStats],
    ppo: list[SeriesStats],
) -> tuple[list[SeriesStats], list[SeriesStats]]:
    """Clip both series to the shorter observed x horizon."""
    if not sac or not ppo:
        return sac, ppo
    shared_x_max = min(sac[-1].x, ppo[-1].x)
    return (
        [item for item in sac if item.x <= shared_x_max],
        [item for item in ppo if item.x <= shared_x_max],
    )


def main() -> None:
    args = parse_args()
    trained_sac = summarize(load_runs(args.trained_sac), args.variance, y_scale=100.0)
    ppo = (
        summarize(load_runs(args.ppo), args.variance, y_scale=100.0)
        if args.include_ppo
        else None
    )
    untrained = summarize(load_runs(args.untrained), args.variance, y_scale=100.0)
    egc2f = load_single_trace(args.egc2f, y_scale=100.0)
    write_png_plot(args.plot, trained_sac, ppo, untrained, egc2f)
    print(f"Wrote plot: {args.plot}")

    val_reward_sac = summarize(
        load_runs(args.val_reward_sac, x_scale=args.sac_batch_size),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    val_reward_ppo = summarize(
        load_runs(args.val_reward_ppo, x_scale=args.ppo_batch_size),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    write_reward_plot(
        args.val_reward_plot,
        sac=val_reward_sac,
        ppo=val_reward_ppo,
        title="Validation Reward",
        xlabel="glimpses",
    )
    print(f"Wrote plot: {args.val_reward_plot}")

    val_reward_sac_batches = summarize(
        load_runs(args.val_reward_sac),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    val_reward_ppo_batches = summarize(
        load_runs(args.val_reward_ppo),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    # Problem: raw batch-axis reward plots looked uneven when one algorithm's
    # export continued beyond the other's final batch.
    # Solution: clip both batch-axis series to the shorter observed batch
    # horizon before plotting.
    # Result: SAC/PPO batch plots compare the same training-step range while
    # glimpse-axis plots still show sample-efficiency exposure.
    val_reward_sac_batches, val_reward_ppo_batches = trim_to_shared_x(
        val_reward_sac_batches,
        val_reward_ppo_batches,
    )
    write_reward_plot(
        args.val_reward_batch_plot,
        sac=val_reward_sac_batches,
        ppo=val_reward_ppo_batches,
        title="Validation Reward",
        xlabel="batches",
    )
    print(f"Wrote plot: {args.val_reward_batch_plot}")

    online_reward_sac = summarize(
        load_runs(args.online_reward_sac, x_scale=args.sac_batch_size),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    online_reward_ppo = summarize(
        load_runs(args.online_reward_ppo, x_scale=args.ppo_batch_size),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    write_reward_plot(
        args.online_reward_plot,
        sac=online_reward_sac,
        ppo=online_reward_ppo,
        title="Online Reward",
        xlabel="glimpses",
    )
    print(f"Wrote plot: {args.online_reward_plot}")

    online_reward_sac_batches = summarize(
        load_runs(args.online_reward_sac),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    online_reward_ppo_batches = summarize(
        load_runs(args.online_reward_ppo),
        args.variance,
        min_runs=args.reward_min_runs,
    )
    online_reward_sac_batches, online_reward_ppo_batches = trim_to_shared_x(
        online_reward_sac_batches,
        online_reward_ppo_batches,
    )
    write_reward_plot(
        args.online_reward_batch_plot,
        sac=online_reward_sac_batches,
        ppo=online_reward_ppo_batches,
        title="Online Reward",
        xlabel="batches",
    )
    print(f"Wrote plot: {args.online_reward_batch_plot}")


if __name__ == "__main__":
    main()
