"""Plot trained, untrained, and EG-C2F baseline mIoU over timesteps.

The input files are Plotly-style JSON traces from ``results/json``. Source mIoU
values are stored as fractions; this script plots them as percentages.
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

BASELINE_MIOU = {
    0: 0.3958,
    1: 0.4223,
    2: 0.4330,
}


@dataclass(frozen=True)
class SeriesStats:
    """Mean and run-to-run spread for one model at each timestep."""

    timestep: int
    mean_pct: float
    variance_pct2: float
    std_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trained",
        type=Path,
        default=Path("results/json/trained.json"),
        help="JSON file containing trained-model mIoU traces.",
    )
    parser.add_argument(
        "--untrained",
        type=Path,
        default=Path("results/json/untrained.json"),
        help="JSON file containing untrained-model mIoU traces.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/json/miou_trained_vs_untrained.png"),
        help="Output PNG plot path.",
    )
    parser.add_argument(
        "--variance",
        choices=["sample", "population"],
        default="sample",
        help="Variance estimator for the five seed runs.",
    )
    return parser.parse_args()


def load_runs(path: Path) -> dict[int, list[float]]:
    """Load traces and group mIoU fractions by timestep."""
    traces = json.loads(path.read_text())
    grouped: dict[int, list[float]] = {}
    for trace_idx, trace in enumerate(traces):
        xs = trace.get("x")
        ys = trace.get("y")
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
            raise ValueError(f"{path} trace {trace_idx} must contain equal-length x/y lists.")
        for timestep, miou in zip(xs, ys, strict=True):
            grouped.setdefault(int(timestep), []).append(float(miou))
    return grouped


def summarize(grouped: dict[int, list[float]], variance_mode: str) -> list[SeriesStats]:
    """Convert run-level mIoU fractions into percent mean/variance per timestep."""
    stats: list[SeriesStats] = []
    for timestep in sorted(grouped):
        values_pct = [value * 100.0 for value in grouped[timestep]]
        if len(values_pct) < 2:
            spread = 0.0
        elif variance_mode == "population":
            avg = mean(values_pct)
            spread = sum((value - avg) ** 2 for value in values_pct) / len(values_pct)
        else:
            spread = variance(values_pct)
        stats.append(
            SeriesStats(
                timestep=timestep,
                mean_pct=mean(values_pct),
                variance_pct2=spread,
                std_pct=math.sqrt(spread),
            )
        )
    return stats


def write_png_plot(
    path: Path,
    trained: list[SeriesStats],
    untrained: list[SeriesStats],
) -> None:
    """Draw mean curves with low-alpha spread regions and a solid EG-C2F line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Problem: error bars made the three-timestep plot visually busy.
    # Solution: render mean +/- standard deviation as translucent bands.
    # Result: the variance remains visible while the mean curves stay readable.
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    _plot_series(ax, trained, label="Trained", color="#577D33")
    _plot_series(ax, untrained, label="Untrained", color="#467ed7")

    baseline_timesteps = sorted(BASELINE_MIOU)
    baseline_values = [BASELINE_MIOU[timestep] * 100.0 for timestep in baseline_timesteps]
    ax.plot(
        baseline_timesteps,
        baseline_values,
        color="#B75F95",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="EG-C2F",
    )

    all_stats = trained + untrained
    y_values = [item.mean_pct + item.std_pct for item in all_stats]
    y_values += [item.mean_pct - item.std_pct for item in all_stats]
    y_values += baseline_values
    y_min = math.floor((min(y_values) - 0.35) * 2.0) / 2.0
    y_max = math.ceil((max(y_values) + 0.35) * 2.0) / 2.0

    ax.set_xlabel("timestep t")
    ax.set_ylabel("mIoU (%)")
    ax.set_xticks(sorted({item.timestep for item in all_stats} | set(BASELINE_MIOU)))
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_series(axes, stats: list[SeriesStats], *, label: str, color: str) -> None:
    timesteps = [item.timestep for item in stats]
    means = [item.mean_pct for item in stats]
    lower = [item.mean_pct - item.std_pct for item in stats]
    upper = [item.mean_pct + item.std_pct for item in stats]
    axes.fill_between(timesteps, lower, upper, color=color, alpha=0.15, linewidth=0)
    axes.plot(
        timesteps,
        means,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        label=label,
    )


def main() -> None:
    args = parse_args()
    trained = summarize(load_runs(args.trained), args.variance)
    untrained = summarize(load_runs(args.untrained), args.variance)
    write_png_plot(args.plot, trained, untrained)
    print(f"Wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
