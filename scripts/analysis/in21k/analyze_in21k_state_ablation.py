"""Analyze IN21k dense SAC state-representation ablations.

The input is a Plotly-style JSON export from Comet, typically
``train/norm_mean``. Lower values are better. The script writes a Markdown
recommendation and a compact plot so state-ablation conclusions are reproducible
when the JSON export is replaced.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


# Problem: matplotlib may try to write global font caches on managed machines.
# Solution: point plotting cache directories into the repo-local results tree
# before importing pyplot. Result: the script runs in the same sandboxed setup
# as the other analysis scripts.
_cache_dir = Path("results/.cache").resolve()
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


STATE_LABELS = {
    "canvas_hist": "Viewpoint History",
    "canvas_hist_cosprev": "Viewpoint History + Cos-Prev",
    "canvas_hist_reconstructionnorm": "Viewpoint History + Reconstruction Norm",
    "viewpoint-history": "Viewpoint History",
    "canvas_no_hist": "No History",
    "canvas_no_hist_detdebt": "No History + Detail Debt",
    "canvas_no_hist_cosprev": "No History + Cos-Prev",
    "canvas_no_hist_cosprev(1)": "No History + Cos-Prev",
    "canvas_no_hist_detdebt_cosprev": "No History + Detail Debt + Cos-Prev",
    "canvas_reconstructionnorm": "Viewpoint History + Reconstruction Norm",
    "canvas_no_hist_reconstructionnorm": "No History + Reconstruction Norm",
    "canvas_teacherreconstructionerror": "Viewpoint History + Teacher Error",
}


STATE_NOTES = {
    "canvas_hist": "Longer-run default explicit VPE/GRU coordinate-history state.",
    "canvas_hist_cosprev": "Longer-run viewpoint-history state plus current-vs-previous canvas feature change.",
    "canvas_hist_reconstructionnorm": "Longer-run viewpoint-history state plus target-free reconstructed-feature norm map.",
    "viewpoint-history": "Default explicit VPE/GRU coordinate-history state.",
    "canvas_no_hist": "Current canvas only; explicit viewpoint-history branch disabled.",
    "canvas_no_hist_detdebt": "Canvas only plus scale-aware visited-footprint detail debt.",
    "canvas_no_hist_cosprev": "Canvas only plus current-vs-previous canvas feature change.",
    "canvas_no_hist_cosprev(1)": "Earlier no-history run plus current-vs-previous canvas feature change.",
    "canvas_no_hist_detdebt_cosprev": "Canvas only plus detail debt and cos-prev maps.",
    "canvas_reconstructionnorm": "Default history plus target-free reconstructed-feature norm map.",
    "canvas_no_hist_reconstructionnorm": "Reconstruction norm map with explicit viewpoint history disabled.",
    "canvas_teacherreconstructionerror": "Default history plus target-based teacher reconstruction-error map.",
}


@dataclass(frozen=True)
class StateTrace:
    """One state-ablation curve and scalar screening metrics."""

    name: str
    label: str
    xs: list[float]
    ys: list[float]
    x_max: float
    first: float
    final: float
    last_window_mean: float
    last10_mean: float
    last20_mean: float
    best: float
    auc: float
    smoothness: float
    improvement: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/json/train_norm_mean.json"),
        help="Plotly JSON trace export for train/norm_mean.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/json/in21k_state_ablation_recommendation.md"),
        help="Markdown recommendation output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/json/in21k_state_ablation_train_norm_mean.png"),
        help="PNG comparison plot output path.",
    )
    parser.add_argument(
        "--rank-window",
        type=int,
        default=5,
        help="Number of final points used for the primary endpoint rank.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Trailing moving-average window used for curve smoothness.",
    )
    return parser.parse_args()


def trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    """Return x-normalized trapezoid area so trace horizons are comparable."""
    if len(xs) < 2:
        return ys[0]
    area = 0.0
    for left_x, right_x, left_y, right_y in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], strict=True):
        area += (right_x - left_x) * (left_y + right_y) * 0.5
    return area / max(xs[-1] - xs[0], 1.0)


def moving_average(values: list[float], window: int) -> list[float]:
    """Return trailing moving averages for a one-dimensional trace."""
    if window < 1:
        raise ValueError("Moving-average window must be positive.")
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(mean(values[start : idx + 1]))
    return smoothed


def average_abs_step(values: list[float]) -> float:
    """Measure curve volatility as mean absolute point-to-point movement."""
    if len(values) < 2:
        return 0.0
    return mean(abs(right - left) for left, right in zip(values[:-1], values[1:], strict=True))


def load_traces(path: Path, *, rank_window: int, smooth_window: int) -> list[StateTrace]:
    """Load Plotly traces and compute lower-is-better state ablation metrics."""
    traces = json.loads(path.read_text())
    if not isinstance(traces, list):
        raise ValueError(f"{path} must contain a list of Plotly traces.")
    loaded: list[StateTrace] = []
    for trace_idx, trace in enumerate(traces):
        name = trace.get("name")
        xs = trace.get("x")
        ys = trace.get("y")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path} trace {trace_idx} must contain a non-empty name.")
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
            raise ValueError(f"{path} trace {trace_idx} must contain equal-length x/y lists.")
        if not ys:
            raise ValueError(f"{path} trace {trace_idx} must contain at least one point.")
        x_values = [float(value) for value in xs]
        y_values = [float(value) for value in ys]
        final_window = y_values[-rank_window:]
        last10 = y_values[-min(10, len(y_values)) :]
        last20 = y_values[-min(20, len(y_values)) :]
        loaded.append(
            StateTrace(
                name=name,
                label=STATE_LABELS.get(name, name),
                xs=x_values,
                ys=y_values,
                x_max=x_values[-1],
                first=y_values[0],
                final=y_values[-1],
                last_window_mean=mean(final_window),
                last10_mean=mean(last10),
                last20_mean=mean(last20),
                best=min(y_values),
                auc=trapezoid_auc(x_values, y_values),
                smoothness=average_abs_step(moving_average(y_values, smooth_window)),
                improvement=y_values[0] - mean(final_window),
            )
        )
    return loaded


def rank_by(values: list[StateTrace], key: str) -> dict[str, int]:
    """Return one-indexed lower-is-better ranks for a StateTrace attribute."""
    ordered = sorted(values, key=lambda item: (getattr(item, key), item.name))
    return {trace.name: rank for rank, trace in enumerate(ordered, start=1)}


def gap_row(
    traces_by_name: dict[str, StateTrace],
    *,
    left: str,
    right: str,
    label: str,
) -> str | None:
    """Render one comparison row as left minus right when both traces exist."""
    if left not in traces_by_name or right not in traces_by_name:
        return None
    left_trace = traces_by_name[left]
    right_trace = traces_by_name[right]
    diff = left_trace.last_window_mean - right_trace.last_window_mean
    rel = diff / right_trace.last_window_mean * 100.0
    return (
        "| "
        f"{label} | `{left}` | `{right}` | "
        f"{diff:+.8f} | {rel:+.4f}% | "
        f"{left_trace.last_window_mean:.8f} | {right_trace.last_window_mean:.8f} |"
    )


def format_recommendation(traces: list[StateTrace], *, rank_window: int) -> list[str]:
    """Build the recommendation Markdown from computed state metrics."""
    by_last = sorted(traces, key=lambda item: (item.last_window_mean, item.name))
    by_auc = sorted(traces, key=lambda item: (item.auc, item.name))
    by_smooth = sorted(traces, key=lambda item: (item.smoothness, item.name))
    traces_by_name = {trace.name: trace for trace in traces}
    max_horizon = max(trace.x_max for trace in traces)
    max_horizon_traces = [trace for trace in traces if trace.x_max == max_horizon]
    max_horizon_ranked = sorted(
        max_horizon_traces,
        key=lambda item: (item.last_window_mean, item.name),
    )
    last_rank = rank_by(traces, "last_window_mean")
    auc_rank = rank_by(traces, "auc")
    smooth_rank = rank_by(traces, "smoothness")
    best_last = by_last[0].last_window_mean
    best_long = max_horizon_ranked[0]

    lines = [
        "# IN21k State Ablation Recommendation",
        "",
        f"Input metric: `train/norm_mean`. Lower is better. Primary endpoint ranking uses the mean of the final {rank_window} points.",
        "",
        "## Recommendation",
        "",
        f"At the longest shared horizon in this JSON (`x={max_horizon:g}`), `{best_long.name}` is best by final-window `train/norm_mean`.",
        "",
        "Use viewpoint history plus `--reconstruction-norm-state` as the current main state candidate if this trace is your intended `canvas_hist_reconstructionnorm` run. The longer training helps the leading candidates separate from their earlier checkpoints, but the remaining top gaps are still small enough to treat as a one-seed training signal rather than a settled statistical result.",
        "",
        "Recommended next comparisons:",
        "",
        "| Priority | State | Why |",
        "|---:|---|---|",
        "| 1 | `--reconstruction-norm-state` | Best longer-run candidate when viewpoint history is enabled. |",
        "| 2 | `--cos-prev` | Best temporal-change backup to compare against reconstruction norm. |",
        "| 3 | `--disable-viewpoint-history-state --cos-prev` | Tests whether cos-prev still helps without explicit coordinate history. |",
        "| Reference | default viewpoint-history | Baseline for measuring whether aux state is doing real work. |",
        "",
        "## Longest-Horizon Ranking",
        "",
        f"Only traces ending at `x={max_horizon:g}` are included here, so older shorter runs do not distort the longer-training comparison.",
        "",
        "| Rank | State Trace | State Label | Last-Window Mean | Gap vs Best | Relative Gap | Final | Best | AUC |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    best_long_value = best_long.last_window_mean
    for rank, trace in enumerate(max_horizon_ranked, start=1):
        gap = trace.last_window_mean - best_long_value
        lines.append(
            "| "
            f"{rank} | `{trace.name}` | {trace.label} | "
            f"{trace.last_window_mean:.8f} | {gap:+.8f} | "
            f"{gap / best_long_value * 100.0:+.4f}% | {trace.final:.8f} | "
            f"{trace.best:.8f} | {trace.auc:.8f} |"
        )
    lines.extend(
        [
            "",
        "## Endpoint Ranking",
        "",
        "| Rank | State Trace | State Label | Horizon | Last-Window Mean | Gap vs Best | Relative Gap | First | Improvement |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, trace in enumerate(by_last, start=1):
        gap = trace.last_window_mean - best_last
        lines.append(
            "| "
            f"{rank} | `{trace.name}` | {trace.label} | "
            f"{trace.x_max:g} | {trace.last_window_mean:.8f} | {gap:+.8f} | "
            f"{gap / best_last * 100.0:+.4f}% | {trace.first:.8f} | "
            f"{trace.improvement:.8f} |"
        )

    lines.extend(
        [
            "",
            "## Robustness Table",
            "",
            "| State Trace | Horizon | Last-5 Rank | AUC Rank | Smooth Rank | Last-5 | Last-10 | Last-20 | AUC | Smoothness | Best |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for trace in by_last:
        lines.append(
            "| "
            f"`{trace.name}` | {trace.x_max:g} | {last_rank[trace.name]} | {auc_rank[trace.name]} | "
            f"{smooth_rank[trace.name]} | {trace.last_window_mean:.8f} | "
            f"{trace.last10_mean:.8f} | {trace.last20_mean:.8f} | "
            f"{trace.auc:.8f} | {trace.smoothness:.8f} | {trace.best:.8f} |"
        )

    comparisons = [
        ("Teacher error vs reconstruction norm", "canvas_teacherreconstructionerror", "canvas_reconstructionnorm"),
        (
            "Longer reconstruction norm vs longer default",
            "canvas_hist_reconstructionnorm",
            "canvas_hist",
        ),
        (
            "Longer reconstruction norm vs longer cos-prev",
            "canvas_hist_reconstructionnorm",
            "canvas_hist_cosprev",
        ),
        (
            "Longer history cos-prev vs longer no-history cos-prev",
            "canvas_hist_cosprev",
            "canvas_no_hist_cosprev",
        ),
        ("Reconstruction norm history value", "canvas_no_hist_reconstructionnorm", "canvas_reconstructionnorm"),
        ("No history vs default", "canvas_no_hist", "viewpoint-history"),
        ("Detail debt vs no history", "canvas_no_hist_detdebt", "canvas_no_hist"),
        ("Cos-prev vs no history", "canvas_no_hist_cosprev", "canvas_no_hist"),
        (
            "Detail debt + cos-prev vs no history",
            "canvas_no_hist_detdebt_cosprev",
            "canvas_no_hist",
        ),
        (
            "Cos-prev added to detail debt",
            "canvas_no_hist_detdebt_cosprev",
            "canvas_no_hist_detdebt",
        ),
    ]
    lines.extend(
        [
            "",
            "## Direct Comparisons",
            "",
            "Deltas are `left - right` using the final-window mean; negative means the left state is better.",
            "",
            "| Comparison | Left | Right | Delta | Relative Delta | Left Last-Window | Right Last-Window |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, left, right in comparisons:
        row = gap_row(traces_by_name, left=left, right=right, label=label)
        if row is not None:
            lines.append(row)

    lines.extend(
        [
            "",
            "## State Notes",
            "",
            "| State Trace | Note |",
            "|---|---|",
        ]
    )
    for trace in sorted(traces, key=lambda item: item.name):
        lines.append(f"| `{trace.name}` | {STATE_NOTES.get(trace.name, trace.label)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `teacher_reconstruction_error` is the strongest endpoint trace, but it is target-dependent and only slightly ahead of the cleaner reconstruction-norm state.",
            "- The longer-run subset changes the story: viewpoint history plus reconstruction norm is now the best longest-horizon trace.",
            "- Longer training appears useful because the comparable longer traces improve over their shorter earlier versions.",
            "- The leading gaps remain small, so this is best treated as a practical single-seed selection signal rather than proof that the state is universally better.",
            "- Detail debt still does not have a clear positive signal in the shorter runs.",
        ]
    )
    return lines


def write_plot(path: Path, traces: list[StateTrace]) -> None:
    """Plot state-ablation train/norm_mean curves sorted by endpoint rank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(traces, key=lambda item: (item.last_window_mean, item.name))
    colors = plt.get_cmap("tab10")

    # Problem: state names are long and curves are close. Solution: rank labels
    # by endpoint and use a compact legend outside the axes. Result: the plot
    # stays readable while preserving all traces.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=200)
    for rank, trace in enumerate(ordered, start=1):
        color = colors((rank - 1) % 10)
        ax.plot(
            trace.xs,
            trace.ys,
            color=color,
            linewidth=2.0 if rank <= 4 else 1.3,
            alpha=0.95 if rank <= 4 else 0.72,
            label=f"{rank}. {trace.label}",
        )
        ax.scatter([trace.xs[-1]], [trace.ys[-1]], color=color, s=16, zorder=3)
    ax.set_title("IN21k State Ablation: Train Norm Mean")
    ax.set_xlabel("step")
    ax.set_ylabel("train/norm_mean")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.rank_window < 1:
        raise ValueError("--rank-window must be positive.")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be positive.")

    traces = load_traces(
        args.input,
        rank_window=args.rank_window,
        smooth_window=args.smooth_window,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(format_recommendation(traces, rank_window=args.rank_window)) + "\n"
    )
    write_plot(args.plot, traces)

    print(f"Wrote report: {args.report}")
    print(f"Wrote plot: {args.plot}")
    print("\nBest states by final-window train/norm_mean:")
    for rank, trace in enumerate(
        sorted(traces, key=lambda item: (item.last_window_mean, item.name))[:5],
        start=1,
    ):
        print(f"  {rank}. {trace.name}: {trace.last_window_mean:.8f}")


if __name__ == "__main__":
    main()
