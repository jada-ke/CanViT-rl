"""Plot and rank IN21k reward-mode loss curves.

The input files are Plotly-style JSON traces exported from Comet. Each trace is
one reward-mode run. Lower loss is better; screening rankings prefer the mean
of the final evaluation window over a single last point.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


# Problem: Matplotlib/fontconfig can try to write user-level caches in sandboxed
# runs. Solution: route plotting caches into the repo-local results directory
# before importing pyplot. Result: the script stays self-contained.
_cache_dir = Path("results/.cache").resolve()
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


@dataclass(frozen=True)
class LossTrace:
    """One reward-mode loss curve plus screening metrics."""

    mode: str
    xs: list[int]
    ys: list[float]
    final_x: int
    final_loss: float
    best_loss: float
    last_window_mean: float
    last_window_std: float
    last_window_range: float
    auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--norm",
        type=Path,
        default=Path("results/json/eval_final_loss_norm VS step_chart_data.json"),
        help="Plotly JSON file for eval/final_loss_norm traces.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/json/eval_final_loss_raw VS step_chart_data.json"),
        help="Plotly JSON file for eval/final_loss_raw traces.",
    )
    parser.add_argument(
        "--norm-plot",
        type=Path,
        default=Path("results/json/in21k_reward_modes_final_loss_norm.png"),
        help="Output PNG path for normalized final-loss curves.",
    )
    parser.add_argument(
        "--raw-plot",
        type=Path,
        default=Path("results/json/in21k_reward_modes_final_loss_raw.png"),
        help="Output PNG path for raw final-loss curves.",
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        default=Path("results/json/in21k_reward_mode_loss_rankings.md"),
        help="Output Markdown ranking table path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of best reward modes to emphasize in each plot.",
    )
    parser.add_argument(
        "--rank-window",
        type=int,
        default=5,
        help="Number of final evaluation points to average for screening rank.",
    )
    return parser.parse_args()


def load_traces(path: Path, *, rank_window: int) -> list[LossTrace]:
    """Load Plotly traces and rank by final-window mean loss."""
    traces = json.loads(path.read_text())
    loaded: list[LossTrace] = []
    for trace_idx, trace in enumerate(traces):
        xs = trace.get("x")
        ys = trace.get("y")
        name = trace.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path} trace {trace_idx} must contain a non-empty name.")
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
            raise ValueError(f"{path} trace {trace_idx} must contain equal-length x/y lists.")
        if not xs:
            raise ValueError(f"{path} trace {trace_idx} must contain at least one point.")
        x_values = [int(value) for value in xs]
        y_values = [float(value) for value in ys]
        final_window = y_values[-rank_window:]
        loaded.append(
            LossTrace(
                mode=name,
                xs=x_values,
                ys=y_values,
                final_x=x_values[-1],
                final_loss=y_values[-1],
                best_loss=min(y_values),
                last_window_mean=mean(final_window),
                last_window_std=pstdev(final_window),
                last_window_range=max(final_window) - min(final_window),
                auc=trapezoid_auc(x_values, y_values),
            )
        )
    return sorted(
        loaded,
        key=lambda item: (
            item.last_window_mean,
            item.last_window_std,
            item.auc,
            item.mode,
        ),
    )


def trapezoid_auc(xs: list[int], ys: list[float]) -> float:
    """Return x-normalized trapezoid area so shorter/longer traces are comparable."""
    if len(xs) < 2:
        return ys[0]
    area = 0.0
    for left_x, right_x, left_y, right_y in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], strict=True):
        area += (right_x - left_x) * (left_y + right_y) * 0.5
    return area / max(xs[-1] - xs[0], 1)


def write_loss_plot(
    path: Path,
    traces: list[LossTrace],
    *,
    ylabel: str,
    title: str,
    top_k: int,
) -> None:
    """Plot all reward modes while highlighting the best screening curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    top_modes = {trace.mode for trace in traces[:top_k]}
    colors = plt.get_cmap("tab10")

    # Problem: nine reward-mode curves overlap enough that a full-color plot is
    # hard to rank visually. Solution: draw non-top modes in light gray and
    # emphasize the best final-window-mean modes with labels. Result: the plot
    # stays readable while preserving every curve.
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=200)
    for trace in reversed(traces):
        if trace.mode not in top_modes:
            ax.plot(trace.xs, trace.ys, color="#c7c7c7", linewidth=1.1, alpha=0.58)

    for rank, trace in enumerate(traces[:top_k], start=1):
        ax.plot(
            trace.xs,
            trace.ys,
            color=colors((rank - 1) % 10),
            linewidth=2.2,
            label=f"{rank}. {trace.mode}",
        )
        ax.scatter(
            [trace.final_x],
            [trace.final_loss],
            color=colors((rank - 1) % 10),
            s=18,
            zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def format_ranking_section(title: str, traces: list[LossTrace]) -> list[str]:
    """Render one Markdown ranking table."""
    lines = [
        f"## {title}",
        "",
        "| Rank | Reward Mode | Last-Window Mean | Last-Window Std | Final Loss | Best Loss | AUC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, trace in enumerate(traces, start=1):
        lines.append(
            "| "
            f"{rank} | `{trace.mode}` | {trace.last_window_mean:.8f} | "
            f"{trace.last_window_std:.8f} | {trace.final_loss:.8f} | "
            f"{trace.best_loss:.8f} | {trace.auc:.8f} |"
        )
    return lines


def ranks_by_mode(traces: list[LossTrace]) -> dict[str, int]:
    """Map reward mode to one-indexed rank."""
    return {trace.mode: rank for rank, trace in enumerate(traces, start=1)}


def format_consensus_section(norm: list[LossTrace], raw: list[LossTrace]) -> list[str]:
    """Rank modes by agreement across normalized and raw loss spaces."""
    norm_rank = ranks_by_mode(norm)
    raw_rank = ranks_by_mode(raw)
    norm_by_mode = {trace.mode: trace for trace in norm}
    raw_by_mode = {trace.mode: trace for trace in raw}
    rows = []
    for mode in sorted(set(norm_rank) & set(raw_rank)):
        rows.append(
            (
                norm_rank[mode] + raw_rank[mode],
                max(norm_rank[mode], raw_rank[mode]),
                mode,
                norm_rank[mode],
                raw_rank[mode],
                norm_by_mode[mode].last_window_mean,
                raw_by_mode[mode].last_window_mean,
                norm_by_mode[mode].last_window_std,
                raw_by_mode[mode].last_window_std,
            )
        )
    rows.sort()
    lines = [
        "## Consensus Screening Rank",
        "",
        "Ranks sum the normalized-loss and raw-loss last-window-mean ranks; lower is better.",
        "",
        "| Rank | Reward Mode | Rank Sum | Worst Rank | Norm Rank | Raw Rank | Norm Last-Window Mean | Raw Last-Window Mean | Norm Last-Window Std | Raw Last-Window Std |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        rank_sum, worst_rank, mode, norm_mode_rank, raw_mode_rank, norm_mean, raw_mean, norm_std, raw_std = row
        lines.append(
            "| "
            f"{rank} | `{mode}` | {rank_sum} | {worst_rank} | "
            f"{norm_mode_rank} | {raw_mode_rank} | {norm_mean:.8f} | "
            f"{raw_mean:.8f} | {norm_std:.8f} | {raw_std:.8f} |"
        )
    return lines


def write_rankings(path: Path, *, norm: list[LossTrace], raw: list[LossTrace]) -> None:
    """Write lower-is-better screening rankings for both loss spaces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IN21k Reward Mode Loss Rankings",
        "",
        "Lower is better. Primary rankings use the mean of the final evaluation window, not a single final point.",
        "",
    ]
    lines.extend(format_consensus_section(norm, raw))
    lines.extend(["", ""])
    lines.extend(format_ranking_section("Normalized Final Loss", norm))
    lines.extend(["", ""])
    lines.extend(format_ranking_section("Raw Final Loss", raw))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    if args.rank_window < 1:
        raise ValueError("--rank-window must be positive.")

    norm_traces = load_traces(args.norm, rank_window=args.rank_window)
    raw_traces = load_traces(args.raw, rank_window=args.rank_window)

    write_loss_plot(
        args.norm_plot,
        norm_traces,
        ylabel="eval/final_loss_norm",
        title="IN21k Reward Modes: Normalized Final Loss",
        top_k=args.top_k,
    )
    print(f"Wrote plot: {args.norm_plot}")

    write_loss_plot(
        args.raw_plot,
        raw_traces,
        ylabel="eval/final_loss_raw",
        title="IN21k Reward Modes: Raw Final Loss",
        top_k=args.top_k,
    )
    print(f"Wrote plot: {args.raw_plot}")

    write_rankings(args.ranking, norm=norm_traces, raw=raw_traces)
    print(f"Wrote ranking: {args.ranking}")

    print(f"\nBest normalized loss modes by last-{args.rank_window} mean:")
    for rank, trace in enumerate(norm_traces[: args.top_k], start=1):
        print(
            f"  {rank}. {trace.mode}: "
            f"mean={trace.last_window_mean:.8f} std={trace.last_window_std:.8f}"
        )

    print(f"\nBest raw loss modes by last-{args.rank_window} mean:")
    for rank, trace in enumerate(raw_traces[: args.top_k], start=1):
        print(
            f"  {rank}. {trace.mode}: "
            f"mean={trace.last_window_mean:.8f} std={trace.last_window_std:.8f}"
        )


if __name__ == "__main__":
    main()
