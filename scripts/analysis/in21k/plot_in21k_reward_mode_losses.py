"""Plot and rank IN21k reward-mode loss curves.

The input files are Plotly-style JSON traces exported from Comet. Each trace is
one reward-mode run. Lower loss is better; screening rankings prefer the mean
of the final evaluation window over a single last point.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    seed: int | None
    xs: list[int]
    ys: list[float]
    final_x: int
    final_loss: float
    best_loss: float
    last_window_mean: float
    last_window_std: float
    last_window_range: float
    auc: float


@dataclass(frozen=True)
class ModeSummary:
    """Seed-aggregated metrics for one reward formulation."""

    mode: str
    xs: list[int]
    mean_ys: list[float]
    std_ys: list[float]
    seed_count: int
    last_window_mean: float
    last_window_std: float
    final_mean: float
    final_std: float
    best_mean: float
    auc_mean: float
    auc_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--norm",
        type=Path,
        default=Path("results/json/eval_final_loss_norm.json"),
        help="Plotly JSON file for eval/final_loss_norm traces.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/json/eval_final_loss_raw.json"),
        help="Plotly JSON file for eval/final_loss_raw traces.",
    )
    parser.add_argument(
        "--eval-norm-mean",
        type=Path,
        default=Path("results/json/eval_norm_mean.json"),
        help="Plotly JSON file for eval/norm_mean traces.",
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
        "--eval-norm-mean-plot",
        type=Path,
        default=Path("results/json/in21k_reward_modes_eval_norm_mean.png"),
        help="Output PNG path for pretraining-objective normalized mean curves.",
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
    parser.add_argument(
        "--robust-windows",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20],
        help="Final evaluation windows to compare in the robustness table.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window used for the smoothness/volatility rank.",
    )
    return parser.parse_args()


def reward_mode_from_trace_name(name: str) -> str:
    """Strip run-size and seed suffixes from a Comet trace name."""
    # Problem: seed exports include names like ``mode_test_512_s42``, which
    # would be ranked as separate modes. Solution: normalize known suffixes
    # back to just the reward formulation. Result: metrics aggregate across
    # seeds 42/43/44 for each candidate.
    normalized = re.sub(r"_test_\d+_s\d+$", "", name)
    normalized = re.sub(r"_s\d+$", "", normalized)
    return normalized


def seed_from_trace_name(name: str) -> int | None:
    """Return the Comet seed suffix when trace names include one."""
    match = re.search(r"_s(\d+)$", name)
    if match is None:
        return None
    return int(match.group(1))


def load_traces(path: Path, *, rank_window: int) -> list[LossTrace]:
    """Load Plotly traces and rank individual runs by final-window mean loss."""
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
                mode=reward_mode_from_trace_name(name),
                seed=seed_from_trace_name(name),
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


def summarize_traces(traces: list[LossTrace]) -> list[ModeSummary]:
    """Aggregate loaded traces by reward mode across seeds and rank lower-is-better."""
    by_mode: dict[str, list[LossTrace]] = {}
    for trace in traces:
        by_mode.setdefault(trace.mode, []).append(trace)

    summaries: list[ModeSummary] = []
    for mode, traces in by_mode.items():
        grouped_xs = sorted(set.intersection(*(set(trace.xs) for trace in traces)))
        if not grouped_xs:
            raise ValueError(f"No shared x values found for mode {mode!r}.")
        values_by_x: dict[int, list[float]] = {x_value: [] for x_value in grouped_xs}
        for trace in traces:
            y_by_x = dict(zip(trace.xs, trace.ys, strict=True))
            for x_value in grouped_xs:
                values_by_x[x_value].append(y_by_x[x_value])

        mean_ys = [mean(values_by_x[x_value]) for x_value in grouped_xs]
        std_ys = [pstdev(values_by_x[x_value]) for x_value in grouped_xs]
        last_window_values = [trace.last_window_mean for trace in traces]
        final_values = [trace.final_loss for trace in traces]
        best_values = [trace.best_loss for trace in traces]
        auc_values = [trace.auc for trace in traces]
        summaries.append(
            ModeSummary(
                mode=mode,
                xs=grouped_xs,
                mean_ys=mean_ys,
                std_ys=std_ys,
                seed_count=len(traces),
                last_window_mean=mean(last_window_values),
                last_window_std=pstdev(last_window_values),
                final_mean=mean(final_values),
                final_std=pstdev(final_values),
                best_mean=mean(best_values),
                auc_mean=mean(auc_values),
                auc_std=pstdev(auc_values),
            )
        )
    return sorted(
        summaries,
        key=lambda item: (
            item.last_window_mean,
            item.last_window_std,
            item.auc_mean,
            item.mode,
        ),
    )


def load_mode_summaries(path: Path, *, rank_window: int) -> list[ModeSummary]:
    """Load traces, aggregate by reward mode across seeds, and rank lower-is-better."""
    return summarize_traces(load_traces(path, rank_window=rank_window))


def trapezoid_auc(xs: list[int], ys: list[float]) -> float:
    """Return x-normalized trapezoid area so shorter/longer traces are comparable."""
    if len(xs) < 2:
        return ys[0]
    area = 0.0
    for left_x, right_x, left_y, right_y in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], strict=True):
        area += (right_x - left_x) * (left_y + right_y) * 0.5
    return area / max(xs[-1] - xs[0], 1)


def moving_average(values: list[float], window: int) -> list[float]:
    """Return a simple trailing moving average for curve smoothness checks."""
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


def rank_values(scores: dict[str, float]) -> dict[str, int]:
    """Rank lower-is-better scalar scores with deterministic mode-name tie breaks."""
    return {
        mode: rank
        for rank, (mode, _score) in enumerate(sorted(scores.items(), key=lambda item: (item[1], item[0])), start=1)
    }


def final_window_scores(traces: list[LossTrace], window: int) -> dict[str, float]:
    """Average per-seed final-window means by mode for one candidate window."""
    by_mode: dict[str, list[float]] = {}
    for trace in traces:
        final_values = trace.ys[-window:]
        by_mode.setdefault(trace.mode, []).append(mean(final_values))
    return {mode: mean(values) for mode, values in by_mode.items()}


def smoothness_scores(traces: list[LossTrace], *, smooth_window: int) -> dict[str, float]:
    """Average moving-average volatility by mode; lower means smoother."""
    by_mode: dict[str, list[float]] = {}
    for trace in traces:
        smoothed = moving_average(trace.ys, smooth_window)
        by_mode.setdefault(trace.mode, []).append(average_abs_step(smoothed))
    return {mode: mean(values) for mode, values in by_mode.items()}


def write_loss_plot(
    path: Path,
    summaries: list[ModeSummary],
    *,
    ylabel: str,
    title: str,
    top_k: int,
) -> None:
    """Plot all reward modes while highlighting the best screening curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    top_modes = {summary.mode for summary in summaries[:top_k]}
    colors = plt.get_cmap("tab10")

    # Problem: multi-seed reward-mode curves overlap enough that plotting every
    # seed separately hides the actual comparison. Solution: aggregate by reward
    # mode and draw mean +/- seed std bands, highlighting the best final-window
    # means. Result: the plots show both central tendency and seed sensitivity.
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=200)
    for summary in reversed(summaries):
        if summary.mode not in top_modes:
            lower = [
                mean_value - std_value
                for mean_value, std_value in zip(summary.mean_ys, summary.std_ys, strict=True)
            ]
            upper = [
                mean_value + std_value
                for mean_value, std_value in zip(summary.mean_ys, summary.std_ys, strict=True)
            ]
            ax.fill_between(summary.xs, lower, upper, color="#c7c7c7", alpha=0.11, linewidth=0)
            ax.plot(summary.xs, summary.mean_ys, color="#a8a8a8", linewidth=1.0, alpha=0.68)

    for rank, summary in enumerate(summaries[:top_k], start=1):
        color = colors((rank - 1) % 10)
        lower = [
            mean_value - std_value
            for mean_value, std_value in zip(summary.mean_ys, summary.std_ys, strict=True)
        ]
        upper = [
            mean_value + std_value
            for mean_value, std_value in zip(summary.mean_ys, summary.std_ys, strict=True)
        ]
        ax.fill_between(summary.xs, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            summary.xs,
            summary.mean_ys,
            color=colors((rank - 1) % 10),
            linewidth=2.2,
            label=f"{rank}. {summary.mode}",
        )
        ax.scatter(
            [summary.xs[-1]],
            [summary.final_mean],
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


def format_ranking_section(title: str, summaries: list[ModeSummary]) -> list[str]:
    """Render one Markdown ranking table."""
    lines = [
        f"## {title}",
        "",
        "| Rank | Reward Mode | Seeds | Last-Window Mean | Seed Std | Final Mean | Final Std | Best Mean | AUC Mean | AUC Std |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, summary in enumerate(summaries, start=1):
        lines.append(
            "| "
            f"{rank} | `{summary.mode}` | {summary.seed_count} | "
            f"{summary.last_window_mean:.8f} | {summary.last_window_std:.8f} | "
            f"{summary.final_mean:.8f} | {summary.final_std:.8f} | "
            f"{summary.best_mean:.8f} | {summary.auc_mean:.8f} | {summary.auc_std:.8f} |"
        )
    return lines


def ranks_by_mode(summaries: list[ModeSummary]) -> dict[str, int]:
    """Map reward mode to one-indexed rank."""
    return {summary.mode: rank for rank, summary in enumerate(summaries, start=1)}


def format_consensus_section(
    *,
    norm: list[ModeSummary],
    raw: list[ModeSummary],
    eval_norm_mean: list[ModeSummary],
) -> list[str]:
    """Rank modes by agreement across all selected loss spaces."""
    rows = consensus_rows(norm=norm, raw=raw, eval_norm_mean=eval_norm_mean)
    lines = [
        "## Consensus Screening Rank",
        "",
        "Ranks sum the final-loss-norm, final-loss-raw, and eval-norm-mean last-window ranks; lower is better.",
        "",
        "| Rank | Reward Mode | Rank Sum | Worst Rank | Final Norm Rank | Final Raw Rank | Eval Norm Mean Rank | Final Norm Mean | Final Raw Mean | Eval Norm Mean | Final Norm Seed Std | Final Raw Seed Std | Eval Norm Seed Std |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        (
            rank_sum,
            worst_rank,
            mode,
            norm_mode_rank,
            raw_mode_rank,
            eval_norm_mode_rank,
            norm_mean,
            raw_mean,
            eval_norm_mean_value,
            norm_std,
            raw_std,
            eval_norm_std,
        ) = row
        lines.append(
            "| "
            f"{rank} | `{mode}` | {rank_sum} | {worst_rank} | "
            f"{norm_mode_rank} | {raw_mode_rank} | {eval_norm_mode_rank} | "
            f"{norm_mean:.8f} | {raw_mean:.8f} | {eval_norm_mean_value:.8f} | "
            f"{norm_std:.8f} | {raw_std:.8f} | {eval_norm_std:.8f} |"
        )
    return lines


def consensus_rows(
    *,
    norm: list[ModeSummary],
    raw: list[ModeSummary],
    eval_norm_mean: list[ModeSummary],
) -> list[tuple[int, int, str, int, int, int, float, float, float, float, float, float]]:
    """Return sortable consensus rows shared by the table and paired analysis."""
    norm_rank = ranks_by_mode(norm)
    raw_rank = ranks_by_mode(raw)
    eval_norm_rank = ranks_by_mode(eval_norm_mean)
    norm_by_mode = {summary.mode: summary for summary in norm}
    raw_by_mode = {summary.mode: summary for summary in raw}
    eval_norm_by_mode = {summary.mode: summary for summary in eval_norm_mean}
    rows: list[tuple[int, int, str, int, int, int, float, float, float, float, float, float]] = []
    for mode in sorted(set(norm_rank) & set(raw_rank) & set(eval_norm_rank)):
        rows.append(
            (
                norm_rank[mode] + raw_rank[mode] + eval_norm_rank[mode],
                max(norm_rank[mode], raw_rank[mode], eval_norm_rank[mode]),
                mode,
                norm_rank[mode],
                raw_rank[mode],
                eval_norm_rank[mode],
                norm_by_mode[mode].last_window_mean,
                raw_by_mode[mode].last_window_mean,
                eval_norm_by_mode[mode].last_window_mean,
                norm_by_mode[mode].last_window_std,
                raw_by_mode[mode].last_window_std,
                eval_norm_by_mode[mode].last_window_std,
            )
        )
    rows.sort()
    return rows


def format_paired_difference_section(
    *,
    norm_traces: list[LossTrace],
    raw_traces: list[LossTrace],
    eval_norm_mean_traces: list[LossTrace],
    norm: list[ModeSummary],
    raw: list[ModeSummary],
    eval_norm_mean: list[ModeSummary],
) -> list[str]:
    """Render paired seed deltas for the top consensus modes."""
    # Problem: aggregate means can overstate tiny differences when each reward
    # mode was run on the same seeds. Solution: compare the top mode against
    # the next two consensus modes seed-by-seed using the same final-window
    # score used for ranking. Result: the table shows whether the winner is
    # consistently better or just slightly ahead after averaging.
    ordered_modes = [row[2] for row in consensus_rows(norm=norm, raw=raw, eval_norm_mean=eval_norm_mean)]
    if len(ordered_modes) < 3:
        return []

    baseline = ordered_modes[0]
    comparisons = ordered_modes[1:3]
    metric_traces = [
        ("Normalized Final Loss", norm_traces),
        ("Raw Final Loss", raw_traces),
        ("Pretraining-Objective Norm Mean", eval_norm_mean_traces),
    ]
    lines = [
        "## Paired Top-3 Seed Differences",
        "",
        f"Baseline is `{baseline}`. Deltas are `baseline - comparison` using each run's final-window mean; negative means the baseline was lower/better on that same seed.",
        "",
        "| Metric | Comparison | Shared Seeds | Baseline Wins | Mean Delta | Delta Std | Per-Seed Deltas |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for metric_name, traces in metric_traces:
        values = {
            (trace.mode, trace.seed): trace.last_window_mean
            for trace in traces
            if trace.seed is not None
        }
        for comparison in comparisons:
            shared_seeds = sorted(
                seed
                for mode, seed in values
                if mode == baseline and (comparison, seed) in values
            )
            deltas = [values[(baseline, seed)] - values[(comparison, seed)] for seed in shared_seeds]
            if deltas:
                seed_text = ", ".join(str(seed) for seed in shared_seeds)
                delta_text = ", ".join(
                    f"s{seed}: {delta:+.8f}" for seed, delta in zip(shared_seeds, deltas, strict=True)
                )
                win_count = sum(delta < 0.0 for delta in deltas)
                lines.append(
                    "| "
                    f"{metric_name} | `{comparison}` | {seed_text} | "
                    f"{win_count}/{len(deltas)} | {mean(deltas):+.8f} | {pstdev(deltas):.8f} | "
                    f"{delta_text} |"
                )
            else:
                lines.append(
                    "| "
                    f"{metric_name} | `{comparison}` | 0 | 0/0 | n/a | n/a | n/a |"
                )
    return lines


def format_selected_reward_section() -> list[str]:
    """Render the final reward-choice conclusion at the top of the report."""
    # Problem: the ranking tables are regenerated as new JSON exports arrive,
    # but the experimental conclusion lives in the discussion unless we write it
    # into the artifact. Solution: emit a concise selected-reward section before
    # the detailed tables. Result: the Markdown report preserves the final
    # decision and the reason for choosing it.
    return [
        "## Selected Reward Formulation",
        "",
        "Final decision: use `norm_loss_reduction` as the default dense IN21k reward formulation.",
        "",
        "This mode is selected because training uses `gamma=0`, so the myopic objective should reward the glimpse that gives the largest fractional improvement from the current canvas state:",
        "",
        "```text",
        "r_t = (L_{t-1} - L_t) / max(L_{t-1}, reward_eps)",
        "```",
        "",
        "The epsilon-regularized variant `norm_loss_eps_reduction` is still useful as a stability ablation, especially `--reward-reduction-eps 0.005`, but the current paired full-seed sweep did not beat exact `norm_loss_reduction` on endpoint metrics. Treat incomplete-seed rows in the global ranking as reference only; the main decision is based on same-seed comparisons across seeds 42/43/44.",
        "",
    ]


def format_robustness_section(
    *,
    norm_traces: list[LossTrace],
    raw_traces: list[LossTrace],
    eval_norm_mean_traces: list[LossTrace],
    norm: list[ModeSummary],
    raw: list[ModeSummary],
    eval_norm_mean: list[ModeSummary],
    robust_windows: list[int],
    smooth_window: int,
) -> list[str]:
    """Render rank sensitivity and smoothness checks for the consensus modes."""
    # Problem: the top reward modes are closer than seed noise, so a single
    # last-5 endpoint rank is too brittle. Solution: report rank sensitivity
    # across several final windows, whole-curve AUC, and moving-average
    # volatility. Result: near-ties are visible before choosing a scale-up mode.
    ordered_modes = [row[2] for row in consensus_rows(norm=norm, raw=raw, eval_norm_mean=eval_norm_mean)]
    metric_inputs = [
        ("Normalized Final Loss", norm_traces, norm),
        ("Raw Final Loss", raw_traces, raw),
        ("Pretraining-Objective Norm Mean", eval_norm_mean_traces, eval_norm_mean),
    ]
    window_labels = [f"Last-{window} Rank" for window in robust_windows]
    lines = [
        "## Robustness Checks",
        "",
        f"Window ranks use the mean of each mode's final N eval points across seeds. Smoothness rank uses mean absolute point-to-point movement after a trailing moving average with window {smooth_window}; lower is smoother.",
        "",
        "| Metric | Reward Mode | "
        + " | ".join(window_labels)
        + " | AUC Rank | Smooth Rank | Smoothness |",
        "|---|---|"
        + "|".join("---:" for _ in window_labels)
        + "|---:|---:|---:|",
    ]
    for metric_name, traces, summaries in metric_inputs:
        window_ranks = {
            window: rank_values(final_window_scores(traces, window))
            for window in robust_windows
        }
        auc_ranks = rank_values({summary.mode: summary.auc_mean for summary in summaries})
        smooth_scores = smoothness_scores(traces, smooth_window=smooth_window)
        smooth_ranks = rank_values(smooth_scores)
        for mode in ordered_modes:
            if mode not in smooth_scores:
                continue
            line_values = [
                str(window_ranks[window][mode])
                for window in robust_windows
            ]
            lines.append(
                "| "
                f"{metric_name} | `{mode}` | "
                + " | ".join(line_values)
                + f" | {auc_ranks[mode]} | {smooth_ranks[mode]} | {smooth_scores[mode]:.10f} |"
            )
    return lines


def write_rankings(
    path: Path,
    *,
    norm_traces: list[LossTrace],
    raw_traces: list[LossTrace],
    eval_norm_mean_traces: list[LossTrace],
    norm: list[ModeSummary],
    raw: list[ModeSummary],
    eval_norm_mean: list[ModeSummary],
    robust_windows: list[int],
    smooth_window: int,
) -> None:
    """Write lower-is-better screening rankings for both loss spaces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IN21k Reward Mode Loss Rankings",
        "",
        "Lower is better. Primary rankings use the mean of the final evaluation window, not a single final point.",
        "",
    ]
    lines.extend(format_selected_reward_section())
    lines.extend([""])
    lines.extend(format_consensus_section(norm=norm, raw=raw, eval_norm_mean=eval_norm_mean))
    lines.extend(["", ""])
    lines.extend(
        format_robustness_section(
            norm_traces=norm_traces,
            raw_traces=raw_traces,
            eval_norm_mean_traces=eval_norm_mean_traces,
            norm=norm,
            raw=raw,
            eval_norm_mean=eval_norm_mean,
            robust_windows=robust_windows,
            smooth_window=smooth_window,
        )
    )
    lines.extend(["", ""])
    lines.extend(
        format_paired_difference_section(
            norm_traces=norm_traces,
            raw_traces=raw_traces,
            eval_norm_mean_traces=eval_norm_mean_traces,
            norm=norm,
            raw=raw,
            eval_norm_mean=eval_norm_mean,
        )
    )
    lines.extend(["", ""])
    lines.extend(format_ranking_section("Normalized Final Loss", norm))
    lines.extend(["", ""])
    lines.extend(format_ranking_section("Raw Final Loss", raw))
    lines.extend(["", ""])
    lines.extend(format_ranking_section("Pretraining-Objective Norm Mean", eval_norm_mean))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    if args.rank_window < 1:
        raise ValueError("--rank-window must be positive.")
    if any(window < 1 for window in args.robust_windows):
        raise ValueError("--robust-windows values must be positive.")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be positive.")

    norm_runs = load_traces(args.norm, rank_window=args.rank_window)
    raw_runs = load_traces(args.raw, rank_window=args.rank_window)
    eval_norm_mean_runs = load_traces(
        args.eval_norm_mean,
        rank_window=args.rank_window,
    )
    norm_traces = summarize_traces(norm_runs)
    raw_traces = summarize_traces(raw_runs)
    eval_norm_mean_traces = summarize_traces(eval_norm_mean_runs)

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

    write_loss_plot(
        args.eval_norm_mean_plot,
        eval_norm_mean_traces,
        ylabel="eval/norm_mean",
        title="IN21k Reward Modes: Pretraining-Objective Norm Mean",
        top_k=args.top_k,
    )
    print(f"Wrote plot: {args.eval_norm_mean_plot}")

    write_rankings(
        args.ranking,
        norm_traces=norm_runs,
        raw_traces=raw_runs,
        eval_norm_mean_traces=eval_norm_mean_runs,
        norm=norm_traces,
        raw=raw_traces,
        eval_norm_mean=eval_norm_mean_traces,
        robust_windows=args.robust_windows,
        smooth_window=args.smooth_window,
    )
    print(f"Wrote ranking: {args.ranking}")

    print(f"\nBest normalized loss modes by last-{args.rank_window} mean:")
    for rank, trace in enumerate(norm_traces[: args.top_k], start=1):
        print(
            f"  {rank}. {trace.mode}: "
            f"mean={trace.last_window_mean:.8f} seed_std={trace.last_window_std:.8f}"
        )

    print(f"\nBest raw loss modes by last-{args.rank_window} mean:")
    for rank, trace in enumerate(raw_traces[: args.top_k], start=1):
        print(
            f"  {rank}. {trace.mode}: "
            f"mean={trace.last_window_mean:.8f} seed_std={trace.last_window_std:.8f}"
        )

    print(f"\nBest pretraining-objective norm mean modes by last-{args.rank_window} mean:")
    for rank, trace in enumerate(eval_norm_mean_traces[: args.top_k], start=1):
        print(
            f"  {rank}. {trace.mode}: "
            f"mean={trace.last_window_mean:.8f} seed_std={trace.last_window_std:.8f}"
        )


if __name__ == "__main__":
    main()
