"""Paired fixed-eval bootstrap for IN21k dense SAC state ablation checkpoints.

This script evaluates already-trained ``latest.pt`` checkpoints on the same
fixed dense-feature subset and bootstraps per-image losses. It is intentionally
separate from training so state-ablation runs do not need to be retrained.

Example:
    uv run python scripts/diagnostics/in21k/bootstrap_in21k_state_checkpoints.py \
        --feature-base-dir /features \
        --feature-image-root /data/train \
        --model-repo "$CANVIT_CHECKPOINT" \
        --eval-images 512 \
        --eval-subset-seed 10042 \
        --eval-batch-size 32 \
        --checkpoint-root checkpoints/in1k_dense_sac
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from tqdm import tqdm

# Problem: matplotlib/torch helper imports may consult global cache paths on
# managed machines. Solution: put cache writes under results/.cache before
# importing the dense SAC helper module. Result: this analysis script behaves
# like the other local plotting/eval tools.
_cache_dir = Path("results/.cache").resolve()
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

from canvit_rl.vision.precision import resolve_canvit_dtype  # noqa: E402
from canvit_rl.in21k.dense_train_batch import (  # noqa: E402
    dense_glimpse_images,
    init_normalizer_stats_from_shard,
    load_dense_train_batch,
)
from canvit_rl.in21k.rewards import dense_distillation_metrics  # noqa: E402
from canvit_rl.policies.canvas_state.models import CanvasStateActor  # noqa: E402
from canvit_rl.policies.image_independent.viewpoint import action_to_viewpoint  # noqa: E402
from scripts.training.in21k.train_i21k_dense_sac import (  # noqa: E402
    append_viewpoint_history,
    build_dense_eval_loader,
    build_pretrain_config,
    canvas_aux_channels,
    canvas_aux_state_map,
    canvas_layernorm_spatial,
    empty_viewpoint_history,
    load_frozen_hf_model,
    uses_canvas_aux_state,
)


DEFAULT_EXPERIMENTS = [
    "viewpoint-history",
    "canvas_no_hist",
    "canvas_no_hist_detdebt",
    "canvas_no_hist_cosprev",
    "canvas_no_hist_detdebt_cosprev",
    "canvas_reconstructionnorm",
    "canvas_no_hist_reconstructionnorm",
    "canvas_teacherreconstructionerror",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """Name and checkpoint path for one already-trained state-ablation run."""

    name: str
    checkpoint: Path


@dataclass(frozen=True)
class EvalResult:
    """Per-image losses from one deterministic checkpoint rollout."""

    name: str
    checkpoint: Path
    final_loss_norm: np.ndarray
    final_loss_raw: np.ndarray
    norm_mean: np.ndarray
    initial_loss_norm: np.ndarray
    initial_loss_raw: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-base-dir", type=Path, required=True)
    parser.add_argument("--feature-image-root", type=Path, default=None)
    parser.add_argument("--tar-dir", type=Path, default=None)
    parser.add_argument("--eval-feature-base-dir", type=Path, default=None)
    parser.add_argument("--eval-feature-image-root", type=Path, default=None)
    parser.add_argument("--paired-hidden-feature-base-dir", type=Path, default=None)
    parser.add_argument("--paired-hidden-feature-image-root", type=Path, default=None)
    parser.add_argument("--paired-hidden-tar-dir", type=Path, default=None)
    parser.add_argument(
        "--model-repo",
        type=str,
        default="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02",
    )
    parser.add_argument("--teacher-name", type=str, default="dinov3_vitb16")
    parser.add_argument("--scene-resolution", type=int, default=512)
    parser.add_argument("--glimpse-grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-shards", type=int, default=1)
    parser.add_argument("--eval-images", type=int, default=512)
    parser.add_argument("--eval-subset-seed", type=int, default=10042)
    parser.add_argument("--normalizer-max-samples", type=int, default=0)
    parser.add_argument("--reset-normalizer", action="store_true")
    parser.add_argument("--canvit-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--t",
        type=int,
        default=None,
        help="Optional rollout length override. Defaults to each checkpoint's saved t.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/in1k_dense_sac"),
        help="Root containing <experiment>/<checkpoint-name> files.",
    )
    parser.add_argument("--checkpoint-name", type=str, default="latest.pt")
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help=(
            "Experiment to evaluate. Use either a name under --checkpoint-root "
            "or name=path/to/checkpoint.pt. Repeat to customize the set."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument(
        "--metric",
        choices=["final_loss_norm", "norm_mean", "final_loss_raw"],
        default="norm_mean",
        help="Primary metric for ranking and pairwise CIs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/json/state_checkpoint_bootstrap"),
    )
    args = parser.parse_args()
    if args.eval_images < 1:
        raise ValueError("--eval-images must be positive.")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive.")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")
    if not (0.0 < args.ci < 100.0):
        raise ValueError("--ci must be between 0 and 100.")
    return args


def parse_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    """Resolve experiment names to checkpoint files."""
    requested = args.experiment or DEFAULT_EXPERIMENTS
    specs: list[ExperimentSpec] = []
    for item in requested:
        if "=" in item:
            name, raw_path = item.split("=", 1)
            checkpoint = Path(raw_path)
        else:
            name = item
            checkpoint = args.checkpoint_root / item / args.checkpoint_name
        if not name:
            raise ValueError(f"Invalid empty experiment name in {item!r}.")
        specs.append(ExperimentSpec(name=name, checkpoint=checkpoint))
    return specs


def checkpoint_value(
    checkpoint_args: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    """Read a saved CLI arg with a fallback for older checkpoints."""
    return checkpoint_args.get(key.replace("-", "_"), fallback)


def merged_eval_args(
    cli_args: argparse.Namespace,
    checkpoint_args: dict[str, Any],
) -> argparse.Namespace:
    """Combine shared eval settings with architecture flags saved in a checkpoint."""
    values = {
        "feature_base_dir": cli_args.feature_base_dir,
        "feature_image_root": cli_args.feature_image_root,
        "tar_dir": cli_args.tar_dir,
        "eval_feature_base_dir": cli_args.eval_feature_base_dir,
        "eval_feature_image_root": cli_args.eval_feature_image_root,
        "paired_hidden_feature_base_dir": cli_args.paired_hidden_feature_base_dir,
        "paired_hidden_feature_image_root": cli_args.paired_hidden_feature_image_root,
        "paired_hidden_tar_dir": cli_args.paired_hidden_tar_dir,
        "model_repo": cli_args.model_repo,
        "teacher_name": cli_args.teacher_name,
        "scene_resolution": cli_args.scene_resolution,
        "glimpse_grid_size": cli_args.glimpse_grid_size,
        "batch_size": cli_args.batch_size,
        "eval_batch_size": cli_args.eval_batch_size,
        "num_workers": cli_args.num_workers,
        "subset_shards": cli_args.subset_shards,
        "eval_images": cli_args.eval_images,
        "eval_subset_seed": cli_args.eval_subset_seed,
        "normalizer_max_samples": cli_args.normalizer_max_samples,
        "reset_normalizer": cli_args.reset_normalizer,
        "batches": 1,
        "canvit_dtype": cli_args.canvit_dtype,
        "t": int(cli_args.t if cli_args.t is not None else checkpoint_value(checkpoint_args, "t", 4)),
        "max_history": int(checkpoint_value(checkpoint_args, "max_history", 5)),
        "min_scale": float(checkpoint_value(checkpoint_args, "min_scale", 0.25)),
        "scene_reward_weight": float(checkpoint_value(checkpoint_args, "scene_reward_weight", 1.0)),
        "cls_reward_weight": float(checkpoint_value(checkpoint_args, "cls_reward_weight", 1.0)),
        "d_model": int(checkpoint_value(checkpoint_args, "d_model", 256)),
        "rff_dim": int(checkpoint_value(checkpoint_args, "rff_dim", 128)),
        "rff_seed": int(checkpoint_value(checkpoint_args, "rff_seed", 0)),
        "canvas_entropy_state": bool(checkpoint_value(checkpoint_args, "canvas_entropy_state", False)),
        "reconstruction_norm_state": bool(
            checkpoint_value(checkpoint_args, "reconstruction_norm_state", False)
        ),
        "teacher_reconstruction_error_state": bool(
            checkpoint_value(checkpoint_args, "teacher_reconstruction_error_state", False)
        ),
        "detail_debt": bool(checkpoint_value(checkpoint_args, "detail_debt", False)),
        "cos_prev": bool(checkpoint_value(checkpoint_args, "cos_prev", False)),
        "disable_canvas_avg_pool": bool(
            checkpoint_value(checkpoint_args, "disable_canvas_avg_pool", False)
        ),
        "disable_canvas_max_pool": bool(
            checkpoint_value(checkpoint_args, "disable_canvas_max_pool", False)
        ),
        "disable_viewpoint_history_state": bool(
            checkpoint_value(checkpoint_args, "disable_viewpoint_history_state", False)
        ),
    }
    if values["max_history"] < values["t"] + 1:
        raise ValueError(
            f"Checkpoint max_history={values['max_history']} is smaller than t+1={values['t'] + 1}."
        )
    return argparse.Namespace(**values)


def load_actor(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    eval_args: argparse.Namespace,
    canvas_feature_dim: int,
    device: torch.device,
) -> CanvasStateActor:
    """Build and load the actor shape encoded by the checkpoint args."""
    if "actor" not in checkpoint:
        raise ValueError(f"Expected dense SAC checkpoint with an actor key: {checkpoint_path}")
    saved_canvas_feature_dim = checkpoint.get("canvas_feature_dim")
    if saved_canvas_feature_dim is not None and int(saved_canvas_feature_dim) != canvas_feature_dim:
        raise ValueError(
            f"{checkpoint_path} was saved with canvas_feature_dim={saved_canvas_feature_dim}, "
            f"but the loaded model exposes {canvas_feature_dim}."
        )
    actor = CanvasStateActor(
        canvas_feature_dim=canvas_feature_dim,
        d_model=eval_args.d_model,
        rff_dim=eval_args.rff_dim,
        rff_seed=eval_args.rff_seed,
        use_entropy_state=uses_canvas_aux_state(eval_args),
        aux_state_channels=canvas_aux_channels(eval_args),
        use_canvas_avg_pool=not eval_args.disable_canvas_avg_pool,
        use_canvas_max_pool=not eval_args.disable_canvas_max_pool,
        use_viewpoint_history=not eval_args.disable_viewpoint_history_state,
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def evaluate_checkpoint_per_image(
    *,
    eval_args: argparse.Namespace,
    eval_loader,
    actor: CanvasStateActor,
    model,
    scene_norm,
    cls_norm,
    canvas_grid_size: int,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    device: torch.device,
    experiment_name: str,
    checkpoint_path: Path,
) -> EvalResult:
    """Mirror dense SAC eval, but retain per-image losses for paired bootstrap."""
    if hasattr(eval_loader, "reset"):
        eval_loader.reset()

    final_norm: list[torch.Tensor] = []
    final_raw: list[torch.Tensor] = []
    norm_mean_sums: list[torch.Tensor] = []
    initial_norm: list[torch.Tensor] = []
    initial_raw: list[torch.Tensor] = []
    eval_batch_size = min(int(eval_args.eval_batch_size), max(eval_args.eval_images, 1))
    eval_batches = max(1, math.ceil(eval_args.eval_images / max(eval_batch_size, 1)))

    with torch.inference_mode():
        iterator = tqdm(
            range(eval_batches),
            desc=f"eval {experiment_name}",
            leave=False,
        )
        for _ in iterator:
            batch = load_dense_train_batch(
                train_loader=eval_loader,
                device=device,
                scene_norm=scene_norm,
                cls_norm=cls_norm,
                non_blocking=True,
            )
            batch_size = batch.images.shape[0]
            state = model.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)
            coords, lengths = empty_viewpoint_history(
                batch_size=batch_size,
                max_steps=eval_args.max_history,
                device=device,
            )

            full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
            initial_canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=canvas_grid_size,
            )
            full_glimpse = sample_at_viewpoint(
                spatial=batch.images,
                viewpoint=full_vp,
                glimpse_size_px=glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
            state = out.state
            current_metrics = dense_distillation_metrics(
                model=model,
                state=state,
                batch=batch,
                scene_denorm=scene_norm.destandardize,
                cls_denorm=cls_norm.destandardize,
                scene_weight=eval_args.scene_reward_weight,
                cls_weight=eval_args.cls_reward_weight,
            )
            initial_norm.append(current_metrics.loss_norm.detach().cpu())
            initial_raw.append(current_metrics.loss_raw.detach().cpu())
            norm_mean_sum = current_metrics.scene_loss_norm + current_metrics.cls_loss_norm
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=canvas_grid_size,
            )
            coords, lengths = append_viewpoint_history(
                coords=coords,
                lengths=lengths,
                viewpoint=full_vp,
                step=0,
            )
            canvas_entropy = canvas_aux_state_map(
                args=eval_args,
                model=model,
                state=state,
                batch=batch,
                coords=coords,
                lengths=lengths,
                canvas_grid_size=canvas_grid_size,
                canvas_summary=canvas_summary,
                prev_canvas_summary=initial_canvas_summary,
            )

            for step_idx in range(eval_args.t):
                obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
                if canvas_entropy is not None:
                    obs["entropy"] = canvas_entropy
                action = actor.deterministic_action(obs)
                vp = action_to_viewpoint(action, min_scale=eval_args.min_scale)
                glimpse = sample_at_viewpoint(
                    spatial=dense_glimpse_images(batch),
                    viewpoint=vp,
                    glimpse_size_px=glimpse_size_px,
                ).to(dtype=canvit_dtype)
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state
                current_metrics = dense_distillation_metrics(
                    model=model,
                    state=state,
                    batch=batch,
                    scene_denorm=scene_norm.destandardize,
                    cls_denorm=cls_norm.destandardize,
                    scene_weight=eval_args.scene_reward_weight,
                    cls_weight=eval_args.cls_reward_weight,
                )
                norm_mean_sum = norm_mean_sum + (
                    current_metrics.scene_loss_norm + current_metrics.cls_loss_norm
                )
                prev_canvas_summary = canvas_summary
                canvas_summary = canvas_layernorm_spatial(
                    model=model,
                    state=state,
                    canvas_grid_size=canvas_grid_size,
                )
                coords, lengths = append_viewpoint_history(
                    coords=coords,
                    lengths=lengths,
                    viewpoint=vp,
                    step=step_idx + 1,
                )
                canvas_entropy = canvas_aux_state_map(
                    args=eval_args,
                    model=model,
                    state=state,
                    batch=batch,
                    coords=coords,
                    lengths=lengths,
                    canvas_grid_size=canvas_grid_size,
                    canvas_summary=canvas_summary,
                    prev_canvas_summary=prev_canvas_summary,
                )

            final_norm.append(current_metrics.loss_norm.detach().cpu())
            final_raw.append(current_metrics.loss_raw.detach().cpu())
            norm_mean_sums.append((norm_mean_sum / (eval_args.t + 1)).detach().cpu())

    return EvalResult(
        name=experiment_name,
        checkpoint=checkpoint_path,
        final_loss_norm=torch.cat(final_norm).numpy()[: eval_args.eval_images],
        final_loss_raw=torch.cat(final_raw).numpy()[: eval_args.eval_images],
        norm_mean=torch.cat(norm_mean_sums).numpy()[: eval_args.eval_images],
        initial_loss_norm=torch.cat(initial_norm).numpy()[: eval_args.eval_images],
        initial_loss_raw=torch.cat(initial_raw).numpy()[: eval_args.eval_images],
    )


def bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval for a one-dimensional mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector.")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (100.0 - ci) * 0.5
    low, high = np.percentile(means, [alpha, 100.0 - alpha])
    return float(values.mean()), float(low), float(high)


def metric_values(result: EvalResult, metric: str) -> np.ndarray:
    """Return the selected per-image metric from an EvalResult."""
    return getattr(result, metric)


def write_outputs(
    *,
    args: argparse.Namespace,
    results: list[EvalResult],
    t: int,
) -> None:
    """Write Markdown, CSV, and JSON reports for checkpoint bootstrap results."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    metric = args.metric
    ordered = sorted(results, key=lambda item: float(metric_values(item, metric).mean()))
    best = ordered[0]

    summary_rows = []
    for result in ordered:
        values = metric_values(result, metric)
        mean_value, ci_low, ci_high = bootstrap_ci(
            values,
            samples=args.bootstrap_samples,
            ci=args.ci,
            rng=rng,
        )
        summary_rows.append(
            {
                "rank": len(summary_rows) + 1,
                "experiment": result.name,
                "checkpoint": str(result.checkpoint),
                "metric": metric,
                "mean": mean_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "std": float(values.std(ddof=0)),
                "n": int(values.size),
            }
        )

    pair_rows = []
    for left in ordered:
        for right in ordered:
            if left.name == right.name:
                continue
            deltas = metric_values(left, metric) - metric_values(right, metric)
            mean_delta, ci_low, ci_high = bootstrap_ci(
                deltas,
                samples=args.bootstrap_samples,
                ci=args.ci,
                rng=rng,
            )
            pair_rows.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "metric": metric,
                    "delta_mean": mean_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "left_better": bool(ci_high < 0.0),
                    "right_better": bool(ci_low > 0.0),
                }
            )

    (args.out_dir / "summary.csv").write_text("")
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.out_dir / "pairwise.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    payload = {
        "metric": metric,
        "eval_images": args.eval_images,
        "eval_subset_seed": args.eval_subset_seed,
        "t": t,
        "bootstrap_samples": args.bootstrap_samples,
        "ci": args.ci,
        "summary": summary_rows,
        "pairwise": pair_rows,
    }
    (args.out_dir / "bootstrap_results.json").write_text(json.dumps(payload, indent=2))

    best_pairs = [row for row in pair_rows if row["right"] == best.name]
    md_lines = [
        "# IN21k State Checkpoint Bootstrap",
        "",
        f"Metric: `{metric}`. Lower is better.",
        f"Eval images: `{args.eval_images}`. Eval subset seed: `{args.eval_subset_seed}`. Rollout T: `{t}`.",
        f"Bootstrap samples: `{args.bootstrap_samples}`. CI: `{args.ci:.1f}%`.",
        "",
        "## Summary Ranking",
        "",
        "| Rank | Experiment | Mean | CI Low | CI High | Std | N |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            "| "
            f"{row['rank']} | `{row['experiment']}` | {row['mean']:.8f} | "
            f"{row['ci_low']:.8f} | {row['ci_high']:.8f} | {row['std']:.8f} | {row['n']} |"
        )

    md_lines.extend(
        [
            "",
            f"## Paired Deltas Vs Best: `{best.name}`",
            "",
            "Delta is `candidate - best`; negative means the candidate beats the current best.",
            "",
            "| Candidate | Delta Mean | CI Low | CI High | Interpretation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(best_pairs, key=lambda item: item["delta_mean"]):
        if row["ci_low"] <= 0.0 <= row["ci_high"]:
            interpretation = "overlaps zero"
        elif row["ci_low"] > 0.0:
            interpretation = "worse than best on this eval subset"
        else:
            interpretation = "better than best on this eval subset"
        md_lines.append(
            "| "
            f"`{row['left']}` | {row['delta_mean']:+.8f} | {row['ci_low']:+.8f} | "
            f"{row['ci_high']:+.8f} | {interpretation} |"
        )

    md_lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If a paired delta CI crosses zero, this fixed eval subset does not separate those two checkpoints reliably.",
            "- This controls eval-image uncertainty only; it does not measure training-seed uncertainty.",
            "- When top candidates overlap, pick the cleaner state representation rather than over-reading the rank.",
        ]
    )
    (args.out_dir / "recommendation.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    args = parse_args()
    specs = parse_experiments(args)
    missing = [spec.checkpoint for spec in specs if not spec.checkpoint.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s):\n" + "\n".join(f"  {path}" for path in missing)
        )

    loaded_checkpoints: dict[str, dict[str, Any]] = {}
    eval_args_by_name: dict[str, argparse.Namespace] = {}
    for spec in specs:
        checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected dict checkpoint: {spec.checkpoint}")
        checkpoint_args = dict(checkpoint.get("args", {}))
        loaded_checkpoints[spec.name] = checkpoint
        eval_args_by_name[spec.name] = merged_eval_args(args, checkpoint_args)

    t_values = {eval_args.t for eval_args in eval_args_by_name.values()}
    if len(t_values) != 1:
        formatted = ", ".join(
            f"{name}:T={eval_args.t}" for name, eval_args in eval_args_by_name.items()
        )
        raise ValueError(f"Paired bootstrap expects the same T for all checkpoints ({formatted}).")
    shared_t = t_values.pop()

    first_args = next(iter(eval_args_by_name.values()))
    cfg, _modules = build_pretrain_config(first_args)
    device = cfg.device
    model, glimpse_size_px = load_frozen_hf_model(first_args, cfg)
    canvit_dtype = resolve_canvit_dtype(first_args.canvit_dtype, device)
    model.to(device=device, dtype=canvit_dtype)
    for module in model.modules():
        if module.__class__.__name__ == "VPEEncoder":
            module.to(device=device, dtype=torch.float32)
    canvas_grid_size = int(cfg.canvas_patch_grid_size)
    cls_norm, scene_norm = model.standardizers(canvas_grid_size)
    if first_args.reset_normalizer or not scene_norm.initialized:
        shards_dir = (
            cfg.feature_base_dir
            / cfg.teacher_name
            / str(cfg.scene_resolution)
            / "shards"
        )
        init_normalizer_stats_from_shard(
            shards_dir=shards_dir,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            device=device,
            max_samples=first_args.normalizer_max_samples,
        )

    results: list[EvalResult] = []
    for spec in specs:
        eval_args = eval_args_by_name[spec.name]
        eval_loader = build_dense_eval_loader(eval_args, cfg)
        actor = load_actor(
            checkpoint_path=spec.checkpoint,
            checkpoint=loaded_checkpoints[spec.name],
            eval_args=eval_args,
            canvas_feature_dim=int(model.canvas_dim),
            device=device,
        )
        result = evaluate_checkpoint_per_image(
            eval_args=eval_args,
            eval_loader=eval_loader,
            actor=actor,
            model=model,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            canvas_grid_size=canvas_grid_size,
            glimpse_size_px=glimpse_size_px,
            canvit_dtype=canvit_dtype,
            device=device,
            experiment_name=spec.name,
            checkpoint_path=spec.checkpoint,
        )
        results.append(result)
        print(f"{spec.name}: {args.metric}={metric_values(result, args.metric).mean():.8f}")

    write_outputs(args=args, results=results, t=shared_t)
    print(f"Wrote bootstrap reports to {args.out_dir}")


if __name__ == "__main__":
    main()
