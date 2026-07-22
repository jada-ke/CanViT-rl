"""Train an IN21k dense Canvas critic from random-k candidate rewards.

This is a critic-only diagnostic for the dense SAC architecture: sample ``--k``
random Viewpoint candidates per state, compute the true dense reward for each,
and regress Q(state, action) to those one-step rewards without SAC bootstrapping
or actor updates.

Example:
    uv run python scripts/train_in21k_critic.py \
        --feature-base-dir /features \
        --feature-image-root /data/train \
        --model-repo canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02 \
        --reward-mode raw_mse_log_delta \
        --batch-size 4 \
        --batches 1000 \
        --t 2 \
        --k 32 \
        --critic-local-action-features \
        --canvas-entropy-state
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

if "--comet" in sys.argv and "--no-comet" not in sys.argv:
    try:
        from comet_ml import Experiment
    except ImportError:
        Experiment = None
else:
    Experiment = None

import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from tqdm import tqdm

from canvit_rl.canvas.state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    empty_viewpoint_history,
)
from canvit_rl.canvit_precision import resolve_canvit_dtype
from canvit_rl.greedy import _index_state_batch, _repeat_state_chunks
from canvit_rl.pretrain_IN21k.dense_train_batch import (
    DenseTrainBatch,
    dense_glimpse_images,
    init_normalizer_stats_from_shard,
    load_dense_train_batch,
)
from canvit_rl.pretrain_IN21k.reward import DenseDistillationMetrics, dense_reward
from canvit_rl.sac_models import CanvasStateCritic
from canvit_rl.viewpoint_policy import viewpoint_to_action
from scripts.train_i21k_dense_sac import (
    build_dense_eval_loader,
    build_dense_loader,
    build_pretrain_config,
    dense_canvas_entropy_map,
    load_frozen_hf_model,
)

REWARD_MODES = [
    "raw_mse_delta",
    "raw_mse_log_delta",
    "raw_mse_log_delta_clipped",
    "raw_mse_log_delta_tanh",
    "raw_mse_reduction",
    "raw_mse_l0_delta",
    "raw_mse_clipped_l0_delta",
    "raw_mse_tanh_l0_delta",
    "norm_loss_delta",
    "norm_loss_log_delta",
    "norm_loss_log_delta_clipped",
    "norm_loss_log_delta_tanh",
    "norm_loss_reduction",
    "norm_loss_l0_delta",
    "norm_loss_clipped_l0_delta",
    "norm_loss_tanh_l0_delta",
]


def make_comet_experiment(args: argparse.Namespace):
    """Create a Comet experiment only when explicitly enabled."""
    if args.no_comet:
        return None
    if Experiment is None:
        raise RuntimeError("Install comet-ml or run without --comet.")
    experiment = Experiment(
        project_name=args.comet_project,
        workspace=args.comet_workspace,
        auto_param_logging=True,
        auto_metric_logging=True,
    )
    experiment.set_name(args.experiment_name or "in21k-critic")
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    return experiment


def parse_args() -> argparse.Namespace:
    """Parse critic-only dense IN21k training arguments."""
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--subset-size",
        "--train-images",
        dest="subset_size",
        type=int,
        default=0,
        help=(
            "Number of fixed training images to materialize. 0 streams shards; "
            "--train-images is a clearer alias for critic-only runs."
        ),
    )
    parser.add_argument(
        "--subset-seed",
        "--train-subset-seed",
        dest="subset_seed",
        type=int,
        default=42,
        help="Seed for selecting the fixed training image subset.",
    )
    parser.add_argument("--subset-shards", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--rollout-policy",
        choices=["oracle", "critic", "random"],
        default="oracle",
        help="Action used to advance to the next state after each supervised target set.",
    )
    parser.add_argument("--max-history", type=int, default=5)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="raw_mse_log_delta")
    parser.add_argument("--reward-eps", type=float, default=1e-6)
    parser.add_argument("--reward-log-clip", type=float, default=1.0)
    parser.add_argument("--reward-l0-clip", type=float, default=1.0)
    parser.add_argument("--reward-tanh-scale", type=float, default=1.0)
    parser.add_argument("--scene-reward-weight", type=float, default=1.0)
    parser.add_argument("--cls-reward-weight", type=float, default=1.0)
    parser.add_argument("--canvit-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--rff-dim", type=int, default=128)
    parser.add_argument("--rff-seed", type=int, default=42)
    parser.add_argument("--critic-d-model", type=int, default=256)
    parser.add_argument("--critic-rff-dim", type=int, default=128)
    parser.add_argument("--critic-local-action-features", action="store_true")
    parser.add_argument("--canvas-entropy-state", action="store_true")
    parser.add_argument("--disable-canvas-avg-pool", action="store_true")
    parser.add_argument("--disable-canvas-max-pool", action="store_true")
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-images", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-subset-seed", type=int, default=10042)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/in21k_critic"))
    parser.add_argument("--reset-normalizer", action="store_true")
    parser.add_argument("--normalizer-max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-comet", action="store_true", default=True)
    parser.add_argument("--comet", dest="no_comet", action="store_false")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="in21k-critic")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="in21k-critic")
    args = parser.parse_args()
    if args.t < 1:
        raise ValueError("--t must be positive.")
    if args.max_history < args.t + 1:
        raise ValueError("--max-history must be at least --t + 1.")
    if args.k < 2:
        raise ValueError("--k must be at least 2.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.subset_size < 0:
        raise ValueError("--train-images/--subset-size must be non-negative.")
    if args.eval_batch_size < 0 or args.eval_images < 0:
        raise ValueError("--eval-batch-size/--eval-images must be non-negative.")
    if args.critic_d_model < 1 or args.critic_rff_dim < 1:
        raise ValueError("--critic-d-model and --critic-rff-dim must be positive.")
    if args.disable_canvas_avg_pool and args.disable_canvas_max_pool:
        raise ValueError("At least one canvas pooling branch must remain enabled.")
    return args


def _episode_l0_for_reward_mode(mode: str, metrics: DenseDistillationMetrics) -> torch.Tensor:
    """Return the reset denominator in the same loss space as reward mode."""
    if mode.startswith("norm_loss"):
        return metrics.loss_norm.detach().clone()
    return metrics.loss_raw.detach().clone()


def _repeat_dense_batch_chunks(batch: DenseTrainBatch, chunks: int) -> DenseTrainBatch:
    """Repeat a full dense batch in chunk order to match random candidate actions."""
    repeat_shape = (chunks,) + (1,)
    return DenseTrainBatch(
        images=batch.images.repeat(repeat_shape + (1,) * (batch.images.ndim - 2)),
        labels=batch.labels.repeat(chunks),
        scene_target=batch.scene_target.repeat(
            repeat_shape + (1,) * (batch.scene_target.ndim - 2)
        ),
        cls_target=batch.cls_target.repeat(
            repeat_shape + (1,) * (batch.cls_target.ndim - 2)
        ),
        raw_scene_target=batch.raw_scene_target.repeat(
            repeat_shape + (1,) * (batch.raw_scene_target.ndim - 2)
        ),
        raw_cls_target=batch.raw_cls_target.repeat(
            repeat_shape + (1,) * (batch.raw_cls_target.ndim - 2)
        ),
        glimpse_images=(
            None
            if batch.glimpse_images is None
            else batch.glimpse_images.repeat(
                repeat_shape + (1,) * (batch.glimpse_images.ndim - 2)
            )
        ),
    )


def _repeat_metrics_chunks(metrics: DenseDistillationMetrics, chunks: int) -> DenseDistillationMetrics:
    """Repeat per-sample metrics in chunk order."""
    return DenseDistillationMetrics(
        scene_loss_norm=metrics.scene_loss_norm.repeat(chunks),
        cls_loss_norm=metrics.cls_loss_norm.repeat(chunks),
        loss_norm=metrics.loss_norm.repeat(chunks),
        scene_loss_raw=metrics.scene_loss_raw.repeat(chunks),
        cls_loss_raw=metrics.cls_loss_raw.repeat(chunks),
        loss_raw=metrics.loss_raw.repeat(chunks),
    )


def _random_candidate_viewpoints(
    *,
    batch_size: int,
    k: int,
    min_scale: float,
    device: torch.device,
) -> Viewpoint:
    """Sample random in-bounds candidate Viewpoints in chunk order [k, batch]."""
    scales = torch.rand(k, batch_size, device=device) * (1.0 - min_scale) + min_scale
    bounds = (1.0 - scales).clamp_min(0.0)
    centers = (torch.rand(k, batch_size, 2, device=device) * 2.0 - 1.0) * bounds[..., None]
    return Viewpoint(centers=centers.reshape(k * batch_size, 2), scales=scales.reshape(-1))


def _dense_metrics(
    *,
    model,
    state,
    batch: DenseTrainBatch,
    scene_norm,
    cls_norm,
    args: argparse.Namespace,
) -> DenseDistillationMetrics:
    """Compute dense distillation metrics with the active reward weights."""
    from canvit_rl.pretrain_IN21k.reward import dense_distillation_metrics

    return dense_distillation_metrics(
        model=model,
        state=state,
        batch=batch,
        scene_denorm=scene_norm.destandardize,
        cls_denorm=cls_norm.destandardize,
        scene_weight=args.scene_reward_weight,
        cls_weight=args.cls_reward_weight,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values with average ranks for ties; lowest value gets rank 1."""
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or nan for constant inputs."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman correlation without scipy."""
    return _pearson(_average_ranks(x), _average_ranks(y))


def _mean_valid(values: list[float]) -> float:
    """Mean that ignores nan values."""
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(valid.mean()) if valid.size else float("nan")


def _critic_batch(
    *,
    canvas_summary: torch.Tensor,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    canvas_entropy: torch.Tensor | None,
    k: int,
) -> dict[str, torch.Tensor]:
    """Repeat one state batch to align with chunk-ordered candidate actions."""
    batch = {
        "canvas": canvas_summary.repeat(k, 1, 1, 1).detach().clone(),
        "coords": coords.repeat(k, 1, 1).detach().clone(),
        "lengths": lengths.repeat(k).detach().clone(),
    }
    if canvas_entropy is not None:
        batch["entropy"] = canvas_entropy.repeat(k, 1, 1, 1).detach().clone()
    return batch


def _candidate_rewards_and_next(
    *,
    args: argparse.Namespace,
    model,
    scene_norm,
    cls_norm,
    batch: DenseTrainBatch,
    state,
    current_metrics: DenseDistillationMetrics,
    episode_l0: torch.Tensor,
    candidates: Viewpoint,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
) -> tuple[torch.Tensor, object, DenseDistillationMetrics]:
    """Evaluate candidate next states and dense rewards in chunk order."""
    k = candidates.centers.shape[0] // batch.images.shape[0]
    candidate_batch = _repeat_dense_batch_chunks(batch, k)
    candidate_state = _repeat_state_chunks(state, k)
    with torch.inference_mode():
        glimpse = sample_at_viewpoint(
            spatial=dense_glimpse_images(candidate_batch),
            viewpoint=candidates,
            glimpse_size_px=glimpse_size_px,
        ).to(dtype=canvit_dtype)
        out = model(glimpse=glimpse, state=candidate_state, viewpoint=candidates)
        after_metrics = _dense_metrics(
            model=model,
            state=out.state,
            batch=candidate_batch,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            args=args,
        )
        reward = dense_reward(
            mode=args.reward_mode,
            before=_repeat_metrics_chunks(current_metrics, k),
            after=after_metrics,
            l0=episode_l0.repeat(k),
            eps=args.reward_eps,
            log_clip=args.reward_log_clip,
            l0_clip=args.reward_l0_clip,
            tanh_scale=args.reward_tanh_scale,
        )
    return reward.detach(), out.state, after_metrics


def _select_candidate_indices(
    *,
    rewards: torch.Tensor,
    q_values: torch.Tensor,
    policy: str,
    batch_size: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Choose one candidate per sample for rolling to the next state."""
    reward_grid = rewards.view(k, batch_size)
    q_grid = q_values.view(k, batch_size)
    if policy == "oracle":
        chosen_k = reward_grid.argmax(dim=0)
    elif policy == "critic":
        chosen_k = q_grid.argmax(dim=0)
    else:
        chosen_k = torch.randint(0, k, (batch_size,), device=device)
    sample_ids = torch.arange(batch_size, device=device)
    return chosen_k * batch_size + sample_ids


def _batch_metrics_from_predictions(
    *,
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    batch_size: int,
    k: int,
) -> dict[str, float]:
    """Compute ranking/regret metrics from candidate Q predictions."""
    q_np = q_values.detach().cpu().view(k, batch_size).numpy().astype(np.float64)
    r_np = rewards.detach().cpu().view(k, batch_size).numpy().astype(np.float64)
    pearsons: list[float] = []
    spearmans: list[float] = []
    mses: list[float] = []
    regrets: list[float] = []
    top_percentiles: list[float] = []
    best_rewards: list[float] = []
    pred_best_rewards: list[float] = []
    for sample_idx in range(batch_size):
        q = q_np[:, sample_idx]
        r = r_np[:, sample_idx]
        q_argmax = int(np.argmax(q))
        best = float(np.max(r))
        chosen = float(r[q_argmax])
        pearsons.append(_pearson(q, r))
        spearmans.append(_spearman(q, r))
        mses.append(float(np.mean((q - r) ** 2)))
        regrets.append(best - chosen)
        top_percentiles.append(float(np.mean(r <= chosen)))
        best_rewards.append(best)
        pred_best_rewards.append(chosen)
    return {
        "reward_mse": float(np.mean(mses)),
        "pearson": _mean_valid(pearsons),
        "spearman": _mean_valid(spearmans),
        "top1_true_percentile": float(np.mean(top_percentiles)),
        "regret": float(np.mean(regrets)),
        "best_reward": float(np.mean(best_rewards)),
        "pred_best_reward": float(np.mean(pred_best_rewards)),
    }


def _build_critics(args: argparse.Namespace, canvas_feature_dim: int, device: torch.device):
    """Create q1/q2 with the same architecture family as dense SAC."""
    common = dict(
        canvas_feature_dim=canvas_feature_dim,
        d_model=args.critic_d_model,
        rff_dim=args.critic_rff_dim,
        rff_seed=args.rff_seed,
        use_entropy_state=args.canvas_entropy_state,
        use_canvas_avg_pool=not args.disable_canvas_avg_pool,
        use_canvas_max_pool=not args.disable_canvas_max_pool,
        use_action_location_features=args.critic_local_action_features,
    )
    return CanvasStateCritic(**common).to(device), CanvasStateCritic(**common).to(device)


def run_epoch(
    *,
    args: argparse.Namespace,
    loader,
    model,
    scene_norm,
    cls_norm,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    optimizer: torch.optim.Optimizer | None,
    canvas_grid_size: int,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    """Run supervised critic train/eval batches."""
    training = optimizer is not None
    q1.train(training)
    q2.train(training)
    accum: dict[str, list[float]] = {
        "loss": [],
        "q1_loss": [],
        "q2_loss": [],
        "reward_mse": [],
        "pearson": [],
        "spearman": [],
        "top1_true_percentile": [],
        "regret": [],
        "best_reward": [],
        "pred_best_reward": [],
    }
    if hasattr(loader, "reset") and not training:
        loader.reset()
    for _ in range(batches):
        batch = load_dense_train_batch(
            train_loader=loader,
            device=device,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            non_blocking=True,
        )
        batch_size = batch.images.shape[0]
        state = model.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)
        coords, lengths = empty_viewpoint_history(
            batch_size=batch_size,
            max_steps=args.max_history,
            device=device,
        )
        full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
        with torch.inference_mode():
            full_glimpse = sample_at_viewpoint(
                spatial=batch.images,
                viewpoint=full_vp,
                glimpse_size_px=glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
            state = out.state
            current_metrics = _dense_metrics(
                model=model,
                state=state,
                batch=batch,
                scene_norm=scene_norm,
                cls_norm=cls_norm,
                args=args,
            )
            episode_l0 = _episode_l0_for_reward_mode(args.reward_mode, current_metrics)
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=canvas_grid_size,
            )
            canvas_entropy = (
                dense_canvas_entropy_map(
                    model=model,
                    state=state,
                    batch=batch,
                    canvas_grid_size=canvas_grid_size,
                )
                if args.canvas_entropy_state
                else None
            )
        coords, lengths = append_viewpoint_history(
            coords=coords,
            lengths=lengths,
            viewpoint=full_vp,
            step=0,
        )
        for step_idx in range(args.t):
            candidates = _random_candidate_viewpoints(
                batch_size=batch_size,
                k=args.k,
                min_scale=args.min_scale,
                device=device,
            )
            rewards, candidate_next_state, _ = _candidate_rewards_and_next(
                args=args,
                model=model,
                scene_norm=scene_norm,
                cls_norm=cls_norm,
                batch=batch,
                state=state,
                current_metrics=current_metrics,
                episode_l0=episode_l0,
                candidates=candidates,
                glimpse_size_px=glimpse_size_px,
                canvit_dtype=canvit_dtype,
            )
            action = viewpoint_to_action(candidates, min_scale=args.min_scale)
            critic_batch = _critic_batch(
                canvas_summary=canvas_summary,
                coords=coords,
                lengths=lengths,
                canvas_entropy=canvas_entropy,
                k=args.k,
            )
            q1_pred = q1(critic_batch, action)
            q2_pred = q2(critic_batch, action)
            target = rewards.float().detach().clone()
            q1_loss = F.mse_loss(q1_pred, target)
            q2_loss = F.mse_loss(q2_pred, target)
            loss = q1_loss + q2_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(q1.parameters()) + list(q2.parameters()),
                        args.grad_clip,
                    )
                optimizer.step()
            with torch.no_grad():
                q_min = torch.minimum(q1_pred, q2_pred)
                metrics = _batch_metrics_from_predictions(
                    q_values=q_min,
                    rewards=target,
                    batch_size=batch_size,
                    k=args.k,
                )
                accum["loss"].append(float(loss.detach().cpu().item()))
                accum["q1_loss"].append(float(q1_loss.detach().cpu().item()))
                accum["q2_loss"].append(float(q2_loss.detach().cpu().item()))
                for key, value in metrics.items():
                    accum[key].append(value)
                selected = _select_candidate_indices(
                    rewards=target,
                    q_values=q_min.detach(),
                    policy=args.rollout_policy,
                    batch_size=batch_size,
                    k=args.k,
                    device=device,
                )
                selected_vp = Viewpoint(
                    centers=candidates.centers.index_select(0, selected),
                    scales=candidates.scales.index_select(0, selected),
                )
                state = _index_state_batch(candidate_next_state, selected)
                current_metrics = _dense_metrics(
                    model=model,
                    state=state,
                    batch=batch,
                    scene_norm=scene_norm,
                    cls_norm=cls_norm,
                    args=args,
                )
                canvas_summary = canvas_layernorm_spatial(
                    model=model,
                    state=state,
                    canvas_grid_size=canvas_grid_size,
                )
                canvas_entropy = (
                    dense_canvas_entropy_map(
                        model=model,
                        state=state,
                        batch=batch,
                        canvas_grid_size=canvas_grid_size,
                    )
                    if args.canvas_entropy_state
                    else None
                )
                coords, lengths = append_viewpoint_history(
                    coords=coords,
                    lengths=lengths,
                    viewpoint=selected_vp,
                    step=step_idx + 1,
                )
    prefix = "train" if training else "eval"
    return {
        f"{prefix}/critic_{key}": _mean_valid(values)
        for key, values in accum.items()
    }


def save_checkpoint(
    *,
    path: Path,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    batch: int,
    metrics: dict[str, float],
) -> None:
    """Save critic-only training state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "batch": batch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    """Train dense IN21k critics with supervised random-k candidate rewards."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg, modules = build_pretrain_config(args)
    device = cfg.device
    train_loader = build_dense_loader(args, cfg, modules)
    eval_loader = build_dense_eval_loader(args, cfg)
    model, glimpse_size_px = load_frozen_hf_model(args, cfg)
    canvit_dtype = resolve_canvit_dtype(args.canvit_dtype, device)
    model.to(device=device, dtype=canvit_dtype)
    for module in model.modules():
        if module.__class__.__name__ == "VPEEncoder":
            module.to(device=device, dtype=torch.float32)
    canvas_grid_size = cfg.canvas_patch_grid_size
    cls_norm, scene_norm = model.standardizers(canvas_grid_size)
    if args.reset_normalizer or not scene_norm.initialized:
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
            max_samples=args.normalizer_max_samples,
        )
    q1, q2 = _build_critics(args, int(model.canvas_dim), device)
    optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()),
        lr=args.critic_lr,
        weight_decay=args.weight_decay,
    )
    comet_exp = make_comet_experiment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_metrics: dict[str, float] = {}
    pbar = tqdm(range(1, args.batches + 1), desc="Training IN21k dense critic")
    for batch_idx in pbar:
        latest_metrics = run_epoch(
            args=args,
            loader=train_loader,
            model=model,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            q1=q1,
            q2=q2,
            optimizer=optimizer,
            canvas_grid_size=canvas_grid_size,
            glimpse_size_px=glimpse_size_px,
            canvit_dtype=canvit_dtype,
            device=device,
            batches=1,
        )
        if batch_idx % args.log_interval == 0 or batch_idx == 1:
            if comet_exp is not None:
                # Problem: the critic-only diagnostic printed useful ranking
                # metrics but did not preserve curves in Comet. Solution: log
                # the same compact train dict at the existing log interval.
                # Result: critic capacity tests can be compared without full
                # SAC runs.
                comet_exp.log_metrics(latest_metrics, step=batch_idx)
            pbar.write(
                "train "
                f"batch={batch_idx} "
                f"loss={latest_metrics.get('train/critic_loss', float('nan')):.4f} "
                f"pearson={latest_metrics.get('train/critic_pearson', float('nan')):+.3f} "
                f"spearman={latest_metrics.get('train/critic_spearman', float('nan')):+.3f} "
                "top1_pct="
                f"{latest_metrics.get('train/critic_top1_true_percentile', float('nan')):.3f} "
                f"regret={latest_metrics.get('train/critic_regret', float('nan')):+.4f}"
            )
        if eval_loader is not None and batch_idx % args.eval_interval == 0:
            eval_batch_size = int(args.eval_batch_size or args.batch_size)
            eval_batches = max(1, math.ceil(args.eval_images / max(eval_batch_size, 1)))
            eval_metrics = run_epoch(
                args=args,
                loader=eval_loader,
                model=model,
                scene_norm=scene_norm,
                cls_norm=cls_norm,
                q1=q1,
                q2=q2,
                optimizer=None,
                canvas_grid_size=canvas_grid_size,
                glimpse_size_px=glimpse_size_px,
                canvit_dtype=canvit_dtype,
                device=device,
                batches=eval_batches,
            )
            latest_metrics.update(eval_metrics)
            if comet_exp is not None:
                comet_exp.log_metrics(eval_metrics, step=batch_idx)
            pbar.write(
                "eval "
                f"batch={batch_idx} "
                f"mse={eval_metrics.get('eval/critic_reward_mse', float('nan')):.4f} "
                f"pearson={eval_metrics.get('eval/critic_pearson', float('nan')):+.3f} "
                f"spearman={eval_metrics.get('eval/critic_spearman', float('nan')):+.3f} "
                "top1_pct="
                f"{eval_metrics.get('eval/critic_top1_true_percentile', float('nan')):.3f} "
                f"regret={eval_metrics.get('eval/critic_regret', float('nan')):+.4f}"
            )
        if batch_idx % args.checkpoint_interval == 0:
            save_checkpoint(
                path=args.checkpoint_dir / "latest.pt",
                q1=q1,
                q2=q2,
                optimizer=optimizer,
                args=args,
                batch=batch_idx,
                metrics=latest_metrics,
            )
            with (args.checkpoint_dir / "latest_metrics.json").open("w") as f:
                json.dump(latest_metrics, f, indent=2, sort_keys=True)
    save_checkpoint(
        path=args.checkpoint_dir / "latest.pt",
        q1=q1,
        q2=q2,
        optimizer=optimizer,
        args=args,
        batch=args.batches,
        metrics=latest_metrics,
    )


if __name__ == "__main__":
    main()
