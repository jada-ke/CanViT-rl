"""Evaluate non-learned IN21k dense-view baselines with SAC-compatible metrics.

Example:
    python -u scripts/eval_in21k_baseline.py \
        --policy random \
        --feature-base-dir /features \
        --feature-image-root /data/train \
        --model-repo "$CANVIT_CHECKPOINT" \
        --batch-size 32 \
        --t 2 \
        --reward-mode raw_mse_log_delta \
        --eval-images 20 \
        --eval-subset-seed 24 \
        --eval-batch-size 16 \
        --actor-reward-percentile-grid-size 11 \
        --comet \
        --experiment-name random-baseline
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.train_i21k_dense_sac import (
    add_dense_sac_comet_args,
    build_dense_eval_loader,
    build_pretrain_config,
    evaluate_dense_sac,
    load_frozen_hf_model,
    make_dense_comet_experiment,
    parse_reward_map_scales,
)
from canvit_rl.canvit_precision import resolve_canvit_dtype
from canvit_rl.viewpoint_policy import viewpoint_to_action
from canvit_pytorch import Viewpoint
from canvit_pytorch.policies import random_viewpoints


def _level_viewpoints(level: int) -> list[tuple[float, float, float]]:
    if level == 0:
        return [(0.0, 0.0, 1.0)]
    side = 2**level
    scale = 1.0 / side
    centers = []
    for row in range(side):
        center_y = -1.0 + scale + 2.0 * scale * row
        for col in range(side):
            center_x = -1.0 + scale + 2.0 * scale * col
            centers.append((center_y, center_x, scale))
    return centers


def _build_tile_masks(
    tiles: list[tuple[float, float, float]],
    *,
    canvas_grid: int,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.linspace(
        -1.0 + 1.0 / canvas_grid,
        1.0 - 1.0 / canvas_grid,
        canvas_grid,
        device=device,
    )
    crops = torch.tensor(tiles, device=device)
    cy, cx, scale = crops[:, 0], crops[:, 1], crops[:, 2]
    row_in = (coords.unsqueeze(0) - cy.unsqueeze(1)).abs() <= scale.unsqueeze(1)
    col_in = (coords.unsqueeze(0) - cx.unsqueeze(1)).abs() <= scale.unsqueeze(1)
    return row_in.unsqueeze(2) & col_in.unsqueeze(1)


class RandomDensePolicy(nn.Module):
    """Sample one valid random Viewpoint per eval step."""

    def __init__(self, *, min_scale: float) -> None:
        super().__init__()
        self.min_scale = float(min_scale)

    def deterministic_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        device = batch["coords"].device
        batch_size = batch["coords"].shape[0]
        viewpoint = random_viewpoints(
            batch_size=batch_size,
            device=device,
            n_viewpoints=1,
            min_scale=self.min_scale,
            max_scale=1.0,
            start_with_full_scene=False,
        )[0]
        return viewpoint_to_action(
            viewpoint,
            min_scale=self.min_scale,
        )


class DenseEntropyCoarseToFinePolicy(nn.Module):
    """Use the built-in EG-C2F tile schedule with dense-error scores."""

    def __init__(self, *, min_scale: float, canvas_grid: int, device: torch.device) -> None:
        super().__init__()
        self.min_scale = float(min_scale)
        self.levels = [_level_viewpoints(level) for level in range(3)]
        level_starts = []
        step = 0
        for level in self.levels:
            level_starts.append(step)
            step += len(level)
        self.level_starts = level_starts
        self.tile_masks = [
            None,
            _build_tile_masks(self.levels[1], canvas_grid=canvas_grid, device=device),
            _build_tile_masks(self.levels[2], canvas_grid=canvas_grid, device=device),
        ]
        self.visited = [
            None,
            torch.zeros(0, len(self.levels[1]), dtype=torch.bool, device=device),
            torch.zeros(0, len(self.levels[2]), dtype=torch.bool, device=device),
        ]

    def deterministic_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "entropy" not in batch:
            raise ValueError("EG-C2F baseline requires the internal dense-error map.")
        entropy = batch["entropy"].float()
        device = entropy.device
        batch_size, _, grid_h, grid_w = entropy.shape
        if grid_h != grid_w:
            raise ValueError("EG-C2F expects a square canvas entropy grid.")
        step_idx = int(batch["lengths"].min().detach().item())
        if not torch.all(batch["lengths"] == step_idx):
            raise ValueError("EG-C2F baseline expects synchronized eval episodes.")
        if step_idx >= sum(len(level) for level in self.levels):
            raise ValueError("EG-C2F has 21 built-in timesteps; require --t <= 20.")
        level_idx = sum(1 for start in self.level_starts[1:] if step_idx >= start)
        pos_in_level = step_idx - self.level_starts[level_idx]
        crops = self.levels[level_idx]
        if level_idx == 0:
            selected = torch.tensor(crops[0], device=device).expand(batch_size, -1)
        else:
            visited = self.visited[level_idx]
            if visited is None or visited.shape[0] != batch_size or pos_in_level == 0:
                visited = torch.zeros(
                    batch_size,
                    len(crops),
                    dtype=torch.bool,
                    device=device,
                )
                self.visited[level_idx] = visited
            masks = self.tile_masks[level_idx]
            assert masks is not None
            n_cells = masks.sum(dim=(1, 2)).clamp(min=1).float()
            # Problem: IN21k dense eval has no ADE segmentation probe entropy,
            # but we still want the same EG-C2F action schedule. Solution:
            # score the built-in C2F tiles with the dense reconstruction-error
            # map and mark visited tiles exactly like canvit-eval does. Result:
            # the baseline follows the ADE policy's fixed 1+4+16 tile sequence
            # without introducing a custom scale schedule.
            scores = (entropy * masks.unsqueeze(0).float()).sum(dim=(2, 3))
            scores = scores / n_cells.unsqueeze(0)
            scores = scores.masked_fill(visited, float("-inf"))
            chosen = scores.argmax(dim=1)
            visited.scatter_(1, chosen.unsqueeze(1), True)
            selected = torch.tensor(crops, device=device).index_select(0, chosen)
        centers = selected[:, :2]
        scales = selected[:, 2]
        return viewpoint_to_action(
            Viewpoint(centers=centers, scales=scales.to(device=device)),
            min_scale=self.min_scale,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["random", "egc2f"], default="random")
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
    parser.add_argument("--reset-normalizer", action="store_true")
    parser.add_argument("--normalizer-max-samples", type=int, default=0)
    parser.add_argument("--teacher-name", type=str, default="dinov3_vitb16")
    parser.add_argument("--scene-resolution", type=int, default=512)
    parser.add_argument("--glimpse-grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-shards", type=int, default=1)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=5)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--scene-reward-weight", type=float, default=1.0)
    parser.add_argument("--cls-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        choices=[
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
            "norm_loss_tanh_reduction",
            "norm_loss_l0_delta",
            "norm_loss_clipped_l0_delta",
            "norm_loss_tanh_l0_delta",
        ],
        default="raw_mse_log_delta",
    )
    parser.add_argument("--reward-eps", type=float, default=1e-6)
    parser.add_argument("--reward-log-clip", type=float, default=1.0)
    parser.add_argument("--reward-l0-clip", type=float, default=1.0)
    parser.add_argument("--reward-tanh-scale", type=float, default=1.0)
    parser.add_argument("--canvit-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewpoint-entropy-bins", type=int, default=8)
    parser.add_argument("--eval-images", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-subset-seed", type=int, default=10042)
    parser.add_argument("--actor-reward-percentile-grid-size", type=int, default=11)
    parser.add_argument(
        "--actor-reward-percentile-scales",
        type=str,
        default="0.30,0.40,0.50,0.65,0.80",
    )
    parser.add_argument("--reward-map-chunk-size", type=int, default=16)
    parser.add_argument(
        "--comet-step",
        type=int,
        default=None,
        help="Metric step for Comet. Defaults to eval_images * (t + 1) glimpses.",
    )
    parser.add_argument(
        "--comet-steps",
        type=str,
        default=None,
        help=(
            "Optional comma-separated Comet steps for drawing a flat baseline "
            "reference on the same eval metric plots."
        ),
    )
    add_dense_sac_comet_args(parser)
    args = parser.parse_args()
    if args.max_history < args.t + 1:
        raise ValueError("--max-history must be at least --t + 1.")
    if args.eval_images < 1:
        raise ValueError("--eval-images must be positive for baseline eval.")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be non-negative.")
    if args.actor_reward_percentile_grid_size < 0:
        raise ValueError("--actor-reward-percentile-grid-size must be non-negative.")
    if args.actor_reward_percentile_grid_size == 1:
        raise ValueError("--actor-reward-percentile-grid-size must be 0 or >= 2.")
    if args.reward_map_chunk_size < 1:
        raise ValueError("--reward-map-chunk-size must be positive.")
    if args.viewpoint_entropy_bins < 1:
        raise ValueError("--viewpoint-entropy-bins must be positive.")
    if args.reward_log_clip <= 0.0:
        raise ValueError("--reward-log-clip must be positive.")
    if args.reward_l0_clip <= 0.0:
        raise ValueError("--reward-l0-clip must be positive.")
    if args.reward_tanh_scale <= 0.0:
        raise ValueError("--reward-tanh-scale must be positive.")
    if args.subset_shards < 1:
        raise ValueError("--subset-shards must be positive.")
    if args.policy == "egc2f" and args.t + 1 > 21:
        raise ValueError("EG-C2F has 21 built-in timesteps; require --t <= 20.")
    if args.policy == "egc2f" and args.min_scale > 0.25:
        raise ValueError(
            "EG-C2F includes quarter-scale tiles; require --min-scale <= 0.25 "
            "when using the SAC-compatible action adapter."
        )
    parse_reward_map_scales(args.actor_reward_percentile_scales)
    if args.paired_hidden_feature_base_dir is not None:
        if args.feature_image_root is None:
            raise ValueError(
                "Paired shard loading currently requires --feature-image-root "
                "for oracle/non-t0 glimpse pixels."
            )
        if (args.paired_hidden_feature_image_root is None) == (
            args.paired_hidden_tar_dir is None
        ):
            raise ValueError(
                "Exactly one of --paired-hidden-feature-image-root or "
                "--paired-hidden-tar-dir is required with "
                "--paired-hidden-feature-base-dir."
            )
    return args


def build_baseline_policy(
    args: argparse.Namespace,
    *,
    canvas_grid: int,
    device: torch.device,
) -> nn.Module:
    if args.policy == "random":
        return RandomDensePolicy(min_scale=args.min_scale)
    return DenseEntropyCoarseToFinePolicy(
        min_scale=args.min_scale,
        canvas_grid=canvas_grid,
        device=device,
    )


def parse_comet_steps(args: argparse.Namespace) -> list[int]:
    if args.comet_steps:
        steps = [int(item) for item in args.comet_steps.split(",") if item.strip()]
        if not steps or any(step < 0 for step in steps):
            raise ValueError("--comet-steps must contain non-negative integers.")
        return steps
    default_step = (
        int(args.comet_step)
        if args.comet_step is not None
        else int(args.eval_images) * int(args.t + 1)
    )
    return [default_step]


def main() -> None:
    args = parse_args()
    args.batches = 1
    args.canvas_entropy_state = args.policy == "egc2f"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg, _modules = build_pretrain_config(args)
    device = cfg.device
    eval_loader = build_dense_eval_loader(args, cfg)
    model, glimpse_size_px = load_frozen_hf_model(args, cfg)
    canvit_dtype = resolve_canvit_dtype(args.canvit_dtype, device)
    model.to(device=device, dtype=canvit_dtype)
    for module in model.modules():
        if module.__class__.__name__ == "VPEEncoder":
            module.to(device=device, dtype=torch.float32)

    grid_size = cfg.canvas_patch_grid_size
    cls_norm, scene_norm = model.standardizers(grid_size)
    if args.reset_normalizer or not scene_norm.initialized:
        from canvit_rl.pretrain_IN21k.dense_train_batch import (
            init_normalizer_stats_from_shard,
        )

        shards_dir = (
            cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
        )
        init_normalizer_stats_from_shard(
            shards_dir=shards_dir,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            device=device,
            max_samples=args.normalizer_max_samples,
        )

    policy = build_baseline_policy(args, canvas_grid=grid_size, device=device).to(device)
    metrics = evaluate_dense_sac(
        args=args,
        eval_loader=eval_loader,
        actor=policy,
        model=model,
        scene_norm=scene_norm,
        cls_norm=cls_norm,
        canvas_grid_size=grid_size,
        glimpse_size_px=glimpse_size_px,
        canvit_dtype=canvit_dtype,
        device=device,
        compute_actor_reward_percentiles=args.actor_reward_percentile_grid_size > 0,
    )
    metrics["eval/policy_baseline_id"] = float(0 if args.policy == "random" else 1)
    comet_exp = make_dense_comet_experiment(args)
    if comet_exp is not None:
        for comet_step in parse_comet_steps(args):
            comet_exp.log_metrics(metrics, step=comet_step)
        comet_exp.end()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
