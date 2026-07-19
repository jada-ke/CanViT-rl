"""Train Canvas SAC policies on IN21k DINOv3 dense-feature distillation rewards.

The data path is owned by CanViT-pretrain's shard loader. This script only
wraps that batch contract in the existing Canvas SAC actor/critic/replay code.

Example:
    uv run python scripts/train_i21k_dense_sac.py \
        --feature-base-dir datasets/mnist_glimpse_dense_oracle/features \
        --feature-image-root datasets/mnist_glimpse_export/oracle \
        --paired-hidden-feature-base-dir datasets/mnist_glimpse_dense_hidden/features \
        --paired-hidden-feature-image-root datasets/mnist_glimpse_export/hidden \
        --model-repo canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02 \
        --batches 1000 --batch-size 8 --t 4 --no-comet

    uv run python scripts/train_i21k_dense_sac.py \
        --feature-base-dir datasets/imagenet_ood/features \
        --feature-image-root datasets/imagenet_ood/images \
        --model-repo "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02" \
        --subset-size 4 \
        --subset-seed 42 \
        --subset-shards 1 \
        --batch-size 4 \
        --batches 201 \
        --debug-viz-dir results/i21k_dense_debug \
        --debug-viz-images 4 \
        --t 2 \
        --critic-local-action-features \
        --canvas-entropy-state \
        --disable-canvas-max-pool \
        --reward-mode raw_mse_l0_delta \
        --reward-map-interval 200 \
        --reward-map-images 4 \
        --comet \
        --eval-interval 50 \
        --eval-images 4 \
        --eval-subset-seed 42 \
        --eval-batch-size 4 

"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
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
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from tqdm import tqdm

from canvit_rl.canvas.sac import (
    REPLAY_STORAGE_DTYPE,
    CanvasReplayBuffer,
    CanvasSAC,
    replay_canvas_bytes,
    resolve_replay_device,
    should_pin_replay_memory,
    validate_replay_memory,
)
from canvit_rl.canvas.eval import viewpoint_entropy
from canvit_rl.canvas.state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    empty_viewpoint_history,
)
from canvit_rl.canvit_precision import resolve_canvit_dtype
from canvit_rl.env import get_device
from canvit_rl.greedy import _index_state_batch, _repeat_state_chunks
from canvit_rl.pretrain_IN21k.checkpoints import (
    load_dense_sac_resume,
    save_dense_sac_checkpoint,
)
from canvit_rl.pretrain_IN21k.dense_train_batch import (
    DenseTrainBatch,
    FixedDenseSubsetLoader,
    PairedDenseShardLoader,
    apply_dense_feature_config,
    dense_glimpse_images,
    init_normalizer_stats_from_shard,
    load_dense_train_batch,
)
from canvit_rl.pretrain_IN21k.pretrain_modules import load_pretrain_modules
from canvit_rl.pretrain_IN21k.reward import (
    DenseDistillationMetrics,
    dense_distillation_metrics,
    dense_reward,
)
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic
from canvit_rl.viewpoint_policy import action_to_viewpoint, viewpoint_to_action

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def add_dense_sac_comet_args(parser: argparse.ArgumentParser) -> None:
    """Register Comet flags while keeping experiment creation opt-in."""
    parser.add_argument("--comet-log-interval", type=int, default=50)
    parser.add_argument("--no-comet", action="store_true", default=True)
    parser.add_argument("--comet", dest="no_comet", action="store_false")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="i21k-dense-sac")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="i21k-dense-sac")


def parse_reward_map_scales(value: str) -> list[float]:
    """Parse comma-separated dense reward-map scales."""
    scales = [float(item) for item in value.split(",") if item.strip()]
    if not scales or any(scale <= 0.0 or scale > 1.0 for scale in scales):
        raise ValueError("--reward-map-scales must contain values in (0, 1].")
    return scales


def make_dense_comet_experiment(args: argparse.Namespace):
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
    experiment.set_name(args.experiment_name or "i21k-dense-sac")
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    return experiment


def parse_args() -> argparse.Namespace:
    """Parse dense IN21k SAC training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-base-dir", type=Path, required=True)
    parser.add_argument("--feature-image-root", type=Path, default=None)
    parser.add_argument("--tar-dir", type=Path, default=None)
    parser.add_argument(
        "--paired-hidden-feature-base-dir",
        type=Path,
        default=None,
        help=(
            "Optional feature shard base for paired active-view data. Targets "
            "come from --feature-base-dir, while t0 images come from this "
            "hidden source."
        ),
    )
    parser.add_argument("--paired-hidden-feature-image-root", type=Path, default=None)
    parser.add_argument("--paired-hidden-tar-dir", type=Path, default=None)
    parser.add_argument(
        "--model-repo",
        type=str,
        default="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02",
        help="Hugging Face CanViT pretraining repo used as the frozen backbone.",
    )
    parser.add_argument("--reset-normalizer", action="store_true")
    parser.add_argument("--normalizer-max-samples", type=int, default=0)
    parser.add_argument("--teacher-name", type=str, default="dinov3_vitb16")
    parser.add_argument("--scene-resolution", type=int, default=512)
    parser.add_argument("--glimpse-grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=0,
        help=(
            "If >0, materialize a deterministic random subset from dense "
            "shards and train only on that subset."
        ),
    )
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--subset-shards",
        type=int,
        default=1,
        help="Number of shard files to sample candidates from for --subset-size.",
    )
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=5)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--scene-reward-weight", type=float, default=1.0)
    parser.add_argument("--cls-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        choices=[
            "raw_mse_delta",
            "raw_mse_reduction",
            "raw_mse_l0_delta",
            "norm_loss_delta",
            "norm_loss_reduction",
            "norm_loss_l0_delta",
        ],
        default="raw_mse_l0_delta",
    )
    parser.add_argument("--reward-eps", type=float, default=1e-6)
    parser.add_argument("--canvit-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--rff-dim", type=int, default=128)
    parser.add_argument("--rff-seed", type=int, default=42)
    parser.add_argument("--critic-local-action-features", action="store_true")
    parser.add_argument(
        "--canvas-entropy-state",
        action="store_true",
        help=(
            "Append a normalized dense-feature reconstruction-error map to "
            "the CanvasStateActor/Critic state under the existing entropy key."
        ),
    )
    parser.add_argument("--viewpoint-entropy-bins", type=int, default=8)
    parser.add_argument("--disable-canvas-avg-pool", action="store_true")
    parser.add_argument("--disable-canvas-max-pool", action="store_true")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--target-entropy", type=float, default=-3.0)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--learning-starts", type=int, default=1)
    parser.add_argument("--updates-per-batch", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-images", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-subset-seed", type=int, default=10042)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/i21k_dense_sac"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--debug-viz-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG rollout overlays from early train batches.",
    )
    parser.add_argument("--debug-viz-images", type=int, default=0)
    parser.add_argument("--debug-viz-batches", type=int, default=1)
    parser.add_argument(
        "--reward-map-images",
        type=int,
        default=0,
        help="If >0, save dense true-reward vs critic-Q maps for this many batch images.",
    )
    parser.add_argument("--reward-map-grid-size", type=int, default=11)
    parser.add_argument("--reward-map-scales", type=str, default="0.25,0.50")
    parser.add_argument("--reward-map-chunk-size", type=int, default=16)
    parser.add_argument(
        "--reward-map-interval",
        type=int,
        default=None,
        help="Batch interval for dense reward maps. Defaults to --log-interval.",
    )
    parser.add_argument(
        "--reward-map-output-dir",
        type=Path,
        default=Path("results/i21k_dense_reward_maps"),
    )
    add_dense_sac_comet_args(parser)
    args = parser.parse_args()
    if args.max_history < args.t + 1:
        raise ValueError("--max-history must be at least --t + 1.")
    if args.disable_canvas_avg_pool and args.disable_canvas_max_pool:
        raise ValueError("At least one canvas pooling branch must remain enabled.")
    if args.debug_viz_images < 0 or args.debug_viz_batches < 0:
        raise ValueError("--debug-viz-images and --debug-viz-batches must be non-negative.")
    if args.reward_map_images < 0:
        raise ValueError("--reward-map-images must be non-negative.")
    if args.reward_map_grid_size < 2:
        raise ValueError("--reward-map-grid-size must be >= 2.")
    if args.reward_map_chunk_size < 1:
        raise ValueError("--reward-map-chunk-size must be positive.")
    if args.reward_map_interval is not None and args.reward_map_interval < 1:
        raise ValueError("--reward-map-interval must be positive.")
    if args.viewpoint_entropy_bins < 1:
        raise ValueError("--viewpoint-entropy-bins must be positive.")
    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be positive.")
    if args.eval_images < 0:
        raise ValueError("--eval-images must be non-negative.")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be non-negative.")
    parse_reward_map_scales(args.reward_map_scales)
    if args.subset_size < 0:
        raise ValueError("--subset-size must be non-negative.")
    if args.subset_shards < 1:
        raise ValueError("--subset-shards must be positive.")
    if args.paired_hidden_feature_base_dir is not None:
        if args.subset_size > 0:
            raise ValueError("--subset-size is not supported with paired shard loading.")
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


def _denormalized_uint8_image(image: torch.Tensor) -> np.ndarray:
    """Convert one preprocessed ImageNet tensor into an RGB uint8 image."""
    image = image.detach().cpu().float()
    restored = image * IMAGENET_STD + IMAGENET_MEAN
    restored = restored.clamp(0.0, 1.0)
    return (restored.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _viewpoint_box(
    viewpoint: Viewpoint,
    *,
    batch_idx: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Convert a CanViT Viewpoint into a clamped PIL rectangle."""
    cy, cx = viewpoint.centers[batch_idx].detach().cpu().float().tolist()
    scale = float(viewpoint.scales[batch_idx].detach().cpu().float().item())
    center_x = (cx + 1.0) * 0.5 * (width - 1)
    center_y = (cy + 1.0) * 0.5 * (height - 1)
    box_w = scale * (width - 1)
    box_h = scale * (height - 1)
    left = int(round(max(0.0, center_x - box_w * 0.5)))
    top = int(round(max(0.0, center_y - box_h * 0.5)))
    right = int(round(min(width - 1.0, center_x + box_w * 0.5)))
    bottom = int(round(min(height - 1.0, center_y + box_h * 0.5)))
    return left, top, right, bottom


def maybe_save_debug_rollout_viz(
    *,
    args: argparse.Namespace,
    comet_exp,
    update_count: int,
    images: torch.Tensor,
    viewpoints: list[Viewpoint],
    rewards_by_step: list[torch.Tensor],
    batch_idx: int,
    start_batch: int,
) -> None:
    """Save early rollout viewpoint overlays when debug visualization is enabled."""
    if (
        args.debug_viz_dir is None
        or args.debug_viz_images == 0
        or batch_idx - start_batch >= args.debug_viz_batches
    ):
        return
    from PIL import Image, ImageDraw

    args.debug_viz_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        "white",
        "red",
        "lime",
        "dodgerblue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
    ]
    max_images = min(args.debug_viz_images, images.shape[0])
    for sample_idx in range(max_images):
        pil_image = Image.fromarray(_denormalized_uint8_image(images[sample_idx]))
        draw = ImageDraw.Draw(pil_image)
        width, height = pil_image.size
        reward_text: list[str] = []
        for step_idx, viewpoint in enumerate(viewpoints):
            color = colors[step_idx % len(colors)]
            box = _viewpoint_box(
                viewpoint,
                batch_idx=sample_idx,
                height=height,
                width=width,
            )
            for inset in range(2):
                draw.rectangle(
                    (
                        box[0] + inset,
                        box[1] + inset,
                        box[2] - inset,
                        box[3] - inset,
                    ),
                    outline=color,
                )
            label = "full" if step_idx == 0 else f"t{step_idx}"
            draw.text((box[0] + 3, box[1] + 3), label, fill=color)
            if step_idx > 0 and step_idx - 1 < len(rewards_by_step):
                reward = float(rewards_by_step[step_idx - 1][sample_idx].detach().cpu())
                reward_text.append(f"{label}:{reward:+.3f}")
        if reward_text:
            draw.rectangle((0, 0, width, 14 * len(reward_text) + 4), fill=(0, 0, 0))
            for line_idx, text in enumerate(reward_text):
                draw.text((4, 2 + 14 * line_idx), text, fill="white")
        output = args.debug_viz_dir / f"batch_{batch_idx:06d}_sample_{sample_idx:03d}.png"
        pil_image.save(output)
        if comet_exp is not None and hasattr(comet_exp, "log_image"):
            comet_exp.log_image(
                str(output),
                name=f"debug/i21k_dense_rollout/{output.name}",
                step=update_count,
            )


def _candidate_grid(*, scale: float, grid_size: int, device: torch.device) -> Viewpoint:
    """Build an in-bounds y/x candidate grid for one scale."""
    bound = max(1.0 - scale, 0.0)
    values = torch.linspace(-bound, bound, grid_size, device=device)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    centers = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    scales = torch.full((centers.shape[0],), float(scale), device=device)
    return Viewpoint(centers=centers, scales=scales)


def _slice_dense_batch(batch: DenseTrainBatch, sample_idx: int) -> DenseTrainBatch:
    """Return a one-sample dense batch while preserving all target fields."""
    return DenseTrainBatch(
        images=batch.images[sample_idx : sample_idx + 1],
        labels=batch.labels[sample_idx : sample_idx + 1],
        scene_target=batch.scene_target[sample_idx : sample_idx + 1],
        cls_target=batch.cls_target[sample_idx : sample_idx + 1],
        raw_scene_target=batch.raw_scene_target[sample_idx : sample_idx + 1],
        raw_cls_target=batch.raw_cls_target[sample_idx : sample_idx + 1],
        glimpse_images=(
            None
            if batch.glimpse_images is None
            else batch.glimpse_images[sample_idx : sample_idx + 1]
        ),
    )


def _repeat_dense_batch(batch: DenseTrainBatch, repeats: int) -> DenseTrainBatch:
    """Repeat a one-sample dense batch to match candidate-grid chunks."""
    return DenseTrainBatch(
        images=batch.images.repeat((repeats,) + (1,) * (batch.images.ndim - 1)),
        labels=batch.labels.repeat(repeats),
        scene_target=batch.scene_target.repeat(
            (repeats,) + (1,) * (batch.scene_target.ndim - 1)
        ),
        cls_target=batch.cls_target.repeat(
            (repeats,) + (1,) * (batch.cls_target.ndim - 1)
        ),
        raw_scene_target=batch.raw_scene_target.repeat(
            (repeats,) + (1,) * (batch.raw_scene_target.ndim - 1)
        ),
        raw_cls_target=batch.raw_cls_target.repeat(
            (repeats,) + (1,) * (batch.raw_cls_target.ndim - 1)
        ),
        glimpse_images=(
            None
            if batch.glimpse_images is None
            else batch.glimpse_images.repeat(
                (repeats,) + (1,) * (batch.glimpse_images.ndim - 1)
            )
        ),
    )


def _slice_dense_metrics(
    metrics: DenseDistillationMetrics,
    sample_idx: int,
) -> DenseDistillationMetrics:
    """Keep one sample's before-metrics for dense reward dispatch."""
    return DenseDistillationMetrics(
        scene_loss_norm=metrics.scene_loss_norm[sample_idx : sample_idx + 1],
        cls_loss_norm=metrics.cls_loss_norm[sample_idx : sample_idx + 1],
        loss_norm=metrics.loss_norm[sample_idx : sample_idx + 1],
        scene_loss_raw=metrics.scene_loss_raw[sample_idx : sample_idx + 1],
        cls_loss_raw=metrics.cls_loss_raw[sample_idx : sample_idx + 1],
        loss_raw=metrics.loss_raw[sample_idx : sample_idx + 1],
    )


def _episode_l0_for_reward_mode(
    *,
    mode: str,
    metrics: DenseDistillationMetrics,
) -> torch.Tensor:
    """Return the reset denominator in the same space as the reward mode.

    Problem: raw and normalized dense losses live on different scales, so
    sharing one l0 tensor across both l0-delta modes would silently corrupt one
    reward. Solution: choose loss_raw for raw-space modes and loss_norm for
    normalized-space modes at reset. Result: l0 remains fixed per episode and
    always matches the delta being divided.
    """
    if mode.startswith("norm_loss"):
        return metrics.loss_norm.detach().clone()
    return metrics.loss_raw.detach().clone()


def _eval_display_loss_for_reward_mode(mode: str) -> tuple[str, str]:
    """
    Return the eval final-loss metric aligned with the reward's loss space.
    """
    if mode.startswith("norm_loss"):
        return "final_loss_norm", "eval/final_loss_norm"
    return "final_loss_raw", "eval/final_loss_raw"


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation for flattened maps, or nan for constant maps."""
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")
    x_flat = x[finite].reshape(-1)
    y_flat = y[finite].reshape(-1)
    if np.std(x_flat) == 0 or np.std(y_flat) == 0:
        return float("nan")
    return float(np.corrcoef(x_flat, y_flat)[0, 1])


def _finite_map_max(values: np.ndarray) -> float:
    """Return the finite map maximum, or nan without triggering all-NaN warnings."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(finite_values.max())


def _show_reward_map_background(ax, image_np: np.ndarray) -> None:
    """Draw an RGB image behind dense reward maps in Viewpoint coordinates."""
    ax.imshow(image_np, extent=[-1.0, 1.0, 1.0, -1.0])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(1.0, -1.0)


def dense_canvas_entropy_map(
    *,
    model,
    state,
    batch: DenseTrainBatch,
    canvas_grid_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return normalized dense-feature error as [B, 1, G, G] uncertainty."""
    scene_pred = model.predict_teacher_scene(state.canvas).float()
    scene_target = batch.scene_target.float()
    per_patch_error = (scene_pred - scene_target).pow(2).mean(dim=-1)
    # Problem: IN21k dense SAC has no segmentation probe entropy Solution: feed a
    # per-image min/max-normalized teacher-feature reconstruction error map
    # through that branch. Result: the actor/critics receive spatial
    # uncertainty
    error_map = per_patch_error.reshape(
        per_patch_error.shape[0],
        1,
        canvas_grid_size,
        canvas_grid_size,
    )
    flat = error_map.flatten(1)
    min_val = flat.min(dim=1).values[:, None, None, None]
    max_val = flat.max(dim=1).values[:, None, None, None]
    return ((error_map - min_val) / (max_val - min_val).clamp_min(eps)).contiguous()


def _evaluate_dense_reward_grid(
    *,
    args: argparse.Namespace,
    image: torch.Tensor,
    batch: DenseTrainBatch,
    state,
    before_metrics: DenseDistillationMetrics,
    l0: torch.Tensor,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    canvas_summary: torch.Tensor,
    canvas_entropy: torch.Tensor | None,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    model,
    scene_denorm,
    cls_denorm,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate true dense reward and critic Q for a center grid."""
    vp_all = _candidate_grid(
        scale=scale,
        grid_size=args.reward_map_grid_size,
        device=image.device,
    )
    rewards: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    total = vp_all.centers.shape[0]
    with torch.inference_mode():
        for start in range(0, total, args.reward_map_chunk_size):
            stop = min(start + args.reward_map_chunk_size, total)
            repeats = stop - start
            vp = Viewpoint(
                centers=vp_all.centers[start:stop],
                scales=vp_all.scales[start:stop],
            )
            candidate_batch = _repeat_dense_batch(batch, repeats)
            candidate_state = _repeat_state_chunks(state, repeats)
            glimpse = sample_at_viewpoint(
                spatial=dense_glimpse_images(candidate_batch),
                viewpoint=vp,
                glimpse_size_px=glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(glimpse=glimpse, state=candidate_state, viewpoint=vp)
            after_metrics = dense_distillation_metrics(
                model=model,
                state=out.state,
                batch=candidate_batch,
                scene_denorm=scene_denorm,
                cls_denorm=cls_denorm,
                scene_weight=args.scene_reward_weight,
                cls_weight=args.cls_reward_weight,
            )

            reward = dense_reward(
                mode=args.reward_mode,
                before=DenseDistillationMetrics(
                    scene_loss_norm=before_metrics.scene_loss_norm.expand(repeats),
                    cls_loss_norm=before_metrics.cls_loss_norm.expand(repeats),
                    loss_norm=before_metrics.loss_norm.expand(repeats),
                    scene_loss_raw=before_metrics.scene_loss_raw.expand(repeats),
                    cls_loss_raw=before_metrics.cls_loss_raw.expand(repeats),
                    loss_raw=before_metrics.loss_raw.expand(repeats),
                ),
                after=after_metrics,
                l0=l0.expand(repeats),
                eps=args.reward_eps,
            )
            critic_batch = {
                "canvas": canvas_summary.repeat(repeats, 1, 1, 1),
                "coords": coords.repeat(repeats, 1, 1),
                "lengths": lengths.repeat(repeats),
            }
            if canvas_entropy is not None:
                critic_batch["entropy"] = canvas_entropy.repeat(repeats, 1, 1, 1)
            action = viewpoint_to_action(vp, min_scale=args.min_scale)
            q_pred = torch.minimum(q1(critic_batch, action), q2(critic_batch, action))
            rewards.append(reward.detach().cpu().reshape(-1))
            q_values.append(q_pred.detach().cpu().reshape(-1))
    reward_map = torch.cat(rewards).view(
        args.reward_map_grid_size,
        args.reward_map_grid_size,
    )
    q_map = torch.cat(q_values).view(args.reward_map_grid_size, args.reward_map_grid_size)
    return reward_map.numpy(), q_map.numpy()


def maybe_save_dense_reward_maps(
    *,
    args: argparse.Namespace,
    comet_exp,
    update_count: int,
    batch_idx: int,
    batch: DenseTrainBatch,
    state,
    current_metrics: DenseDistillationMetrics,
    l0: torch.Tensor,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    canvas_summary: torch.Tensor,
    canvas_entropy: torch.Tensor | None,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    model,
    scene_denorm,
    cls_denorm,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
) -> Path | None:
    """Save dense true-reward/Q heatmaps for the current warmup state."""
    if args.reward_map_images <= 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to save dense reward-map figures.") from exc

    args.reward_map_output_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_reward_map_scales(args.reward_map_scales)
    max_images = min(args.reward_map_images, batch.images.shape[0])
    n_cols = 1 + 2 * len(scales)
    fig, axes = plt.subplots(
        max_images,
        n_cols,
        figsize=(3.4 * n_cols, max(3.2 * max_images, 4.0)),
        dpi=150,
        squeeze=False,
    )
    for sample_idx in range(max_images):
        sample_index = torch.tensor([sample_idx], device=batch.images.device)
        sample_batch = _slice_dense_batch(batch, sample_idx)
        sample_state = _index_state_batch(state, sample_index)
        sample_coords = coords[sample_idx : sample_idx + 1]
        sample_lengths = lengths[sample_idx : sample_idx + 1]
        sample_canvas = canvas_summary[sample_idx : sample_idx + 1]
        sample_entropy = (
            None
            if canvas_entropy is None
            else canvas_entropy[sample_idx : sample_idx + 1]
        )
        sample_metrics = _slice_dense_metrics(current_metrics, sample_idx)
        sample_l0 = l0[sample_idx : sample_idx + 1]
        with torch.inference_mode():
            actor_batch = {
                "canvas": sample_canvas,
                "coords": sample_coords,
                "lengths": sample_lengths,
            }
            if sample_entropy is not None:
                actor_batch["entropy"] = sample_entropy
            actor_action = actor.deterministic_action(actor_batch)
        actor_vp = action_to_viewpoint(actor_action, min_scale=args.min_scale)
        image_np = _denormalized_uint8_image(sample_batch.images[0])
        axes[sample_idx, 0].imshow(image_np)
        axes[sample_idx, 0].set_title(
            f"batch={batch_idx} sample={sample_idx}\n"
            f"{args.reward_mode} loss={float(sample_metrics.loss_norm.item()):.4f}"
        )
        axes[sample_idx, 0].axis("off")
        for scale_idx, scale in enumerate(scales):
            reward_map, q_map = _evaluate_dense_reward_grid(
                args=args,
                image=sample_batch.images,
                batch=sample_batch,
                state=sample_state,
                before_metrics=sample_metrics,
                l0=sample_l0,
                coords=sample_coords,
                lengths=sample_lengths,
                canvas_summary=sample_canvas,
                canvas_entropy=sample_entropy,
                q1=q1,
                q2=q2,
                model=model,
                scene_denorm=scene_denorm,
                cls_denorm=cls_denorm,
                glimpse_size_px=glimpse_size_px,
                canvit_dtype=canvit_dtype,
                scale=scale,
            )
            bound = max(1.0 - scale, 0.0)
            extent = [-bound, bound, bound, -bound]
            reward_ax = axes[sample_idx, 1 + 2 * scale_idx]
            q_ax = axes[sample_idx, 2 + 2 * scale_idx]
            _show_reward_map_background(reward_ax, image_np)
            reward_im = reward_ax.imshow(
                reward_map,
                origin="upper",
                extent=extent,
                cmap="coolwarm",
                alpha=0.58,
            )
            reward_ax.set_title(
                f"scale={scale:.2f} true reward\n"
                f"max={_finite_map_max(reward_map):+.4f}"
            )
            fig.colorbar(reward_im, ax=reward_ax, fraction=0.046, pad=0.04)
            _show_reward_map_background(q_ax, image_np)
            q_im = q_ax.imshow(
                q_map,
                origin="upper",
                extent=extent,
                cmap="coolwarm",
                alpha=0.58,
            )
            q_ax.set_title(
                f"scale={scale:.2f} predicted Q\ncorr={_corr(reward_map, q_map):+.3f}"
            )
            fig.colorbar(q_im, ax=q_ax, fraction=0.046, pad=0.04)
            for ax in (reward_ax, q_ax):
                ax.scatter(
                    [float(actor_vp.centers[0, 1].detach().cpu().item())],
                    [float(actor_vp.centers[0, 0].detach().cpu().item())],
                    c="black",
                    s=38,
                    marker="x",
                    linewidths=1.8,
                )
                ax.set_xlabel("x center")
                ax.set_ylabel("y center")
    fig.suptitle(f"IN21k dense SAC reward maps ({args.reward_mode}) update={update_count}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output = args.reward_map_output_dir / (
        f"dense_reward_maps_batch_{batch_idx:06d}_update_{update_count:06d}.png"
    )
    fig.savefig(output)
    plt.close(fig)
    if comet_exp is not None and hasattr(comet_exp, "log_image"):
        comet_exp.log_image(
            str(output),
            name=f"reward_maps/i21k_dense/{output.name}",
            step=update_count,
        )
    return output


def maybe_save_dense_policy_glimpses(
    *,
    args: argparse.Namespace,
    comet_exp,
    update_count: int,
    batch_idx: int,
    batch: DenseTrainBatch,
    state,
    current_metrics: DenseDistillationMetrics,
    l0: torch.Tensor,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    canvas_summary: torch.Tensor,
    canvas_entropy: torch.Tensor | None,
    actor: CanvasStateActor,
    model,
    scene_denorm,
    cls_denorm,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    canvas_grid_size: int,
) -> Path | None:
    """Save deterministic policy glimpse overlays at the reward-map cadence."""
    if args.reward_map_images <= 0 or args.t <= 0:
        return None
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to save dense glimpse figures.") from exc

    args.reward_map_output_dir.mkdir(parents=True, exist_ok=True)
    max_images = min(args.reward_map_images, batch.images.shape[0])
    n_steps = args.t + 1
    n_cols = n_steps + 1
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n_steps))
    fig, axes = plt.subplots(
        max_images,
        n_cols,
        figsize=(4.0 * n_cols, max(3.2 * max_images, 4.0)),
        dpi=150,
        squeeze=False,
    )
    was_training = actor.training
    actor.eval()
    try:
        with torch.inference_mode():
            for sample_idx in range(max_images):
                sample_index = torch.tensor([sample_idx], device=batch.images.device)
                sample_batch = _slice_dense_batch(batch, sample_idx)
                sample_state = _index_state_batch(state, sample_index)
                sample_coords = coords[sample_idx : sample_idx + 1].clone()
                sample_lengths = lengths[sample_idx : sample_idx + 1].clone()
                sample_canvas = canvas_summary[sample_idx : sample_idx + 1]
                sample_entropy = (
                    None
                    if canvas_entropy is None
                    else canvas_entropy[sample_idx : sample_idx + 1]
                )
                sample_metrics = _slice_dense_metrics(current_metrics, sample_idx)
                sample_l0 = l0[sample_idx : sample_idx + 1]
                image_np = _denormalized_uint8_image(sample_batch.images[0])
                height, width = image_np.shape[:2]
                full_vp = Viewpoint.full_scene(batch_size=1, device=batch.images.device)
                # Problem: the dense actor visualizer showed zoomed crop
                # panels, unlike the ADE Canvas SAC policy contact sheet.
                # Solution: render every timestep on the full source image,
                # plus one combined overlay column that shows all rectangles.
                # Result: dense SAC policy diagnostics use the same scan
                # pattern as ADE20K while making overlap easy to inspect.
                viewpoints = [full_vp]
                titles = [
                    f"batch={batch_idx} sample={sample_idx} t0\n"
                    f"loss={float(sample_metrics.loss_norm[0].detach().cpu()):.4f}"
                ]
                for actor_step_idx in range(args.t):
                    obs = {
                        "canvas": sample_canvas,
                        "coords": sample_coords,
                        "lengths": sample_lengths,
                    }
                    if sample_entropy is not None:
                        obs["entropy"] = sample_entropy
                    action = actor.deterministic_action(obs)
                    vp = action_to_viewpoint(action, min_scale=args.min_scale)
                    glimpse = sample_at_viewpoint(
                        spatial=dense_glimpse_images(sample_batch),
                        viewpoint=vp,
                        glimpse_size_px=glimpse_size_px,
                    )
                    out = model(
                        glimpse=glimpse.to(dtype=canvit_dtype),
                        state=sample_state,
                        viewpoint=vp,
                    )
                    next_metrics = dense_distillation_metrics(
                        model=model,
                        state=out.state,
                        batch=sample_batch,
                        scene_denorm=scene_denorm,
                        cls_denorm=cls_denorm,
                        scene_weight=args.scene_reward_weight,
                        cls_weight=args.cls_reward_weight,
                    )
                    reward = dense_reward(
                        mode=args.reward_mode,
                        before=sample_metrics,
                        after=next_metrics,
                        l0=sample_l0,
                        eps=args.reward_eps,
                    )
                    center = vp.centers[0].detach().cpu().tolist()
                    viewpoints.append(vp)
                    titles.append(
                        f"batch={batch_idx} sample={sample_idx} t{actor_step_idx + 1}\n"
                        f"s={float(vp.scales[0].detach().cpu()):.2f} "
                        f"c=({center[0]:+.2f},{center[1]:+.2f})\n"
                        f"reward={float(reward[0].detach().cpu()):+.4f}"
                    )
                    sample_state = out.state
                    sample_metrics = next_metrics
                    sample_canvas = canvas_layernorm_spatial(
                        model=model,
                        state=sample_state,
                        canvas_grid_size=canvas_grid_size,
                    )
                    sample_entropy = (
                        dense_canvas_entropy_map(
                            model=model,
                            state=sample_state,
                            batch=sample_batch,
                            canvas_grid_size=canvas_grid_size,
                        )
                        if args.canvas_entropy_state
                        else None
                    )
                    sample_coords, sample_lengths = append_viewpoint_history(
                        coords=sample_coords,
                        lengths=sample_lengths,
                        viewpoint=vp,
                        step=actor_step_idx + 1,
                    )
                overview_ax = axes[sample_idx, 0]
                overview_ax.imshow(image_np)
                for step_idx, vp in enumerate(viewpoints):
                    left, top, right, bottom = _viewpoint_box(
                        vp,
                        batch_idx=0,
                        height=height,
                        width=width,
                    )
                    rect = patches.Rectangle(
                        (left, top),
                        max(right - left, 1),
                        max(bottom - top, 1),
                        linewidth=2.5,
                        edgecolor=colors[step_idx],
                        facecolor="none",
                    )
                    overview_ax.add_patch(rect)
                    overview_ax.text(
                        left + 3,
                        top + 12,
                        f"t{step_idx}",
                        color=colors[step_idx],
                        fontsize=9,
                        weight="bold",
                    )
                overview_ax.set_title(
                    f"batch={batch_idx} sample={sample_idx}\nall viewpoints"
                )
                overview_ax.axis("off")
                for step_idx, vp in enumerate(viewpoints):
                    ax = axes[sample_idx, step_idx + 1]
                    ax.imshow(image_np)
                    left, top, right, bottom = _viewpoint_box(
                        vp,
                        batch_idx=0,
                        height=height,
                        width=width,
                    )
                    rect = patches.Rectangle(
                        (left, top),
                        max(right - left, 1),
                        max(bottom - top, 1),
                        linewidth=3.0,
                        edgecolor=colors[step_idx],
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.set_title(titles[step_idx])
                    ax.axis("off")
    finally:
        if was_training:
            actor.train()
    fig.suptitle(f"IN21k dense SAC policy glimpses update={update_count}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output = args.reward_map_output_dir / (
        f"dense_policy_glimpses_batch_{batch_idx:06d}_update_{update_count:06d}.png"
    )
    fig.savefig(output)
    plt.close(fig)
    if comet_exp is not None and hasattr(comet_exp, "log_image"):
        comet_exp.log_image(
            str(output),
            name=f"glimpses/i21k_dense/{output.name}",
            step=update_count,
        )
    return output


def build_pretrain_config(args: argparse.Namespace):
    """Create a CanViT-pretrain Config using only dense-feature training fields."""
    modules = load_pretrain_modules()
    cfg = modules.Config()
    apply_dense_feature_config(
        cfg,
        feature_base_dir=args.feature_base_dir,
        feature_image_root=args.feature_image_root,
        tar_dir=args.tar_dir,
    )
    cfg.teacher_name = args.teacher_name
    cfg.scene_resolution = args.scene_resolution
    cfg.glimpse_grid_size = args.glimpse_grid_size
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.steps_per_job = args.batches
    cfg.normalizer_max_samples = args.normalizer_max_samples
    cfg.reset_normalizer = args.reset_normalizer
    cfg.device = get_device()
    return cfg, modules


def build_dense_loader(args: argparse.Namespace, cfg, modules):
    """Create CanViT-pretrain's shard loader without constructing a val loader."""
    shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
    if args.paired_hidden_feature_base_dir is not None:
        hidden_shards_dir = (
            args.paired_hidden_feature_base_dir
            / cfg.teacher_name
            / str(cfg.scene_resolution)
            / "shards"
        )
        # Problem: paired MNIST active-view training needs t0 pixels from the
        # hidden shard source but dense distillation targets from the oracle
        # shard source. Solution: use a paired loader that joins hidden/oracle
        # rows by staged sample basename instead of relying on shard row order.
        # Result: each rollout sees a hidden warmup image and oracle-target
        # reward for the same sample even after independent parquet shuffles.
        return PairedDenseShardLoader(
            hidden_shards_dir=hidden_shards_dir,
            target_shards_dir=shards_dir,
            image_size=cfg.scene_resolution,
            batch_size=args.batch_size,
            hidden_image_root=args.paired_hidden_feature_image_root,
            hidden_tar_dir=args.paired_hidden_tar_dir,
            target_image_root=cfg.feature_image_root,
            shuffle_seed=args.seed,
        )
    if args.subset_size > 0:
        return FixedDenseSubsetLoader(
            shards_dir=shards_dir,
            image_size=cfg.scene_resolution,
            batch_size=args.batch_size,
            subset_size=args.subset_size,
            subset_seed=args.subset_seed,
            subset_shards=args.subset_shards,
            image_root=cfg.feature_image_root,
            tar_dir=cfg.tar_dir,
        )
    return modules.ShardedFeatureLoader(
        shards_dir=shards_dir,
        image_size=cfg.scene_resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        start_step=0,
        image_root=cfg.feature_image_root,
        tar_dir=cfg.tar_dir,
        steps_per_job=args.batches,
    )


def build_dense_eval_loader(args: argparse.Namespace, cfg):
    """Create a deterministic dense-feature eval subset from the shard source."""
    if args.eval_images <= 0:
        return None
    shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
    eval_batch_size = args.eval_batch_size or args.batch_size
    eval_batch_size = min(eval_batch_size, max(args.eval_images, 1))
    if args.paired_hidden_feature_base_dir is not None:
        hidden_shards_dir = (
            args.paired_hidden_feature_base_dir
            / cfg.teacher_name
            / str(cfg.scene_resolution)
            / "shards"
        )
        return PairedDenseShardLoader(
            hidden_shards_dir=hidden_shards_dir,
            target_shards_dir=shards_dir,
            image_size=cfg.scene_resolution,
            batch_size=eval_batch_size,
            hidden_image_root=args.paired_hidden_feature_image_root,
            hidden_tar_dir=args.paired_hidden_tar_dir,
            target_image_root=cfg.feature_image_root,
            shuffle_seed=args.eval_subset_seed,
        )
    # Problem: dense IN21k SAC previously had no validation pass, only online
    # train-batch rewards. Solution: materialize a fixed eval subset with its
    # own seed through the same image/feature loading path. Result: metrics are
    # stable across intervals, though they are only a true held-out validation
    # split when the configured shard/image source itself is held out.
    return FixedDenseSubsetLoader(
        shards_dir=shards_dir,
        image_size=cfg.scene_resolution,
        batch_size=eval_batch_size,
        subset_size=args.eval_images,
        subset_seed=args.eval_subset_seed,
        subset_shards=args.subset_shards,
        image_root=cfg.feature_image_root,
        tar_dir=cfg.tar_dir,
    )


def load_frozen_hf_model(args: argparse.Namespace, cfg):
    """Load the frozen CanViT pretraining model from Hugging Face."""
    model = CanViTForPretrainingHFHub.from_pretrained(args.model_repo).to(cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch_size_px = model.backbone.patch_size_px
    glimpse_size_px = cfg.glimpse_grid_size * patch_size_px
    cfg.model = model.cfg
    cfg.canvas_patch_grid_size = model.canvas_patch_grid_sizes[0]
    print(f"Loaded frozen CanViT from Hugging Face: {args.model_repo}")
    return model, glimpse_size_px


def build_agent(args: argparse.Namespace, canvas_feature_dim: int, device: torch.device):
    """Construct Canvas SAC actor/critics using the existing model classes."""
    common = dict(
        canvas_feature_dim=canvas_feature_dim,
        d_model=args.d_model,
        rff_dim=args.rff_dim,
        rff_seed=args.rff_seed,
        use_entropy_state=args.canvas_entropy_state,
        use_canvas_avg_pool=not args.disable_canvas_avg_pool,
        use_canvas_max_pool=not args.disable_canvas_max_pool,
    )
    actor = CanvasStateActor(**common).to(device)
    critic_common = dict(
        common,
        use_action_location_features=args.critic_local_action_features,
    )
    q1 = CanvasStateCritic(**critic_common).to(device)
    q2 = CanvasStateCritic(**critic_common).to(device)
    target_q1 = CanvasStateCritic(**critic_common).to(device)
    target_q2 = CanvasStateCritic(**critic_common).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    agent = CanvasSAC(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        init_alpha=args.init_alpha,
        target_entropy=args.target_entropy,
    )
    return actor, q1, q2, target_q1, target_q2, agent


def evaluate_dense_sac(
    *,
    args: argparse.Namespace,
    eval_loader,
    actor: CanvasStateActor,
    model,
    scene_norm,
    cls_norm,
    canvas_grid_size: int,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate deterministic dense SAC rollout on a fixed dense-feature subset."""
    if eval_loader is None:
        return {}
    actor_was_training = actor.training
    actor.eval()
    eval_images = int(args.eval_images)
    eval_batch_size = min(int(args.eval_batch_size or args.batch_size), max(eval_images, 1))
    eval_batches = max(1, math.ceil(eval_images / max(eval_batch_size, 1)))
    initial_norm: list[torch.Tensor] = []
    final_norm: list[torch.Tensor] = []
    initial_raw: list[torch.Tensor] = []
    final_raw: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    entropy_points: list[np.ndarray] = []
    scale_sums = [0.0 for _ in range(args.t)]
    scale_counts = [0 for _ in range(args.t)]
    with torch.inference_mode():
        for _ in range(eval_batches):
            batch = load_dense_train_batch(
                train_loader=eval_loader,
                device=device,
                scene_norm=scene_norm,
                cls_norm=cls_norm,
                non_blocking=True,
            )
            batch_size = batch.images.shape[0]
            state = model.init_state(
                batch_size=batch_size,
                canvas_grid_size=canvas_grid_size,
            )
            coords, lengths = empty_viewpoint_history(
                batch_size=batch_size,
                max_steps=args.max_history,
                device=device,
            )
            full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
            full_glimpse = sample_at_viewpoint(
                spatial=batch.images,
                viewpoint=full_vp,
                glimpse_size_px=glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
            state = out.state
            initial_metrics = dense_distillation_metrics(
                model=model,
                state=state,
                batch=batch,
                scene_denorm=scene_norm.destandardize,
                cls_denorm=cls_norm.destandardize,
                scene_weight=args.scene_reward_weight,
                cls_weight=args.cls_reward_weight,
            )
            # Problem: l0-delta rewards need a fixed denominator in the same
            # loss space as their delta. Solution: capture the post-reset raw
            # or normalized loss once, immediately after the full-scene warmup.
            # Result: eval rewards never divide norm deltas by raw references.
            episode_l0 = _episode_l0_for_reward_mode(
                mode=args.reward_mode,
                metrics=initial_metrics,
            )
            current_metrics = initial_metrics
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
            episode_reward = torch.zeros(batch_size, device=device)
            for step_idx in range(args.t):
                obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
                if canvas_entropy is not None:
                    obs["entropy"] = canvas_entropy
                action = actor.deterministic_action(obs)
                vp = action_to_viewpoint(action, min_scale=args.min_scale)
                entropy_points.append(
                    torch.cat([vp.centers, vp.scales[:, None]], dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                scale_sums[step_idx] += float(vp.scales.detach().sum().item())
                scale_counts[step_idx] += batch_size
                glimpse = sample_at_viewpoint(
                    spatial=dense_glimpse_images(batch),
                    viewpoint=vp,
                    glimpse_size_px=glimpse_size_px,
                ).to(dtype=canvit_dtype)
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state
                next_metrics = dense_distillation_metrics(
                    model=model,
                    state=state,
                    batch=batch,
                    scene_denorm=scene_norm.destandardize,
                    cls_denorm=cls_norm.destandardize,
                    scene_weight=args.scene_reward_weight,
                    cls_weight=args.cls_reward_weight,
                )
                episode_reward = episode_reward + dense_reward(
                    mode=args.reward_mode,
                    before=current_metrics,
                    after=next_metrics,
                    l0=episode_l0,
                    eps=args.reward_eps,
                )
                current_metrics = next_metrics
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
                    viewpoint=vp,
                    step=step_idx + 1,
                )
            initial_norm.append(initial_metrics.loss_norm.detach().cpu())
            final_norm.append(current_metrics.loss_norm.detach().cpu())
            initial_raw.append(initial_metrics.loss_raw.detach().cpu())
            final_raw.append(current_metrics.loss_raw.detach().cpu())
            rewards.append(episode_reward.detach().cpu())
    if actor_was_training:
        actor.train()
    initial_norm_t = torch.cat(initial_norm)
    final_norm_t = torch.cat(final_norm)
    initial_raw_t = torch.cat(initial_raw)
    final_raw_t = torch.cat(final_raw)
    reward_t = torch.cat(rewards)
    metrics = {
        "eval/reward": float(reward_t.mean().item()),
        "eval/reward_std": float(reward_t.std(unbiased=False).item()),
        "eval/initial_loss_norm": float(initial_norm_t.mean().item()),
        "eval/final_loss_norm": float(final_norm_t.mean().item()),
        "eval/loss_norm_reduction": float(
            (initial_norm_t.mean() - final_norm_t.mean()).item()
        ),
        "eval/initial_loss_raw": float(initial_raw_t.mean().item()),
        "eval/final_loss_raw": float(final_raw_t.mean().item()),
        "eval/loss_raw_reduction": float(
            (initial_raw_t.mean() - final_raw_t.mean()).item()
        ),
        "eval/viewpoint_entropy": viewpoint_entropy(
            entropy_points,
            bins=args.viewpoint_entropy_bins,
        ),
    }
    for step in range(args.t):
        metrics[f"eval/mean_scale_by_t{step + 1}"] = (
            scale_sums[step] / max(scale_counts[step], 1)
        )
    return metrics


def train_once(args: argparse.Namespace) -> None:
    """Run dense-feature Canvas SAC training."""
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
    print(f"Frozen CanViT inference dtype: {canvit_dtype}")

    G = cfg.canvas_patch_grid_size
    cls_norm, scene_norm = model.standardizers(G)
    if args.reset_normalizer or not scene_norm.initialized:
        shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
        init_normalizer_stats_from_shard(
            shards_dir=shards_dir,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            device=device,
            max_samples=args.normalizer_max_samples,
        )

    canvas_feature_dim = int(model.canvas_dim)
    actor, q1, q2, target_q1, target_q2, agent = build_agent(
        args,
        canvas_feature_dim,
        device,
    )
    start_batch, update_count = load_dense_sac_resume(
        path=args.resume,
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        agent=agent,
    )
    replay_bytes = replay_canvas_bytes(
        capacity=args.buffer_size,
        canvas_feature_dim=canvas_feature_dim,
        canvas_grid_size=G,
        include_entropy=args.canvas_entropy_state,
    )
    replay_device = resolve_replay_device(train_device=device, replay_bytes=replay_bytes)
    validate_replay_memory(storage_device=replay_device, replay_bytes=replay_bytes)
    replay_pin_memory = should_pin_replay_memory(
        storage_device=replay_device,
        train_device=device,
    )
    replay = CanvasReplayBuffer(
        capacity=args.buffer_size,
        max_history=args.max_history,
        canvas_feature_dim=canvas_feature_dim,
        canvas_grid_size=G,
        storage_device=replay_device,
        store_entropy=args.canvas_entropy_state,
        pin_memory=replay_pin_memory,
    )
    print(
        "Replay storage: "
        f"device={replay_device}, dtype={REPLAY_STORAGE_DTYPE}, "
        f"canvas_bytes={replay_bytes / 1024**3:.2f} GiB, "
        f"pin_memory={replay_pin_memory}"
    )

    comet_exp = make_dense_comet_experiment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reward_window: list[float] = []
    entropy_points: list[np.ndarray] = []
    scale_sums = [0.0 for _ in range(args.t)]
    scale_counts = [0 for _ in range(args.t)]
    latest_metrics: dict[str, float] = {}
    elapsed = 0.0
    glimpses = 0
    reward_map_interval = max(args.reward_map_interval or args.log_interval, 1)
    next_reward_map_batch = start_batch
    next_eval_batch = max(args.eval_interval, start_batch)
    last_eval_batch: int | None = None
    pbar = tqdm(range(start_batch, args.batches + 1), desc="Training IN21k dense SAC")
    for batch_idx in pbar:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_start = time.perf_counter()
        batch = load_dense_train_batch(
            train_loader=train_loader,
            device=device,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            non_blocking=True,
        )
        batch_size = batch.images.shape[0]
        state = model.init_state(batch_size=batch_size, canvas_grid_size=G)
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
            current_metrics = dense_distillation_metrics(
                model=model,
                state=state,
                batch=batch,
                scene_denorm=scene_norm.destandardize,
                cls_denorm=cls_norm.destandardize,
                scene_weight=args.scene_reward_weight,
                cls_weight=args.cls_reward_weight,
            )
            # Problem: multi-step l0 deltas must not renormalize against the
            # changing current loss or a loss from the wrong feature space.
            # Solution: store the matching reset loss once per batch episode
            # and pass it through all later reward computations.
            episode_l0 = _episode_l0_for_reward_mode(
                mode=args.reward_mode,
                metrics=current_metrics,
            )
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=G,
            )
            canvas_entropy = (
                dense_canvas_entropy_map(
                    model=model,
                    state=state,
                    batch=batch,
                    canvas_grid_size=G,
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
        should_log_reward_maps = args.reward_map_images > 0 and batch_idx >= next_reward_map_batch
        if should_log_reward_maps:
            maybe_save_dense_reward_maps(
                args=args,
                comet_exp=comet_exp,
                update_count=update_count,
                batch_idx=batch_idx,
                batch=batch,
                state=state,
                current_metrics=current_metrics,
                l0=episode_l0,
                coords=coords,
                lengths=lengths,
                canvas_summary=canvas_summary,
                canvas_entropy=canvas_entropy,
                actor=actor,
                q1=q1,
                q2=q2,
                model=model,
                scene_denorm=scene_norm.destandardize,
                cls_denorm=cls_norm.destandardize,
                glimpse_size_px=glimpse_size_px,
                canvit_dtype=canvit_dtype,
            )
            maybe_save_dense_policy_glimpses(
                args=args,
                comet_exp=comet_exp,
                update_count=update_count,
                batch_idx=batch_idx,
                batch=batch,
                state=state,
                current_metrics=current_metrics,
                l0=episode_l0,
                coords=coords,
                lengths=lengths,
                canvas_summary=canvas_summary,
                canvas_entropy=canvas_entropy,
                actor=actor,
                model=model,
                scene_denorm=scene_norm.destandardize,
                cls_denorm=cls_norm.destandardize,
                glimpse_size_px=glimpse_size_px,
                canvit_dtype=canvit_dtype,
                canvas_grid_size=G,
            )
            while next_reward_map_batch <= batch_idx:
                next_reward_map_batch += reward_map_interval
        rollout_viewpoints = [full_vp]
        rollout_rewards: list[torch.Tensor] = []

        for step_idx in range(args.t):
            obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
            if canvas_entropy is not None:
                obs["entropy"] = canvas_entropy
            if replay.size < args.learning_starts:
                action = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)
            else:
                with torch.no_grad():
                    action, _ = actor.sample(obs)
            vp = action_to_viewpoint(action, min_scale=args.min_scale)
            entropy_points.append(
                torch.cat([vp.centers, vp.scales[:, None]], dim=1).detach().cpu().numpy()
            )
            scale_sums[step_idx] += float(vp.scales.detach().sum().item())
            scale_counts[step_idx] += batch_size
            prev_canvas = canvas_summary.clone()
            prev_coords = coords.clone()
            prev_lengths = lengths.clone()
            with torch.inference_mode():
                glimpse = sample_at_viewpoint(
                    spatial=dense_glimpse_images(batch),
                    viewpoint=vp,
                    glimpse_size_px=glimpse_size_px,
                ).to(dtype=canvit_dtype)
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
                next_metrics = dense_distillation_metrics(
                    model=model,
                    state=out.state,
                    batch=batch,
                    scene_denorm=scene_norm.destandardize,
                    cls_denorm=cls_norm.destandardize,
                    scene_weight=args.scene_reward_weight,
                    cls_weight=args.cls_reward_weight,
                )
                next_canvas_summary = canvas_layernorm_spatial(
                    model=model,
                    state=out.state,
                    canvas_grid_size=G,
                )
                next_canvas_entropy = (
                    dense_canvas_entropy_map(
                        model=model,
                        state=out.state,
                        batch=batch,
                        canvas_grid_size=G,
                    )
                    if args.canvas_entropy_state
                    else None
                )
            reward = dense_reward(
                mode=args.reward_mode,
                before=current_metrics,
                after=next_metrics,
                l0=episode_l0,
                eps=args.reward_eps,
            )
            rollout_viewpoints.append(
                Viewpoint(
                    centers=vp.centers.detach().clone(),
                    scales=vp.scales.detach().clone(),
                )
            )
            rollout_rewards.append(reward.detach().clone())
            coords, lengths = append_viewpoint_history(
                coords=coords,
                lengths=lengths,
                viewpoint=vp,
                step=step_idx + 1,
            )
            done = torch.full(
                (batch_size,),
                float(step_idx == args.t - 1),
                device=device,
            )
            replay.add_batch(
                canvas=prev_canvas,
                coords=prev_coords,
                lengths=prev_lengths,
                actions=action.detach().clone(),
                rewards=reward.detach().clone(),
                next_canvas=next_canvas_summary,
                next_coords=coords,
                next_lengths=lengths,
                dones=done,
                entropy=canvas_entropy,
                next_entropy=next_canvas_entropy,
            )
            reward_window.extend(reward.detach().cpu().numpy().astype(float).tolist())
            state = out.state
            current_metrics = next_metrics
            canvas_summary = next_canvas_summary
            canvas_entropy = next_canvas_entropy

        maybe_save_debug_rollout_viz(
            args=args,
            comet_exp=comet_exp,
            update_count=update_count,
            images=batch.images,
            viewpoints=rollout_viewpoints,
            rewards_by_step=rollout_rewards,
            batch_idx=batch_idx,
            start_batch=start_batch,
        )

        if replay.size >= args.learning_starts:
            for _ in range(args.updates_per_batch):
                agent.update(replay.sample(args.replay_batch_size, device))
                update_count += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - batch_start
        glimpses += batch_size * (args.t + 1)
        if batch_idx % args.log_interval == 0 or batch_idx == start_batch:
            latest_metrics = {}
            if reward_window:
                reward_np = np.asarray(reward_window, dtype=np.float64)
                latest_metrics.update(
                    {
                        "train/online_reward/mean": float(reward_np.mean()),
                        "train/online_reward/std": float(reward_np.std()),
                        "train/online_reward/max": float(reward_np.max()),
                        "train/online_reward/min": float(reward_np.min()),
                    }
                )
            latest_metrics["throughput/glimpses_per_sec"] = glimpses / max(elapsed, 1e-12)
            latest_metrics["train/viewpoint_entropy"] = viewpoint_entropy(
                entropy_points,
                bins=args.viewpoint_entropy_bins,
            )
            for step in range(args.t):
                latest_metrics[f"train/mean_scale_by_t{step + 1}"] = (
                    scale_sums[step] / max(scale_counts[step], 1)
                )
            if comet_exp is not None and latest_metrics:
                comet_exp.log_metrics(latest_metrics, step=update_count)
            reward_window.clear()
            entropy_points.clear()
            scale_sums = [0.0 for _ in range(args.t)]
            scale_counts = [0 for _ in range(args.t)]
            pbar.set_postfix(
                reward=f"{latest_metrics.get('train/online_reward/mean', 0.0):+.4f}",
                updates=update_count,
            )

        if eval_loader is not None and batch_idx >= next_eval_batch:
            eval_metrics = evaluate_dense_sac(
                args=args,
                eval_loader=eval_loader,
                actor=actor,
                model=model,
                scene_norm=scene_norm,
                cls_norm=cls_norm,
                canvas_grid_size=G,
                glimpse_size_px=glimpse_size_px,
                canvit_dtype=canvit_dtype,
                device=device,
            )
            latest_metrics.update(eval_metrics)
            if comet_exp is not None and eval_metrics:
                comet_exp.log_metrics(eval_metrics, step=update_count)
            display_loss_name, display_loss_key = _eval_display_loss_for_reward_mode(
                args.reward_mode
            )
            pbar.write(
                "eval "
                f"batch={batch_idx} update={update_count} "
                f"reward={eval_metrics.get('eval/reward', 0.0):+.4f} "
                f"{display_loss_name}={eval_metrics.get(display_loss_key, 0.0):.4f}"
            )
            last_eval_batch = batch_idx
            while next_eval_batch <= batch_idx:
                next_eval_batch += args.eval_interval

        if batch_idx % args.checkpoint_interval == 0:
            save_dense_sac_checkpoint(
                path=args.checkpoint_dir / "latest.pt",
                actor=actor,
                q1=q1,
                q2=q2,
                target_q1=target_q1,
                target_q2=target_q2,
                agent=agent,
                args=args,
                canvas_feature_dim=canvas_feature_dim,
                batch=batch_idx,
                updates=update_count,
                metrics=latest_metrics,
            )

    if eval_loader is not None and last_eval_batch != args.batches:
        eval_metrics = evaluate_dense_sac(
            args=args,
            eval_loader=eval_loader,
            actor=actor,
            model=model,
            scene_norm=scene_norm,
            cls_norm=cls_norm,
            canvas_grid_size=G,
            glimpse_size_px=glimpse_size_px,
            canvit_dtype=canvit_dtype,
            device=device,
        )
        latest_metrics.update(eval_metrics)
        if comet_exp is not None and eval_metrics:
            comet_exp.log_metrics(eval_metrics, step=update_count)

    save_dense_sac_checkpoint(
        path=args.checkpoint_dir / "final.pt",
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        agent=agent,
        args=args,
        canvas_feature_dim=canvas_feature_dim,
        batch=args.batches,
        updates=update_count,
        metrics=latest_metrics,
    )


def main() -> None:
    """CLI entrypoint."""
    train_once(parse_args())


if __name__ == "__main__":
    main()
