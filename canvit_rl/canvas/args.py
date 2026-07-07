"""Canvas SAC command-line argument helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from canvit_rl.canvas.logging import add_canvas_sac_comet_args


def add_canvas_sac_args(parser: argparse.ArgumentParser) -> None:
    """Register core Canvas SAC training arguments."""
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ADE20k",
        help=(
            "ADE20K root, or synthetic root containing images/<split> and "
            "masks/<split> folders."
        ),
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "ade20k", "synthetic"],
        default="auto",
        help=(
            "auto detects folder data with images/<split>/masks/<split> or "
            "flat images/ and masks/ subdirectories; use ade20k or synthetic "
            "to force a format."
        ),
    )
    parser.add_argument(
        "--synthetic-image-dir",
        type=str,
        default=None,
        help="Optional image directory for --dataset-format synthetic.",
    )
    parser.add_argument(
        "--synthetic-mask-dir",
        type=str,
        default=None,
        help="Optional mask directory for --dataset-format synthetic.",
    )
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument(
        "--eval-split",
        choices=["training", "validation"],
        default="validation",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-images", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument(
        "--canvit-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Inference dtype for frozen CanViT only; RL networks/probe stay fp32.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--rff-dim", type=int, default=128)
    parser.add_argument("--rff-seed", type=int, default=42)
    parser.add_argument("--max-history", type=int, default=6)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument(
        "--randomize-actor-init",
        action="store_true",
        help=(
            "Initialize the deterministic actor mean to a random near-center "
            "Viewpoint instead of the default zero-action midpoint-scale prior."
        ),
    )
    parser.add_argument(
        "--actor-init-center-radius",
        type=float,
        default=0.25,
        help=(
            "Uniform radius for --randomize-actor-init center coordinates; "
            "centers are sampled from [-radius, radius]."
        ),
    )
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--target-entropy", type=float, default=-3.0)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1)
    parser.add_argument("--updates-per-batch", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument(
        "--skip-final-full-validation-miou",
        action="store_true",
        help=(
            "Skip the post-training best.pt evaluation on the full validation "
            "split with mIoUAccumulator."
        ),
    )
    parser.add_argument(
        "--skip-eval-random",
        action="store_true",
        help="Skip random-policy baseline rollouts during eval.",
    )
    parser.add_argument(
        "--skip-eval-egc2f",
        action="store_true",
        help="Skip entropy-guided coarse-to-fine baseline rollouts during eval.",
    )
    parser.add_argument("--viewpoint-entropy-bins", type=int, default=8)
    parser.add_argument(
        "--reward-map-images",
        type=int,
        default=0,
        help=(
            "If >0, save true-reward vs critic-Q maps for this many validation "
            "images every --reward-map-interval SAC updates."
        ),
    )
    parser.add_argument("--reward-map-grid-size", type=int, default=11)
    parser.add_argument("--reward-map-scales", type=str, default="0.25,0.50")
    parser.add_argument("--reward-map-chunk-size", type=int, default=16)
    parser.add_argument(
        "--reward-map-interval",
        type=int,
        default=None,
        help=(
            "SAC update interval for live reward maps. Defaults to "
            "--eval-interval when omitted."
        ),
    )
    parser.add_argument(
        "--reward-map-output-dir",
        type=Path,
        default=Path("results/sac_canvas_reward_maps"),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-actor-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional pretrained CanvasStateActor checkpoint for a fresh SAC "
            "run. Accepts a latest.pt payload with an actor key or a bare "
            "actor_final.pt state dict."
        ),
    )
    parser.add_argument(
        "--init-critic-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional pretrained CanvasStateCritic checkpoint for a fresh SAC "
            "run. Expects q1 and optional q2 keys; target critics are synced."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/canvas_sac"),
    )
    add_canvas_sac_comet_args(parser)


def validate_canvas_sac_args(args: argparse.Namespace) -> None:
    """Validate cross-argument constraints for Canvas SAC runs."""
    if args.t < 0:
        raise ValueError("--t must be non-negative.")
    if args.max_history < args.t + 1:
        raise ValueError("--max-history must be at least t+1.")
    if args.t + 1 > 21:
        raise ValueError("EG-C2F evaluation requires --t <= 20.")
    if not 0.0 <= args.actor_init_center_radius < 1.0:
        raise ValueError("--actor-init-center-radius must be in [0, 1).")
    if args.reward_map_images < 0:
        raise ValueError("--reward-map-images must be non-negative.")
    if args.reward_map_grid_size < 2:
        raise ValueError("--reward-map-grid-size must be >= 2.")
    if args.reward_map_chunk_size < 1:
        raise ValueError("--reward-map-chunk-size must be positive.")
    if args.reward_map_interval is not None and args.reward_map_interval < 1:
        raise ValueError("--reward-map-interval must be positive.")
    if args.resume is not None and (
        args.init_actor_checkpoint is not None
        or args.init_critic_checkpoint is not None
    ):
        # Problem: resume restores optimizer/alpha/replay counters while
        # initializer checkpoints are intended for fresh runs. Solution: reject
        # mixed modes before any model or optimizer state is touched.
        raise ValueError(
            "--resume cannot be combined with --init-actor-checkpoint or "
            "--init-critic-checkpoint."
        )
