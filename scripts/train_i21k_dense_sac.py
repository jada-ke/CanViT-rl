"""Train Canvas SAC policies on IN21k DINOv3 dense-feature distillation rewards.

The data path is owned by CanViT-pretrain's shard loader. This script only
wraps that batch contract in the existing Canvas SAC actor/critic/replay code.

Example:
    uv run python scripts/train_i21k_dense_sac.py \
        --feature-base-dir /path/to/features \
        --feature-image-root /datasets/imagenet21k \
        --model-repo canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02 \
        --batches 1000 --batch-size 8 --t 4 --no-comet
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Problem: Comet's framework integrations must be imported before torch, but
# importing comet_ml in this headless environment can initialize plotting/font
# caches even for --help. Solution: pre-scan argv and import Comet before torch
# only when --comet is explicitly requested. Result: Comet runs get the correct
# import order while default/no-Comet runs stay lightweight.
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
    validate_replay_memory,
)
from canvit_rl.canvas.state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    empty_viewpoint_history,
)
from canvit_rl.canvit_precision import resolve_canvit_dtype
from canvit_rl.env import get_device
from canvit_rl.pretrain_IN21k.checkpoints import (
    load_dense_sac_resume,
    save_dense_sac_checkpoint,
)
from canvit_rl.pretrain_IN21k.dense_train_batch import (
    FixedDenseSubsetLoader,
    apply_dense_feature_config,
    init_normalizer_stats_from_shard,
    load_dense_train_batch,
)
from canvit_rl.pretrain_IN21k.pretrain_modules import load_pretrain_modules
from canvit_rl.pretrain_IN21k.reward import (
    dense_distillation_metrics,
    dense_reward,
)
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic
from canvit_rl.viewpoint_policy import action_to_viewpoint

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def add_dense_sac_comet_args(parser: argparse.ArgumentParser) -> None:
    """Register Comet flags while keeping experiment creation opt-in."""
    parser.add_argument("--comet-log-interval", type=int, default=20)
    parser.add_argument("--no-comet", action="store_true", default=True)
    parser.add_argument("--comet", dest="no_comet", action="store_false")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="i21k-dense-sac")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="i21k-dense-sac")


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
    parser.add_argument("--t", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=5)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--scene-reward-weight", type=float, default=1.0)
    parser.add_argument("--cls-reward-weight", type=float, default=0.25)
    parser.add_argument(
        "--reward-mode",
        choices=["raw_mse_reduction", "norm_loss_reduction"],
        default="raw_mse_reduction",
    )
    parser.add_argument("--reward-eps", type=float, default=1e-6)
    parser.add_argument("--canvit-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--rff-dim", type=int, default=128)
    parser.add_argument("--rff-seed", type=int, default=42)
    parser.add_argument("--critic-local-action-features", action="store_true")
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
    add_dense_sac_comet_args(parser)
    args = parser.parse_args()
    if args.max_history < args.t + 1:
        raise ValueError("--max-history must be at least --t + 1.")
    if args.disable_canvas_avg_pool and args.disable_canvas_max_pool:
        raise ValueError("At least one canvas pooling branch must remain enabled.")
    if args.debug_viz_images < 0 or args.debug_viz_batches < 0:
        raise ValueError("--debug-viz-images and --debug-viz-batches must be non-negative.")
    if args.subset_size < 0:
        raise ValueError("--subset-size must be non-negative.")
    if args.subset_shards < 1:
        raise ValueError("--subset-shards must be positive.")
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
            # Problem: quick smoke tests need human-readable reward context
            # without a plotting stack. Solution: draw compact per-step reward
            # text directly on the image. Result: saved PNGs show both where
            # the policy looked and whether the dense reward improved.
            draw.rectangle((0, 0, width, 14 * len(reward_text) + 4), fill=(0, 0, 0))
            for line_idx, text in enumerate(reward_text):
                draw.text((4, 2 + 14 * line_idx), text, fill="white")
        output = args.debug_viz_dir / f"batch_{batch_idx:06d}_sample_{sample_idx:03d}.png"
        pil_image.save(output)
        if comet_exp is not None and hasattr(comet_exp, "log_image"):
            # Problem: local debug PNGs are useful for smoke tests but easy to
            # miss in remote runs. Solution: log the same bounded set of
            # overlays to Comet when enabled. Result: small-subset experiments
            # can inspect policy viewpoint choices without downloading files.
            comet_exp.log_image(
                str(output),
                name=f"debug/i21k_dense_rollout/{output.name}",
                step=update_count,
            )


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


def load_frozen_hf_model(args: argparse.Namespace, cfg):
    """Load the frozen CanViT pretraining model from Hugging Face."""
    # Problem: the dense SAC path previously accepted local canvit-pretrain
    # .pt checkpoints, which split it from the ADE20K scripts' HF checkpoint
    # convention. Solution: load the same CanViT pretraining model directly
    # from a Hugging Face repo. Result: one model source is used for policy
    # pretraining, and only SAC actor/critic checkpoints are written locally.
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


def train_once(args: argparse.Namespace) -> None:
    """Run dense-feature Canvas SAC training."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg, modules = build_pretrain_config(args)
    device = cfg.device
    train_loader = build_dense_loader(args, cfg, modules)
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
    )
    replay_device = resolve_replay_device(train_device=device, replay_bytes=replay_bytes)
    validate_replay_memory(storage_device=replay_device, replay_bytes=replay_bytes)
    replay = CanvasReplayBuffer(
        capacity=args.buffer_size,
        max_history=args.max_history,
        canvas_feature_dim=canvas_feature_dim,
        canvas_grid_size=G,
        storage_device=replay_device,
    )
    print(
        "Replay storage: "
        f"device={replay_device}, dtype={REPLAY_STORAGE_DTYPE}, "
        f"canvas_bytes={replay_bytes / 1024**3:.2f} GiB"
    )

    comet_exp = make_dense_comet_experiment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_windows: dict[str, list[float]] = defaultdict(list)
    reward_window: list[float] = []
    latest_metrics: dict[str, float] = {}
    elapsed = 0.0
    glimpses = 0
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
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=G,
            )
        coords, lengths = append_viewpoint_history(
            coords=coords,
            lengths=lengths,
            viewpoint=full_vp,
            step=0,
        )
        rollout_viewpoints = [full_vp]
        rollout_rewards: list[torch.Tensor] = []

        for step_idx in range(args.t):
            obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
            if replay.size < args.learning_starts:
                action = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)
            else:
                with torch.no_grad():
                    action, log_prob = actor.sample(obs)
                train_windows["actor/log_prob"].append(float(log_prob.mean().item()))
                train_windows["actor/entropy"].append(float((-log_prob).mean().item()))
            vp = action_to_viewpoint(action, min_scale=args.min_scale)
            prev_canvas = canvas_summary.clone()
            prev_coords = coords.clone()
            prev_lengths = lengths.clone()
            with torch.inference_mode():
                glimpse = sample_at_viewpoint(
                    spatial=batch.images,
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
            reward = dense_reward(
                mode=args.reward_mode,
                before=current_metrics,
                after=next_metrics,
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
            )
            reward_window.extend(reward.detach().cpu().numpy().astype(float).tolist())
            state = out.state
            current_metrics = next_metrics
            canvas_summary = next_canvas_summary

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
                metrics = agent.update(replay.sample(args.replay_batch_size, device))
                update_count += 1
                for key, value in metrics.items():
                    train_windows[key].append(value)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - batch_start
        glimpses += batch_size * (args.t + 1)
        if batch_idx % args.log_interval == 0 or batch_idx == start_batch:
            latest_metrics = {
                key: float(np.mean(values))
                for key, values in train_windows.items()
                if values
            }
            if reward_window:
                reward_np = np.asarray(reward_window, dtype=np.float64)
                latest_metrics.update(
                    {
                        "train/reward_mean": float(reward_np.mean()),
                        "train/reward_std": float(reward_np.std()),
                        "train/reward_min": float(reward_np.min()),
                        "train/reward_max": float(reward_np.max()),
                    }
                )
            latest_metrics["throughput/glimpses_per_sec"] = glimpses / max(elapsed, 1e-12)
            if comet_exp is not None and latest_metrics:
                comet_exp.log_metrics(latest_metrics, step=update_count)
            train_windows.clear()
            reward_window.clear()
            pbar.set_postfix(
                reward=f"{latest_metrics.get('train/reward_mean', 0.0):+.4f}",
                updates=update_count,
            )

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
