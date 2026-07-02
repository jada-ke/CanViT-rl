"""
Train image-dependent SAC over the current layernorm-pooled CanViT canvas.

This is the canvas-state analogue of scripts/train_viewpoint_sac.py:

    state  = current CanViT canvas summary + compact viewpoint history
    action = next Viewpoint
    reward = (CE_before - CE_after) / CE_before

Example:
    python scripts/train_canvas_sac.py \
    --dataset synthetic_segmentation \
    --batches 1000 --batch-size 7 --max-samples 7 --t 1 \
    --eval-images 3 --eval-batch-size 3 --eval-split validation\
    --replay-batch-size 4 \
    --checkpoint-dir checkpoints/canvas_sac/synthetic-im7-t1_1000 \
    --experiment-name synthetic-im7-t1_1000

    uv run python scripts/train_canvas_sac.py \
        --dataset synthetic_segmentation \
        --batches 1000 --batch-size 1 --max-samples 1 --t 1 \
        --eval-images 1 --eval-batch-size 1 \
        --replay-batch-size 8 \
        --checkpoint-dir checkpoints/canvas_sac/synthetic-ade-im1-t1 \
        --experiment-name synthetic-ade-im1-t2

    uv run python scripts/train_canvas_sac.py \
        --dataset synthetic_segmentation \
        --batches 100 \
        --batch-size 1 \
        --eval-images 3 \
        --eval-batch-size 1 \
        --optuna-trials 5 \
        --no-comet
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_pytorch.policies import random_viewpoints
from canvit_specialize.datasets.ade20k import (
    ADE20kDataset,
    IGNORE_LABEL,
    NUM_CLASSES,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from canvit_rl.ade_labels import remap_ade_mask_labels
from canvit_rl.canvas_state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    empty_viewpoint_history,
)
from canvit_rl.canvit_precision import configure_frozen_canvit_precision
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import _segmentation_cross_entropy_losses
from canvit_rl.reward import relative_ce_reduction
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic
from canvit_rl.viewpoint_policy import action_to_viewpoint

try:
    from visualize_sac_reward_maps import visualize_reward_maps_for_indices
except ImportError:
    from scripts.visualize_sac_reward_maps import visualize_reward_maps_for_indices
try:
    from visualize_policy_glimpses import visualize_canvas_policy_for_indices
except ImportError:
    from scripts.visualize_policy_glimpses import visualize_canvas_policy_for_indices
try:
    from canvas_sac_optuna import add_canvas_sac_optuna_args, run_canvas_sac_optuna
except ImportError:
    from scripts.canvas_sac_optuna import (
        add_canvas_sac_optuna_args,
        run_canvas_sac_optuna,
    )

EVAL_REPO = Path(__file__).resolve().parents[1] / "CanViT-eval"
if EVAL_REPO.is_dir() and str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))


REPLAY_STORAGE_DTYPE = torch.float16
REPLAY_STORAGE_DTYPE_BYTES = 2
REPLAY_GPU_FRACTION = 0.55
MAX_CPU_REPLAY_BYTES =96 * 1024**3
DATALOADER_PREFETCH_FACTOR = 4


def _sync_for_timing(device: torch.device) -> None:
    """Synchronize CUDA kernels before reading throughput timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _limit_dataset(dataset, max_samples: int | None, *, offset: int = 0):
    """Restrict datasets while preserving deterministic validation subsets."""
    if max_samples is None:
        return dataset
    start = min(offset, len(dataset))
    stop = min(start + max_samples, len(dataset))
    return torch.utils.data.Subset(dataset, range(start, stop))


class SyntheticSegmentationDataset(torch.utils.data.Dataset):
    """Image/mask folder dataset for ADE-embedded synthetic segmentation."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        *,
        image_dir: Path,
        mask_dir: Path,
        scene_size_px: int,
        img_transform,
    ) -> None:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Synthetic image directory not found: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Synthetic mask directory not found: {mask_dir}")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.scene_size_px = scene_size_px
        self.img_transform = img_transform
        self.images = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if not self.images:
            raise ValueError(f"No synthetic images found in {image_dir}")
        mask_by_stem = {
            path.stem: path
            for path in mask_dir.iterdir()
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        }
        missing = [path.name for path in self.images if path.stem not in mask_by_stem]
        if missing:
            raise ValueError(
                "Missing synthetic masks with matching stems for: "
                + ", ".join(missing[:10])
            )
        self.masks = [mask_by_stem[path.stem] for path in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        image_tensor = self.img_transform(image)
        resample_nearest = getattr(Image, "Resampling", Image).NEAREST
        mask = mask.resize(
            (self.scene_size_px, self.scene_size_px),
            resample=resample_nearest,
        )
        # Fixed by Codex on 2026-06-23
        # Fixed by Codex on 2026-06-24
        # Problem: Older synthetic masks may contain raw ADE ids 1..150, and
        # label 150 is out of bounds for a 150-channel CE target.
        # Solution: preserve semantic labels but normalize raw ADE ids to
        # zero-based 0..149 while keeping IGNORE_LABEL=255 padding.
        # Result: The frozen ADE probe is trained/evaluated against valid CE
        # targets in the inserted region only.
        mask_tensor = torch.from_numpy(
            remap_ade_mask_labels(np.asarray(mask)).astype(np.int64)
        )
        return image_tensor, mask_tensor


def _build_segmentation_dataset(
    *,
    args: argparse.Namespace,
    cfg: CanViTEnvConfig,
    split: str,
    img_tf,
    mask_tf,
):
    """Build either ADE20K or folder-based synthetic segmentation data."""
    dataset_root = Path(args.dataset)
    split_image_dir = dataset_root / "images" / split
    split_mask_dir = dataset_root / "masks" / split
    inferred_synthetic = (
        (split_image_dir.is_dir() and split_mask_dir.is_dir())
        or ((dataset_root / "images").is_dir() and (dataset_root / "masks").is_dir())
    )
    dataset_format = (
        "synthetic"
        if args.dataset_format == "auto" and inferred_synthetic
        else "ade20k"
        if args.dataset_format == "auto"
        else args.dataset_format
    )
    if dataset_format == "ade20k":
        return ADE20kDataset(
            root=dataset_root,
            split=split,
            img_transform=img_tf,
            mask_transform=mask_tf,
        )
    if args.synthetic_image_dir:
        image_dir = Path(args.synthetic_image_dir)
    elif split_image_dir.is_dir():
        image_dir = split_image_dir
    else:
        image_dir = dataset_root / "images"
    if args.synthetic_mask_dir:
        mask_dir = Path(args.synthetic_mask_dir)
    elif split_mask_dir.is_dir():
        mask_dir = split_mask_dir
    else:
        mask_dir = dataset_root / "masks"
    return SyntheticSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        scene_size_px=cfg.scene_size_px,
        img_transform=img_tf,
    )


def _segmentation_metrics(
    *,
    model,
    probe: torch.nn.Module,
    state,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    acc: mIoUAccumulator | None = None,
) -> tuple[torch.Tensor, float | None]:
    """Return per-image CE and optionally update a dataset-level mIoU acc."""
    losses = _segmentation_cross_entropy_losses(
        model=model,
        state=state,
        probe=probe,
        canvas_grid_size=cfg.canvas_grid_size,
        mask=masks,
        batch_size=masks.shape[0],
    )
    miou = None
    if acc is not None:
        spatial = model.get_spatial(state.canvas).reshape(
            masks.shape[0],
            cfg.canvas_grid_size,
            cfg.canvas_grid_size,
            -1,
        )
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            logits = probe(spatial.float()).float()
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        acc.update(logits.argmax(dim=1), masks)
        miou = float(acc.compute())
    return losses, miou


def _viewpoint_entropy(values: list[np.ndarray], *, bins: int) -> float:
    """Entropy of visited (y, x, scale) bins, normalized to [0, 1]."""
    if not values:
        return 0.0
    points = np.concatenate(values, axis=0)
    if points.shape[0] <= 1:
        return 0.0
    hist, _ = np.histogramdd(
        points,
        bins=bins,
        range=[[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
    )
    probs = hist.reshape(-1).astype(np.float64)
    probs = probs[probs > 0]
    probs = probs / probs.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / max(float(np.log(hist.size)), 1e-12)


def _parse_scales(value: str) -> list[float]:
    """Parse comma-separated reward-map scales."""
    scales = [float(item) for item in value.split(",") if item.strip()]
    if not scales or any(scale <= 0 or scale > 1 for scale in scales):
        raise ValueError("--reward-map-scales must contain values in (0, 1].")
    return scales


def _replay_canvas_bytes(
    *,
    capacity: int,
    canvas_feature_dim: int,
    canvas_grid_size: int,
) -> int:
    """Return bytes for current + next canvas replay tensors."""
    return (
        2
        * capacity
        * canvas_feature_dim
        * canvas_grid_size
        * canvas_grid_size
        * REPLAY_STORAGE_DTYPE_BYTES
    )


def _resolve_replay_device(
    *,
    train_device: torch.device,
    replay_bytes: int,
) -> torch.device:
    """Pick replay storage placement without surprising CPU or VRAM OOMs."""
    if train_device.type != "cuda":
        return torch.device("cpu")

    free_bytes, _ = torch.cuda.mem_get_info(train_device)
    if replay_bytes <= int(free_bytes * REPLAY_GPU_FRACTION):
        return train_device
    return torch.device("cpu")


def _validate_replay_memory(
    *,
    storage_device: torch.device,
    replay_bytes: int,
) -> None:
    """Fail early when replay would exceed the configured CPU RAM budget."""
    if storage_device.type != "cpu":
        return
    if replay_bytes <= MAX_CPU_REPLAY_BYTES:
        return
    actual_gb = replay_bytes / 1024**3
    max_gb = MAX_CPU_REPLAY_BYTES / 1024**3
    raise ValueError(
        "Canvas replay would allocate "
        f"{actual_gb:.2f} GiB on CPU, exceeding the {max_gb:.2f} GiB safety "
        "limit. Reduce --buffer-size or use a GPU with enough free VRAM for "
        "auto CUDA replay."
    )


def _dataloader_kwargs(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Build DataLoader settings that overlap input loading with CUDA work."""
    kwargs: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR
    return kwargs


def _eval_random_batch(
    *,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvit_dtype: torch.dtype,
    acc: mIoUAccumulator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out random learned viewpoints after a full-scene warm-up."""
    device = images.device
    state = model.init_state(batch_size=images.shape[0], canvas_grid_size=cfg.canvas_grid_size)
    full_vp = Viewpoint.full_scene(batch_size=images.shape[0], device=device)
    with torch.inference_mode():
        full_glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        ).to(dtype=canvit_dtype)
        full_out = model(
            glimpse=full_glimpse,
            state=state,
            viewpoint=full_vp,
        )
        state = full_out.state
        initial_ce, _ = _segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg
        )
        for _ in range(args.t):
            vp = random_viewpoints(
                batch_size=images.shape[0],
                device=device,
                n_viewpoints=1,
                min_scale=args.min_scale,
                max_scale=1.0,
                start_with_full_scene=False,
            )[0]
            glimpse = sample_at_viewpoint(
                spatial=images,
                viewpoint=vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(
                glimpse=glimpse,
                state=state,
                viewpoint=vp,
            )
            state = out.state
        final_ce, _ = _segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg, acc=acc
        )
    return initial_ce, final_ce


def _eval_egc2f_batch(
    *,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvit_dtype: torch.dtype,
    acc: mIoUAccumulator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate canvit-eval entropy-guided coarse-to-fine on the same horizon."""
    try:
        from canvit_eval.episode import run_episode
        from canvit_eval.policies import make_policy
    except ImportError as exc:
        raise RuntimeError(
            "EG-C2F evaluation requires canvit-eval import support. "
            "Place CanViT-eval next to this repo or install it."
        ) from exc
    if args.t + 1 > 21:
        raise ValueError("EG-C2F has 21 built-in timesteps; require --t <= 20.")
    batch_size = images.shape[0]
    policy = make_policy(
        "entropy_coarse_to_fine",
        batch_size=batch_size,
        device=images.device,
        n_viewpoints=args.t + 1,
        canvas_grid=cfg.canvas_grid_size,
        probe=probe,
        get_spatial_fn=model.get_spatial,
    )
    with torch.inference_mode():
        steps = run_episode(
            model=model,
            images=images.to(dtype=canvit_dtype),
            policy=policy,
            n_timesteps=args.t + 1,
            canvas_grid=cfg.canvas_grid_size,
            glimpse_px=cfg.glimpse_size_px,
        )
        initial_ce, _ = _segmentation_metrics(
            model=model, probe=probe, state=steps[0].state, masks=masks, cfg=cfg
        )
        final_ce, _ = _segmentation_metrics(
            model=model,
            probe=probe,
            state=steps[-1].state,
            masks=masks,
            cfg=cfg,
            acc=acc,
        )
    return initial_ce, final_ce


def _make_comet_experiment(args: argparse.Namespace):
    """Create a Comet experiment unless disabled for local dry runs."""
    if args.no_comet:
        return None
    if Experiment is None:
        raise RuntimeError(
            "Comet logging is enabled by default, but comet_ml is not installed. "
            "Install comet-ml or run with --no-comet."
        )
    kwargs = dict(
        project_name=args.comet_project,
        auto_param_logging=True,
        auto_metric_logging=True,
    )
    if args.comet_workspace:
        kwargs["workspace"] = args.comet_workspace
    experiment = Experiment(**kwargs)
    experiment.set_name(args.experiment_name or "canvas-state-sac")
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    return experiment


class CanvasReplayBuffer:
    """Replay buffer for image-dependent current-canvas SAC transitions."""

    def __init__(
        self,
        *,
        capacity: int,
        max_history: int,
        canvas_feature_dim: int,
        canvas_grid_size: int,
        storage_device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.storage_device = storage_device
        alloc_kwargs = {"device": storage_device}
        self.canvas = torch.zeros(
            (capacity, canvas_feature_dim, canvas_grid_size, canvas_grid_size),
            dtype=REPLAY_STORAGE_DTYPE,
            **alloc_kwargs,
        )
        self.next_canvas = torch.zeros_like(self.canvas)
        self.coords = torch.zeros(
            (capacity, max_history, 3),
            dtype=torch.float32,
            **alloc_kwargs,
        )
        self.next_coords = torch.zeros_like(self.coords)
        self.lengths = torch.zeros(capacity, dtype=torch.long, **alloc_kwargs)
        self.next_lengths = torch.zeros_like(self.lengths)
        self.actions = torch.zeros(
            (capacity, 3),
            dtype=torch.float32,
            **alloc_kwargs,
        )
        self.rewards = torch.zeros(
            capacity,
            dtype=torch.float32,
            **alloc_kwargs,
        )
        self.dones = torch.zeros_like(self.rewards)
        self.pos = 0
        self.size = 0

    def _copy_rows(self, tensor: torch.Tensor, values: torch.Tensor) -> None:
        """Copy a batch into circular replay slots without host NumPy staging."""
        batch_size = values.shape[0]
        end = self.pos + batch_size
        values = values.detach().to(
            device=self.storage_device,
            dtype=tensor.dtype,
            non_blocking=self.storage_device.type == "cuda",
        )
        if end <= self.capacity:
            tensor[self.pos:end].copy_(
                values,
                non_blocking=self.storage_device.type == "cuda",
            )
            return
        first = self.capacity - self.pos
        tensor[self.pos:].copy_(
            values[:first],
            non_blocking=self.storage_device.type == "cuda",
        )
        tensor[: end - self.capacity].copy_(
            values[first:],
            non_blocking=self.storage_device.type == "cuda",
        )

    def add_batch(
        self,
        *,
        canvas: torch.Tensor,
        coords: torch.Tensor,
        lengths: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_canvas: torch.Tensor,
        next_coords: torch.Tensor,
        next_lengths: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        batch_size = canvas.shape[0]
        if batch_size > self.capacity:
            raise ValueError(
                f"Replay batch_size={batch_size} exceeds capacity={self.capacity}; "
                "increase --buffer-size or reduce --batch-size."
            )
        self._copy_rows(self.canvas, canvas)
        self._copy_rows(self.coords, coords)
        self._copy_rows(self.lengths, lengths)
        self._copy_rows(self.actions, actions)
        self._copy_rows(self.rewards, rewards)
        self._copy_rows(self.next_canvas, next_canvas)
        self._copy_rows(self.next_coords, next_coords)
        self._copy_rows(self.next_lengths, next_lengths)
        self._copy_rows(self.dones, dones)
        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.storage_device)

        def move(values: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
            return values.index_select(0, idx).to(
                device=device,
                dtype=dtype or values.dtype,
                non_blocking=self.storage_device.type == device.type,
            )

        return {
            "canvas": move(self.canvas, dtype=torch.float32),
            "coords": move(self.coords),
            "lengths": move(self.lengths),
            "actions": move(self.actions),
            "rewards": move(self.rewards),
            "next_canvas": move(self.next_canvas, dtype=torch.float32),
            "next_coords": move(self.next_coords),
            "next_lengths": move(self.next_lengths),
            "dones": move(self.dones),
        }


class CanvasSAC:
    """Continuous SAC for current-canvas actor and critic networks."""

    def __init__(
        self,
        *,
        actor: CanvasStateActor,
        q1: CanvasStateCritic,
        q2: CanvasStateCritic,
        target_q1: CanvasStateCritic,
        target_q2: CanvasStateCritic,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        init_alpha: float,
        target_entropy: float,
    ) -> None:
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        self.actor_opt = torch.optim.AdamW(actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.AdamW(
            list(q1.parameters()) + list(q2.parameters()),
            lr=critic_lr,
        )
        device = next(actor.parameters()).device
        self.log_alpha = torch.nn.Parameter(
            torch.log(torch.tensor(init_alpha, device=device))
        )
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, sample: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = {
            "canvas": sample["canvas"],
            "coords": sample["coords"],
            "lengths": sample["lengths"],
        }
        next_obs = {
            "canvas": sample["next_canvas"],
            "coords": sample["next_coords"],
            "lengths": sample["next_lengths"],
        }
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_min = torch.minimum(
                self.target_q1(next_obs, next_action),
                self.target_q2(next_obs, next_action),
            )
            target_q = sample["rewards"] + self.gamma * (1.0 - sample["dones"]) * (
                target_min - self.alpha.detach() * next_log_prob
            )

        q1_pred = self.q1(obs, sample["actions"])
        q2_pred = self.q2(obs, sample["actions"])
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        self.q_opt.zero_grad(set_to_none=True)
        (q1_loss + q2_loss).backward()
        self.q_opt.step()

        action, log_prob = self.actor.sample(obs)
        q_min = torch.minimum(self.q1(obs, action), self.q2(obs, action))
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        entropy = -log_prob.detach()
        return {
            "actor/loss": float(actor_loss.detach().item()),
            "actor/entropy": float(entropy.mean().item()),
            "actor/log_prob": float(log_prob.detach().mean().item()),
            "actor/action_std": float(action.detach().std(unbiased=False).item()),
            "critic/q1_loss": float(q1_loss.detach().item()),
            "critic/q2_loss": float(q2_loss.detach().item()),
            "critic/target_q": float(target_q.detach().mean().item()),
            "critic/q_mean": float(q_min.detach().mean().item()),
            "critic/q_std": float(q_min.detach().std(unbiased=False).item()),
            "sac/alpha": float(self.alpha.detach().item()),
            "sac/entropy": float(entropy.mean().item()),
            "sac/target_entropy_gap": float(
                entropy.mean().item() - abs(self.target_entropy)
            ),
            "alpha/loss": float(alpha_loss.detach().item()),
        }

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        with torch.no_grad():
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.mul_(1.0 - self.tau).add_(src_param, alpha=self.tau)


def _build_networks(
    *,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    device: torch.device,
) -> tuple[
    CanvasStateActor,
    CanvasStateCritic,
    CanvasStateCritic,
    CanvasStateCritic,
    CanvasStateCritic,
]:
    """Construct current-canvas actor, twin critics, and target critics."""
    kwargs = dict(
        canvas_feature_dim=canvas_feature_dim,
        d_model=args.d_model,
        rff_dim=args.rff_dim,
        rff_seed=args.rff_seed,
    )
    actor = CanvasStateActor(**kwargs).to(device)
    q1 = CanvasStateCritic(**kwargs).to(device)
    q2 = CanvasStateCritic(**kwargs).to(device)
    target_q1 = copy.deepcopy(q1).to(device)
    target_q2 = copy.deepcopy(q2).to(device)
    return actor, q1, q2, target_q1, target_q2


def _save_checkpoint(
    *,
    path: Path,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
    agent: CanvasSAC,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    batch: int,
    updates: int,
    best_relative_ce_gain: float,
    eval_metrics: dict[str, float] | None,
) -> None:
    """Save canvas SAC state; best selection matches relative CE reward."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "target_q1": target_q1.state_dict(),
            "target_q2": target_q2.state_dict(),
            "actor_opt": agent.actor_opt.state_dict(),
            "q_opt": agent.q_opt.state_dict(),
            "alpha_opt": agent.alpha_opt.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "args": vars(args),
            "canvas_feature_dim": canvas_feature_dim,
            "batch": batch,
            "updates": updates,
            "best_relative_ce_gain": best_relative_ce_gain,
            "selection_metric": "eval/reward",
            "eval_metrics": eval_metrics or {},
            "state_representation": "current_canvas_layernorm_with_viewpoint_history",
        },
        path,
    )


def _load_resume(
    *,
    args: argparse.Namespace,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
    agent: CanvasSAC,
) -> tuple[int, int, float]:
    """Resume a canvas SAC checkpoint while keeping every network trainable."""
    if args.resume is None:
        return 1, 0, float("-inf")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint["q2"])
    target_q1.load_state_dict(checkpoint.get("target_q1", q1.state_dict()))
    target_q2.load_state_dict(checkpoint.get("target_q2", q2.state_dict()))
    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "q_opt" in checkpoint:
        agent.q_opt.load_state_dict(checkpoint["q_opt"])
    if "alpha_opt" in checkpoint:
        agent.alpha_opt.load_state_dict(checkpoint["alpha_opt"])
    if "log_alpha" in checkpoint:
        agent.log_alpha.data.copy_(checkpoint["log_alpha"])
    return (
        int(checkpoint.get("batch", 0)) + 1,
        int(checkpoint.get("updates", 0)),
        float(
            checkpoint.get(
                "best_relative_ce_gain",
                checkpoint.get("best_ce_gain", float("-inf")),
            )
        ),
    )


def _checkpoint_module_state(
    checkpoint: object,
    key: str,
    *,
    path: Path,
) -> dict[str, torch.Tensor]:
    """Extract either a named module state dict or a bare state dict."""
    if isinstance(checkpoint, dict) and key in checkpoint:
        state = checkpoint[key]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict for {key} in {path}.")
    return state


def _load_pretrained_initializers(
    *,
    args: argparse.Namespace,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    target_q1: CanvasStateCritic,
    target_q2: CanvasStateCritic,
) -> None:
    """Initialize SAC modules from pretrained actor/critic checkpoints.

    This deliberately avoids optimizer, alpha, replay, and batch counters so
    pretrained BC/critic weights can seed a fresh SAC run without pretending to
    be a full training resume.
    """
    if args.resume is not None:
        if (
            args.init_actor_checkpoint is not None
            or args.init_critic_checkpoint is not None
        ):
            raise ValueError(
                "--resume cannot be combined with --init-actor-checkpoint or "
                "--init-critic-checkpoint; resume already restores all SAC state."
            )
        return

    if args.init_actor_checkpoint is not None:
        checkpoint = torch.load(
            args.init_actor_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        actor.load_state_dict(
            _checkpoint_module_state(
                checkpoint,
                "actor",
                path=args.init_actor_checkpoint,
            )
        )
        print(f"Initialized canvas SAC actor from {args.init_actor_checkpoint}")

    if args.init_critic_checkpoint is not None:
        checkpoint = torch.load(
            args.init_critic_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "--init-critic-checkpoint expects a dict checkpoint with q1/q2 keys."
            )
        if "q1" not in checkpoint:
            raise ValueError(
                f"Expected q1 in critic checkpoint: {args.init_critic_checkpoint}"
            )
        q1.load_state_dict(checkpoint["q1"])
        q2.load_state_dict(checkpoint.get("q2", checkpoint["q1"]))
        target_q1.load_state_dict(q1.state_dict())
        target_q2.load_state_dict(q2.state_dict())
        print(f"Initialized canvas SAC critics from {args.init_critic_checkpoint}")


def _eval_canvas_sac_batch(
    *,
    actor: CanvasStateActor,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    canvit_dtype: torch.dtype,
    acc: mIoUAccumulator,
    scale_sums: list[float],
    scale_counts: list[int],
    entropy_points: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out deterministic image-dependent SAC over a validation batch."""
    device = images.device
    batch_size = images.shape[0]
    state = model.init_state(batch_size=batch_size, canvas_grid_size=cfg.canvas_grid_size)
    coords, lengths = empty_viewpoint_history(
        batch_size=batch_size,
        max_steps=args.max_history,
        device=device,
    )
    full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
    with torch.inference_mode():
        full_glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        ).to(dtype=canvit_dtype)
        full_out = model(
            glimpse=full_glimpse,
            state=state,
            viewpoint=full_vp,
        )
        state = full_out.state
        initial_ce, _ = _segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg
        )
        canvas_summary = canvas_layernorm_spatial(
            model=model,
            state=state,
            canvas_grid_size=cfg.canvas_grid_size,
        )
    coords, lengths = append_viewpoint_history(
        coords=coords,
        lengths=lengths,
        viewpoint=full_vp,
        step=0,
    )
    for step_idx in range(args.t):
        obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
        with torch.no_grad():
            action = actor.deterministic_action(obs)
        vp = action_to_viewpoint(action, min_scale=args.min_scale)
        entropy_points.append(
            torch.cat([vp.centers, vp.scales[:, None]], dim=1).detach().cpu().numpy()
        )
        scale_sums[step_idx] += float(vp.scales.detach().sum().item())
        scale_counts[step_idx] += batch_size
        with torch.inference_mode():
            glimpse = sample_at_viewpoint(
                spatial=images,
                viewpoint=vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ).to(dtype=canvit_dtype)
            out = model(
                glimpse=glimpse,
                state=state,
                viewpoint=vp,
            )
            state = out.state
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=cfg.canvas_grid_size,
            )
        coords, lengths = append_viewpoint_history(
            coords=coords,
            lengths=lengths,
            viewpoint=vp,
            step=step_idx + 1,
        )
    with torch.inference_mode():
        final_ce, _ = _segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg, acc=acc
        )
    return initial_ce, final_ce


def evaluate(
    *,
    actor: CanvasStateActor,
    eval_loader: DataLoader,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    canvit_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate Random, EG-C2F, and canvas SAC on a fixed validation subset."""
    eval_random = not args.skip_eval_random
    eval_egc2f = not args.skip_eval_egc2f
    random_acc = (
        mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) if eval_random else None
    )
    egc2f_acc = (
        mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) if eval_egc2f else None
    )
    sac_acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    ce_sums = defaultdict(float)
    n_images = 0
    sac_scale_sums = [0.0 for _ in range(args.t)]
    sac_scale_counts = [0 for _ in range(args.t)]
    sac_entropy_points: list[np.ndarray] = []

    for images, masks in tqdm(eval_loader, desc="Evaluating canvas SAC", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]
        n_images += batch_size

        if eval_random:
            assert random_acc is not None
            rand_initial, rand_final = _eval_random_batch(
                model=model,
                probe=probe,
                images=images,
                masks=masks,
                cfg=cfg,
                args=args,
                canvit_dtype=canvit_dtype,
                acc=random_acc,
            )
            ce_sums["random_initial"] += float(rand_initial.sum().item())
            ce_sums["random_final"] += float(rand_final.sum().item())
        if eval_egc2f:
            assert egc2f_acc is not None
            eg_initial, eg_final = _eval_egc2f_batch(
                model=model,
                probe=probe,
                images=images,
                masks=masks,
                cfg=cfg,
                args=args,
                canvit_dtype=canvit_dtype,
                acc=egc2f_acc,
            )
            ce_sums["egc2f_initial"] += float(eg_initial.sum().item())
            ce_sums["egc2f_final"] += float(eg_final.sum().item())
        sac_initial, sac_final = _eval_canvas_sac_batch(
            actor=actor,
            model=model,
            probe=probe,
            images=images,
            masks=masks,
            cfg=cfg,
            args=args,
            canvas_feature_dim=canvas_feature_dim,
            canvit_dtype=canvit_dtype,
            acc=sac_acc,
            scale_sums=sac_scale_sums,
            scale_counts=sac_scale_counts,
            entropy_points=sac_entropy_points,
        )
        ce_sums["sac_initial"] += float(sac_initial.sum().item())
        ce_sums["sac_final"] += float(sac_final.sum().item())

    random_miou = float(random_acc.compute()) if random_acc is not None else math.nan
    egc2f_miou = float(egc2f_acc.compute()) if egc2f_acc is not None else math.nan
    sac_miou = float(sac_acc.compute())
    random_ce = ce_sums["random_final"] / max(n_images, 1) if eval_random else math.nan
    egc2f_ce = ce_sums["egc2f_final"] / max(n_images, 1) if eval_egc2f else math.nan
    sac_initial_ce = ce_sums["sac_initial"] / max(n_images, 1)
    sac_ce = ce_sums["sac_final"] / max(n_images, 1)
    ce_gain = sac_initial_ce - sac_ce
    eval_reward = ce_gain / max(sac_initial_ce, 1e-12)
    metrics = {
        "eval/random_miou": random_miou,
        "eval/egc2f_miou": egc2f_miou,
        "eval/sac_miou": sac_miou,
        "eval/random_final_ce": random_ce,
        "eval/egc2f_final_ce": egc2f_ce,
        "eval/sac_final_ce": sac_ce,
        "eval/final_miou": sac_miou,
        "eval/final_ce": sac_ce,
        "eval/miou_gain": sac_miou - random_miou,
        "eval/ce_gain": ce_gain,
        "eval/reward": eval_reward,
        "eval/relative_ce_gain": eval_reward,
        "eval/random_ce_gain": (
            ce_sums["random_initial"] / max(n_images, 1) - random_ce
            if eval_random
            else math.nan
        ),
        "eval/egc2f_ce_gain": (
            ce_sums["egc2f_initial"] / max(n_images, 1) - egc2f_ce
            if eval_egc2f
            else math.nan
        ),
        "eval/sac_vs_random": sac_miou - random_miou,
        "eval/sac_vs_egc2f": sac_miou - egc2f_miou,
        "eval/sac_viewpoint_entropy": _viewpoint_entropy(
            sac_entropy_points,
            bins=args.viewpoint_entropy_bins,
        ),
    }
    metrics.update(
        {
            "final_miou": metrics["eval/final_miou"],
            "final_ce": metrics["eval/final_ce"],
            "miou_gain": metrics["eval/miou_gain"],
            "ce_gain": metrics["eval/ce_gain"],
            "reward": metrics["eval/reward"],
            "relative_ce_gain": metrics["eval/relative_ce_gain"],
            "sac_vs_random": metrics["eval/sac_vs_random"],
            "sac_vs_egc2f": metrics["eval/sac_vs_egc2f"],
            "viewpoint_entropy": metrics["eval/sac_viewpoint_entropy"],
        }
    )
    for step_idx in range(args.t):
        metrics[f"eval/sac_mean_scale_by_t{step_idx + 1}"] = (
            sac_scale_sums[step_idx] / max(sac_scale_counts[step_idx], 1)
        )
    return metrics


def _split_eval_metrics(metrics: dict[str, float], split: str) -> dict[str, float]:
    """Return explicit split-prefixed aliases for eval metrics."""
    prefix = f"eval/{split}/"
    return {
        f"{prefix}{name.removeprefix('eval/')}": value
        for name, value in metrics.items()
        if name.startswith("eval/")
    }


def _combine_eval_metrics(
    *,
    selected_metrics: dict[str, float],
    train_metrics: dict[str, float],
    selected_split: str,
    train_split: str,
) -> dict[str, float]:
    """Combine eval metrics with clear split names while preserving old aliases."""
    combined = dict(selected_metrics)
    combined.update(_split_eval_metrics(selected_metrics, selected_split))
    combined.update(_split_eval_metrics(train_metrics, train_split))
    return combined


def _maybe_visualize_reward_maps(
    *,
    actor: CanvasStateActor,
    q1: CanvasStateCritic,
    q2: CanvasStateCritic,
    eval_dataset,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    device: torch.device,
    canvit_dtype: torch.dtype,
    update_count: int,
    comet_exp,
) -> None:
    """Optionally save live canvas SAC reward/Q maps after validation."""
    if args.reward_map_images <= 0:
        return
    # Fixed by Codex on 2026-06-23
    # Problem: Canvas SAC validation logged scalar metrics but could not show
    # whether the image-dependent critic's Q landscape matched true CE gains.
    # Solution: reuse the reward-map renderer with live CanvasStateActor/Critic
    # modules and explicitly select the canvas policy path.
    # Result: Each enabled eval pass can save and optionally Comet-log reward
    # maps for the same canvas-dependent networks being validated.
    indices = list(range(min(args.reward_map_images, len(eval_dataset))))
    paths = visualize_reward_maps_for_indices(
        actor=actor,
        q1=q1,
        q2=q2,
        dataset=eval_dataset,
        indices=indices,
        model=model,
        probe=probe,
        cfg=cfg,
        device=device,
        min_scale=args.min_scale,
        scales=_parse_scales(args.reward_map_scales),
        grid_size=args.reward_map_grid_size,
        chunk_size=args.reward_map_chunk_size,
        output_dir=args.reward_map_output_dir,
        split_label=args.eval_split,
        title_prefix=f"Canvas SAC validation reward map update={update_count}",
        policy_kind="canvas",
        max_history=args.max_history,
        output_name_suffix=f"update_{update_count:06d}",
    )
    paths.extend(
        visualize_canvas_policy_for_indices(
            actor=actor,
            dataset=eval_dataset,
            indices=indices,
            model=model,
            probe=probe,
            cfg=cfg,
            device=device,
            t=args.t,
            max_history=args.max_history,
            min_scale=args.min_scale,
            output_dir=args.reward_map_output_dir,
            split_label=args.eval_split,
            title_prefix=f"Canvas SAC validation policy update={update_count}",
            canvit_dtype=canvit_dtype,
            output_name_suffix=f"update_{update_count:06d}",
        )
    )
    if comet_exp is not None:
        for path in paths:
            comet_exp.log_image(str(path), name=path.name, step=update_count)


def train_once(args: argparse.Namespace) -> float:
    """Run full current-canvas SAC and return best relative eval CE gain."""
    if args.t < 0:
        raise ValueError("--t must be non-negative.")
    if args.max_history < args.t + 1:
        raise ValueError("--max-history must be at least t+1.")
    if args.t + 1 > 21:
        raise ValueError("EG-C2F evaluation requires --t <= 20.")
    if args.reward_map_images < 0:
        raise ValueError("--reward-map-images must be non-negative.")
    if args.reward_map_grid_size < 2:
        raise ValueError("--reward-map-grid-size must be >= 2.")
    if args.reward_map_chunk_size < 1:
        raise ValueError("--reward-map-chunk-size must be positive.")
    if args.reward_map_interval is not None and args.reward_map_interval < 1:
        raise ValueError("--reward-map-interval must be positive.")
    _parse_scales(args.reward_map_scales)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    cfg = CanViTEnvConfig()
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    train_dataset = _build_segmentation_dataset(
        args=args,
        cfg=cfg,
        split=args.split,
        img_tf=img_tf,
        mask_tf=mask_tf,
    )
    train_dataset = _limit_dataset(train_dataset, args.max_samples)
    train_eval_dataset = _limit_dataset(train_dataset, args.eval_images)
    eval_dataset = _build_segmentation_dataset(
        args=args,
        cfg=cfg,
        split=args.eval_split,
        img_tf=img_tf,
        mask_tf=mask_tf,
    )
    eval_dataset = _limit_dataset(eval_dataset, args.eval_images)
    if len(train_dataset) == 0 or len(train_eval_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("Train and validation datasets must be non-empty.")
    loader_kwargs = _dataloader_kwargs(args, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=True),
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    probe_repo = args.probe_repo or resolve_canvit_repo(
        f"probe-ade20k-40k-s512-c{cfg.canvas_grid_size}-in21k"
    )
    print(f"Loading CanViT segmentation model with probe: {probe_repo}")
    seg = (
        CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo=cfg.checkpoint,
            probe_repo=probe_repo,
        )
        .eval()
        .to(device)
    )
    model = seg.canvit
    probe = seg.head
    canvit_dtype = configure_frozen_canvit_precision(
        model=model,
        probe=probe,
        requested=args.canvit_dtype,
        device=device,
    )
    print(f"CanViT inference dtype: {canvit_dtype}")
    for param in model.parameters():
        param.requires_grad_(False)
    for param in probe.parameters():
        param.requires_grad_(False)

    canvas_feature_dim = int(model.canvas_dim)
    actor, q1, q2, target_q1, target_q2 = _build_networks(
        args=args,
        canvas_feature_dim=canvas_feature_dim,
        device=device,
    )
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
    start_batch, update_count, best_relative_ce_gain = _load_resume(
        args=args,
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        agent=agent,
    )
    _load_pretrained_initializers(
        args=args,
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
    )
    replay_canvas_bytes = _replay_canvas_bytes(
        capacity=args.buffer_size,
        canvas_feature_dim=canvas_feature_dim,
        canvas_grid_size=cfg.canvas_grid_size,
    )
    replay_device = _resolve_replay_device(
        train_device=device,
        replay_bytes=replay_canvas_bytes,
    )
    _validate_replay_memory(
        storage_device=replay_device,
        replay_bytes=replay_canvas_bytes,
    )
    print(
        "Replay storage: "
        f"device={replay_device}, dtype={REPLAY_STORAGE_DTYPE}, "
        f"canvas_bytes={replay_canvas_bytes / 1024**3:.2f} GiB"
    )
    replay = CanvasReplayBuffer(
        capacity=args.buffer_size,
        max_history=args.max_history,
        canvas_feature_dim=canvas_feature_dim,
        canvas_grid_size=cfg.canvas_grid_size,
        storage_device=replay_device,
    )
    comet_exp = _make_comet_experiment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_iter = iter(train_loader)
    train_windows: dict[str, list[float]] = defaultdict(list)
    reward_window: list[float] = []
    entropy_points: list[np.ndarray] = []
    scale_sums = [0.0 for _ in range(args.t)]
    scale_counts = [0 for _ in range(args.t)]
    latest_metrics: dict[str, float] | None = None
    next_eval_update = max(args.eval_interval, 1)
    reward_map_interval = max(args.reward_map_interval or args.eval_interval, 1)
    next_reward_map_update = reward_map_interval
    last_reward_map_update: int | None = None
    elapsed_seconds = 0.0
    committed_glimpses = 0

    pbar = tqdm(range(start_batch, args.batches + 1), desc="Training canvas SAC")
    for batch_idx in pbar:
        _sync_for_timing(device)
        batch_start = time.perf_counter()
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]

        state = model.init_state(batch_size=batch_size, canvas_grid_size=cfg.canvas_grid_size)
        coords, lengths = empty_viewpoint_history(
            batch_size=batch_size,
            max_steps=args.max_history,
            device=device,
        )
        full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
        with torch.inference_mode():
            full_glimpse = sample_at_viewpoint(
                spatial=images,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ).to(dtype=canvit_dtype)
            full_out = model(
                glimpse=full_glimpse,
                state=state,
                viewpoint=full_vp,
            )
            state = full_out.state
            current_ce, _ = _segmentation_metrics(
                model=model, probe=probe, state=state, masks=masks, cfg=cfg
            )
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=cfg.canvas_grid_size,
            )
        coords, lengths = append_viewpoint_history(
            coords=coords,
            lengths=lengths,
            viewpoint=full_vp,
            step=0,
        )

        for step_idx in range(args.t):
            obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
            if replay.size < args.learning_starts:
                action = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)
            else:
                with torch.no_grad():
                    action, log_prob = actor.sample(obs)
                    train_windows["actor/log_prob"].append(float(log_prob.mean().item()))
                    train_windows["actor/entropy"].append(float((-log_prob).mean().item()))
                    train_windows["actor/action_std"].append(
                        float(action.std(unbiased=False).item())
                    )
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
                    spatial=images,
                    viewpoint=vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                ).to(dtype=canvit_dtype)
                out = model(
                    glimpse=glimpse,
                    state=state,
                    viewpoint=vp,
                )
                next_ce, _ = _segmentation_metrics(
                    model=model, probe=probe, state=out.state, masks=masks, cfg=cfg
                )
                next_canvas_summary = canvas_layernorm_spatial(
                    model=model,
                    state=out.state,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
            reward = relative_ce_reduction(current_ce, next_ce)
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
            current_ce = next_ce
            canvas_summary = next_canvas_summary

        _sync_for_timing(device)
        elapsed_seconds += time.perf_counter() - batch_start
        committed_glimpses += batch_size * (args.t + 1)
        glimpses_per_sec = committed_glimpses / max(elapsed_seconds, 1e-12)

        if replay.size >= args.learning_starts:
            for _ in range(args.updates_per_batch):
                metrics = agent.update(replay.sample(args.replay_batch_size, device))
                update_count += 1
                for key, value in metrics.items():
                    train_windows[key].append(value)

                if update_count % args.comet_log_interval == 0:
                    train_metrics = {
                        key: float(np.mean(values))
                        for key, values in train_windows.items()
                        if values
                    }
                    if reward_window:
                        rewards_np = np.asarray(reward_window, dtype=np.float64)
                        train_metrics.update(
                            {
                                "train/online_reward/mean": float(
                                    np.mean(rewards_np)
                                ),
                                "train/online_reward/std": float(
                                    np.std(rewards_np)
                                ),
                                "train/online_reward/max": float(
                                    np.max(rewards_np)
                                ),
                                "train/online_reward/min": float(
                                    np.min(rewards_np)
                                ),
                            }
                        )
                    train_metrics["train/batch"] = float(batch_idx)
                    train_metrics["train/updates"] = float(update_count)
                    train_metrics["train/replay_size"] = float(replay.size)
                    train_metrics["throughput/glimpses_per_sec"] = glimpses_per_sec
                    train_metrics["throughput/committed_glimpses_per_sec"] = glimpses_per_sec
                    train_metrics["train/viewpoint_entropy"] = _viewpoint_entropy(
                        entropy_points,
                        bins=args.viewpoint_entropy_bins,
                    )
                    for step in range(args.t):
                        train_metrics[f"train/mean_scale_by_t{step + 1}"] = (
                            scale_sums[step] / max(scale_counts[step], 1)
                        )
                    if comet_exp is not None:
                        comet_exp.log_metrics(train_metrics, step=update_count)
                    train_windows.clear()
                    reward_window.clear()
                    entropy_points.clear()
                    scale_sums = [0.0 for _ in range(args.t)]
                    scale_counts = [0 for _ in range(args.t)]

                if update_count >= next_eval_update:
                    train_eval_metrics = evaluate(
                        actor=actor,
                        eval_loader=train_eval_loader,
                        model=model,
                        probe=probe,
                        cfg=cfg,
                        args=args,
                        canvas_feature_dim=canvas_feature_dim,
                        canvit_dtype=canvit_dtype,
                        device=device,
                    )
                    if args.eval_split == args.split:
                        selected_eval_metrics = train_eval_metrics
                    else:
                        selected_eval_metrics = evaluate(
                            actor=actor,
                            eval_loader=eval_loader,
                            model=model,
                            probe=probe,
                            cfg=cfg,
                            args=args,
                            canvas_feature_dim=canvas_feature_dim,
                            canvit_dtype=canvit_dtype,
                            device=device,
                        )
                    eval_metrics = _combine_eval_metrics(
                        selected_metrics=selected_eval_metrics,
                        train_metrics=train_eval_metrics,
                        selected_split=args.eval_split,
                        train_split=args.split,
                    )
                    latest_metrics = eval_metrics
                    if comet_exp is not None:
                        comet_exp.log_metrics(eval_metrics, step=update_count)
                    current_eval_reward = eval_metrics["eval/reward"]
                    current_train_reward = eval_metrics[f"eval/{args.split}/reward"]
                    current_selected_reward = eval_metrics[
                        f"eval/{args.eval_split}/reward"
                    ]
                    _save_checkpoint(
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
                        best_relative_ce_gain=best_relative_ce_gain,
                        eval_metrics=eval_metrics,
                    )
                    if current_eval_reward > best_relative_ce_gain:
                        best_relative_ce_gain = current_eval_reward
                        _save_checkpoint(
                            path=args.checkpoint_dir / "best.pt",
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
                            best_relative_ce_gain=best_relative_ce_gain,
                            eval_metrics=eval_metrics,
                        )
                    print(
                        f"update={update_count} batch={batch_idx} "
                        f"{args.split}_reward={current_train_reward:+.4f} "
                        f"{args.eval_split}_reward={current_selected_reward:+.4f} "
                        f"ce_gain={eval_metrics['eval/ce_gain']:+.4f} "
                        f"sac_miou={eval_metrics['eval/sac_miou']:.4f}"
                    )
                    next_eval_update += args.eval_interval

                if (
                    args.reward_map_images > 0
                    and update_count >= next_reward_map_update
                ):
                    _maybe_visualize_reward_maps(
                        actor=actor,
                        q1=q1,
                        q2=q2,
                        eval_dataset=eval_dataset,
                        model=model,
                        probe=probe,
                        cfg=cfg,
                        args=args,
                        device=device,
                        canvit_dtype=canvit_dtype,
                        update_count=update_count,
                        comet_exp=comet_exp,
                    )
                    last_reward_map_update = update_count
                    while next_reward_map_update <= update_count:
                        next_reward_map_update += reward_map_interval

        pbar.set_postfix(
            {
                "updates": update_count,
                "replay": replay.size,
                "glimpses/s": f"{glimpses_per_sec:.1f}",
                "best_reward": f"{best_relative_ce_gain:+.4f}"
                if best_relative_ce_gain != float("-inf")
                else "nan",
            }
        )

    if latest_metrics is None:
        train_eval_metrics = evaluate(
            actor=actor,
            eval_loader=train_eval_loader,
            model=model,
            probe=probe,
            cfg=cfg,
            args=args,
            canvas_feature_dim=canvas_feature_dim,
            canvit_dtype=canvit_dtype,
            device=device,
        )
        if args.eval_split == args.split:
            selected_eval_metrics = train_eval_metrics
        else:
            selected_eval_metrics = evaluate(
                actor=actor,
                eval_loader=eval_loader,
                model=model,
                probe=probe,
                cfg=cfg,
                args=args,
                canvas_feature_dim=canvas_feature_dim,
                canvit_dtype=canvit_dtype,
                device=device,
            )
        latest_metrics = _combine_eval_metrics(
            selected_metrics=selected_eval_metrics,
            train_metrics=train_eval_metrics,
            selected_split=args.eval_split,
            train_split=args.split,
        )
        best_relative_ce_gain = max(
            best_relative_ce_gain,
            latest_metrics["eval/reward"],
        )
        if comet_exp is not None:
            comet_exp.log_metrics(latest_metrics, step=update_count)
        if args.reward_map_images > 0 and last_reward_map_update != update_count:
            _maybe_visualize_reward_maps(
                actor=actor,
                q1=q1,
                q2=q2,
                eval_dataset=eval_dataset,
                model=model,
                probe=probe,
                cfg=cfg,
                args=args,
                device=device,
                canvit_dtype=canvit_dtype,
                update_count=update_count,
                comet_exp=comet_exp,
            )
    final_relative_ce_gain = latest_metrics["eval/reward"]
    if final_relative_ce_gain >= best_relative_ce_gain:
        best_relative_ce_gain = final_relative_ce_gain
        _save_checkpoint(
            path=args.checkpoint_dir / "best.pt",
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
            best_relative_ce_gain=best_relative_ce_gain,
            eval_metrics=latest_metrics,
        )
    _save_checkpoint(
        path=args.checkpoint_dir / "latest.pt",
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
        best_relative_ce_gain=best_relative_ce_gain,
        eval_metrics=latest_metrics,
    )
    torch.save(actor.state_dict(), args.checkpoint_dir / "actor_final.pt")
    if comet_exp is not None:
        comet_exp.log_metric("final/reward", latest_metrics["eval/reward"], step=update_count)
        comet_exp.log_metric("final/ce_gain", latest_metrics["eval/ce_gain"], step=update_count)
        comet_exp.log_metric("final/miou", latest_metrics["eval/sac_miou"], step=update_count)
        comet_exp.end()
    print(f"Saved canvas SAC latest checkpoint to {args.checkpoint_dir / 'latest.pt'}")
    print(f"Best eval/reward: {best_relative_ce_gain:+.4f}")
    return best_relative_ce_gain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--eval-split", choices=["training", "validation"], default="validation")
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
        "--skip-eval-random",
        action="store_true",
        help="Skip random-policy baseline rollouts during eval.",
    )
    parser.add_argument(
        "--skip-eval-egc2f",
        action="store_true",
        help="Skip entropy-guided coarse-to-fine baseline rollouts during eval.",
    )
    parser.add_argument("--comet-log-interval", type=int, default=20)
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
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="canvas-sac")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="canvas-sac")
    add_canvas_sac_optuna_args(parser)
    args = parser.parse_args()
    if args.resume is not None:
        if (
            args.init_actor_checkpoint is not None
            or args.init_critic_checkpoint is not None
        ):
            raise ValueError(
                "--resume cannot be combined with --init-actor-checkpoint or "
                "--init-critic-checkpoint."
            )
    return args


def main() -> None:
    args = parse_args()
    if args.optuna_trials:
        run_canvas_sac_optuna(args, train_once)
        return
    train_once(args)


if __name__ == "__main__":
    main()
