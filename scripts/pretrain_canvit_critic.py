"""
Train continuous SAC critics from k-candidate CE-improvement oracle labels.

Usage:
    uv run python scripts/pretrain_canvit_critic.py --batches 100 --max-samples 1 --batch-size 1 --t 1 --k 16 \
        --test-images 8 --checkpoint-dir checkpoints/canvit_critic --experiment-name critic-im1-k16-t1
    uv run python scripts/pretrain_canvit_critic.py --optuna-trials 20 --batches 50
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Any

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
    make_val_transforms,
)
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.sac_models import ViewpointHistoryCritic


def _sync_for_timing(device: torch.device) -> None:
    """Synchronize CUDA kernels before reading throughput timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _viewpoint_to_action(viewpoint: Viewpoint, *, min_scale: float) -> torch.Tensor:
    """Map an upstream Viewpoint back to the SAC tanh action range."""
    centers = viewpoint.centers.float()
    scale_action = 2.0 * (viewpoint.scales.float() - min_scale) / (1.0 - min_scale)
    scale_action = scale_action - 1.0
    return torch.cat([centers, scale_action[:, None]], dim=-1).clamp(-1.0, 1.0)


def _repeat_batch(
    batch: dict[str, torch.Tensor],
    repeats: int,
) -> dict[str, torch.Tensor]:
    """Repeat a one-state actor/critic batch for K candidate actions."""
    return {
        key: value.repeat_interleave(repeats, dim=0)
        for key, value in batch.items()
    }


def _concat_batches(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate per-image critic states into one train batch."""
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in batches[0]
    }


def _empty_history(*, max_steps: int) -> dict[str, Any]:
    """Create an empty fixed-slot viewpoint history on CPU."""
    return {"coords": torch.zeros(max_steps, 3), "length": 0}


def _append_history(
    *,
    history: dict[str, Any],
    viewpoint: Viewpoint,
    max_steps: int,
) -> dict[str, Any]:
    """Append one Viewpoint to the fixed-slot history state."""
    length = int(history["length"])
    if length >= max_steps:
        raise ValueError(
            f"History length {length} reached max_steps={max_steps}; "
            "increase --max-history."
        )
    coords = history["coords"].clone()
    coords[length] = torch.cat(
        [
            viewpoint.centers[0].detach().float().cpu(),
            viewpoint.scales[0:1].detach().float().cpu(),
        ],
        dim=0,
    )
    return {"coords": coords, "length": length + 1}


def _batch_from_history(
    history: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert one CPU viewpoint history into a one-item critic batch."""
    return {
        "coords": history["coords"][None].to(device),
        "lengths": torch.as_tensor([history["length"]], device=device),
    }


def _segmentation_ce(
    *,
    model,
    probe: torch.nn.Module,
    state,
    mask: torch.Tensor,
    canvas_grid_size: int,
) -> float:
    """Decode a canvas state and return pixel-averaged segmentation CE."""
    spatial = model.get_spatial(state.canvas).view(
        mask.shape[0],
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float())
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return float(
        F.cross_entropy(logits, mask.long(), ignore_index=IGNORE_LABEL).detach()
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


def _mean(values: list[float]) -> float:
    """Return 0 for empty metric windows."""
    return sum(values) / len(values) if values else 0.0


def _grad_norm(parameters) -> float:
    """Compute a total gradient norm for one critic."""
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().norm(2).item() ** 2)
    return total ** 0.5


def _make_comet_experiment(args: argparse.Namespace, trial_number: int | None = None):
    """Create a Comet experiment unless disabled for local dry runs."""
    if args.no_comet:
        return None
    if Experiment is None:
        raise RuntimeError(
            "Comet logging is enabled by default, but comet_ml is not installed. "
            "Install comet-ml or run with --no-comet."
        )
    comet_kwargs = dict(
        project_name=args.comet_project,
        auto_param_logging=True,
        auto_metric_logging=True,
    )
    if args.comet_workspace:
        comet_kwargs["workspace"] = args.comet_workspace
    experiment = Experiment(**comet_kwargs)
    name = args.experiment_name or "canvit-critic-ce-greedy"
    if trial_number is not None:
        name = f"{name}-trial-{trial_number}"
    experiment.set_name(name)
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    if trial_number is not None:
        experiment.log_parameter("optuna_trial", trial_number)
    return experiment


def _limit_dataset(dataset, max_samples: int | None, *, offset: int = 0):
    """Restrict datasets like the BC trainer while allowing a holdout offset."""
    if max_samples is None:
        return dataset
    start = min(offset, len(dataset))
    stop = min(start + max_samples, len(dataset))
    return torch.utils.data.Subset(dataset, range(start, stop))


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return latest/best checkpoint paths inside the configured directory."""
    return args.checkpoint_dir / "latest.pt", args.checkpoint_dir / "best.pt"


def _save_checkpoint(
    *,
    path: Path,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    batch: int,
    n_labels: int,
    metric: float,
    probe_repo: str,
    cfg: CanViTEnvConfig,
) -> None:
    """Save critic training state for resume/eval."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "batch": batch,
            "episode": batch,
            "n_labels": n_labels,
            "metric": metric,
            "selection_metric": "critic/top1_match",
            "metadata": {
                "probe_repo": probe_repo,
                "model_repo": cfg.checkpoint,
                "canvas_grid_size": cfg.canvas_grid_size,
                "glimpse_size_px": cfg.glimpse_size_px,
                "n_labels": n_labels,
                "target": "gamma0_ce_reduction",
                "gamma": 0.0,
                "state_representation": "viewpoint_history",
            },
        },
        path,
    )


def _load_resume_checkpoint(
    *,
    path: Path,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    opt: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Load critic training state and return next batch plus label count."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "q1" not in checkpoint:
        raise ValueError(f"Expected critic checkpoint with q1/q2 keys: {path}")
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint.get("q2", checkpoint["q1"]))
    if "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])
    saved_batch = int(checkpoint.get("batch", checkpoint.get("episode", 0)))
    n_labels = int(checkpoint.get("n_labels", 0))
    return saved_batch + 1, n_labels


def _build_critics(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ViewpointHistoryCritic, ViewpointHistoryCritic]:
    """Construct twin Q(history, action) critics for global-prior validation."""
    kwargs = dict(
        d_model=args.d_model,
        max_steps=args.max_history,
        rff_dim=args.rff_dim,
        rff_seed=args.rff_seed,
    )
    q1 = ViewpointHistoryCritic(**kwargs).to(device)
    q2 = ViewpointHistoryCritic(**kwargs).to(device)
    return q1, q2


def _collect_state_metrics(
    *,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    batch: dict[str, torch.Tensor],
    action_batch: torch.Tensor,
    reward_batch: torch.Tensor,
) -> dict[str, float]:
    """Compute reward, Q calibration, ranking, and oracle-order metrics."""
    with torch.no_grad():
        q1_values = q1(batch, action_batch)
        q2_values = q2(batch, action_batch)
        q_values = torch.minimum(q1_values, q2_values)

    rewards_np = reward_batch.detach().cpu().numpy().astype(np.float64)
    q_np = q_values.detach().cpu().numpy().astype(np.float64)
    q1_np = q1_values.detach().cpu().numpy().astype(np.float64)
    q2_np = q2_values.detach().cpu().numpy().astype(np.float64)
    best_reward_idx = int(np.argmax(rewards_np))
    worst_reward_idx = int(np.argmin(rewards_np))
    best_q_idx = int(np.argmax(q_np))
    random_idx = random.randrange(len(rewards_np))
    reward_desc = np.argsort(rewards_np)[::-1]
    top3 = set(reward_desc[: min(3, len(reward_desc))])
    top5 = set(reward_desc[: min(5, len(reward_desc))])
    q_best = float(q_np[best_reward_idx])
    q_worst = float(q_np[worst_reward_idx])
    return {
        "reward/mean": float(np.mean(rewards_np)),
        "reward/std": float(np.std(rewards_np)),
        "reward/min": float(np.min(rewards_np)),
        "reward/max": float(np.max(rewards_np)),
        "reward/candidate_std": float(np.std(rewards_np)),
        "reward/candidate_range": float(np.max(rewards_np) - np.min(rewards_np)),
        "q1/mean": float(np.mean(q1_np)),
        "q1/std": float(np.std(q1_np)),
        "q2/mean": float(np.mean(q2_np)),
        "q2/std": float(np.std(q2_np)),
        "critic/spearman": _spearman(q_np, rewards_np),
        "critic/pearson": _pearson(q_np, rewards_np),
        "critic/top1_match": float(best_q_idx == best_reward_idx),
        "critic/top3_match": float(best_q_idx in top3),
        "critic/top5_match": float(best_q_idx in top5),
        "critic/top1_vs_random": float(rewards_np[best_q_idx] > rewards_np[random_idx]),
        "q/best": q_best,
        "q/random": float(q_np[random_idx]),
        "q/worst": q_worst,
        "q_gap_best_worst": q_best - q_worst,
    }


def _sample_candidates(
    *,
    image: torch.Tensor,
    mask: torch.Tensor,
    model,
    probe: torch.nn.Module,
    state,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    current_ce: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[float, Viewpoint, Any, float]]]:
    """Generate K actions and CE-reduction labels from the same current state."""
    candidates = random_viewpoints(
        batch_size=1,
        device=device,
        n_viewpoints=args.k,
        min_scale=args.min_scale,
        max_scale=1.0,
        start_with_full_scene=False,
    )
    actions = []
    rewards = []
    records = []
    with torch.inference_mode():
        for candidate_vp in candidates:
            glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=candidate_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            )
            out = model(glimpse=glimpse, state=state, viewpoint=candidate_vp)
            ce_after = _segmentation_ce(
                model=model,
                state=out.state,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            reward = current_ce - ce_after
            action = _viewpoint_to_action(candidate_vp, min_scale=args.min_scale)
            actions.append(action.squeeze(0))
            rewards.append(reward)
            records.append((reward, candidate_vp, out, ce_after))
    return (
        torch.stack(actions).to(device),
        torch.as_tensor(rewards, device=device, dtype=torch.float32),
        records,
    )


def _evaluate_once(
    *,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    eval_iter,
    eval_loader: DataLoader,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], Any]:
    """Sample one held-out image and evaluate ranking on CE-improvement labels."""
    try:
        image, mask = next(eval_iter)
    except StopIteration:
        eval_iter = iter(eval_loader)
        image, mask = next(eval_iter)
    image = image.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
    history = _empty_history(max_steps=args.max_history)

    full_vp = Viewpoint.full_scene(batch_size=1, device=device)
    with torch.inference_mode():
        full_glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        full_out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
        state = full_out.state
        current_ce = _segmentation_ce(
            model=model,
            state=state,
            probe=probe,
            mask=mask,
            canvas_grid_size=cfg.canvas_grid_size,
        )
    history = _append_history(
        history=history,
        viewpoint=full_vp,
        max_steps=args.max_history,
    )
    batch = _batch_from_history(history, device=device)
    actions, rewards, _ = _sample_candidates(
        image=image,
        mask=mask,
        model=model,
        probe=probe,
        state=state,
        cfg=cfg,
        args=args,
        current_ce=current_ce,
        device=device,
    )
    metrics = _collect_state_metrics(
        q1=q1,
        q2=q2,
        batch=_repeat_batch(batch, args.k),
        action_batch=actions,
        reward_batch=rewards,
    )
    return {
        "eval/critic_spearman": metrics["critic/spearman"],
        "eval/critic_pearson": metrics["critic/pearson"],
        "eval/critic_top1_match": metrics["critic/top1_match"],
        "eval/critic_top3_match": metrics["critic/top3_match"],
        "eval/critic_top5_match": metrics["critic/top5_match"],
        "eval/critic_top1_vs_random": metrics["critic/top1_vs_random"],
        "eval/q_gap_best_worst": metrics["q_gap_best_worst"],
    }, eval_iter


def _evaluate_many(
    *,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    eval_iter,
    eval_loader: DataLoader,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], Any]:
    """Average held-out ranking metrics across --test-images images."""
    windows: dict[str, list[float]] = {}
    for _ in range(args.test_images):
        metrics, eval_iter = _evaluate_once(
            q1=q1,
            q2=q2,
            eval_iter=eval_iter,
            eval_loader=eval_loader,
            model=model,
            probe=probe,
            cfg=cfg,
            args=args,
            device=device,
        )
        for key, value in metrics.items():
            if not np.isnan(value):
                windows.setdefault(key, []).append(float(value))
    windows["eval/test_images"] = [float(args.test_images)]
    return {key: _mean(values) for key, values in windows.items()}, eval_iter


def train_once(
    args: argparse.Namespace,
    *,
    trial_number: int | None = None,
) -> float:
    """Run one gamma=0 CE critic validation experiment."""
    if args.t < 0 or args.k <= 1:
        raise ValueError("Require --t >= 0 and --k > 1.")
    if args.batches <= 0 or args.batch_size <= 0:
        raise ValueError("Require --batches > 0 and --batch-size > 0.")
    if args.min_scale <= 0 or args.min_scale >= 1:
        raise ValueError("Require 0 < --min-scale < 1.")
    if args.comet_log_interval <= 0:
        raise ValueError("--comet-log-interval must be positive.")
    if args.test_images <= 0:
        raise ValueError("--test-images must be positive.")
    if args.max_history < args.t + 1:
        raise ValueError(
            f"--max-history ({args.max_history}) must be >= t+1 ({args.t + 1})."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    cfg = CanViTEnvConfig()
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    train_dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    train_dataset = _limit_dataset(train_dataset, args.max_samples)
    eval_dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.eval_split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    eval_offset = args.max_samples if args.eval_split == args.split else 0
    eval_dataset = _limit_dataset(
        eval_dataset,
        args.eval_samples,
        offset=eval_offset,
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("Train and eval datasets must each contain at least one image.")
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=True),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
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
    for param in model.parameters():
        param.requires_grad_(False)
    for param in probe.parameters():
        param.requires_grad_(False)

    q1, q2 = _build_critics(args, device)
    latest_path, best_path = _checkpoint_paths(args)
    opt = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    comet_exp = _make_comet_experiment(args, trial_number)

    data_iter = iter(loader)
    eval_iter = iter(eval_loader)
    start_batch = 1
    n_labels = 0
    if args.resume is not None:
        print(f"Resuming critic checkpoint: {args.resume}")
        start_batch, n_labels = _load_resume_checkpoint(
            path=args.resume,
            q1=q1,
            q2=q2,
            opt=opt,
        )
        print(f"Resumed at batch={start_batch} labels={n_labels}")

    metric_windows: dict[str, list[float]] = {}
    best_top1 = float("-inf")
    elapsed_seconds = 0.0
    committed_glimpses = 0
    candidate_glimpses = 0

    pbar = tqdm(
        range(start_batch, args.batches + 1),
        desc="Training CE critic",
        miniters=args.comet_log_interval,
        maxinterval=float("inf"),
    )
    for batch_idx in pbar:
        _sync_for_timing(device)
        batch_start = time.perf_counter()
        try:
            images, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, masks = next(data_iter)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        actual_batch_size = int(images.shape[0])
        samples: list[dict[str, Any]] = []

        for sample_idx in range(actual_batch_size):
            image = images[sample_idx : sample_idx + 1]
            mask = masks[sample_idx : sample_idx + 1]
            state = model.init_state(
                batch_size=1,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            history = _empty_history(max_steps=args.max_history)

            full_vp = Viewpoint.full_scene(batch_size=1, device=device)
            full_glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            )
            with torch.inference_mode():
                full_out = model(
                    glimpse=full_glimpse,
                    state=state,
                    viewpoint=full_vp,
                )
                state = full_out.state
                current_ce = _segmentation_ce(
                    model=model,
                    state=state,
                    probe=probe,
                    mask=mask,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
            history = _append_history(
                history=history,
                viewpoint=full_vp,
                max_steps=args.max_history,
            )
            samples.append(
                {
                    "image": image,
                    "mask": mask,
                    "state": state,
                    "history": history,
                    "current_ce": current_ce,
                }
            )

        batch_loss = 0.0
        for step_idx in range(args.t):
            step_obs: list[dict[str, torch.Tensor]] = []
            step_actions: list[torch.Tensor] = []
            step_rewards: list[torch.Tensor] = []
            step_records: list[list[tuple[float, Viewpoint, Any, float]]] = []

            for sample in samples:
                state_batch = _batch_from_history(sample["history"], device=device)
                actions, rewards, candidate_records = _sample_candidates(
                    image=sample["image"],
                    mask=sample["mask"],
                    model=model,
                    probe=probe,
                    state=sample["state"],
                    cfg=cfg,
                    args=args,
                    current_ce=sample["current_ce"],
                    device=device,
                )
                step_obs.append(state_batch)
                step_actions.append(actions)
                step_rewards.append(rewards)
                step_records.append(candidate_records)

            obs_batch = _concat_batches(
                [_repeat_batch(state_batch, args.k) for state_batch in step_obs]
            )
            action_batch = torch.cat(step_actions, dim=0)
            reward_batch = torch.cat(step_rewards, dim=0)
            q1_pred = q1(obs_batch, action_batch)
            q2_pred = q2(obs_batch, action_batch)
            q1_loss = F.mse_loss(q1_pred, reward_batch)
            q2_loss = F.mse_loss(q2_pred, reward_batch)
            loss = q1_loss + q2_loss
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(q1.parameters()) + list(q2.parameters()),
                    args.grad_clip,
                )
            grad_q1 = _grad_norm(q1.parameters())
            grad_q2 = _grad_norm(q2.parameters())
            opt.step()
            n_labels += actual_batch_size * args.k
            batch_loss += float(loss.detach().item())

            for key, value in {
                "critic/q1_mse": float(q1_loss.detach().item()),
                "critic/q2_mse": float(q2_loss.detach().item()),
                "critic/mean_mse": float(
                    0.5 * (q1_loss.detach().item() + q2_loss.detach().item())
                ),
                "grad/q1_norm": grad_q1,
                "grad/q2_norm": grad_q2,
            }.items():
                metric_windows.setdefault(key, []).append(value)

            for sample, state_batch, actions, rewards, candidate_records in zip(
                samples,
                step_obs,
                step_actions,
                step_rewards,
                step_records,
                strict=True,
            ):
                state_metrics = _collect_state_metrics(
                    q1=q1,
                    q2=q2,
                    batch=_repeat_batch(state_batch, args.k),
                    action_batch=actions,
                    reward_batch=rewards,
                )
                state_metrics.update(
                    {
                        "state/timestep": float(step_idx + 1),
                        "state/history_length": float(
                            state_batch["lengths"][0].item()
                        ),
                    }
                )
                for key, value in state_metrics.items():
                    if not np.isnan(value):
                        metric_windows.setdefault(key, []).append(float(value))

                if args.rollout_policy == "best":
                    _, next_vp, next_out, next_ce = max(
                        candidate_records,
                        key=lambda item: item[0],
                    )
                else:
                    _, next_vp, next_out, next_ce = random.choice(candidate_records)
                sample["state"] = next_out.state
                sample["history"] = _append_history(
                    history=sample["history"],
                    viewpoint=next_vp,
                    max_steps=args.max_history,
                )
                sample["current_ce"] = next_ce

        _sync_for_timing(device)
        elapsed_seconds += time.perf_counter() - batch_start
        committed_glimpses += actual_batch_size * (args.t + 1)
        candidate_glimpses += actual_batch_size * (1 + args.t * args.k)
        committed_gps = committed_glimpses / max(elapsed_seconds, 1e-12)
        candidate_gps = candidate_glimpses / max(elapsed_seconds, 1e-12)
        pbar.set_postfix(
            {
                "loss": f"{batch_loss:.4f}",
                "glimpses/s": f"{committed_gps:.1f}",
                "cand/s": f"{candidate_gps:.1f}",
            }
        )

        should_log = batch_idx % args.comet_log_interval == 0
        if should_log:
            eval_metrics, eval_iter = _evaluate_many(
                q1=q1,
                q2=q2,
                eval_iter=eval_iter,
                eval_loader=eval_loader,
                model=model,
                probe=probe,
                cfg=cfg,
                args=args,
                device=device,
            )
            metrics = {key: _mean(values) for key, values in metric_windows.items()}
            metrics.update(eval_metrics)
            metrics["train/n_labels"] = float(n_labels)
            metrics["train/batch"] = float(batch_idx)
            metrics["throughput/committed_glimpses_per_sec"] = committed_gps
            metrics["throughput/candidate_glimpses_per_sec"] = candidate_gps
            if comet_exp is not None:
                comet_exp.log_metrics(metrics, step=batch_idx)
            print(
                f"batch={batch_idx} labels={n_labels} "
                f"top1={metrics.get('critic/top1_match', 0.0):.3f} "
                f"spearman={metrics.get('critic/spearman', float('nan')):+.3f} "
                f"glimpses/s={committed_gps:.1f} cand/s={candidate_gps:.1f}"
            )

            current_top1 = metrics.get("critic/top1_match", float("-inf"))
            _save_checkpoint(
                path=latest_path,
                q1=q1,
                q2=q2,
                opt=opt,
                args=args,
                batch=batch_idx,
                n_labels=n_labels,
                metric=current_top1,
                probe_repo=probe_repo,
                cfg=cfg,
            )
            if current_top1 > best_top1:
                # Fixed by Codex on 2026-06-15
                # Problem: Lowest MSE can select critics with poor action rankings.
                # Solution: Track the best checkpoint by top-1 oracle agreement.
                # Result: Saved best.pt optimizes action selection quality for SAC validation.
                best_top1 = current_top1
                _save_checkpoint(
                    path=best_path,
                    q1=q1,
                    q2=q2,
                    opt=opt,
                    args=args,
                    batch=batch_idx,
                    n_labels=n_labels,
                    metric=best_top1,
                    probe_repo=probe_repo,
                    cfg=cfg,
                )
            metric_windows.clear()

    if metric_windows:
        eval_metrics, eval_iter = _evaluate_many(
            q1=q1,
            q2=q2,
            eval_iter=eval_iter,
            eval_loader=eval_loader,
            model=model,
            probe=probe,
            cfg=cfg,
            args=args,
            device=device,
        )
        metrics = {key: _mean(values) for key, values in metric_windows.items()}
        metrics.update(eval_metrics)
        metrics["train/n_labels"] = float(n_labels)
        metrics["train/batch"] = float(args.batches)
        metrics["throughput/committed_glimpses_per_sec"] = (
            committed_glimpses / max(elapsed_seconds, 1e-12)
        )
        metrics["throughput/candidate_glimpses_per_sec"] = (
            candidate_glimpses / max(elapsed_seconds, 1e-12)
        )
        if comet_exp is not None:
            comet_exp.log_metrics(metrics, step=args.batches)
        current_top1 = metrics.get("critic/top1_match", float("-inf"))
        if current_top1 > best_top1:
            best_top1 = current_top1
            _save_checkpoint(
                path=best_path,
                q1=q1,
                q2=q2,
                opt=opt,
                args=args,
                batch=args.batches,
                n_labels=n_labels,
                metric=best_top1,
                probe_repo=probe_repo,
                cfg=cfg,
            )

    final_metric = best_top1 if best_top1 != float("-inf") else 0.0
    _save_checkpoint(
        path=latest_path,
        q1=q1,
        q2=q2,
        opt=opt,
        args=args,
        batch=args.batches,
        n_labels=n_labels,
        metric=final_metric,
        probe_repo=probe_repo,
        cfg=cfg,
    )
    if comet_exp is not None:
        comet_exp.log_metric("final/critic_top1_match", final_metric)
        comet_exp.end()
    print(f"Saved latest CE critic to {latest_path}")
    print(f"Best top1 checkpoint: {best_path} ({final_metric:.4f})")
    return final_metric


def run_optuna(args: argparse.Namespace) -> None:
    """Run Optuna sweeps using top-1 match as the validation objective."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optuna or run without --optuna-trials.") from exc

    def objective(trial: Any) -> float:
        trial_args = copy.deepcopy(args)
        trial_args.lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        trial_args.weight_decay = trial.suggest_float(
            "weight_decay",
            1e-7,
            1e-2,
            log=True,
        )
        trial_args.d_model = trial.suggest_categorical("d_model", [128, 256, 384])
        trial_args.rff_dim = trial.suggest_categorical("rff_dim", [64, 128, 256])
        trial_args.rff_seed = trial.suggest_int("rff_seed", 1, 10_000)
        trial_args.seed = args.seed + trial.number
        trial_args.checkpoint_dir = args.checkpoint_dir / f"trial_{trial.number}"
        return train_once(trial_args, trial_number=trial.number)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage),
    )
    study.optimize(objective, n_trials=args.optuna_trials)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Deprecated alias for --batches.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Restrict training dataset to first N images, matching actor BC.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=128,
        help="Restrict held-out evaluation images.",
    )
    parser.add_argument(
        "--test-images",
        type=int,
        default=1,
        help="Held-out images to average at each validation/logging interval.",
    )
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
    )
    parser.add_argument(
        "--eval-split",
        choices=["training", "validation"],
        default="validation",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument(
        "--rff-dim",
        type=int,
        default=128,
        help="Output dimension for the upstream CanViT VPEEncoder.",
    )
    parser.add_argument(
        "--rff-seed",
        type=int,
        default=42,
        help="Seed for the upstream CanViT VPEEncoder RFF matrix.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=16,
        help=(
            "Maximum number of viewpoint history slots. Must be >= t+1 "
            "(one warmup full-scene glimpse plus t learned steps)."
        ),
    )
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--comet-log-interval", type=int, default=10)
    parser.add_argument(
        "--rollout-policy",
        choices=["best", "random"],
        default="best",
        help="How to advance the state after labeling each K-candidate set.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/canvit_critic"),
    )
    parser.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--comet-project", type=str, default="sac-critic-ce")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="critic-ce-greedy")
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-study-name", type=str, default="critic-ce-greedy")
    parser.add_argument("--optuna-storage", type=str, default=None)
    args = parser.parse_args()
    if args.output is not None:
        args.checkpoint_dir = args.output.parent
    if args.best_output is not None:
        args.checkpoint_dir = args.best_output.parent
    return args


def main() -> None:
    args = parse_args()
    if args.optuna_trials:
        run_optuna(args)
    else:
        train_once(args)


if __name__ == "__main__":
    main()
