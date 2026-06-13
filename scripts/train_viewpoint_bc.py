"""
Behavior-clone an image-independent SAC actor from the k-greedy heuristic.

The actor sees only previous Viewpoints, encoded as VPE with an optional
timestep feature. CanViT and the ADE20K probe stay frozen; k-greedy supplies
batched teacher actions and mIoU diagnostics.

Example:
    uv run python scripts/train_viewpoint_bc.py --batches 100  --max-samples 1\
        --batch-size 1 --k 16 --log-std-penalty 0 \
        --experiment-name bc-im1-k16 --comet-log-interval 10
    uv run python scripts/train_viewpoint_bc.py --optuna-trials 20 --batches 50
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
from canvit_specialize.datasets.ade20k import (
    ADE20kDataset,
    IGNORE_LABEL,
    NUM_CLASSES,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import greedy_step_batch
from canvit_rl.viewpoint_policy import (
    ViewpointGaussianActor,
    action_to_viewpoint,
    viewpoint_to_action,
)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _sync_for_timing(device: torch.device) -> None:
    """Synchronize CUDA kernels before reading wall-clock throughput."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    name = args.experiment_name or "viewpoint-bc"
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


def _append_history(
    coords: torch.Tensor,
    lengths: torch.Tensor,
    viewpoint: Viewpoint,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write one Viewpoint into slot `step` of the fixed-length history tensor.

    Using an explicit ``step`` counter (rather than ``lengths.max()``) is
    safer: ``lengths`` can legitimately differ across batch items if history
    is ever masked per-sample, and we always want to write into the next
    sequential slot driven by the outer loop.
    """
    if step >= coords.shape[1]:
        raise ValueError(
            f"History slot {step} is out of range; increase --max-history "
            f"(current max_steps={coords.shape[1]})."
        )
    coords[:, step, :2] = viewpoint.centers.detach().float()
    coords[:, step, 2] = viewpoint.scales.detach().float()
    lengths = lengths + 1
    return coords, lengths


def _update_miou_and_ce(
    *,
    acc: mIoUAccumulator,
    model,
    probe: torch.nn.Module,
    state,
    masks: torch.Tensor,
    canvas_grid_size: int,
) -> float:
    """Update mIoU accumulator and return mean cross-entropy loss for this step.

    Returns the scalar CE loss so callers can accumulate it separately from
    the mIoU metric without a second probe forward pass.
    """
    spatial = model.get_spatial(state.canvas).view(
        masks.shape[0],
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float())
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    ce_loss = float(
        F.cross_entropy(logits, masks.long(), ignore_index=IGNORE_LABEL).detach()
    )
    acc.update(logits.argmax(dim=1), masks)
    return ce_loss


def _build_actor(
    args: argparse.Namespace,
    device: torch.device,
) -> ViewpointGaussianActor:
    """Construct the image-independent Gaussian actor."""
    return ViewpointGaussianActor(
        d_model=args.d_model,
        max_steps=args.max_history,
        rff_dim=args.rff_dim,
        rff_seed=args.rff_seed,
    ).to(device)


def _save_checkpoint(
    *,
    actor: ViewpointGaussianActor,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    batch_idx: int,
    metric: float,
) -> None:
    """Persist the current BC actor in a SAC-compatible actor-state checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": actor.state_dict(),
        "actor_opt": optimizer.state_dict(),
        "args": vars(args),
        "batch": batch_idx,
        "metric": metric,
        "state_representation": "viewpoint_history",
    }
    torch.save(payload, checkpoint_dir / "latest.pt")
    torch.save(actor.state_dict(), checkpoint_dir / "actor_final.pt")


def _mean(values: list[float]) -> float:
    """Return 0 for empty metric windows."""
    return sum(values) / len(values) if values else 0.0


def _make_step_windows(n_steps: int) -> list[list[float]]:
    """Allocate one empty window list per learned step (step 0 excluded)."""
    return [[] for _ in range(n_steps)]


def train_once(
    args: argparse.Namespace,
    *,
    trial_number: int | None = None,
) -> float:
    """Run one BC experiment and return the final mean actor mIoU."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = CanViTEnvConfig()
    device = get_device()
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_samples, len(dataset)))
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset, replacement=True),
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

    actor = _build_actor(args, device)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    comet_exp = _make_comet_experiment(args, trial_number)

    loss_window: list[float] = []
    teacher_miou_windows = _make_step_windows(args.t)
    actor_miou_windows = _make_step_windows(args.t)
    teacher_ce_windows = _make_step_windows(args.t)
    actor_ce_windows = _make_step_windows(args.t)
    teacher_scale_windows = _make_step_windows(args.t)

    rand_miou_windows = _make_step_windows(args.t)
    rand_ce_windows = _make_step_windows(args.t)

    action_mean_y_window: list[float] = []
    action_mean_x_window: list[float] = []
    action_mean_scale_window: list[float] = []
    action_std_y_window: list[float] = []
    action_std_x_window: list[float] = []
    action_std_scale_window: list[float] = []
    action_error_y_window: list[float] = []
    action_error_x_window: list[float] = []
    action_error_scale_window: list[float] = []
    grad_norm_window: list[float] = []

    elapsed_seconds = 0.0
    committed_glimpses = 0
    candidate_glimpses = 0
    last_actor_miou = 0.0
    data_iter = iter(loader)
    progress = tqdm(range(1, args.batches + 1), desc="BC k-greedy actor")

    for batch_idx in progress:
        try:
            images, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, masks = next(data_iter)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]

        teacher_state = model.init_state(
            batch_size=batch_size,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        actor_state = model.init_state(
            batch_size=batch_size,
            canvas_grid_size=cfg.canvas_grid_size,
        )

        teacher_coords = torch.zeros(batch_size, args.max_history, 3, device=device)
        actor_coords = torch.zeros_like(teacher_coords)
        teacher_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        actor_lengths = torch.zeros_like(teacher_lengths)

        history_step = 0

        full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
        full_glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        _sync_for_timing(device)
        start_time = time.perf_counter()
        with torch.inference_mode():
            full_teacher_out = model(
                glimpse=full_glimpse,
                state=teacher_state,
                viewpoint=full_vp,
            )
            full_actor_out = model(
                glimpse=full_glimpse,
                state=actor_state,
                viewpoint=full_vp,
            )
        teacher_state = full_teacher_out.state
        actor_state = full_actor_out.state

        rand_baseline_state = full_actor_out.state
        teacher_coords, teacher_lengths = _append_history(
            teacher_coords, teacher_lengths, full_vp, history_step
        )
        actor_coords, actor_lengths = _append_history(
            actor_coords, actor_lengths, full_vp, history_step
        )
        history_step += 1

        teacher_accs = [
            mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
            for _ in range(args.t)
        ]
        actor_accs = [
            mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
            for _ in range(args.t)
        ]
        rand_accs = [
            mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
            for _ in range(args.t)
        ]

        batch_loss = torch.zeros((), device=device)
        for step_idx in range(args.t):
            # --- BC loss: supervise actor on teacher's history ---
            teacher_batch = {"coords": teacher_coords, "lengths": teacher_lengths}
            mean, log_std = actor(teacher_batch)
            pred_action = torch.tanh(mean)

            action_mean_y_window.append(float(pred_action[:, 0].mean().detach()))
            action_mean_x_window.append(float(pred_action[:, 1].mean().detach()))
            decoded_scale = (
                (pred_action[:, 2] + 1.0)
                * 0.5
                * (1.0 - args.min_scale)
                + args.min_scale
            )
            action_mean_scale_window.append(float(decoded_scale.mean().detach()))
            action_std_y_window.append(
                float(pred_action[:, 0].std(unbiased=False).detach())
            )
            action_std_x_window.append(
                float(pred_action[:, 1].std(unbiased=False).detach())
            )
            action_std_scale_window.append(
                float(decoded_scale.std(unbiased=False).detach())
            )

            teacher_vp, teacher_state, _ = greedy_step_batch(
                model=model,
                images=images,
                state=teacher_state,
                k=args.k,
                glimpse_size_px=cfg.glimpse_size_px,
                device=device,
                min_scale=args.min_scale,
                max_scale=1.0,
                sample_seed=args.seed + batch_idx * 10_000 + step_idx,
                probe=probe,
                canvas_grid_size=cfg.canvas_grid_size,
                masks=masks,
            )
            teacher_scale_windows[step_idx].append(
                float(teacher_vp.scales.detach().mean())
            )
            target_action = viewpoint_to_action(teacher_vp, min_scale=args.min_scale)
            action_error = (pred_action - target_action).abs()
            action_error_y_window.append(float(action_error[:, 0].mean().detach()))
            action_error_x_window.append(float(action_error[:, 1].mean().detach()))
            action_error_scale_window.append(
                float(action_error[:, 2].mean().detach())
            )
            step_loss = F.mse_loss(pred_action, target_action)
            if args.log_std_penalty:
                step_loss = step_loss + args.log_std_penalty * log_std.pow(2).mean()
            batch_loss = batch_loss + step_loss

            teacher_coords, teacher_lengths = _append_history(
                teacher_coords, teacher_lengths, teacher_vp, history_step
            )

            # --- Teacher metrics at this learned step ---
            teacher_ce = _update_miou_and_ce(
                acc=teacher_accs[step_idx],
                model=model,
                probe=probe,
                state=teacher_state,
                masks=masks,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            teacher_miou_windows[step_idx].append(
                float(teacher_accs[step_idx].compute())
            )
            teacher_ce_windows[step_idx].append(teacher_ce)

            # --- Actor rollout (inference only) and metrics ---
            with torch.inference_mode():
                actor_action = actor.deterministic_action(
                    {"coords": actor_coords, "lengths": actor_lengths}
                )
                actor_vp = action_to_viewpoint(actor_action, min_scale=args.min_scale)
                actor_glimpse = sample_at_viewpoint(
                    spatial=images,
                    viewpoint=actor_vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                )
                actor_out = model(
                    glimpse=actor_glimpse,
                    state=actor_state,
                    viewpoint=actor_vp,
                )
                actor_state = actor_out.state

            actor_coords, actor_lengths = _append_history(
                actor_coords, actor_lengths, actor_vp, history_step
            )

            actor_ce = _update_miou_and_ce(
                acc=actor_accs[step_idx],
                model=model,
                probe=probe,
                state=actor_state,
                masks=masks,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            actor_miou_windows[step_idx].append(float(actor_accs[step_idx].compute()))
            actor_ce_windows[step_idx].append(actor_ce)

            with torch.inference_mode():
                rand_centers = torch.empty(batch_size, 2, device=device).uniform_(
                    -1.0,
                    1.0,
                )
                rand_scales = torch.empty(batch_size, device=device).uniform_(
                    args.min_scale, 1.0
                )
                rand_vp = Viewpoint(centers=rand_centers, scales=rand_scales)
                rand_glimpse = sample_at_viewpoint(
                    spatial=images,
                    viewpoint=rand_vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                )
                rand_out = model(
                    glimpse=rand_glimpse,
                    state=rand_baseline_state,
                    viewpoint=rand_vp,
                )
            rand_ce = _update_miou_and_ce(
                acc=rand_accs[step_idx],
                model=model,
                probe=probe,
                state=rand_out.state,
                masks=masks,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            rand_miou_windows[step_idx].append(float(rand_accs[step_idx].compute()))
            rand_ce_windows[step_idx].append(rand_ce)

            history_step += 1

        batch_loss = batch_loss / max(args.t, 1)
        optimizer.zero_grad()
        batch_loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            args.grad_clip,
        )
        grad_norm_window.append(float(total_grad_norm))
        optimizer.step()

        _sync_for_timing(device)
        batch_elapsed = time.perf_counter() - start_time
        elapsed_seconds += batch_elapsed
        committed_glimpses += batch_size * (args.t + 1) * 2
        candidate_glimpses += batch_size * (1 + args.t * args.k)
        loss_value = float(batch_loss.detach().item())
        loss_window.append(loss_value)

        # last_actor_miou tracks the final learned step for checkpointing
        last_actor_miou = float(actor_accs[-1].compute()) if args.t > 0 else 0.0

        batch_gps = committed_glimpses / max(elapsed_seconds, 1e-12)
        batch_candidate_gps = candidate_glimpses / max(elapsed_seconds, 1e-12)
        progress.set_postfix(
            {
                "loss": f"{loss_value:.4f}",
                "glimpses/s": f"{batch_gps:.1f}",
                "cand/s": f"{batch_candidate_gps:.1f}",
            }
        )

        if comet_exp is not None and batch_idx % args.comet_log_interval == 0:
            metrics: dict[str, float] = {
                "bc/loss": _mean(loss_window),
                "throughput/committed_glimpses_per_sec": batch_gps,
                "throughput/candidate_glimpses_per_sec": batch_candidate_gps,
                # Action statistics — watch std collapse toward 0 for mode collapse
                "action/mean_y": _mean(action_mean_y_window),
                "action/mean_x": _mean(action_mean_x_window),
                "action/mean_scale": _mean(action_mean_scale_window),
                "teacher/mean_scale": _mean(
                    [
                        value
                        for step_values in teacher_scale_windows
                        for value in step_values
                    ]
                ),
                "action/std_y": _mean(action_std_y_window),
                "action/std_x": _mean(action_std_x_window),
                "action/std_scale": _mean(action_std_scale_window),
                "grad/total_norm": _mean(grad_norm_window),
                "bc/error_y": _mean(action_error_y_window),
                "bc/error_x": _mean(action_error_x_window),
                "bc/error_scale": _mean(action_error_scale_window),
            }
            # Log learned steps only (step 0 warmup excluded — identical for
            # both teacher and actor so it carries no comparative signal).
            for step_idx in range(args.t):
                step_label = step_idx + 1  # human-readable: step 1 = first learned
                actor_miou = _mean(actor_miou_windows[step_idx])
                teacher_miou = _mean(teacher_miou_windows[step_idx])
                rand_miou = _mean(rand_miou_windows[step_idx])
                actor_ce = _mean(actor_ce_windows[step_idx])
                teacher_ce = _mean(teacher_ce_windows[step_idx])
                rand_ce = _mean(rand_ce_windows[step_idx])
                teacher_scale = _mean(teacher_scale_windows[step_idx])

                # mIoU: actor / teacher / random on the same step chart
                metrics[f"miou_step{step_label}_actor"] = actor_miou
                metrics[f"miou_step{step_label}_teacher"] = teacher_miou
                metrics[f"miou_step{step_label}_random"] = rand_miou
                # Gap toward oracle ceiling (positive = actor below teacher)
                metrics[f"miou_step{step_label}_gap"] = teacher_miou - actor_miou

                # CE loss: actor / teacher / random on the same step chart
                # Lower is better; actor and random should converge toward teacher
                metrics[f"ce_step{step_label}_actor"] = actor_ce
                metrics[f"ce_step{step_label}_teacher"] = teacher_ce
                metrics[f"ce_step{step_label}_random"] = rand_ce
                # Gap: positive = actor worse than teacher; converges toward 0
                metrics[f"ce_step{step_label}_gap"] = actor_ce - teacher_ce
                metrics[f"teacher_scale_step{step_label}"] = teacher_scale

            comet_exp.log_metrics(metrics, step=batch_idx)
            loss_window.clear()
            teacher_miou_windows = _make_step_windows(args.t)
            actor_miou_windows = _make_step_windows(args.t)
            teacher_ce_windows = _make_step_windows(args.t)
            actor_ce_windows = _make_step_windows(args.t)
            teacher_scale_windows = _make_step_windows(args.t)
            rand_miou_windows = _make_step_windows(args.t)
            rand_ce_windows = _make_step_windows(args.t)
            action_mean_y_window.clear()
            action_mean_x_window.clear()
            action_mean_scale_window.clear()
            action_std_y_window.clear()
            action_std_x_window.clear()
            action_std_scale_window.clear()
            action_error_y_window.clear()
            action_error_x_window.clear()
            action_error_scale_window.clear()
            grad_norm_window.clear()

        if batch_idx % args.checkpoint_interval == 0:
            _save_checkpoint(
                actor=actor,
                optimizer=optimizer,
                args=args,
                checkpoint_dir=args.checkpoint_dir,
                batch_idx=batch_idx,
                metric=last_actor_miou,
            )

    final_metric = (
        _mean(actor_miou_windows[-1])
        if actor_miou_windows[-1]
        else last_actor_miou
    )
    _save_checkpoint(
        actor=actor,
        optimizer=optimizer,
        args=args,
        checkpoint_dir=args.checkpoint_dir,
        batch_idx=args.batches,
        metric=final_metric,
    )
    if comet_exp is not None:
        comet_exp.log_metric("final/actor_miou_last_step", final_metric)
        comet_exp.end()
    print(f"Saved BC actor to {args.checkpoint_dir / 'actor_final.pt'}")
    return final_metric


def run_optuna(args: argparse.Namespace) -> None:
    """Run Optuna hyperparameter sweeps over the image-independent actor."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optuna or run without --optuna-trials.") from exc

    def objective(trial: Any) -> float:
        trial_args = copy.deepcopy(args)
        trial_args.lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        trial_args.weight_decay = trial.suggest_float(
            "weight_decay",
            1e-6,
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
    """Parse CLI flags for BC and Optuna modes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--t",
        type=int,
        default=1,
        help="Learned glimpses after warmup",
    )
    parser.add_argument("--k", type=int, default=16, help="Greedy candidates per step")
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Restrict dataset to first N images for overfitting sanity checks.",
    )
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-scale", type=float, default=0.05)
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
        "--fourier-bands",
        type=int,
        default=None,
        help="Deprecated alias for --rff-dim; will be removed in a future version.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=2,
        help=(
            "Maximum number of viewpoint history slots. Must be >= t+1 "
            "(one warmup full-scene glimpse plus t learned steps). Default: 16."
        ),
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-std-penalty", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/viewpoint_bc"),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--comet-project", type=str, default="sac-actor-bc")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="viewpoint-bc")
    parser.add_argument("--comet-log-interval", type=int, default=25)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-study-name", type=str, default="viewpoint-bc")
    parser.add_argument("--optuna-storage", type=str, default=None)
    args = parser.parse_args()

    # Wire the deprecated --fourier-bands alias so it isn't silently ignored.
    if args.fourier_bands is not None:
        import warnings
        warnings.warn(
            "--fourier-bands is deprecated; use --rff-dim instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.rff_dim = args.fourier_bands

    # Validate that max_history is large enough for the requested horizon.
    required = args.t + 1
    if args.max_history < required:
        raise ValueError(
            f"--max-history ({args.max_history}) must be >= t+1 ({required}). "
            f"Pass --max-history {required} or higher."
        )

    return args


def main() -> None:
    args = parse_args()
    if args.optuna_trials:
        run_optuna(args)
    else:
        train_once(args)


if __name__ == "__main__":
    main()
