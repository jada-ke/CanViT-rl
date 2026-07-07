"""
Train image-independent SAC over viewpoint history for CanViT active vision.

The learned state is the same fixed-slot viewpoint history used by
scripts/train_viewpoint_bc.py and scripts/pretrain_canvit_critic.py:

    state = previous Viewpoints
    action = next Viewpoint
    reward = CE_before - CE_after

Example:
    uv run python scripts/train_viewpoint_sac.py \
        --actor-checkpoint checkpoints/viewpoint_bc/im1-k16/actor_final.pt \
        --critic-checkpoint checkpoints/canvit_critic/im1-k16-t1/best.pt 
    uv run python scripts/train_viewpoint_sac.py \
        --batches 500 --batch-size 1  --max-samples 1 --t 1 \
        --eval-images 8 \
        --checkpoint-dir checkpoints/viewpoint_sac \
        --no-comet \
        --reward-map-images 2 \
        --reward-map-grid-size 21 \
        --reward-map-scales 0.25,0.50 \
        --reward-map-output-dir results/sac_reward_maps
        --experiment-name pretrain-sac-im1-500 \
        --actor-checkpoint checkpoints/viewpoint_bc/im1-k16/actor_final.pt \
        --critic-checkpoint checkpoints/canvit_critic/im1-k16-t1/best.pt
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from collections import defaultdict
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
    NUM_CLASSES,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

EVAL_REPO = Path(__file__).resolve().parents[1] / "CanViT-eval"
if EVAL_REPO.is_dir() and str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))

try:
    from canvit_eval.episode import run_episode
    from canvit_eval.policies import make_policy
except ImportError:
    run_episode = None
    make_policy = None

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import _segmentation_cross_entropy_losses
from canvit_rl.viewpoint_policy import (
    ViewpointGaussianActor,
    ViewpointHistoryCritic,
    action_to_viewpoint,
    randomize_actor_mean_viewpoint_prior,
)

try:
    from visualize_sac_reward_maps import visualize_reward_maps_for_indices
except ImportError:
    from scripts.visualize_sac_reward_maps import visualize_reward_maps_for_indices


def _sync_for_timing(device: torch.device) -> None:
    """Synchronize CUDA kernels before reading throughput timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_comet_experiment(args: argparse.Namespace):
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
    experiment.set_name(args.experiment_name or "viewpoint-history-sac")
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    return experiment


def _limit_dataset(dataset, max_samples: int | None, *, offset: int = 0):
    """Restrict datasets while preserving deterministic validation subsets."""
    if max_samples is None:
        return dataset
    start = min(offset, len(dataset))
    stop = min(start + max_samples, len(dataset))
    return torch.utils.data.Subset(dataset, range(start, stop))


def _append_history(
    *,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    viewpoint: Viewpoint,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write one batched Viewpoint into the fixed-slot history tensor."""
    if step >= coords.shape[1]:
        raise ValueError(
            f"History slot {step} is out of range for max_history={coords.shape[1]}."
        )
    # Fixed by Codex on 2026-06-18
    # Problem: Full SAC must use the same state contract as actor BC and critic
    # pretraining, otherwise loaded checkpoints answer a different question.
    # Solution: Keep history as batched GPU tensors and append every rollout
    # Viewpoint in lockstep with the environment timestep.
    # Result: Actor, critic, pretraining, and SAC all consume Q(history, action).
    coords[:, step, :2] = viewpoint.centers.detach().float()
    coords[:, step, 2] = viewpoint.scales.detach().float()
    return coords, lengths + 1


def _batch_from_arrays(
    *,
    coords: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert replay arrays into an actor/critic history batch."""
    return {
        "coords": torch.as_tensor(coords, device=device),
        "lengths": torch.as_tensor(lengths, device=device),
    }


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
        spatial = model.get_spatial(state.canvas).view(
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


def _grad_norm(parameters) -> float:
    """Compute total L2 gradient norm for a parameter group."""
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().norm(2).item() ** 2)
    return total ** 0.5


class HistoryReplayBuffer:
    """Replay buffer for image-independent SAC transitions."""

    def __init__(self, *, capacity: int, max_history: int) -> None:
        self.capacity = capacity
        self.max_history = max_history
        self.coords = np.zeros((capacity, max_history, 3), dtype=np.float32)
        self.next_coords = np.zeros((capacity, max_history, 3), dtype=np.float32)
        self.lengths = np.zeros(capacity, dtype=np.int64)
        self.next_lengths = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros((capacity, 3), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add_batch(
        self,
        *,
        coords: torch.Tensor,
        lengths: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_coords: torch.Tensor,
        next_lengths: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        batch_size = coords.shape[0]
        coords_np = coords.detach().cpu().numpy().astype(np.float32)
        next_coords_np = next_coords.detach().cpu().numpy().astype(np.float32)
        lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
        next_lengths_np = next_lengths.detach().cpu().numpy().astype(np.int64)
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        rewards_np = rewards.detach().cpu().numpy().astype(np.float32)
        dones_np = dones.detach().cpu().numpy().astype(np.float32)
        for idx in range(batch_size):
            self.coords[self.pos] = coords_np[idx]
            self.next_coords[self.pos] = next_coords_np[idx]
            self.lengths[self.pos] = lengths_np[idx]
            self.next_lengths[self.pos] = next_lengths_np[idx]
            self.actions[self.pos] = actions_np[idx]
            self.rewards[self.pos] = rewards_np[idx]
            self.dones[self.pos] = dones_np[idx]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "coords": torch.as_tensor(self.coords[idx], device=device),
            "lengths": torch.as_tensor(self.lengths[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_coords": torch.as_tensor(self.next_coords[idx], device=device),
            "next_lengths": torch.as_tensor(self.next_lengths[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


class ViewpointSAC:
    """Continuous SAC for the image-independent viewpoint actor/critic."""

    def __init__(
        self,
        *,
        actor: ViewpointGaussianActor,
        q1: ViewpointHistoryCritic,
        q2: ViewpointHistoryCritic,
        target_q1: ViewpointHistoryCritic,
        target_q2: ViewpointHistoryCritic,
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
        alpha_device = next(actor.parameters()).device
        self.log_alpha = torch.nn.Parameter(
            torch.log(torch.tensor(init_alpha, device=alpha_device))
        )
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, sample: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = {"coords": sample["coords"], "lengths": sample["lengths"]}
        next_obs = {
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
        self.q_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        grad_q1 = _grad_norm(self.q1.parameters())
        grad_q2 = _grad_norm(self.q2.parameters())
        self.q_opt.step()

        action, log_prob = self.actor.sample(obs)
        q_min = torch.minimum(self.q1(obs, action), self.q2(obs, action))
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        grad_actor = _grad_norm(self.actor.parameters())
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
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
            "grad/actor_norm": grad_actor,
            "grad/q1_norm": grad_q1,
            "grad/q2_norm": grad_q2,
            "grad/critic_norm": float((grad_q1**2 + grad_q2**2) ** 0.5),
            "sac/alpha": float(self.alpha.detach().item()),
            "sac/entropy": float(entropy.mean().item()),
            "sac/target_entropy_gap": float(
                entropy.mean().item() - abs(self.target_entropy)
            ),
            "alpha/loss": float(alpha_loss.detach().item()),
        }

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)


def _build_networks(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    ViewpointGaussianActor,
    ViewpointHistoryCritic,
    ViewpointHistoryCritic,
    ViewpointHistoryCritic,
    ViewpointHistoryCritic,
]:
    """Construct actor, twin critics, and target critics."""
    actor = ViewpointGaussianActor(
        d_model=args.d_model,
        max_steps=args.max_history,
        rff_dim=args.rff_dim,
        rff_seed=args.rff_seed,
    ).to(device)
    if (
        args.randomize_actor_init
        and args.resume is None
        and args.actor_checkpoint is None
    ):
        prior = randomize_actor_mean_viewpoint_prior(
            actor,
            min_scale=args.min_scale,
            center_radius=args.actor_init_center_radius,
        )
        print(
            "Randomized viewpoint SAC actor init: "
            f"center=({prior['center_y']:+.3f}, {prior['center_x']:+.3f}) "
            f"scale={prior['scale']:.3f}"
        )
    critic_kwargs = dict(
        d_model=args.d_model,
        max_steps=args.max_history,
        rff_dim=args.rff_dim,
        rff_seed=args.rff_seed,
    )
    q1 = ViewpointHistoryCritic(**critic_kwargs).to(device)
    q2 = ViewpointHistoryCritic(**critic_kwargs).to(device)
    target_q1 = copy.deepcopy(q1).to(device)
    target_q2 = copy.deepcopy(q2).to(device)
    return actor, q1, q2, target_q1, target_q2


def _load_state_dict_flexible(
    module: torch.nn.Module,
    payload: Any,
    key: str,
    path: Path,
) -> None:
    """Load either a plain state_dict or a checkpoint dict containing key."""
    state = payload.get(key) if isinstance(payload, dict) and key in payload else payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a usable {key} state: {path}")
    try:
        module.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(
            f"Could not load {key} checkpoint: {path}\n"
            "This usually means the SAC architecture args do not match the "
            "pretraining run. Check --d-model, --rff-dim, --rff-seed, "
            "--max-history, and --min-scale. If you are loading actor_final.pt, "
            "keep the sibling latest.pt beside it so SAC can auto-read those args."
        ) from exc


def _adopt_pretrained_arch_args(args: argparse.Namespace) -> None:
    """Adopt actor/critic architecture args from initialization checkpoints."""
    candidate_paths: list[Path] = []
    if args.actor_checkpoint is not None:
        candidate_paths.extend([args.actor_checkpoint, args.actor_checkpoint.with_name("latest.pt")])
    if args.critic_checkpoint is not None:
        candidate_paths.append(args.critic_checkpoint)

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
            continue
        saved_args = payload["args"]
        for name in ("d_model", "rff_dim", "rff_seed", "max_history", "min_scale"):
            if name in saved_args:
                old_value = getattr(args, name)
                new_value = saved_args[name]
                if old_value != new_value:
                    # Fixed by Codex on 2026-06-18
                    # Problem: Passing BC actor_final.pt loses architecture
                    # metadata, so SAC defaulted to max_history=16 while BC
                    # checkpoints often use max_history=t+1, e.g. 2.
                    # Solution: Read architecture args from sibling latest.pt
                    # or critic checkpoints before constructing networks.
                    # Result: Pretrained actor/critic checkpoints initialize
                    # SAC with matching History -> VPE -> head dimensions.
                    print(f"Adopting --{name.replace('_', '-')}={new_value} from {path}")
                    setattr(args, name, new_value)
        return


def _load_initialization(
    *,
    args: argparse.Namespace,
    actor: ViewpointGaussianActor,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    target_q1: ViewpointHistoryCritic,
    target_q2: ViewpointHistoryCritic,
    agent: ViewpointSAC,
) -> tuple[int, int, float]:
    """Load resume/pretrained checkpoints while keeping all networks trainable."""
    start_batch = 1
    update_count = 0
    best_ce_gain = float("-inf")
    if args.resume is not None:
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
        start_batch = int(checkpoint.get("batch", 0)) + 1
        update_count = int(checkpoint.get("updates", 0))
        best_ce_gain = float(checkpoint.get("best_ce_gain", best_ce_gain))
        return start_batch, update_count, best_ce_gain

    if args.actor_checkpoint is not None:
        payload = torch.load(args.actor_checkpoint, map_location="cpu", weights_only=False)
        _load_state_dict_flexible(
            actor,
            payload,
            "actor",
            args.actor_checkpoint,
        )
    if args.critic_checkpoint is not None:
        payload = torch.load(args.critic_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "q1" not in payload:
            raise ValueError(f"Expected q1/q2 critic checkpoint: {args.critic_checkpoint}")
        q1.load_state_dict(payload["q1"])
        q2.load_state_dict(payload.get("q2", payload["q1"]))
        target_q1.load_state_dict(q1.state_dict())
        target_q2.load_state_dict(q2.state_dict())
    return start_batch, update_count, best_ce_gain


def _save_checkpoint(
    *,
    path: Path,
    actor: ViewpointGaussianActor,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    target_q1: ViewpointHistoryCritic,
    target_q2: ViewpointHistoryCritic,
    agent: ViewpointSAC,
    args: argparse.Namespace,
    batch: int,
    updates: int,
    best_ce_gain: float,
    eval_metrics: dict[str, float] | None,
) -> None:
    """Save SAC state; best selection is by eval/ce_gain."""
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
            "batch": batch,
            "updates": updates,
            "best_ce_gain": best_ce_gain,
            "selection_metric": "eval/ce_gain",
            "eval_metrics": eval_metrics or {},
            "state_representation": "viewpoint_history",
        },
        path,
    )


def _eval_random_batch(
    *,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    acc: mIoUAccumulator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out random learned viewpoints after a full-scene warm-up."""
    device = images.device
    state = model.init_state(batch_size=images.shape[0], canvas_grid_size=cfg.canvas_grid_size)
    full_vp = Viewpoint.full_scene(batch_size=images.shape[0], device=device)
    with torch.inference_mode():
        full_out = model(
            glimpse=sample_at_viewpoint(
                spatial=images,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ),
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
            out = model(
                glimpse=sample_at_viewpoint(
                    spatial=images,
                    viewpoint=vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                ),
                state=state,
                viewpoint=vp,
            )
            state = out.state
        final_ce, _ = _segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg, acc=acc
        )
    return initial_ce, final_ce


def _eval_sac_batch(
    *,
    actor: ViewpointGaussianActor,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    acc: mIoUAccumulator,
    scale_sums: list[float],
    scale_counts: list[int],
    entropy_points: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out deterministic SAC over a validation batch."""
    device = images.device
    batch_size = images.shape[0]
    state = model.init_state(batch_size=batch_size, canvas_grid_size=cfg.canvas_grid_size)
    coords = torch.zeros(batch_size, args.max_history, 3, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
    with torch.inference_mode():
        full_out = model(
            glimpse=sample_at_viewpoint(
                spatial=images,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ),
            state=state,
            viewpoint=full_vp,
        )
        state = full_out.state
        initial_ce, _ = _segmentation_metrics(
            model=model, probe=probe, state=state, masks=masks, cfg=cfg
        )
    coords, lengths = _append_history(coords=coords, lengths=lengths, viewpoint=full_vp, step=0)
    for step_idx in range(args.t):
        with torch.no_grad():
            action = actor.deterministic_action({"coords": coords, "lengths": lengths})
        vp = action_to_viewpoint(action, min_scale=args.min_scale)
        entropy_points.append(
            torch.cat([vp.centers, vp.scales[:, None]], dim=1).detach().cpu().numpy()
        )
        scale_sums[step_idx] += float(vp.scales.detach().sum().item())
        scale_counts[step_idx] += batch_size
        with torch.inference_mode():
            out = model(
                glimpse=sample_at_viewpoint(
                    spatial=images,
                    viewpoint=vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                ),
                state=state,
                viewpoint=vp,
            )
            state = out.state
        coords, lengths = _append_history(
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


def _eval_egc2f_batch(
    *,
    model,
    probe: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    acc: mIoUAccumulator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate canvit-eval entropy-guided coarse-to-fine on the same horizon."""
    if run_episode is None or make_policy is None:
        raise RuntimeError(
            "EG-C2F evaluation requires canvit-eval import support. "
            "Place CanViT-eval next to this repo or install it."
        )
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
            images=images,
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


def evaluate(
    *,
    actor: ViewpointGaussianActor,
    eval_loader: DataLoader,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate Random, EG-C2F, and SAC on a fixed validation subset."""
    random_acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    egc2f_acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    sac_acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    ce_sums = defaultdict(float)
    n_images = 0
    sac_scale_sums = [0.0 for _ in range(args.t)]
    sac_scale_counts = [0 for _ in range(args.t)]
    sac_entropy_points: list[np.ndarray] = []

    for images, masks in tqdm(eval_loader, desc="Evaluating SAC", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]
        n_images += batch_size

        rand_initial, rand_final = _eval_random_batch(
            model=model,
            probe=probe,
            images=images,
            masks=masks,
            cfg=cfg,
            args=args,
            acc=random_acc,
        )
        eg_initial, eg_final = _eval_egc2f_batch(
            model=model,
            probe=probe,
            images=images,
            masks=masks,
            cfg=cfg,
            args=args,
            acc=egc2f_acc,
        )
        sac_initial, sac_final = _eval_sac_batch(
            actor=actor,
            model=model,
            probe=probe,
            images=images,
            masks=masks,
            cfg=cfg,
            args=args,
            acc=sac_acc,
            scale_sums=sac_scale_sums,
            scale_counts=sac_scale_counts,
            entropy_points=sac_entropy_points,
        )
        ce_sums["random_initial"] += float(rand_initial.sum().item())
        ce_sums["random_final"] += float(rand_final.sum().item())
        ce_sums["egc2f_initial"] += float(eg_initial.sum().item())
        ce_sums["egc2f_final"] += float(eg_final.sum().item())
        ce_sums["sac_initial"] += float(sac_initial.sum().item())
        ce_sums["sac_final"] += float(sac_final.sum().item())

    random_miou = float(random_acc.compute())
    egc2f_miou = float(egc2f_acc.compute())
    sac_miou = float(sac_acc.compute())
    random_ce = ce_sums["random_final"] / max(n_images, 1)
    egc2f_ce = ce_sums["egc2f_final"] / max(n_images, 1)
    sac_initial_ce = ce_sums["sac_initial"] / max(n_images, 1)
    sac_ce = ce_sums["sac_final"] / max(n_images, 1)
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
        "eval/ce_gain": sac_initial_ce - sac_ce,
        "eval/random_ce_gain": ce_sums["random_initial"] / max(n_images, 1) - random_ce,
        "eval/egc2f_ce_gain": ce_sums["egc2f_initial"] / max(n_images, 1) - egc2f_ce,
        "eval/sac_vs_random": sac_miou - random_miou,
        "eval/sac_vs_egc2f": sac_miou - egc2f_miou,
        "eval/sac_viewpoint_entropy": _viewpoint_entropy(
            sac_entropy_points,
            bins=args.viewpoint_entropy_bins,
        )
    }
    metrics.update(
        {
            "final_miou": metrics["eval/final_miou"],
            "final_ce": metrics["eval/final_ce"],
            "miou_gain": metrics["eval/miou_gain"],
            "ce_gain": metrics["eval/ce_gain"],
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


def _maybe_visualize_reward_maps(
    *,
    actor: ViewpointGaussianActor,
    q1: ViewpointHistoryCritic,
    q2: ViewpointHistoryCritic,
    eval_dataset,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    args: argparse.Namespace,
    device: torch.device,
    update_count: int,
    comet_exp,
) -> None:
    """Optionally save live reward/Q maps after a validation pass."""
    if args.reward_map_images <= 0:
        return
    # Fixed by Codex on 2026-06-19
    # Problem: Reward maps were only available as a separate offline script, so
    # Q-landscape sanity checks could drift from the current validation policy.
    # Solution: Reuse the plotting module inside the SAC validation loop with
    # live actor/q1/q2 weights and a fixed prefix of the validation subset.
    # Result: Each enabled validation pass can emit true-vs-predicted reward
    # maps for the same current networks used to compute eval metrics.
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
        title_prefix=f"SAC validation reward map update={update_count}",
    )
    if comet_exp is not None:
        for path in paths:
            comet_exp.log_image(str(path), name=path.name, step=update_count)


def train_once(args: argparse.Namespace) -> float:
    """Run full viewpoint-history SAC and return best eval CE gain."""
    _adopt_pretrained_arch_args(args)
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
    if not 0.0 <= args.actor_init_center_radius < 1.0:
        raise ValueError("--actor-init-center-radius must be in [0, 1).")
    if args.reward_map_chunk_size < 1:
        raise ValueError("--reward-map-chunk-size must be positive.")
    _parse_scales(args.reward_map_scales)
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
    eval_dataset = _limit_dataset(eval_dataset, args.eval_images)
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("Train and validation datasets must be non-empty.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=True),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
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

    actor, q1, q2, target_q1, target_q2 = _build_networks(args, device)
    agent = ViewpointSAC(
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
    start_batch, update_count, best_ce_gain = _load_initialization(
        args=args,
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        agent=agent,
    )
    replay = HistoryReplayBuffer(capacity=args.buffer_size, max_history=args.max_history)
    comet_exp = _make_comet_experiment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_iter = iter(train_loader)
    train_windows: dict[str, list[float]] = defaultdict(list)
    reward_window: list[float] = []
    entropy_points: list[np.ndarray] = []
    scale_sums = [0.0 for _ in range(args.t)]
    scale_counts = [0 for _ in range(args.t)]
    next_eval_update = max(args.eval_interval, 1)
    latest_metrics: dict[str, float] | None = None
    elapsed_seconds = 0.0
    committed_glimpses = 0

    pbar = tqdm(range(start_batch, args.batches + 1), desc="Training viewpoint SAC")
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
        coords = torch.zeros(batch_size, args.max_history, 3, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
        with torch.inference_mode():
            full_out = model(
                glimpse=sample_at_viewpoint(
                    spatial=images,
                    viewpoint=full_vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                ),
                state=state,
                viewpoint=full_vp,
            )
            state = full_out.state
            current_ce, _ = _segmentation_metrics(
                model=model, probe=probe, state=state, masks=masks, cfg=cfg
            )
        coords, lengths = _append_history(coords=coords, lengths=lengths, viewpoint=full_vp, step=0)

        for step_idx in range(args.t):
            obs = {"coords": coords, "lengths": lengths}
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
            prev_coords = coords.clone()
            prev_lengths = lengths.clone()
            with torch.inference_mode():
                out = model(
                    glimpse=sample_at_viewpoint(
                        spatial=images,
                        viewpoint=vp,
                        glimpse_size_px=cfg.glimpse_size_px,
                    ),
                    state=state,
                    viewpoint=vp,
                )
                next_ce, _ = _segmentation_metrics(
                    model=model, probe=probe, state=out.state, masks=masks, cfg=cfg
                )
            reward = current_ce - next_ce
            coords, lengths = _append_history(
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
                coords=prev_coords,
                lengths=prev_lengths,
                actions=action.detach().clone(),
                rewards=reward.detach().clone(),
                next_coords=coords,
                next_lengths=lengths,
                dones=done,
            )
            reward_window.extend(reward.detach().cpu().numpy().astype(float).tolist())
            state = out.state
            current_ce = next_ce

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
                                "reward/mean": float(np.mean(rewards_np)),
                                "reward/std": float(np.std(rewards_np)),
                                "reward/max": float(np.max(rewards_np)),
                                "reward/min": float(np.min(rewards_np)),
                            }
                        )
                    train_metrics["train/batch"] = float(batch_idx)
                    train_metrics["train/updates"] = float(update_count)
                    train_metrics["train/replay_size"] = float(replay.size)
                    train_metrics["throughput/glimpses_per_sec"] = glimpses_per_sec
                    train_metrics["throughput/committed_glimpses_per_sec"] = (
                        glimpses_per_sec
                    )
                    train_metrics["throughput/committed_glimpses"] = float(
                        committed_glimpses
                    )
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
                    eval_metrics = evaluate(
                        actor=actor,
                        eval_loader=eval_loader,
                        model=model,
                        probe=probe,
                        cfg=cfg,
                        args=args,
                        device=device,
                    )
                    latest_metrics = eval_metrics
                    if comet_exp is not None:
                        comet_exp.log_metrics(eval_metrics, step=update_count)
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
                        update_count=update_count,
                        comet_exp=comet_exp,
                    )
                    current_ce_gain = eval_metrics["eval/ce_gain"]
                    _save_checkpoint(
                        path=args.checkpoint_dir / "latest.pt",
                        actor=actor,
                        q1=q1,
                        q2=q2,
                        target_q1=target_q1,
                        target_q2=target_q2,
                        agent=agent,
                        args=args,
                        batch=batch_idx,
                        updates=update_count,
                        best_ce_gain=best_ce_gain,
                        eval_metrics=eval_metrics,
                    )
                    if current_ce_gain > best_ce_gain:
                        best_ce_gain = current_ce_gain
                        _save_checkpoint(
                            path=args.checkpoint_dir / "best.pt",
                            actor=actor,
                            q1=q1,
                            q2=q2,
                            target_q1=target_q1,
                            target_q2=target_q2,
                            agent=agent,
                            args=args,
                            batch=batch_idx,
                            updates=update_count,
                            best_ce_gain=best_ce_gain,
                            eval_metrics=eval_metrics,
                        )
                    print(
                        f"update={update_count} batch={batch_idx} "
                        f"ce_gain={current_ce_gain:+.4f} "
                        f"sac_miou={eval_metrics['eval/sac_miou']:.4f} "
                        f"vs_random={eval_metrics['eval/sac_vs_random']:+.4f} "
                        f"vs_egc2f={eval_metrics['eval/sac_vs_egc2f']:+.4f}"
                    )
                    next_eval_update += args.eval_interval

        pbar.set_postfix(
            {
                "updates": update_count,
                "replay": replay.size,
                "glimpses/s": f"{glimpses_per_sec:.1f}",
                "best_ce_gain": f"{best_ce_gain:+.4f}"
                if best_ce_gain != float("-inf")
                else "nan",
            }
        )

    if latest_metrics is None:
        latest_metrics = evaluate(
            actor=actor,
            eval_loader=eval_loader,
            model=model,
            probe=probe,
            cfg=cfg,
            args=args,
            device=device,
        )
        current_ce_gain = latest_metrics["eval/ce_gain"]
        if current_ce_gain > best_ce_gain:
            best_ce_gain = current_ce_gain
            _save_checkpoint(
                path=args.checkpoint_dir / "best.pt",
                actor=actor,
                q1=q1,
                q2=q2,
                target_q1=target_q1,
                target_q2=target_q2,
                agent=agent,
                args=args,
                batch=args.batches,
                updates=update_count,
                best_ce_gain=best_ce_gain,
                eval_metrics=latest_metrics,
            )
        if comet_exp is not None:
            comet_exp.log_metrics(latest_metrics, step=update_count)
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
            update_count=update_count,
            comet_exp=comet_exp,
        )
    # Problem: end-of-training weights can differ from the last evaluated
    # checkpoint. Solution: only periodic/fallback eval blocks write best.pt,
    # so best.pt always corresponds to the metrics stored with it.
    _save_checkpoint(
        path=args.checkpoint_dir / "latest.pt",
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        agent=agent,
        args=args,
        batch=args.batches,
        updates=update_count,
        best_ce_gain=best_ce_gain,
        eval_metrics=latest_metrics,
    )
    torch.save(actor.state_dict(), args.checkpoint_dir / "actor_final.pt")
    if comet_exp is not None:
        comet_exp.log_metric("final/ce_gain", latest_metrics["eval/ce_gain"], step=update_count)
        comet_exp.log_metric("final/miou", latest_metrics["eval/sac_miou"], step=update_count)
        comet_exp.end()
    print(f"Saved SAC latest checkpoint to {args.checkpoint_dir / 'latest.pt'}")
    print(f"Best eval/ce_gain: {best_ce_gain:+.4f}")
    return best_ce_gain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-batch-size", type=int, default=256)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument("--eval-split", choices=["training", "validation"], default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-images", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-repo", type=str, default=None)
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
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--target-entropy", type=float, default=-3.0)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--learning-starts", type=int, default=10)
    parser.add_argument("--updates-per-batch", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--comet-log-interval", type=int, default=10)
    parser.add_argument("--viewpoint-entropy-bins", type=int, default=8)
    parser.add_argument(
        "--reward-map-images",
        type=int,
        default=0,
        help=(
            "If >0, save true-reward vs critic-Q maps for this many validation "
            "images after each validation pass."
        ),
    )
    parser.add_argument("--reward-map-grid-size", type=int, default=11)
    parser.add_argument("--reward-map-scales", type=str, default="0.10,0.25,0.50")
    parser.add_argument("--reward-map-chunk-size", type=int, default=32)
    parser.add_argument(
        "--reward-map-output-dir",
        type=Path,
        default=Path("results/sac_reward_maps"),
    )
    parser.add_argument("--actor-checkpoint", type=Path, default=None)
    parser.add_argument("--critic-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/viewpoint_sac"),
    )
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="viewpoint-sac")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="viewpoint-sac")
    return parser.parse_args()


def main() -> None:
    train_once(parse_args())


if __name__ == "__main__":
    main()
