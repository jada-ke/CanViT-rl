"""
Train a continuous SAC policy for CanViT active-view selection.

State: sequence of previous CanViT local patch tokens and viewpoint coords.
Action: continuous Viewpoint parameters [cx, cy, scale_raw].
Reward: scaled improvement in ADE20K mIoU.

Example:
    uv run python scripts/train_canvit_sac.py --episodes 1000 --t 5
    uv run python scripts/train_canvit_sac.py \
        --episodes 1000 --t 5 \
        --pretrained-critic checkpoints/canvit_critic/pretrained.pt
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import miou_from_state
from canvit_rl.sac_models import CanViTSequenceEncoder, ContinuousCritic, GaussianActor
from canvit_rl.sac_state import (
    append_glimpse,
    batch_from_sequence,
    empty_sequence,
    extract_local_patches,
    sequence_to_arrays,
)


def _format_step_means(values: list[float]) -> str:
    """Format per-timestep metrics compactly for training logs."""
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _mean_by_step(step_values: deque[list[float]], n_steps: int) -> list[float]:
    """Average fixed-length per-timestep episode metrics over a log window."""
    return [
        sum(values[step] for values in step_values) / len(step_values)
        for step in range(n_steps)
    ]


def _action_to_viewpoint(action: torch.Tensor, *, min_scale: float) -> Viewpoint:
    """Map tanh SAC action [B, 3] into the upstream Viewpoint interface."""
    centers = action[:, :2].float()
    scales = ((action[:, 2] + 1.0) / 2.0 * (1.0 - min_scale) + min_scale).float()
    return Viewpoint(centers=centers, scales=scales)


class SequenceReplayBuffer:
    """Replay buffer for padded CanViT glimpse sequences."""

    def __init__(self, capacity: int, max_steps: int, n_patches: int, patch_dim: int):
        self.capacity = capacity
        self.max_steps = max_steps
        self.n_patches = n_patches
        self.patch_dim = patch_dim
        shape = (capacity, max_steps, n_patches, patch_dim)
        self.patches = np.zeros(shape, dtype=np.float32)
        self.next_patches = np.zeros(shape, dtype=np.float32)
        self.coords = np.zeros((capacity, max_steps, 3), dtype=np.float32)
        self.next_coords = np.zeros((capacity, max_steps, 3), dtype=np.float32)
        self.lengths = np.zeros(capacity, dtype=np.int64)
        self.next_lengths = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros((capacity, 3), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        *,
        seq: dict[str, torch.Tensor],
        action: np.ndarray,
        reward: float,
        next_seq: dict[str, torch.Tensor],
        done: bool,
    ) -> None:
        obs = sequence_to_arrays(
            seq,
            max_steps=self.max_steps,
            n_patches=self.n_patches,
            patch_dim=self.patch_dim,
        )
        next_obs = sequence_to_arrays(
            next_seq,
            max_steps=self.max_steps,
            n_patches=self.n_patches,
            patch_dim=self.patch_dim,
        )
        self.patches[self.pos] = obs[0]
        self.coords[self.pos] = obs[1]
        self.lengths[self.pos] = obs[2]
        self.next_patches[self.pos] = next_obs[0]
        self.next_coords[self.pos] = next_obs[1]
        self.next_lengths[self.pos] = next_obs[2]
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "patches": torch.as_tensor(self.patches[idx], device=device),
            "coords": torch.as_tensor(self.coords[idx], device=device),
            "lengths": torch.as_tensor(self.lengths[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_patches": torch.as_tensor(self.next_patches[idx], device=device),
            "next_coords": torch.as_tensor(self.next_coords[idx], device=device),
            "next_lengths": torch.as_tensor(self.next_lengths[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


class ContinuousSAC:
    """Minimal continuous SAC update for CanViT active-view policies."""

    def __init__(
        self,
        *,
        actor: GaussianActor,
        q1: ContinuousCritic,
        q2: ContinuousCritic,
        target_q1: ContinuousCritic,
        target_q2: ContinuousCritic,
        lr: float,
        alpha: float,
        gamma: float,
        tau: float,
    ) -> None:
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()),
            lr=lr,
        )
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        next_batch = {
            "patches": batch["next_patches"],
            "coords": batch["next_coords"],
            "lengths": batch["next_lengths"],
        }
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_batch)
            target_q = torch.minimum(
                self.target_q1(next_batch, next_action),
                self.target_q2(next_batch, next_action),
            )
            target = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * (
                target_q - self.alpha * next_log_prob
            )

        obs_batch = {
            "patches": batch["patches"],
            "coords": batch["coords"],
            "lengths": batch["lengths"],
        }
        q_loss = F.mse_loss(self.q1(obs_batch, batch["actions"]), target)
        q_loss = q_loss + F.mse_loss(self.q2(obs_batch, batch["actions"]), target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        action, log_prob = self.actor.sample(obs_batch)
        q_min = torch.minimum(self.q1(obs_batch, action), self.q2(obs_batch, action))
        actor_loss = (self.alpha * log_prob - q_min).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        return {
            "q_loss": float(q_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "mean_log_prob": float(log_prob.detach().mean().item()),
        }

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)


def _build_actor_and_critics(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    GaussianActor,
    ContinuousCritic,
    ContinuousCritic,
    ContinuousCritic,
    ContinuousCritic,
]:
    """Construct actor, critics, and target critics with separate encoders."""
    actor_encoder = CanViTSequenceEncoder(
        patch_dim=args.patch_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_steps=args.t,
        n_patches=args.n_patches,
    ).to(device)
    q1_encoder = copy.deepcopy(actor_encoder).to(device)
    q2_encoder = copy.deepcopy(actor_encoder).to(device)
    target_q1_encoder = copy.deepcopy(actor_encoder).to(device)
    target_q2_encoder = copy.deepcopy(actor_encoder).to(device)
    actor = GaussianActor(actor_encoder, args.d_model).to(device)
    q1 = ContinuousCritic(q1_encoder, args.d_model).to(device)
    q2 = ContinuousCritic(q2_encoder, args.d_model).to(device)
    target_q1 = ContinuousCritic(target_q1_encoder, args.d_model).to(device)
    target_q2 = ContinuousCritic(target_q2_encoder, args.d_model).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    return actor, q1, q2, target_q1, target_q2


def _load_pretrained_critic(
    *,
    path: Path,
    q1: ContinuousCritic,
    q2: ContinuousCritic,
    target_q1: ContinuousCritic,
    target_q2: ContinuousCritic,
) -> None:
    """Initialize SAC critics from an independently pretrained checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "q1" not in checkpoint:
        raise ValueError(f"Expected critic checkpoint with q1/q2 keys: {path}")
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint.get("q2", checkpoint["q1"]))
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-patches", type=int, default=64)
    parser.add_argument("--patch-dim", type=int, default=768)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--pretrained-critic", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/canvit_sac"),
    )
    args = parser.parse_args()

    if args.t < 0:
        raise ValueError("--t must be non-negative.")
    if args.min_scale <= 0 or args.min_scale >= 1:
        raise ValueError("Require 0 < --min-scale < 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    cfg = CanViTEnvConfig()
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=RandomSampler(dataset, replacement=True),
        num_workers=0,
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

    actor, q1, q2, target_q1, target_q2 = _build_actor_and_critics(args, device)
    if args.pretrained_critic is not None:
        print(f"Loading pretrained critic: {args.pretrained_critic}")
        _load_pretrained_critic(
            path=args.pretrained_critic,
            q1=q1,
            q2=q2,
            target_q1=target_q1,
            target_q2=target_q2,
        )
    agent = ContinuousSAC(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        lr=args.lr,
        alpha=args.alpha,
        gamma=args.gamma,
        tau=args.tau,
    )
    replay = SequenceReplayBuffer(
        capacity=args.buffer_size,
        max_steps=args.t,
        n_patches=args.n_patches,
        patch_dim=args.patch_dim,
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_returns = deque(maxlen=args.log_interval)
    scaled_returns = deque(maxlen=args.log_interval)
    miou_by_t: deque[list[float]] = deque(maxlen=args.log_interval)
    action_scales = deque(maxlen=args.log_interval * max(args.t, 1))
    data_iter = iter(loader)
    total_steps = 0

    desc = "Training CanViT SAC"
    for episode in tqdm(range(1, args.episodes + 1), desc=desc):
        try:
            image, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            image, mask = next(data_iter)
        image = image.to(device)
        mask = mask.to(device)
        state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
        seq = empty_sequence(n_patches=args.n_patches, patch_dim=args.patch_dim)

        full_vp = Viewpoint.full_scene(batch_size=1, device=device)
        full_glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        with torch.inference_mode():
            full_out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
        state = full_out.state
        full_patches = extract_local_patches(full_out)
        if (
            full_patches.shape[1] != args.n_patches
            or full_patches.shape[2] != args.patch_dim
        ):
            raise ValueError(
                "Unexpected full-scene local patch shape. Pass matching "
                f"--n-patches and --patch-dim. got={tuple(full_patches.shape)}"
            )
        # Fixed by Codex on 2026-06-02
        # Problem: The training loop should expose only the actual mIoU SAC
        # baseline being run.
        # Solution: Keep only full-scene warmup plus patch/coordinate sequence
        # state and train on scaled delta-mIoU.
        # Result: The code exposes the experiment being run without stale
        # unrelated auxiliary reward pathways.
        seq = append_glimpse(seq=seq, patches=full_patches, viewpoint=full_vp)
        episode_mious = [
            miou_from_state(
                model=model,
                state=state,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
        ]
        prev_miou = episode_mious[-1]
        episode_raw_return = 0.0
        episode_scaled_return = 0.0

        for step_idx in range(args.t):
            batch = batch_from_sequence(
                seq,
                max_steps=args.t,
                n_patches=args.n_patches,
                patch_dim=args.patch_dim,
                device=device,
            )
            if total_steps < args.learning_starts:
                action = torch.empty(1, 3, device=device).uniform_(-1.0, 1.0)
            else:
                with torch.no_grad():
                    action, _ = actor.sample(batch)
            vp = _action_to_viewpoint(action, min_scale=args.min_scale)
            glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=vp,
                glimpse_size_px=cfg.glimpse_size_px,
            )
            with torch.inference_mode():
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state
            current_miou = miou_from_state(
                model=model,
                state=state,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            episode_mious.append(current_miou)

            patches = extract_local_patches(out)
            if patches.shape[1] != args.n_patches or patches.shape[2] != args.patch_dim:
                raise ValueError(
                    "Unexpected local patch shape. Pass matching --n-patches and "
                    f"--patch-dim. got={tuple(patches.shape)}"
                )
            next_seq = append_glimpse(seq=seq, patches=patches, viewpoint=vp)
            raw_reward = current_miou - prev_miou
            scaled_reward = raw_reward * args.reward_scale
            replay.add(
                seq=seq,
                action=action.squeeze(0).detach().cpu().numpy(),
                reward=scaled_reward,
                next_seq=next_seq,
                done=step_idx == args.t - 1,
            )
            seq = next_seq
            prev_miou = current_miou
            episode_raw_return += raw_reward
            episode_scaled_return += scaled_reward
            action_scales.append(float(vp.scales.mean().detach().cpu().item()))
            total_steps += 1

            if replay.size >= args.learning_starts:
                for _ in range(args.updates_per_step):
                    agent.update(replay.sample(args.batch_size, device))

        raw_returns.append(episode_raw_return)
        scaled_returns.append(episode_scaled_return)
        miou_by_t.append(episode_mious)
        if episode % args.log_interval == 0:
            mean_miou_by_t = _mean_by_step(miou_by_t, args.t + 1)
            print(
                f"episode={episode} steps={total_steps} "
                f"mean_raw_return={sum(raw_returns) / len(raw_returns):+.6f} "
                f"mean_scaled_return={sum(scaled_returns) / len(scaled_returns):+.4f} "
                f"mean_scale={sum(action_scales) / len(action_scales):.4f} "
                f"mean_miou_by_t={_format_step_means(mean_miou_by_t)}"
            )
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "q1": q1.state_dict(),
                    "q2": q2.state_dict(),
                    "args": vars(args),
                },
                args.checkpoint_dir / "latest.pt",
            )

    torch.save(actor.state_dict(), args.checkpoint_dir / "actor_final.pt")
    print(f"Saved final actor to {args.checkpoint_dir / 'actor_final.pt'}")


if __name__ == "__main__":
    main()
