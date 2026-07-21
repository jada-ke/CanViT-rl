"""On-policy PPO helpers for image-dependent canvas policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic


class CanvasPPOCollapseError(RuntimeError):
    """Raised when a PPO trial hits configured collapse/pruning criteria."""


def _atanh(action: torch.Tensor) -> torch.Tensor:
    """Invert tanh safely for replaying PPO log-probs of stored actions."""
    action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return 0.5 * (torch.log1p(action) - torch.log1p(-action))


def canvas_actor_log_prob(
    actor: CanvasStateActor,
    obs: dict[str, torch.Tensor],
    action: torch.Tensor,
    *,
    log_std_min: float | None = None,
    log_std_max: float | None = None,
) -> torch.Tensor:
    """Return tanh-squashed Gaussian log-prob for an already sampled action."""
    mean, log_std = actor(obs)
    if log_std_min is not None or log_std_max is not None:
        log_std = log_std.clamp(
            min=log_std_min if log_std_min is not None else -float("inf"),
            max=log_std_max if log_std_max is not None else float("inf"),
        )
    raw = _atanh(action)
    dist = torch.distributions.Normal(mean, log_std.exp())
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    return (dist.log_prob(raw) - correction).sum(dim=-1)


def canvas_actor_sample(
    actor: CanvasStateActor,
    obs: dict[str, torch.Tensor],
    *,
    log_std_min: float,
    log_std_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample PPO actions with a tighter policy std cap than SAC."""
    mean, log_std = actor(obs)
    # Problem: PPO can drive SAC's broad log_std cap of 2.0 into tanh
    # saturation, causing boundary actions, KL spikes, and negative squashed
    # log-prob entropy. Solution: clamp std inside PPO's sampling path without
    # changing the shared actor module. Result: SAC keeps its old exploration
    # range while PPO gets bounded, stable rollout actions.
    log_std = log_std.clamp(log_std_min, log_std_max)
    dist = torch.distributions.Normal(mean, log_std.exp())
    raw = dist.rsample()
    action = torch.tanh(raw)
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    log_prob = (dist.log_prob(raw) - correction).sum(dim=-1)
    return action, log_prob, log_std


def canvas_actor_log_prob_and_entropy(
    actor: CanvasStateActor,
    obs: dict[str, torch.Tensor],
    action: torch.Tensor,
    *,
    log_std_min: float,
    log_std_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return action log-prob plus a stable current-policy entropy proxy."""
    mean, log_std = actor(obs)
    log_std = log_std.clamp(log_std_min, log_std_max)
    raw = _atanh(action)
    dist = torch.distributions.Normal(mean, log_std.exp())
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    log_prob = (dist.log_prob(raw) - correction).sum(dim=-1)
    # Problem: using -log_prob(old_action) as PPO's entropy bonus can reward
    # moving away from stale rollout actions instead of keeping the current
    # policy broad. Solution: use the current Gaussian's pre-squash entropy as
    # a stable exploration proxy. Result: entropy regularization acts directly
    # on policy std and is much less prone to sudden viewpoint collapse.
    entropy = dist.entropy().sum(dim=-1)
    return log_prob, entropy, log_std


@dataclass
class CanvasPPORollout:
    """On-policy rollout storage with GAE for Canvas PPO updates."""

    gamma: float
    gae_lambda: float
    canvas: list[torch.Tensor] = field(default_factory=list)
    entropy: list[torch.Tensor] = field(default_factory=list)
    coords: list[torch.Tensor] = field(default_factory=list)
    lengths: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    old_log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def add_batch(
        self,
        *,
        canvas: torch.Tensor,
        coords: torch.Tensor,
        lengths: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> None:
        """Append one vectorized environment step to the on-policy rollout."""
        self.canvas.append(canvas.detach())
        if entropy is not None:
            self.entropy.append(entropy.detach())
        self.coords.append(coords.detach())
        self.lengths.append(lengths.detach())
        self.actions.append(actions.detach())
        self.old_log_probs.append(old_log_probs.detach())
        self.rewards.append(rewards.detach())
        self.dones.append(dones.detach())
        self.values.append(values.detach())

    def __len__(self) -> int:
        return sum(int(reward.numel()) for reward in self.rewards)

    @staticmethod
    def _flatten_time_batch(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1, *tensor.shape[2:])

    def to_training_batch(self) -> dict[str, torch.Tensor]:
        """Flatten rollout tensors and attach normalized GAE advantages."""
        if not self.rewards:
            raise ValueError("Cannot train PPO from an empty rollout.")

        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        values = torch.stack(self.values)
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(rewards[0])
        next_value = torch.zeros_like(rewards[0])
        for step in reversed(range(rewards.shape[0])):
            not_done = 1.0 - dones[step]
            # Problem: PPO needs a low-variance on-policy target while the
            # canvas loop only has sampled action returns. Solution: use GAE
            # against the same action-conditioned critic used by SAC. Result:
            # the architecture stays shared while the optimizer is PPO.
            delta = rewards[step] + self.gamma * not_done * next_value - values[step]
            last_advantage = delta + self.gamma * self.gae_lambda * not_done * last_advantage
            advantages[step] = last_advantage
            next_value = values[step]
        returns = advantages + values
        flat_advantages = advantages.reshape(-1)
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std(unbiased=False).clamp_min(1e-8)
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        batch = {
            "canvas": self._flatten_time_batch(torch.stack(self.canvas)),
            "coords": self._flatten_time_batch(torch.stack(self.coords)),
            "lengths": torch.stack(self.lengths).reshape(-1),
            "actions": self._flatten_time_batch(torch.stack(self.actions)),
            "old_log_probs": torch.stack(self.old_log_probs).reshape(-1),
            "returns": returns.reshape(-1),
            "advantages": flat_advantages,
        }
        if self.entropy:
            batch["entropy"] = self._flatten_time_batch(torch.stack(self.entropy))
        return batch


class CanvasPPO:
    """Clipped PPO for the existing CanvasStateActor/Critic modules."""

    def __init__(
        self,
        *,
        actor: CanvasStateActor,
        critic: CanvasStateCritic,
        actor_lr: float,
        critic_lr: float,
        clip_coef: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        epochs: int,
        minibatch_size: int,
        target_kl: float | None = None,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.actor_opt = torch.optim.AdamW(actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.AdamW(critic.parameters(), lr=critic_lr)
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def update(self, rollout: CanvasPPORollout) -> dict[str, float]:
        """Run PPO epochs over one rollout and return averaged metrics."""
        batch = rollout.to_training_batch()
        total = batch["actions"].shape[0]
        metric_sums: dict[str, float] = {}
        updates = 0
        device = batch["actions"].device

        early_stop = False
        for _ in range(self.epochs):
            order = torch.randperm(total, device=device)
            for start in range(0, total, self.minibatch_size):
                idx = order[start : start + self.minibatch_size]
                obs = {
                    "canvas": batch["canvas"].index_select(0, idx),
                    "coords": batch["coords"].index_select(0, idx),
                    "lengths": batch["lengths"].index_select(0, idx),
                }
                if "entropy" in batch:
                    obs["entropy"] = batch["entropy"].index_select(0, idx)
                actions = batch["actions"].index_select(0, idx)
                old_log_probs = batch["old_log_probs"].index_select(0, idx)
                returns = batch["returns"].index_select(0, idx)
                advantages = batch["advantages"].index_select(0, idx)

                log_probs, policy_entropy, log_std = canvas_actor_log_prob_and_entropy(
                    self.actor,
                    obs,
                    actions,
                    log_std_min=self.log_std_min,
                    log_std_max=self.log_std_max,
                )
                ratio = (log_probs - old_log_probs).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                ) * advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                entropy_bonus = policy_entropy.mean()
                values = self.critic(obs, actions)
                value_loss = F.mse_loss(values, returns)
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                self.actor_opt.step()
                self.critic_opt.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - log_probs).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_coef
                    ).float().mean()
                    metrics = {
                        "actor/loss": float(actor_loss.item()),
                        "actor/entropy": float(entropy_bonus.item()),
                        "actor/log_prob": float(log_probs.mean().item()),
                        "actor/action_std": float(actions.std(unbiased=False).item()),
                        "actor/log_std_mean": float(log_std.mean().item()),
                        "actor/log_std_max": float(log_std.max().item()),
                        "actor/std_mean": float(log_std.exp().mean().item()),
                        "actor/std_max": float(log_std.exp().max().item()),
                        "critic/value_loss": float(value_loss.item()),
                        "critic/value_mean": float(values.mean().item()),
                        "critic/value_std": float(values.std(unbiased=False).item()),
                        "ppo/approx_kl": float(approx_kl.item()),
                        "ppo/clip_fraction": float(clip_fraction.item()),
                        "ppo/ratio_mean": float(ratio.mean().item()),
                        "ppo/advantage_mean": float(advantages.mean().item()),
                        "ppo/advantage_std": float(
                            advantages.std(unbiased=False).item()
                        ),
                    }
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                updates += 1
                if (
                    self.target_kl is not None
                    and self.target_kl > 0.0
                    and metrics["ppo/approx_kl"] > self.target_kl
                ):
                    early_stop = True
                    break
            if early_stop:
                break

        return {key: value / max(updates, 1) for key, value in metric_sums.items()}
