"""Replay storage and SAC update helpers for image-dependent canvas policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic

REPLAY_STORAGE_DTYPE = torch.float16
REPLAY_STORAGE_DTYPE_BYTES = 2
REPLAY_GPU_FRACTION = 0.55
MAX_CPU_REPLAY_BYTES = 96 * 1024**3


def replay_canvas_bytes(
    *,
    capacity: int,
    canvas_feature_dim: int,
    canvas_grid_size: int,
    include_entropy: bool = False,
) -> int:
    """Return bytes for current + next canvas/entropy replay tensors."""
    canvas_bytes = (
        2
        * capacity
        * canvas_feature_dim
        * canvas_grid_size
        * canvas_grid_size
        * REPLAY_STORAGE_DTYPE_BYTES
    )
    if not include_entropy:
        return canvas_bytes
    entropy_bytes = (
        2
        * capacity
        * canvas_grid_size
        * canvas_grid_size
        * REPLAY_STORAGE_DTYPE_BYTES
    )
    return canvas_bytes + entropy_bytes


def resolve_replay_device(
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


def validate_replay_memory(
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
        store_entropy: bool = False,
    ) -> None:
        self.capacity = capacity
        self.storage_device = storage_device
        self.store_entropy = store_entropy
        alloc_kwargs = {"device": storage_device}
        self.canvas = torch.zeros(
            (capacity, canvas_feature_dim, canvas_grid_size, canvas_grid_size),
            dtype=REPLAY_STORAGE_DTYPE,
            **alloc_kwargs,
        )
        self.next_canvas = torch.zeros_like(self.canvas)
        if store_entropy:
            self.entropy = torch.zeros(
                (capacity, 1, canvas_grid_size, canvas_grid_size),
                dtype=REPLAY_STORAGE_DTYPE,
                **alloc_kwargs,
            )
            self.next_entropy = torch.zeros_like(self.entropy)
        else:
            self.entropy = None
            self.next_entropy = None
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
        entropy: torch.Tensor | None = None,
        next_entropy: torch.Tensor | None = None,
    ) -> None:
        batch_size = canvas.shape[0]
        if batch_size > self.capacity:
            raise ValueError(
                f"Replay batch_size={batch_size} exceeds capacity={self.capacity}; "
                "increase --buffer-size or reduce --batch-size."
            )
        self._copy_rows(self.canvas, canvas)
        if self.store_entropy:
            if entropy is None or next_entropy is None:
                raise ValueError(
                    "Entropy replay is enabled but entropy tensors are missing."
                )
            assert self.entropy is not None and self.next_entropy is not None
            # Problem: entropy-state runs need Bellman updates from replay, not
            # only online rollouts. Solution: store current/next entropy maps
            # next to canvas maps. Result: actor and critics see identical state
            # fields during online action selection and sampled SAC updates.
            self._copy_rows(self.entropy, entropy)
            self._copy_rows(self.next_entropy, next_entropy)
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

        batch = {
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
        if self.store_entropy:
            assert self.entropy is not None and self.next_entropy is not None
            batch["entropy"] = move(self.entropy, dtype=torch.float32)
            batch["next_entropy"] = move(self.next_entropy, dtype=torch.float32)
        return batch


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
        if "entropy" in sample:
            obs["entropy"] = sample["entropy"]
            next_obs["entropy"] = sample["next_entropy"]

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
