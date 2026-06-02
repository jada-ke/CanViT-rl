"""
Train an AdaGlimpse-like continuous SAC policy on frozen CanViT.

State: variable-length sequence of previous CanViT glimpse tuples:
    local patches, viewpoint coordinates, and importance scores.
Action: continuous Viewpoint parameters [cx, cy, scale_raw].
Reward: scaled improvement in mIoU, KL, or teacher-CLS cosine similarity.

Example:
    uv run python scripts/train_adaglimpse_canvit_sac.py --episodes 1000 --t 5
    uv run python scripts/train_adaglimpse_canvit_sac.py \
        --episodes 1000 --t 5 --importance-mode zeros
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_pytorch.policies import random_viewpoints
from canvit_pytorch.teacher import load_teacher
from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import miou_from_state


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


def _seg_logits_from_state(
    *,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    state: Any,
    canvas_grid_size: int,
) -> torch.Tensor:
    """Decode segmentation logits from one CanViT recurrent canvas state."""
    spatial = model.get_spatial(state.canvas).view(
        1,
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        return probe(spatial.float()).float()


def _kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student), matching the AdaGlimpse-style task loss."""
    if student_logits.shape[-2:] != teacher_prob.shape[-2:]:
        student_logits = F.interpolate(
            student_logits,
            size=teacher_prob.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def _teacher_prob_from_canvit_full_scene(
    *,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    image: torch.Tensor,
    cfg: CanViTEnvConfig,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """Use frozen CanViT full-scene logits as the KL teacher distribution."""
    state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
    vp = Viewpoint.full_scene(batch_size=1, device=device)
    glimpse = sample_at_viewpoint(
        spatial=image,
        viewpoint=vp,
        glimpse_size_px=cfg.glimpse_size_px,
    )
    out = model(glimpse=glimpse, state=state, viewpoint=vp)
    logits = _seg_logits_from_state(
        model=model,
        probe=probe,
        state=out.state,
        canvas_grid_size=cfg.canvas_grid_size,
    )
    return F.softmax(logits / temperature, dim=1).detach()


def _teacher_prob_from_deeplab(
    *,
    teacher: torch.nn.Module,
    image: torch.Tensor,
    n_classes: int,
    temperature: float,
) -> torch.Tensor:
    """Use a frozen DeepLabV3 teacher distribution for the full image."""
    out = teacher(image)
    logits = out["out"] if isinstance(out, dict) else out
    if logits.shape[1] != n_classes:
        raise ValueError(
            "DeepLab teacher class count does not match the CanViT probe: "
            f"teacher={logits.shape[1]} probe={n_classes}. Pass a matching "
            "--deeplab-checkpoint or use --teacher-mode canvit-full-scene."
        )
    return F.softmax(logits / temperature, dim=1).detach()


def _num_register_tokens(teacher: torch.nn.Module) -> int:
    """Return the teacher's DINO register-token count when exposed."""
    config = getattr(teacher, "config", None)
    value = getattr(config, "num_register_tokens", 0)
    return int(value or 0)


def _probe_embed_dim(probe: torch.nn.Module) -> int | None:
    """Return the expected probe input width when the probe exposes it."""
    for name in ("embed_dim", "in_features", "hidden_size"):
        value = getattr(probe, name, None)
        if isinstance(value, int):
            return value
    return None


def _extract_teacher_patch_tokens(
    features: Any,
    *,
    teacher: torch.nn.Module,
) -> torch.Tensor:
    """Extract spatial patch tokens from DINO-like normalized features."""
    # Fixed by Codex on 2026-06-02
    # Problem: Different DINO/CanViT teacher wrappers expose spatial tokens
    # under different names, and this repo's NormFeatures object does not have
    # the upstream DINOv3 attribute `x_norm_patchtokens`. HuggingFace DINOv3
    # can also expose a full token sequence that includes CLS and registers.
    # Solution: Prefer true patch-token fields; otherwise slice full hidden
    # states after the CLS token and the configured register tokens.
    # Result: `--teacher-mode dinov3-probe` works across teacher wrappers and
    # keeps only local patch embeddings for dense probe decoding.
    for name in (
        "patch_tokens",
        "patchtokens",
        "patches",
        "x_norm_patchtokens",
        "x_norm_patch_tokens",
    ):
        value = getattr(features, name, None)
        if isinstance(value, torch.Tensor):
            if value.ndim == 4:
                return value.flatten(1, 2)
            if value.ndim == 3:
                return value

    sequence = None
    for name in ("last_hidden_state", "hidden_state", "tokens", "x_norm"):
        value = getattr(features, name, None)
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            sequence = value
            break
    hidden_states = getattr(features, "hidden_states", None)
    if sequence is None and isinstance(hidden_states, tuple) and hidden_states:
        value = hidden_states[-1]
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            sequence = value
    if sequence is not None:
        local_start = 1 + _num_register_tokens(teacher)
        if sequence.shape[1] <= local_start:
            raise ValueError(
                "DINO token sequence is too short to contain patch tokens after "
                f"CLS/register slicing: shape={tuple(sequence.shape)} "
                f"local_start={local_start}."
            )
        return sequence[:, local_start:, :]

    available = [
        name
        for name in dir(features)
        if not name.startswith("_") and not callable(getattr(features, name))
    ]
    raise AttributeError(
        "Could not find DINO spatial patch tokens on NormFeatures. "
        f"Available non-callable fields: {available}"
    )


def _teacher_prob_from_dinov3_probe(
    *,
    teacher: torch.nn.Module,
    probe: torch.nn.Module,
    image: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Decode frozen DINOv3 full-scene patch tokens through the frozen probe."""
    with torch.inference_mode():
        features = teacher.forward_norm_features(image)
        patch_tokens = _extract_teacher_patch_tokens(features, teacher=teacher)
        batch_size, n_tokens, dim = patch_tokens.shape
        grid_size = int(n_tokens**0.5)
        if grid_size * grid_size != n_tokens:
            raise ValueError(
                "DINOv3 patch tokens do not form a square spatial grid: "
                f"n_tokens={n_tokens}."
            )
        spatial = patch_tokens.view(batch_size, grid_size, grid_size, dim)
        expected_dim = _probe_embed_dim(probe)
        if expected_dim is not None and dim != expected_dim:
            teacher_name = getattr(getattr(teacher, "config", None), "name_or_path", "")
            raise ValueError(
                "DINO teacher patch-token width does not match the frozen "
                f"segmentation probe: teacher_dim={dim} probe_embed_dim="
                f"{expected_dim} teacher={teacher_name!r}. Use a DINO teacher "
                "whose hidden size matches the CanViT/probe width, e.g. a B/16 "
                "teacher for a 768-dim CanViT-B probe, or switch "
                "--teacher-mode canvit-full-scene."
            )
        # Fixed by Codex on 2026-06-02
        # Problem: A CanViT full-scene teacher becomes degenerate once every
        # episode starts with a full-scene warmup glimpse.
        # Solution: Use frozen DINOv3 full-scene spatial tokens decoded by the
        # same frozen segmentation probe as the student canvas.
        # Result: KL reward compares CanViT's partial canvas to a teacher in
        # the same probe-logit space without using the current canvas as target.
        teacher_logits = probe(spatial.float()).float()
        return F.softmax(teacher_logits / temperature, dim=1).detach()


def _extract_local_patches(out: Any) -> torch.Tensor:
    """Extract CanViT local patch tokens from a forward output."""
    for name in ("local_patches", "patches", "local_tokens"):
        value = getattr(out, name, None)
        if isinstance(value, torch.Tensor):
            patches = value
            break
    else:
        raise AttributeError(
            "CanViT output does not expose local patch tokens. Expected one of "
            "`out.local_patches`, `out.patches`, or `out.local_tokens`."
        )
    if patches.ndim == 2:
        patches = patches.unsqueeze(0)
    if patches.ndim != 3:
        raise ValueError(f"Expected local patches [B, N, D], got {patches.shape}.")
    return patches.detach().float()


def _find_attention_tensor(out: Any) -> torch.Tensor | None:
    """Best-effort extraction of canvas/glimpse attention weights from CanViT."""
    candidates = (
        "canvas_attention_weights",
        "canvas_attn_weights",
        "attention_weights",
        "attn_weights",
        "attn",
    )
    for name in candidates:
        value = getattr(out, name, None)
        if isinstance(value, torch.Tensor):
            return value.detach().float()
    return None


def _attention_to_importance(attention: torch.Tensor, n_patches: int) -> torch.Tensor:
    """Reduce attention-like tensors to one importance score per local patch."""
    if attention.ndim == 2 and attention.shape[-1] == n_patches:
        weights = attention
    else:
        if attention.shape[-1] != n_patches:
            raise ValueError(
                "Attention tensor does not end with the patch dimension: "
                f"shape={tuple(attention.shape)} n_patches={n_patches}."
            )
        weights = attention.reshape(attention.shape[0], -1, n_patches).mean(dim=1)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights / denom


def _extract_importance(
    *,
    out: Any,
    patches: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Return AdaGlimpse-style importance scores for local patches."""
    # Fixed by Codex on 2026-06-02
    # Problem: AdaGlimpse has a learned importance head, while frozen CanViT
    # does not expose the same trained signal.
    # Solution: Use canvas/glimpse attention weights when available, with
    # zeros and patch-norm ablations to test whether importance matters.
    # Result: The SAC state can include an AdaGlimpse-like I_t component
    # without unfreezing CanViT or inventing a new trained head.
    if mode == "zeros":
        return torch.zeros(patches.shape[:2], device=patches.device)
    if mode == "patch-norm":
        weights = patches.norm(dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if mode == "attention":
        attention = _find_attention_tensor(out)
        if attention is None:
            raise AttributeError(
                "Importance mode 'attention' requested, but no attention weights "
                "were found on the CanViT output. Try --importance-mode zeros "
                "for the ablation or --importance-mode patch-norm."
            )
        return _attention_to_importance(attention, patches.shape[1]).to(patches.device)
    raise ValueError(f"Unknown importance mode: {mode}")


def _append_glimpse(
    *,
    seq: dict[str, torch.Tensor],
    patches: torch.Tensor,
    viewpoint: Viewpoint,
    importance: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Append one detached glimpse tuple to a CPU sequence state."""
    coords = torch.cat(
        [
            viewpoint.centers.detach().float(),
            viewpoint.scales[:, None].detach().float(),
        ],
        dim=-1,
    )
    return {
        "patches": torch.cat([seq["patches"], patches.cpu()], dim=0),
        "coords": torch.cat([seq["coords"], coords.cpu()], dim=0),
        "importance": torch.cat([seq["importance"], importance.cpu()], dim=0),
    }


def _empty_sequence(*, n_patches: int, patch_dim: int) -> dict[str, torch.Tensor]:
    """Create an empty AdaGlimpse-like CanViT sequence state."""
    return {
        "patches": torch.zeros(0, n_patches, patch_dim),
        "coords": torch.zeros(0, 3),
        "importance": torch.zeros(0, n_patches),
    }


def _sequence_to_arrays(
    seq: dict[str, torch.Tensor],
    *,
    max_steps: int,
    n_patches: int,
    patch_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad a variable-length sequence for replay storage."""
    length = min(int(seq["patches"].shape[0]), max_steps)
    patches = np.zeros((max_steps, n_patches, patch_dim), dtype=np.float32)
    coords = np.zeros((max_steps, 3), dtype=np.float32)
    importance = np.zeros((max_steps, n_patches), dtype=np.float32)
    if length:
        patches[:length] = seq["patches"][:length].numpy()
        coords[:length] = seq["coords"][:length].numpy()
        importance[:length] = seq["importance"][:length].numpy()
    return patches, coords, importance, length


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
        self.importance = np.zeros((capacity, max_steps, n_patches), dtype=np.float32)
        self.next_importance = np.zeros_like(self.importance)
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
        obs = _sequence_to_arrays(
            seq,
            max_steps=self.max_steps,
            n_patches=self.n_patches,
            patch_dim=self.patch_dim,
        )
        next_obs = _sequence_to_arrays(
            next_seq,
            max_steps=self.max_steps,
            n_patches=self.n_patches,
            patch_dim=self.patch_dim,
        )
        self.patches[self.pos] = obs[0]
        self.coords[self.pos] = obs[1]
        self.importance[self.pos] = obs[2]
        self.lengths[self.pos] = obs[3]
        (
            self.next_patches[self.pos],
            self.next_coords[self.pos],
            self.next_importance[self.pos],
        ) = next_obs[:3]
        self.next_lengths[self.pos] = next_obs[3]
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
            "importance": torch.as_tensor(self.importance[idx], device=device),
            "lengths": torch.as_tensor(self.lengths[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_patches": torch.as_tensor(self.next_patches[idx], device=device),
            "next_coords": torch.as_tensor(self.next_coords[idx], device=device),
            "next_importance": torch.as_tensor(
                self.next_importance[idx],
                device=device,
            ),
            "next_lengths": torch.as_tensor(self.next_lengths[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


class CanViTSequenceEncoder(nn.Module):
    """Transformer encoder over AdaGlimpse-like CanViT glimpse tuples."""

    def __init__(
        self,
        *,
        patch_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_steps: int,
        n_patches: int,
    ) -> None:
        super().__init__()
        # Fixed by Codex on 2026-06-02
        # Problem: AdaGlimpse's actor/critic consume a growing sequence of
        # glimpse tuples, but SAC needs fixed-size tensors in replay.
        # Solution: Store padded CanViT patch/coord/importance sequences and
        # use a Transformer with a padding mask plus learned CLS token.
        # Result: Empty s0 and variable-length later states share one encoder.
        self.max_steps = max_steps
        self.n_patches = n_patches
        self.token_proj = nn.Linear(patch_dim + 4, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_steps * n_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        patches: torch.Tensor,
        coords: torch.Tensor,
        importance: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = patches.shape[0]
        coords_expanded = coords[:, :, None, :].expand(-1, -1, self.n_patches, -1)
        imp = importance[..., None]
        tokens = torch.cat([patches, coords_expanded, imp], dim=-1)
        tokens = self.token_proj(tokens.reshape(batch_size, -1, tokens.shape[-1]))
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.shape[1] + 1]

        step_ids = torch.arange(self.max_steps, device=patches.device)[None, :]
        valid_steps = step_ids < lengths[:, None]
        valid_tokens = valid_steps[:, :, None].expand(-1, -1, self.n_patches)
        pad_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=patches.device),
                ~valid_tokens.reshape(batch_size, -1),
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
        return self.norm(encoded[:, 0])


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor for continuous Viewpoint actions."""

    def __init__(self, encoder: CanViTSequenceEncoder, d_model: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(
            batch["patches"],
            batch["coords"],
            batch["importance"],
            batch["lengths"],
        )
        mean, log_std = self.head(z).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def sample(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(batch)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = (dist.log_prob(raw) - correction).sum(dim=-1)
        return action, log_prob


class ContinuousCritic(nn.Module):
    """Q(s, a) critic for the transformer-encoded CanViT sequence state."""

    def __init__(self, encoder: CanViTSequenceEncoder, d_model: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.q = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(
            batch["patches"],
            batch["coords"],
            batch["importance"],
            batch["lengths"],
        )
        return self.q(torch.cat([z, action], dim=-1)).squeeze(-1)


class ContinuousSAC:
    """Minimal continuous SAC update for the CanViT AdaGlimpse policy."""

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
            "importance": batch["next_importance"],
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
            "importance": batch["importance"],
            "lengths": batch["lengths"],
        }
        q_loss = F.mse_loss(self.q1(obs_batch, batch["actions"]), target) + F.mse_loss(
            self.q2(obs_batch, batch["actions"]),
            target,
        )
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

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)


def _batch_from_sequence(
    seq: dict[str, torch.Tensor],
    *,
    max_steps: int,
    n_patches: int,
    patch_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    patches, coords, importance, length = _sequence_to_arrays(
        seq,
        max_steps=max_steps,
        n_patches=n_patches,
        patch_dim=patch_dim,
    )
    return {
        "patches": torch.as_tensor(patches[None], device=device),
        "coords": torch.as_tensor(coords[None], device=device),
        "importance": torch.as_tensor(importance[None], device=device),
        "lengths": torch.as_tensor([length], device=device),
    }


def _load_deeplab_teacher(
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    """Load a frozen DeepLabV3 teacher, optionally from a checkpoint."""
    from torchvision.models.segmentation import deeplabv3_resnet101

    teacher = deeplabv3_resnet101(num_classes=args.deeplab_num_classes)
    if args.deeplab_checkpoint is not None:
        state = torch.load(args.deeplab_checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state.get("model", state))
        teacher.load_state_dict(state_dict, strict=False)
    teacher = teacher.eval().to(device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _debug_random_cosine_rollout(
    *,
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    image: torch.Tensor,
    cfg: CanViTEnvConfig,
    t: int,
    min_scale: float,
    device: torch.device,
) -> None:
    """Print DINO-CLS cosine deltas for random glimpses before training."""
    with torch.inference_mode():
        teacher_cls = teacher.forward_norm_features(image).cls
        canvas = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
        full_vp = Viewpoint.full_scene(batch_size=1, device=device)
        full_glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        out = model(glimpse=full_glimpse, state=canvas, viewpoint=full_vp)
        canvas = out.state
        prev_score = float(
            F.cosine_similarity(
                canvas.recurrent_cls.squeeze(1).float(),
                teacher_cls.float(),
                dim=-1,
            ).item()
        )
        print(f"debug_random_cosine step=full cosine={prev_score:.6f}")

        # Fixed by Codex on 2026-06-02
        # Problem: Before training, we need to know whether random post-warmup
        # CanViT glimpses improve or degrade the teacher-CLS reward signal.
        # Solution: Run one random rollout and print cosine plus deltas before
        # actor/critic training starts.
        # Result: Reward direction and scale are visible in the terminal before
        # spending time on SAC updates.
        for step_idx in range(t):
            vp = random_viewpoints(
                batch_size=1,
                device=device,
                n_viewpoints=1,
                min_scale=min_scale,
                max_scale=1.0,
                start_with_full_scene=False,
            ).pop()
            glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=vp,
                glimpse_size_px=cfg.glimpse_size_px,
            )
            out = model(glimpse=glimpse, state=canvas, viewpoint=vp)
            canvas = out.state
            score = float(
                F.cosine_similarity(
                    canvas.recurrent_cls.squeeze(1).float(),
                    teacher_cls.float(),
                    dim=-1,
                ).item()
            )
            print(
                f"debug_random_cosine step={step_idx} "
                f"scale={float(vp.scales[0].item()):.4f} "
                f"cosine={score:.6f} delta={score - prev_score:+.6f}"
            )
            prev_score = score


def _debug_random_dinov3_probe_kl_rollout(
    *,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    teacher: torch.nn.Module,
    image: torch.Tensor,
    cfg: CanViTEnvConfig,
    t: int,
    min_scale: float,
    temperature: float,
    device: torch.device,
) -> None:
    """Print DINOv3-probe KL and reward deltas for random glimpses."""
    with torch.inference_mode():
        features = teacher.forward_norm_features(image)
        patch_tokens = _extract_teacher_patch_tokens(features, teacher=teacher)
        batch_size, n_tokens, dim = patch_tokens.shape
        grid_size = int(n_tokens**0.5)
        if grid_size * grid_size != n_tokens:
            raise ValueError(
                "DINOv3 patch tokens do not form a square spatial grid: "
                f"n_tokens={n_tokens}."
            )
        spatial_dino = patch_tokens.view(batch_size, grid_size, grid_size, dim)
        expected_dim = _probe_embed_dim(probe)
        if expected_dim is not None and dim != expected_dim:
            raise ValueError(
                "DINO teacher patch-token width does not match the frozen "
                f"segmentation probe: teacher_dim={dim} probe_embed_dim="
                f"{expected_dim}."
            )
        teacher_logits = probe(spatial_dino.float()).float()
        teacher_prob = F.softmax(teacher_logits / temperature, dim=1)

        canvas = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
        full_vp = Viewpoint.full_scene(batch_size=1, device=device)
        full_glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        out = model(glimpse=full_glimpse, state=canvas, viewpoint=full_vp)
        canvas = out.state
        logits = _seg_logits_from_state(
            model=model,
            probe=probe,
            state=canvas,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        student_prob = F.softmax(logits, dim=1)
        entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-8))).sum(dim=1)
        prev_kl = float(
            _kl_loss(
                student_logits=logits,
                teacher_prob=teacher_prob,
                temperature=temperature,
            ).item()
        )
        # Fixed by Codex on 2026-06-02
        # Problem: The DINOv3-probe KL teacher can silently be incompatible if
        # token grids, probe logits, or teacher entropy look wrong.
        # Solution: Print the teacher spatial shape, teacher/student logits,
        # teacher entropy, and full-scene KL before the random rollout.
        # Result: A single training launch now diagnoses whether the KL signal
        # is degenerate, too large, or shape-mismatched before SAC updates run.
        print(f"debug_dinov3_probe teacher spatial: {tuple(spatial_dino.shape)}")
        print(f"debug_dinov3_probe student logits: {tuple(logits.shape)}")
        print(f"debug_dinov3_probe teacher logits: {tuple(teacher_logits.shape)}")
        print(f"debug_dinov3_probe student prob: {tuple(student_prob.shape)}")
        print(
            "debug_dinov3_probe teacher entropy "
            f"mean={float(entropy.mean().item()):.4f} "
            f"std={float(entropy.std().item()):.4f}"
        )
        print(f"debug_random_dinov3_probe_kl step=full kl={prev_kl:.6f}")

        # Fixed by Codex on 2026-06-02
        # Problem: SAC reward scaling is hard to tune if the raw KL signal is
        # unknown before training starts.
        # Solution: Run one random post-warmup rollout against the DINOv3-probe
        # teacher and print KL plus the exact reward delta, prev_kl - kl.
        # Result: The terminal shows whether per-step rewards are in a useful
        # range before spending time on actor/critic updates.
        for step_idx in range(t):
            vp = random_viewpoints(
                batch_size=1,
                device=device,
                n_viewpoints=1,
                min_scale=min_scale,
                max_scale=1.0,
                start_with_full_scene=False,
            ).pop()
            glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=vp,
                glimpse_size_px=cfg.glimpse_size_px,
            )
            out = model(glimpse=glimpse, state=canvas, viewpoint=vp)
            canvas = out.state
            logits = _seg_logits_from_state(
                model=model,
                probe=probe,
                state=canvas,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            kl = float(
                _kl_loss(
                    student_logits=logits,
                    teacher_prob=teacher_prob,
                    temperature=temperature,
                ).item()
            )
            print(
                f"debug_random_dinov3_probe_kl step={step_idx} "
                f"scale={float(vp.scales[0].item()):.4f} "
                f"kl={kl:.6f} reward_delta={prev_kl - kl:+.6f}"
            )
            prev_kl = kl


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
    parser.add_argument("--teacher-repo", type=str, default=None)
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
    parser.add_argument(
        "--reward-objective",
        choices=["miou", "kl", "cosine"],
        default="miou",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=["dinov3-probe", "canvit-full-scene", "deeplabv3"],
        default="dinov3-probe",
    )
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--deeplab-checkpoint", type=Path, default=None)
    parser.add_argument("--deeplab-num-classes", type=int, default=150)
    parser.add_argument(
        "--importance-mode",
        choices=["attention", "zeros", "patch-norm"],
        default="attention",
    )
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/adaglimpse_canvit_sac"),
    )
    args = parser.parse_args()

    if args.kl_temperature <= 0:
        raise ValueError("--kl-temperature must be positive.")
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

    deeplab_teacher = None
    dino_teacher = None
    if args.reward_objective == "kl" and args.teacher_mode == "deeplabv3":
        deeplab_teacher = _load_deeplab_teacher(args, device)
    teacher_repo = args.teacher_repo or cfg.teacher_repo
    if args.reward_objective == "cosine" or (
        args.reward_objective == "kl" and args.teacher_mode == "dinov3-probe"
    ):
        dino_teacher = load_teacher(teacher_repo, device)

    if args.reward_objective in {"kl", "cosine"}:
        debug_image, _ = dataset[0]
        debug_teacher = dino_teacher or load_teacher(teacher_repo, device)
        debug_image = debug_image.unsqueeze(0).to(device)
        if args.reward_objective == "kl" and args.teacher_mode == "dinov3-probe":
            _debug_random_dinov3_probe_kl_rollout(
                model=model,
                probe=probe,
                teacher=debug_teacher,
                image=debug_image,
                cfg=cfg,
                t=args.t,
                min_scale=args.min_scale,
                temperature=args.kl_temperature,
                device=device,
            )
        else:
            _debug_random_cosine_rollout(
                model=model,
                teacher=debug_teacher,
                image=debug_image,
                cfg=cfg,
                t=args.t,
                min_scale=args.min_scale,
                device=device,
            )

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
    action_scales = deque(maxlen=args.log_interval * args.t)
    data_iter = iter(loader)
    total_steps = 0

    for episode in tqdm(range(1, args.episodes + 1), desc="Training AdaGlimpse SAC"):
        try:
            image, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            image, mask = next(data_iter)
        image = image.to(device)
        mask = mask.to(device)
        canvas = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
        seq = _empty_sequence(n_patches=args.n_patches, patch_dim=args.patch_dim)

        full_vp = Viewpoint.full_scene(batch_size=1, device=device)
        full_glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        with torch.inference_mode():
            full_out = model(glimpse=full_glimpse, state=canvas, viewpoint=full_vp)
        canvas = full_out.state
        full_patches = _extract_local_patches(full_out)
        if (
            full_patches.shape[1] != args.n_patches
            or full_patches.shape[2] != args.patch_dim
        ):
            raise ValueError(
                "Unexpected full-scene local patch shape. Pass matching "
                f"--n-patches and --patch-dim. got={tuple(full_patches.shape)}"
            )
        full_importance = _extract_importance(
            out=full_out,
            patches=full_patches,
            mode=args.importance_mode,
        )
        # Fixed by Codex on 2026-06-02
        # Problem: The continuous AdaGlimpse run must start from the same
        # full-scene context used by the other active-vision baselines.
        # Solution: Commit a full-scene glimpse without SAC reward, append its
        # tuple to the sequence, and use that state as the reward baseline.
        # Result: The actor's first learned action sees full-scene context and
        # mean_miou_by_t[0] is the full-scene mIoU.
        seq = _append_glimpse(
            seq=seq,
            patches=full_patches,
            viewpoint=full_vp,
            importance=full_importance,
        )
        # Fixed by Codex on 2026-06-02
        # Problem: KL/cosine rewards can improve while the ADE20K task metric
        # remains flat, so training logs need the real mIoU trajectory too.
        # Solution: Track mIoU at the full-scene t0 canvas and after every
        # learned action, then average the curve across each logging window.
        # Result: The continuous AdaGlimpse run reports task progress by
        # timestep without changing the reward objective.
        episode_mious = [
            miou_from_state(
                model=model,
                state=canvas,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
        ]

        if args.reward_objective == "miou":
            # Fixed by Codex on 2026-06-02
            # Problem: CLS-cosine rewards were too small/noisy to drive SAC,
            # and the DINOv3-probe KL teacher is not a clean dense target for
            # CanViT's frozen segmentation probe.
            # Solution: Use the already-computed ADE20K mIoU after the
            # full-scene warmup as the reward baseline.
            # Result: The default reward is now scaled delta-mIoU, directly
            # aligned with the metric we care about.
            prev_score = episode_mious[-1]
        elif args.reward_objective == "kl":
            # Fixed by Codex on 2026-06-02
            # Problem: The AdaGlimpse reward is KL improvement toward a frozen
            # full-scene teacher, while earlier CanViT experiments used mIoU or
            # teacher-CLS similarity diagnostics.
            # Solution: Build a frozen full-scene teacher distribution once per
            # episode and reward reductions in KL; keep cosine as an explicit
            # ablation via --reward-objective cosine.
            # Result: The training objective can match the AdaGlimpse-style KL
            # story while preserving a representation-proxy comparison.
            if args.teacher_mode == "canvit-full-scene":
                logits0 = _seg_logits_from_state(
                    model=model,
                    probe=probe,
                    state=canvas,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
                teacher_prob = F.softmax(
                    logits0 / args.kl_temperature,
                    dim=1,
                ).detach()
            elif args.teacher_mode == "dinov3-probe":
                assert dino_teacher is not None
                teacher_prob = _teacher_prob_from_dinov3_probe(
                    teacher=dino_teacher,
                    probe=probe,
                    image=image,
                    temperature=args.kl_temperature,
                )
                logits0 = _seg_logits_from_state(
                    model=model,
                    probe=probe,
                    state=canvas,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
            else:
                assert deeplab_teacher is not None
                teacher_prob = _teacher_prob_from_deeplab(
                    teacher=deeplab_teacher,
                    image=image,
                    n_classes=(
                        probe.num_classes
                        if hasattr(probe, "num_classes")
                        else args.deeplab_num_classes
                    ),
                    temperature=args.kl_temperature,
                )
                logits0 = _seg_logits_from_state(
                    model=model,
                    probe=probe,
                    state=canvas,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
            prev_score = -float(
                _kl_loss(
                    student_logits=logits0,
                    teacher_prob=teacher_prob,
                    temperature=args.kl_temperature,
                ).item()
            )
        else:
            assert dino_teacher is not None
            teacher_cls = dino_teacher.forward_norm_features(image).cls
            prev_score = float(
                F.cosine_similarity(
                    canvas.recurrent_cls.squeeze(1).float(),
                    teacher_cls.float(),
                    dim=-1,
                ).item()
            )

        episode_raw_return = 0.0
        episode_scaled_return = 0.0
        for step_idx in range(args.t):
            batch = _batch_from_sequence(
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
                out = model(glimpse=glimpse, state=canvas, viewpoint=vp)
            canvas = out.state
            current_miou = miou_from_state(
                model=model,
                state=canvas,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            episode_mious.append(current_miou)

            if args.reward_objective == "miou":
                score = current_miou
            elif args.reward_objective == "kl":
                logits = _seg_logits_from_state(
                    model=model,
                    probe=probe,
                    state=canvas,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
                score = -float(
                    _kl_loss(
                        student_logits=logits,
                        teacher_prob=teacher_prob,
                        temperature=args.kl_temperature,
                    ).item()
                )
            else:
                score = float(
                    F.cosine_similarity(
                        canvas.recurrent_cls.squeeze(1).float(),
                        teacher_cls.float(),
                        dim=-1,
                    ).item()
                )

            patches = _extract_local_patches(out)
            if (
                patches.shape[1] != args.n_patches
                or patches.shape[2] != args.patch_dim
            ):
                raise ValueError(
                    "Unexpected local patch shape. Pass matching --n-patches and "
                    f"--patch-dim. got={tuple(patches.shape)}"
                )
            importance = _extract_importance(
                out=out,
                patches=patches,
                mode=args.importance_mode,
            )
            next_seq = _append_glimpse(
                seq=seq,
                patches=patches,
                viewpoint=vp,
                importance=importance,
            )
            raw_reward = score - prev_score
            scaled_reward = raw_reward * args.reward_scale
            replay.add(
                seq=seq,
                action=action.squeeze(0).detach().cpu().numpy(),
                reward=scaled_reward,
                next_seq=next_seq,
                done=step_idx == args.t - 1,
            )
            seq = next_seq
            prev_score = score
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
