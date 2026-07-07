"""Image-independent viewpoint-history actors for SAC/BC experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from canvit_pytorch import VPEEncoder, Viewpoint


def _action_scale_from_viewpoint_scale(
    scale: torch.Tensor,
    *,
    min_scale: float,
) -> torch.Tensor:
    """Map real Viewpoint scales into the actor's tanh action coordinate."""
    return 2.0 * (scale - min_scale) / (1.0 - min_scale) - 1.0


def viewpoint_to_action(
    viewpoint: Viewpoint,
    *,
    min_scale: float,
) -> torch.Tensor:
    """Map a Viewpoint's center/scale tensors into the actor action range."""
    centers = viewpoint.centers.float()
    scales = viewpoint.scales.float()
    scale_raw = _action_scale_from_viewpoint_scale(scales, min_scale=min_scale)
    scale_raw = scale_raw.clamp(-1.0, 1.0)
    return torch.cat([centers.float(), scale_raw[:, None].float()], dim=-1)


def action_to_viewpoint(
    action: torch.Tensor,
    *,
    min_scale: float,
) -> Viewpoint:
    """Map tanh actor actions into CanViT's Viewpoint interface."""
    centers = action[:, :2].float()
    scales = ((action[:, 2] + 1.0) * 0.5 * (1.0 - min_scale) + min_scale).float()
    return Viewpoint(centers=centers, scales=scales)


def randomize_actor_mean_viewpoint_prior(
    actor: nn.Module,
    *,
    min_scale: float,
    center_radius: float,
) -> dict[str, float]:
    """Initialize deterministic actor output to a random near-center Viewpoint.

    The actor emits tanh-squashed actions, so we set the pre-tanh mean bias to
    atanh(target_action). This leaves the log-std rows alone while making the
    initial deterministic policy a controlled random prior instead of the
    default zero-action center/mid-scale prior.
    """
    if not 0.0 <= center_radius < 1.0:
        raise ValueError("--actor-init-center-radius must be in [0, 1).")
    if not 0.0 < min_scale < 1.0:
        raise ValueError("--min-scale must be in (0, 1) for actor init.")
    head = getattr(actor, "head", None)
    if not isinstance(head, nn.Sequential) or not isinstance(head[-1], nn.Linear):
        raise TypeError("actor must expose a Sequential head ending in nn.Linear.")
    output = head[-1]
    if output.out_features < 6:
        raise ValueError("actor head must output mean/log_std with at least 6 values.")

    device = output.weight.device
    center = torch.empty(2, device=device).uniform_(-center_radius, center_radius)
    scale = torch.empty(1, device=device).uniform_(min_scale, 1.0)
    scale_action = _action_scale_from_viewpoint_scale(scale, min_scale=min_scale)
    action = torch.cat([center, scale_action]).clamp(-0.999, 0.999)
    mean_bias = torch.atanh(action)

    with torch.no_grad():
        # Problem: default Linear initialization makes deterministic SAC start
        # at action ~= 0, i.e. centered viewpoint with midpoint scale.
        # Solution: zero only the mean rows and set their bias to a sampled
        # near-center Viewpoint prior in pre-tanh coordinates.
        # Result: log-std exploration remains untouched while update-0 eval can
        # test whether SAC learns from a weaker randomized initial prior.
        output.weight[:3].zero_()
        output.bias[:3].copy_(mean_bias)

    return {
        "center_y": float(center[0].detach().cpu().item()),
        "center_x": float(center[1].detach().cpu().item()),
        "scale": float(scale[0].detach().cpu().item()),
    }


# class ViewpointHistoryEncoder(nn.Module):
#     """Previous encoder kept here temporarily for comparison.
#
#     It projected flattened VPE history into a separate latent vector before
#     the actor head, and optionally appended timestep features. The current
#     actor below removes that middle encoder and feeds flattened VPE directly
#     to the actor head.
#     """
# def __init__(
#         self,
#         *,
#         d_model: int,
#         max_steps: int,
#         state_mode: str,
#         rff_dim: int,
#         rff_seed: int,
#     ) -> None:
#         super().__init__()
#         if state_mode not in {"vpe", "vpe_timestep"}:
#             raise ValueError("state_mode must be 'vpe' or 'vpe_timestep'.")
#         self.max_steps = max_steps
#         self.state_mode = state_mode
#         self.vpe = VPEEncoder(rff_dim=rff_dim, seed=rff_seed)
#         step_dim = 1 if state_mode == "vpe_timestep" else 0
#         slot_dim = self.vpe.output_dim + step_dim
#         self.net = nn.Sequential(
#             nn.LayerNorm(max_steps * slot_dim),
#             nn.Linear(max_steps * slot_dim, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.LayerNorm(d_model),
#         )


class ViewpointGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor over VPE-encoded viewpoint history."""

    def __init__(
        self,
        *,
        d_model: int,
        max_steps: int,
        rff_dim: int,
        rff_seed: int,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.vpe = VPEEncoder(rff_dim=rff_dim, seed=rff_seed)
        slot_dim = self.vpe.output_dim
        # Fixed by Codex on 2026-06-13
        # Problem: the image-independent actor still had a separate history MLP,
        # adding an unnecessary abstraction between VPE history and policy head.
        # Solution: encode each viewed slot with CanViT VPE, mask empty slots,
        # flatten the fixed history, and let the actor head predict the action.
        # Result: the policy matches the intended History -> VPE -> flatten ->
        # ActorHead path without a separate ViewpointHistoryEncoder module.
        self.head = nn.Sequential(
            nn.LayerNorm(max_steps * slot_dim),
            nn.Linear(max_steps * slot_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = batch["coords"]
        lengths = batch["lengths"]
        batch_size, seq_len, _ = coords.shape
        if seq_len > self.max_steps:
            raise ValueError(f"seq_len={seq_len} exceeds max_steps={self.max_steps}.")
        vpe = self.vpe(
            y=coords[..., 0].float(),
            x=coords[..., 1].float(),
            s=coords[..., 2].float().clamp_min(1e-6),
        )
        step_ids = torch.arange(seq_len, device=coords.device)[None, :]
        valid_steps = step_ids < lengths[:, None]
        z = (vpe * valid_steps[..., None].float()).reshape(batch_size, -1)
        mean, log_std = self.head(z).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def deterministic_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return tanh(mean), matching deterministic SAC evaluation."""
        mean, _ = self(batch)
        return torch.tanh(mean)

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


class ViewpointHistoryCritic(nn.Module):
    """Q(history, action) critic over image-independent VPE viewpoint history."""

    def __init__(
        self,
        *,
        d_model: int,
        max_steps: int,
        rff_dim: int,
        rff_seed: int,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.vpe = VPEEncoder(rff_dim=rff_dim, seed=rff_seed)
        slot_dim = self.vpe.output_dim
        self.q = nn.Sequential(
            nn.LayerNorm(max_steps * slot_dim + 3),
            nn.Linear(max_steps * slot_dim + 3, d_model),
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
        coords = batch["coords"]
        lengths = batch["lengths"]
        batch_size, seq_len, _ = coords.shape
        if seq_len > self.max_steps:
            raise ValueError(f"seq_len={seq_len} exceeds max_steps={self.max_steps}.")
        vpe = self.vpe(
            y=coords[..., 0].float(),
            x=coords[..., 1].float(),
            s=coords[..., 2].float().clamp_min(1e-6),
        )
        step_ids = torch.arange(seq_len, device=coords.device)[None, :]
        valid_steps = step_ids < lengths[:, None]
        z = (vpe * valid_steps[..., None].float()).reshape(batch_size, -1)
        return self.q(torch.cat([z, action], dim=-1)).squeeze(-1)
