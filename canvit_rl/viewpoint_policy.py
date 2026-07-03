"""Image-independent viewpoint-history actors for SAC/BC experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from canvit_pytorch import VPEEncoder, Viewpoint


def viewpoint_to_action(
    viewpoint: Viewpoint,
    *,
    min_scale: float,
) -> torch.Tensor:
    """Map a Viewpoint's center/scale tensors into the actor action range."""
    centers = viewpoint.centers.float()
    scales = viewpoint.scales.float()
    scale_raw = (scales - min_scale) / (1.0 - min_scale)
    scale_raw = scale_raw.clamp(0.0, 1.0).mul(2.0).sub(1.0)
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
