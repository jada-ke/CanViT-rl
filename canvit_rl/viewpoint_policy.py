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


class ViewpointHistoryEncoder(nn.Module):
    """MLP encoder over previous viewpoint positions and looked-at timesteps."""

    def __init__(
        self,
        *,
        d_model: int,
        max_steps: int,
        state_mode: str,
        rff_dim: int,
        rff_seed: int,
    ) -> None:
        super().__init__()
        if state_mode not in {"vpe", "vpe_timestep"}:
            raise ValueError("state_mode must be 'vpe' or 'vpe_timestep'.")
        self.max_steps = max_steps
        self.state_mode = state_mode
        self.vpe = VPEEncoder(rff_dim=rff_dim, seed=rff_seed)
        step_dim = 1 if state_mode == "vpe_timestep" else 0
        slot_dim = self.vpe.output_dim + step_dim
        self.net = nn.Sequential(
            nn.LayerNorm(max_steps * slot_dim),
            nn.Linear(max_steps * slot_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, coords: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
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
        vpe = vpe * valid_steps[..., None].float()
        if self.state_mode == "vpe_timestep":
            denom = max(self.max_steps - 1, 1)
            looked_at_t = step_ids.to(coords.dtype) / float(denom)
            time_feature = looked_at_t.expand(batch_size, -1)[..., None]
            time_feature = time_feature * valid_steps[..., None].float()
            vpe = torch.cat([vpe, time_feature], dim=-1)
        return self.net(vpe.reshape(batch_size, -1))


class ViewpointGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor over image-independent viewpoint history."""

    def __init__(self, encoder: ViewpointHistoryEncoder, d_model: int) -> None:
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
        z = self.encoder(batch["coords"], batch["lengths"])
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
