"""Actor and critic modules for continuous CanViT SAC policies."""

from __future__ import annotations

import torch
import torch.nn as nn
from canvit_pytorch import VPEEncoder


class CanViTSequenceEncoder(nn.Module):
    """Transformer encoder over previous CanViT patch/coordinate glimpses."""

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
        # architecture with input dim patch_dim + (cx, cy, scale).
        self.max_steps = max_steps
        self.n_patches = n_patches
        self.token_proj = nn.Linear(patch_dim + 3, d_model)
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
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = patches.shape[0]
        coords_expanded = coords[:, :, None, :].expand(-1, -1, self.n_patches, -1)
        tokens = torch.cat([patches, coords_expanded], dim=-1)
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
        z = self.encoder(batch["patches"], batch["coords"], batch["lengths"])
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
    """Q(s, a) critic for a transformer-encoded CanViT sequence state."""

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
        z = self.encoder(batch["patches"], batch["coords"], batch["lengths"])
        return self.q(torch.cat([z, action], dim=-1)).squeeze(-1)


class CanvasStateEncoder(nn.Module):
    """Encode current CanViT canvas plus compact viewpoint history."""

    def __init__(
        self,
        *,
        canvas_feature_dim: int,
        d_model: int,
        rff_dim: int,
        rff_seed: int,
    ) -> None:
        super().__init__()
        self.canvas_feature_dim = canvas_feature_dim
        self.vpe = VPEEncoder(rff_dim=rff_dim, seed=rff_seed)
        self.canvas_stem = nn.Sequential(
            nn.Conv2d(canvas_feature_dim, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.canvas_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.canvas_max_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.canvas_proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(32 * d_model),
            nn.Linear(32 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.out_norm = nn.LayerNorm(d_model + self.vpe.output_dim)

    @property
    def output_dim(self) -> int:
        return self.out_norm.normalized_shape[0]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        canvas = batch["canvas"]
        coords = batch["coords"]
        lengths = batch["lengths"]
        _, seq_len, _ = coords.shape
        canvas_features = self.canvas_stem(canvas.float())
        canvas_pooled = torch.cat(
            [
                self.canvas_avg_pool(canvas_features),
                self.canvas_max_pool(canvas_features),
            ],
            dim=1,
        )
        canvas_z = self.canvas_proj(canvas_pooled)
        vpe = self.vpe(
            y=coords[..., 0].float(),
            x=coords[..., 1].float(),
            s=coords[..., 2].float().clamp_min(1e-6),
        )
        step_ids = torch.arange(seq_len, device=canvas.device)[None, :]
        valid_steps = step_ids < lengths[:, None]
        vpe = vpe * valid_steps[..., None].float()
        history_z = vpe.sum(dim=1) / lengths.clamp_min(1).float()[:, None]
        history_z = history_z * (lengths > 0).float()[:, None]
        return self.out_norm(torch.cat([canvas_z, history_z], dim=-1))


class CanvasStateActor(nn.Module):
    """Tanh-squashed Gaussian actor over current canvas state."""

    def __init__(
        self,
        *,
        canvas_feature_dim: int,
        d_model: int,
        rff_dim: int,
        rff_seed: int,
    ) -> None:
        super().__init__()
        self.encoder = CanvasStateEncoder(
            canvas_feature_dim=canvas_feature_dim,
            d_model=d_model,
            rff_dim=rff_dim,
            rff_seed=rff_seed,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.head(self.encoder(batch)).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def deterministic_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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


class CanvasStateCritic(nn.Module):
    """Q(current-canvas, action) critic for image-dependent SAC."""

    def __init__(
        self,
        *,
        canvas_feature_dim: int,
        d_model: int,
        rff_dim: int,
        rff_seed: int,
    ) -> None:
        super().__init__()
        self.encoder = CanvasStateEncoder(
            canvas_feature_dim=canvas_feature_dim,
            d_model=d_model,
            rff_dim=rff_dim,
            rff_seed=rff_seed,
        )
        self.q = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim + 3),
            nn.Linear(self.encoder.output_dim + 3, d_model),
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
        z = self.encoder(batch)
        return self.q(torch.cat([z, action], dim=-1)).squeeze(-1)
