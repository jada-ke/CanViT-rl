"""Image-dependent Canvas-state actor and critic modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from canvit_pytorch import VPEEncoder


class CanvasStateEncoder(nn.Module):
    """Encode current CanViT canvas plus optional compact viewpoint history."""

    def __init__(
        self,
        *,
        canvas_feature_dim: int,
        d_model: int,
        rff_dim: int,
        rff_seed: int,
        use_entropy_state: bool = False,
        aux_state_channels: int | None = None,
        use_canvas_avg_pool: bool = True,
        use_canvas_max_pool: bool = True,
        use_viewpoint_history: bool = True,
    ) -> None:
        super().__init__()
        if not use_canvas_avg_pool and not use_canvas_max_pool:
            raise ValueError("At least one canvas pooling branch must be enabled.")
        if aux_state_channels is None:
            aux_state_channels = 1 if use_entropy_state else 0
        if aux_state_channels < 0:
            raise ValueError("aux_state_channels must be non-negative.")
        self.canvas_feature_dim = canvas_feature_dim
        self.use_entropy_state = use_entropy_state
        self.aux_state_channels = aux_state_channels
        self.use_canvas_avg_pool = use_canvas_avg_pool
        self.use_canvas_max_pool = use_canvas_max_pool
        self.use_viewpoint_history = use_viewpoint_history
        self.canvas_stem = nn.Sequential(
            nn.Conv2d(canvas_feature_dim, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        if use_canvas_avg_pool:
            self.canvas_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        if use_canvas_max_pool:
            self.canvas_max_pool = nn.AdaptiveMaxPool2d((4, 4))
        canvas_pool_count = int(use_canvas_avg_pool) + int(use_canvas_max_pool)
        self.canvas_proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(16 * canvas_pool_count * d_model),
            nn.Linear(16 * canvas_pool_count * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        if use_viewpoint_history:
            self.vpe = VPEEncoder(rff_dim=rff_dim, seed=rff_seed)
            self.history_gru = nn.GRU(
                input_size=self.vpe.output_dim,
                hidden_size=d_model,
                batch_first=True,
            )
        if aux_state_channels > 0:
            self.entropy_stem = nn.Sequential(
                nn.Conv2d(aux_state_channels, d_model, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.entropy_pool = nn.AdaptiveAvgPool2d((4, 4))
            self.entropy_proj = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(16 * d_model),
                nn.Linear(16 * d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        state_part_count = 1 + int(use_viewpoint_history) + int(aux_state_channels > 0)
        self.out_norm = nn.LayerNorm(state_part_count * d_model)

    @property
    def output_dim(self) -> int:
        return self.out_norm.normalized_shape[0]

    def encode_with_canvas_features(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled state features plus the pre-pool canvas feature map."""
        canvas = batch["canvas"]
        canvas_features = self.canvas_stem(canvas.float())
        # Problem: avg and max pooling emphasize different canvas evidence,
        # but ablations sometimes need one branch removed. Solution: build the
        # pooled tensor from enabled branches only. Result: experiments can
        # disable avg or max pooling without touching the downstream contract.
        canvas_pool_parts = []
        if self.use_canvas_avg_pool:
            canvas_pool_parts.append(self.canvas_avg_pool(canvas_features))
        if self.use_canvas_max_pool:
            canvas_pool_parts.append(self.canvas_max_pool(canvas_features))
        canvas_pooled = torch.cat(canvas_pool_parts, dim=1)
        canvas_z = self.canvas_proj(canvas_pooled)
        state_parts = [canvas_z]
        if self.use_viewpoint_history:
            coords = batch["coords"]
            lengths = batch["lengths"]
            _, seq_len, _ = coords.shape
            # Problem: core-state ablations need the current canvas without an
            # explicit Viewpoint-history embedding. Solution: gate the VPE/GRU
            # branch behind use_viewpoint_history. Result: rollouts can still
            # maintain history for CanViT bookkeeping while actor/critic state
            # can be canvas-only when requested.
            vpe = self.vpe(
                y=coords[..., 0].float(),
                x=coords[..., 1].float(),
                s=coords[..., 2].float().clamp_min(1e-6),
            )
            step_ids = torch.arange(seq_len, device=canvas.device)[None, :]
            valid_steps = step_ids < lengths[:, None]
            vpe = vpe * valid_steps[..., None].float()
            history_seq, _ = self.history_gru(vpe)
            last_step = lengths.clamp_min(1).sub(1).clamp_max(seq_len - 1)
            batch_ids = torch.arange(coords.shape[0], device=coords.device)
            history_z = history_seq[batch_ids, last_step]
            history_z = history_z * (lengths > 0).float()[:, None]
            state_parts.append(history_z)
        if self.aux_state_channels > 0:
            if "entropy" not in batch:
                raise KeyError("CanvasStateEncoder requires batch['entropy'].")
            if batch["entropy"].shape[1] != self.aux_state_channels:
                raise ValueError(
                    "CanvasStateEncoder expected "
                    f"{self.aux_state_channels} aux channel(s), got "
                    f"{batch['entropy'].shape[1]}."
                )
            entropy_features = self.entropy_stem(batch["entropy"].float())
            entropy_z = self.entropy_proj(self.entropy_pool(entropy_features))
            state_parts.append(entropy_z)
        return self.out_norm(torch.cat(state_parts, dim=-1)), canvas_features

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_with_canvas_features(batch)[0]


class CanvasStateActor(nn.Module):
    """Tanh-squashed Gaussian actor over current canvas state."""

    def __init__(
        self,
        *,
        canvas_feature_dim: int,
        d_model: int,
        rff_dim: int,
        rff_seed: int,
        use_entropy_state: bool = False,
        aux_state_channels: int | None = None,
        use_canvas_avg_pool: bool = True,
        use_canvas_max_pool: bool = True,
        use_viewpoint_history: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = CanvasStateEncoder(
            canvas_feature_dim=canvas_feature_dim,
            d_model=d_model,
            rff_dim=rff_dim,
            rff_seed=rff_seed,
            use_entropy_state=use_entropy_state,
            aux_state_channels=aux_state_channels,
            use_canvas_avg_pool=use_canvas_avg_pool,
            use_canvas_max_pool=use_canvas_max_pool,
            use_viewpoint_history=use_viewpoint_history,
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
        use_action_location_features: bool = False,
        use_entropy_state: bool = False,
        aux_state_channels: int | None = None,
        use_canvas_avg_pool: bool = True,
        use_canvas_max_pool: bool = True,
        use_viewpoint_history: bool = True,
    ) -> None:
        super().__init__()
        self.use_action_location_features = use_action_location_features
        self.encoder = CanvasStateEncoder(
            canvas_feature_dim=canvas_feature_dim,
            d_model=d_model,
            rff_dim=rff_dim,
            rff_seed=rff_seed,
            use_entropy_state=use_entropy_state,
            aux_state_channels=aux_state_channels,
            use_canvas_avg_pool=use_canvas_avg_pool,
            use_canvas_max_pool=use_canvas_max_pool,
            use_viewpoint_history=use_viewpoint_history,
        )
        q_input_dim = self.encoder.output_dim + 3
        if use_action_location_features:
            q_input_dim += d_model
        self.q = nn.Sequential(
            nn.LayerNorm(q_input_dim),
            nn.Linear(q_input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def sample_action_location_features(
        canvas_features: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Sample pre-pool canvas features at the action center."""
        # Problem: the pooled canvas summary discards exact spatial alignment.
        # Solution: treat action[..., :2] as (y, x), flip to grid_sample's
        # (x, y) convention, and read the critic feature map at that location.
        # Result: Q receives features from the region the candidate Viewpoint
        # actually targets, while still preserving the global state summary.
        grid = action[..., :2].flip(-1).to(dtype=canvas_features.dtype)
        grid = grid[:, None, None, :]
        return F.grid_sample(
            canvas_features,
            grid,
            align_corners=False,
        ).squeeze(-1).squeeze(-1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_action_location_features:
            z, canvas_features = self.encoder.encode_with_canvas_features(batch)
            local_z = self.sample_action_location_features(canvas_features, action)
            q_input = torch.cat([z, local_z, action], dim=-1)
        else:
            z = self.encoder(batch)
            q_input = torch.cat([z, action], dim=-1)
        return self.q(q_input).squeeze(-1)
