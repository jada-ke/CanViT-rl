"""
canvit_rl/policy.py

Policy network for active-vision glimpse selection.

Interface contract:
    - Input:  recurrent_cls token, shape (batch, cls_dim)
    - Output: raw action, shape (batch, 3) — [cx, cy, scale_raw] all in [-1, 1]
              scale_raw is remapped to (0, 1] in the environment.

This file contains a simple MLP baseline. 
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    Minimal MLP policy: maps recurrent_cls → action.

    Architecture: LayerNorm → Linear → GELU → Linear → Linear → Tanh
    Output is passed through Tanh to keep actions in [-1, 1].

    Args:
        cls_dim:    Dimension of the recurrent_cls observation (default: 768).
        hidden_dim: Width of the hidden layer (default: 256).
        action_dim: Output dimension — 3 for [cx, cy, scale_raw] (default: 3).
    """

    def __init__(
        self,
        cls_dim: int = 768,
        hidden_dim: int = 256,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cls_dim),
            nn.Linear(cls_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Observation tensor, shape (batch, cls_dim).

        Returns:
            Action tensor, shape (batch, action_dim), values in [-1, 1].
        """
        return self.net(obs)