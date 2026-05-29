"""
canvit_rl/reward.py

Stateless reward functions for CanViT active-vision episodes.

Each function takes tensors in and returns a scalar float out.
No side effects, no model state.

Current rewards:
    - reconstruction_reward: cosine similarity between canvas CLS and teacher CLS
    - delta_reward: gain in reward signal vs previous step (wrapper)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_reward(
    canvas_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
) -> float:
    """
    Cosine similarity between the canvas CLS token and the frozen teacher CLS.

    Args:
        canvas_cls:  Canvas CLS token from CanViT, shape [1, 768] or [768].
        teacher_cls: Teacher CLS token from DINOv3, shape [1, 768] or [768].

    Returns:
        Scalar cosine similarity in [-1, 1]. Higher is better.
    """
    canvas_cls = canvas_cls.float().reshape(1, -1)
    teacher_cls = teacher_cls.float().reshape(1, -1)
    return float(F.cosine_similarity(canvas_cls, teacher_cls, dim=-1).item())


def delta_reward(current_sim: float, previous_sim: float) -> float:
    """
    Reward as gain in cosine similarity vs the previous step.

    Encourages the policy to keep improving the representation
    rather than reaching a plateau.

    Args:
        current_sim:  Cosine similarity after the current glimpse.
        previous_sim: Cosine similarity after the previous glimpse (0.0 at episode start).

    Returns:
        Scalar reward (positive if improved, negative if regressed).
    """
    return current_sim - previous_sim