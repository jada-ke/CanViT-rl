"""Critic objectives for candidate-set action selection diagnostics."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

CriticLossMode = Literal[
    "mse",
    "topk_mse",
    "mse_pairwise",
    "mse_listwise",
]


def topk_reward_weights(
    rewards: torch.Tensor,
    *,
    batch_size: int,
    k: int,
    top_frac: float,
    top_weight: float,
) -> torch.Tensor:
    """Return per-candidate MSE weights that emphasize top true rewards."""
    reward_grid = rewards.view(k, batch_size)
    top_count = max(1, int(round(k * top_frac)))
    top_idx = reward_grid.topk(top_count, dim=0).indices
    weights = torch.ones_like(reward_grid)
    weights.scatter_(0, top_idx, float(top_weight))
    return weights.reshape(-1)


def weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted MSE with stable normalization across top-k settings."""
    per_item = (prediction - target).pow(2) * weights
    return per_item.sum() / weights.sum().clamp_min(1e-12)


def pairwise_best_margin_loss(
    prediction: torch.Tensor,
    rewards: torch.Tensor,
    *,
    batch_size: int,
    k: int,
    margin: float,
) -> torch.Tensor:
    """Encourage the predicted best true-reward candidate to beat alternatives."""
    q_grid = prediction.view(k, batch_size).transpose(0, 1)
    reward_grid = rewards.view(k, batch_size).transpose(0, 1)
    best_idx = reward_grid.argmax(dim=1, keepdim=True)
    best_q = q_grid.gather(1, best_idx)
    mask = torch.ones_like(q_grid, dtype=torch.bool)
    mask.scatter_(1, best_idx, False)
    # Problem: plain MSE can fit the average landscape while ignoring tiny
    # top-action gaps. Solution: compare each sample's true best candidate
    # against the rest with a small margin. Result: training directly penalizes
    # Q functions that know the good region but fail to select its best action.
    return F.relu(float(margin) - (best_q - q_grid))[mask].mean()


def listwise_reward_kl_loss(
    prediction: torch.Tensor,
    rewards: torch.Tensor,
    *,
    batch_size: int,
    k: int,
    temperature: float,
) -> torch.Tensor:
    """Match the candidate-set softmax induced by rewards."""
    temp = max(float(temperature), 1e-6)
    q_grid = prediction.view(k, batch_size).transpose(0, 1)
    reward_grid = rewards.view(k, batch_size).transpose(0, 1)
    target_probs = F.softmax(reward_grid / temp, dim=1).detach()
    log_pred_probs = F.log_softmax(q_grid / temp, dim=1)
    return F.kl_div(log_pred_probs, target_probs, reduction="batchmean")


def candidate_critic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    batch_size: int,
    k: int,
    mode: CriticLossMode,
    top_frac: float,
    top_weight: float,
    aux_weight: float,
    pairwise_margin: float,
    listwise_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the selected critic loss plus detached component metrics."""
    mse = F.mse_loss(prediction, target)
    if mode == "mse":
        return mse, {"mse": mse.detach()}
    if mode == "topk_mse":
        weights = topk_reward_weights(
            target,
            batch_size=batch_size,
            k=k,
            top_frac=top_frac,
            top_weight=top_weight,
        )
        topk_mse = weighted_mse_loss(prediction, target, weights)
        return topk_mse, {"mse": mse.detach(), "topk_mse": topk_mse.detach()}
    if mode == "mse_pairwise":
        pairwise = pairwise_best_margin_loss(
            prediction,
            target,
            batch_size=batch_size,
            k=k,
            margin=pairwise_margin,
        )
        loss = mse + float(aux_weight) * pairwise
        return loss, {"mse": mse.detach(), "pairwise": pairwise.detach()}
    if mode == "mse_listwise":
        listwise = listwise_reward_kl_loss(
            prediction,
            target,
            batch_size=batch_size,
            k=k,
            temperature=listwise_temperature,
        )
        loss = mse + float(aux_weight) * listwise
        return loss, {"mse": mse.detach(), "listwise": listwise.detach()}
    raise ValueError(f"Unknown critic loss mode: {mode}")
