"""Dense distillation reward helpers for IN21k policy pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from canvit_rl.pretrain_IN21k.dense_train_batch import DenseTrainBatch


class TensorDestandardizer(Protocol):
    """Destandardizer contract exposed by CanViT pretraining standardizers."""

    def __call__(self, values: Tensor) -> Tensor:
        """Map standardized predictions back to raw teacher-feature space."""


@dataclass(frozen=True)
class DenseDistillationMetrics:
    """Per-sample dense distillation metrics from the current recurrent state."""

    scene_loss_norm: Tensor
    cls_loss_norm: Tensor
    loss_norm: Tensor
    scene_loss_raw: Tensor
    cls_loss_raw: Tensor
    loss_raw: Tensor


def _validate_weights(*, scene_weight: float, cls_weight: float) -> float:
    """Return positive total reward weight or fail before rollout starts."""
    total_weight = scene_weight + cls_weight
    if total_weight <= 0.0:
        raise ValueError("At least one dense distillation reward weight must be positive.")
    return total_weight


def dense_distillation_metrics(
    *,
    model,
    state,
    batch: DenseTrainBatch,
    scene_denorm: TensorDestandardizer,
    cls_denorm: TensorDestandardizer,
    scene_weight: float,
    cls_weight: float,
) -> DenseDistillationMetrics:
    """Return per-sample loss and raw-cosine metrics for dense distillation."""
    total_weight = _validate_weights(
        scene_weight=scene_weight,
        cls_weight=cls_weight,
    )
    scene_pred = model.predict_teacher_scene(state.canvas)
    cls_pred = model.predict_scene_teacher_cls(state.recurrent_cls)
    scene_loss_norm = (scene_pred.float() - batch.scene_target.float()).pow(2).mean(
        dim=(1, 2)
    )
    cls_loss_norm = (cls_pred.float() - batch.cls_target.float()).pow(2).mean(dim=1)
    # Problem: SAC replay needs one reward per image, but CanViT-pretrain's
    # distillation_loss_fn returns a batch-averaged scalar. Solution: mirror
    # that MSE objective with reduction="none" and reduce only within each
    # sample. Result: each action gets marginal credit for the loss reduction
    # caused by its own selected glimpse.
    loss_norm = (
        scene_weight * scene_loss_norm + cls_weight * cls_loss_norm
    ) / total_weight
    scene_pred_raw = scene_denorm(scene_pred)
    cls_pred_raw = cls_denorm(cls_pred.unsqueeze(1)).squeeze(1)
    # Problem: normalized MSE matches CanViT-pretrain optimization, but
    # position-wise z-scoring can over-weight low-variance teacher dimensions
    # compared with their original feature-scale informativeness. Solution:
    # also compute per-sample MSE in raw DINOv3 feature space. Result:
    # raw_mse_reduction can keep CE-style marginal improvement while avoiding
    # reward shaping from the standardizer's variance weights.
    scene_loss_raw = (
        scene_pred_raw.float() - batch.raw_scene_target.float()
    ).pow(2).mean(dim=(1, 2))
    cls_loss_raw = (cls_pred_raw.float() - batch.raw_cls_target.float()).pow(2).mean(
        dim=1
    )
    loss_raw = (scene_weight * scene_loss_raw + cls_weight * cls_loss_raw) / total_weight
    return DenseDistillationMetrics(
        scene_loss_norm=scene_loss_norm,
        cls_loss_norm=cls_loss_norm,
        loss_norm=loss_norm,
        scene_loss_raw=scene_loss_raw,
        cls_loss_raw=cls_loss_raw,
        loss_raw=loss_raw,
    )


def dense_loss_reduction_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """CE-style reward: relative reduction in normalized distillation loss."""
    return (before.loss_norm - after.loss_norm) / before.loss_norm.clamp_min(eps)


def dense_raw_mse_reduction_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """CE-style reward: relative reduction in raw-space distillation MSE."""
    return (before.loss_raw - after.loss_raw) / before.loss_raw.clamp_min(eps)


def dense_reward(
    *,
    mode: str,
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    eps: float = 1e-6,
) -> Tensor:
    """Dispatch the configured dense SAC reward mode."""
    if mode == "raw_mse_reduction":
        return dense_raw_mse_reduction_reward(before, after, eps=eps)
    if mode == "norm_loss_reduction":
        return dense_loss_reduction_reward(before, after, eps=eps)
    raise ValueError(f"Unsupported dense reward mode: {mode}")
