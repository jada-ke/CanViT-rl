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
    """Step-relative normalized-space reduction: (L[t-1] - L[t]) / L[t-1]."""
    return (before.loss_norm - after.loss_norm) / before.loss_norm.clamp_min(eps)


def dense_loss_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
) -> Tensor:
    """Plain normalized-space loss delta: L[t-1] - L[t]."""
    return before.loss_norm - after.loss_norm


def dense_loss_log_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Normalized-space log improvement: log(L[t-1] + eps) - log(L[t] + eps).

    Problem: plain deltas can be noisy while direct ratios can spike on small
    denominators. Solution: reward proportional loss improvement in log space.
    Result: rewards remain scale-normalized and telescope across multi-step
    episodes without needing an external l0 tensor.
    """
    return before.loss_norm.clamp_min(eps).log() - after.loss_norm.clamp_min(eps).log()


def dense_loss_log_delta_clipped_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
    clip: float = 1.0,
) -> Tensor:
    """Clipped normalized-space log improvement."""
    reward = dense_loss_log_delta_reward(before, after, eps=eps)
    return reward.clamp(min=-clip, max=clip)


def dense_loss_log_delta_tanh_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
    scale: float = 1.0,
) -> Tensor:
    """Tanh-bounded normalized-space log improvement."""
    reward = dense_loss_log_delta_reward(before, after, eps=eps)
    return (scale * reward).tanh()


def dense_raw_mse_reduction_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Step-relative raw-space reduction: (L[t-1] - L[t]) / L[t-1]."""
    return (before.loss_raw - after.loss_raw) / before.loss_raw.clamp_min(eps)


def dense_raw_mse_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
) -> Tensor:
    """Plain raw-space loss delta: L[t-1] - L[t]."""
    return before.loss_raw - after.loss_raw


def dense_raw_mse_log_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Raw-space log improvement: log(L[t-1] + eps) - log(L[t] + eps)."""
    return before.loss_raw.clamp_min(eps).log() - after.loss_raw.clamp_min(eps).log()


def dense_raw_mse_log_delta_clipped_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
    clip: float = 1.0,
) -> Tensor:
    """Clipped raw-space log improvement."""
    reward = dense_raw_mse_log_delta_reward(before, after, eps=eps)
    return reward.clamp(min=-clip, max=clip)


def dense_raw_mse_log_delta_tanh_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    *,
    eps: float = 1e-6,
    scale: float = 1.0,
) -> Tensor:
    """Tanh-bounded raw-space log improvement."""
    reward = dense_raw_mse_log_delta_reward(before, after, eps=eps)
    return (scale * reward).tanh()


def dense_raw_mse_l0_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    l0: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Potential-based raw-space loss delta normalized by episode reset loss.

    Problem: dividing raw MSE deltas by each step's own ``before`` loss makes
    multi-step rewards path-dependent. Solution: hold the t=0 reset loss fixed
    as ``l0`` for the whole episode. Result: per-sample rewards stay difficulty
    normalized while preserving the telescoping potential-difference form.
    """
    return (before.loss_raw - after.loss_raw) / l0.clamp_min(eps)


def dense_raw_mse_clipped_l0_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    l0: Tensor,
    *,
    eps: float = 1e-6,
    clip: float = 1.0,
) -> Tensor:
    """Hard-clipped raw-space l0-normalized delta."""
    reward = dense_raw_mse_l0_delta_reward(before, after, l0, eps=eps)
    return reward.clamp(min=-clip, max=clip)


def dense_raw_mse_tanh_l0_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    l0: Tensor,
    *,
    eps: float = 1e-6,
    scale: float = 1.0,
) -> Tensor:
    """Bounded raw-space l0-normalized delta via tanh(scale * delta / l0)."""
    reward = dense_raw_mse_l0_delta_reward(before, after, l0, eps=eps)
    return (scale * reward).tanh()


def dense_loss_l0_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    l0: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Potential-based normalized-space loss delta scaled by reset loss.

    Problem: l0-normalized shaping is useful in standardized DINOv3 space too,
    but mixing a normalized loss delta with raw-space l0 would silently distort
    reward scale. Solution: require callers to pass the episode t=0
    ``loss_norm`` reference for this mode. Result: normalized-space rewards get
    the same policy-invariant telescoping behavior as raw-space l0 deltas.
    """
    return (before.loss_norm - after.loss_norm) / l0.clamp_min(eps)


def dense_loss_clipped_l0_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    l0: Tensor,
    *,
    eps: float = 1e-6,
    clip: float = 1.0,
) -> Tensor:
    """Hard-clipped normalized-space l0-normalized delta."""
    reward = dense_loss_l0_delta_reward(before, after, l0, eps=eps)
    return reward.clamp(min=-clip, max=clip)


def dense_loss_tanh_l0_delta_reward(
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    l0: Tensor,
    *,
    eps: float = 1e-6,
    scale: float = 1.0,
) -> Tensor:
    """Bounded normalized-space l0 delta via tanh(scale * delta / l0)."""
    reward = dense_loss_l0_delta_reward(before, after, l0, eps=eps)
    return (scale * reward).tanh()


def dense_reward(
    *,
    mode: str,
    before: DenseDistillationMetrics,
    after: DenseDistillationMetrics,
    l0: Tensor | None = None,
    eps: float = 1e-6,
    log_clip: float = 1.0,
    l0_clip: float = 1.0,
    tanh_scale: float = 1.0,
) -> Tensor:
    """Dispatch the configured dense SAC reward mode."""
    if mode == "raw_mse_delta":
        return dense_raw_mse_delta_reward(before, after)
    if mode == "norm_loss_delta":
        return dense_loss_delta_reward(before, after)
    if mode == "raw_mse_log_delta":
        return dense_raw_mse_log_delta_reward(before, after, eps=eps)
    if mode == "norm_loss_log_delta":
        return dense_loss_log_delta_reward(before, after, eps=eps)
    if mode == "raw_mse_log_delta_clipped":
        return dense_raw_mse_log_delta_clipped_reward(
            before,
            after,
            eps=eps,
            clip=log_clip,
        )
    if mode == "norm_loss_log_delta_clipped":
        return dense_loss_log_delta_clipped_reward(
            before,
            after,
            eps=eps,
            clip=log_clip,
        )
    if mode == "raw_mse_log_delta_tanh":
        return dense_raw_mse_log_delta_tanh_reward(
            before,
            after,
            eps=eps,
            scale=tanh_scale,
        )
    if mode == "norm_loss_log_delta_tanh":
        return dense_loss_log_delta_tanh_reward(
            before,
            after,
            eps=eps,
            scale=tanh_scale,
        )
    if mode == "raw_mse_reduction":
        return dense_raw_mse_reduction_reward(before, after, eps=eps)
    if mode == "norm_loss_reduction":
        return dense_loss_reduction_reward(before, after, eps=eps)
    if mode == "raw_mse_l0_delta":
        if l0 is None:
            raise ValueError(
                "raw_mse_l0_delta requires l0 (episode t=0 reference loss)."
            )
        return dense_raw_mse_l0_delta_reward(before, after, l0, eps=eps)
    if mode == "raw_mse_clipped_l0_delta":
        if l0 is None:
            raise ValueError(
                "raw_mse_clipped_l0_delta requires l0 (episode t=0 reference loss)."
            )
        return dense_raw_mse_clipped_l0_delta_reward(
            before,
            after,
            l0,
            eps=eps,
            clip=l0_clip,
        )
    if mode == "raw_mse_tanh_l0_delta":
        if l0 is None:
            raise ValueError(
                "raw_mse_tanh_l0_delta requires l0 (episode t=0 reference loss)."
            )
        return dense_raw_mse_tanh_l0_delta_reward(
            before,
            after,
            l0,
            eps=eps,
            scale=tanh_scale,
        )
    if mode == "norm_loss_l0_delta":
        if l0 is None:
            raise ValueError(
                "norm_loss_l0_delta requires l0 (episode t=0 reference loss)."
            )
        return dense_loss_l0_delta_reward(before, after, l0, eps=eps)
    if mode == "norm_loss_clipped_l0_delta":
        if l0 is None:
            raise ValueError(
                "norm_loss_clipped_l0_delta requires l0 (episode t=0 reference loss)."
            )
        return dense_loss_clipped_l0_delta_reward(
            before,
            after,
            l0,
            eps=eps,
            clip=l0_clip,
        )
    if mode == "norm_loss_tanh_l0_delta":
        if l0 is None:
            raise ValueError(
                "norm_loss_tanh_l0_delta requires l0 (episode t=0 reference loss)."
            )
        return dense_loss_tanh_l0_delta_reward(
            before,
            after,
            l0,
            eps=eps,
            scale=tanh_scale,
        )
    raise ValueError(f"Unsupported dense reward mode: {mode}")
