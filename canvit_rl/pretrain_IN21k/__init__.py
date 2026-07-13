"""Helpers for IN21k dense-feature policy pretraining."""

from canvit_rl.pretrain_IN21k.dense_train_batch import (
    DenseTrainBatch,
    FixedDenseSubsetLoader,
    apply_dense_feature_config,
    init_normalizer_stats_from_shard,
    load_dense_train_batch,
    validate_dense_feature_source,
)
from canvit_rl.pretrain_IN21k.pretrain_modules import (
    install_pretrain_train_shim,
    load_pretrain_modules,
)
from canvit_rl.pretrain_IN21k.reward import (
    DenseDistillationMetrics,
    dense_distillation_metrics,
    dense_loss_reduction_reward,
    dense_raw_mse_reduction_reward,
    dense_reward,
)

__all__ = [
    "DenseDistillationMetrics",
    "DenseTrainBatch",
    "FixedDenseSubsetLoader",
    "apply_dense_feature_config",
    "dense_distillation_metrics",
    "dense_loss_reduction_reward",
    "dense_raw_mse_reduction_reward",
    "dense_reward",
    "init_normalizer_stats_from_shard",
    "install_pretrain_train_shim",
    "load_dense_train_batch",
    "load_pretrain_modules",
    "validate_dense_feature_source",
]
