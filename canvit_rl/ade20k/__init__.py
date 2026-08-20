"""ADE20K-specific datasets, rewards, baselines, and evaluation helpers."""

from canvit_rl.ade20k.datasets import (
    DatasetFormat,
    SyntheticSegmentationDataset,
    build_segmentation_dataset,
    infer_dataset_format,
)
from canvit_rl.ade20k.labels import remap_ade_mask_labels
from canvit_rl.ade20k.rewards import (
    delta_reward,
    reconstruction_reward,
    relative_ce_reduction,
)

__all__ = [
    "DatasetFormat",
    "SyntheticSegmentationDataset",
    "build_segmentation_dataset",
    "delta_reward",
    "infer_dataset_format",
    "reconstruction_reward",
    "relative_ce_reduction",
    "remap_ade_mask_labels",
]
