"""Compatibility wrapper for ADE20K/synthetic segmentation datasets.

New code should import from `canvit_rl.ade20k.datasets`.
"""

from canvit_rl.ade20k.datasets import (
    DatasetFormat,
    SyntheticSegmentationDataset,
    build_segmentation_dataset,
    infer_dataset_format,
)

__all__ = [
    "DatasetFormat",
    "SyntheticSegmentationDataset",
    "build_segmentation_dataset",
    "infer_dataset_format",
]
