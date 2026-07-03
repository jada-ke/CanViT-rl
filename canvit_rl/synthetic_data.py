"""Shared dataset helpers for ADE20K and generated synthetic segmentation roots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from canvit_specialize.datasets.ade20k import ADE20kDataset
from PIL import Image

from canvit_rl.ade_labels import remap_ade_mask_labels

DatasetFormat = Literal["auto", "ade20k", "synthetic"]


class SyntheticSegmentationDataset(torch.utils.data.Dataset):
    """Split-aware image/mask folder dataset for ADE-embedded synthetic samples."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        *,
        root: Path | None = None,
        split: str = "training",
        image_dir: Path | None = None,
        mask_dir: Path | None = None,
        scene_size_px: int,
        img_transform,
    ) -> None:
        if root is not None:
            split_image_dir = root / "images" / split
            split_mask_dir = root / "masks" / split
            image_dir = image_dir or (
                split_image_dir if split_image_dir.is_dir() else root / "images"
            )
            mask_dir = mask_dir or (
                split_mask_dir if split_mask_dir.is_dir() else root / "masks"
            )
        if image_dir is None or mask_dir is None:
            raise ValueError("SyntheticSegmentationDataset requires image and mask dirs.")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Synthetic image directory not found: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Synthetic mask directory not found: {mask_dir}")

        self.scene_size_px = scene_size_px
        self.img_transform = img_transform
        self.images = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if not self.images:
            raise ValueError(f"No synthetic images found in {image_dir}")
        mask_by_stem = {
            path.stem: path
            for path in mask_dir.iterdir()
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        }
        missing = [path.name for path in self.images if path.stem not in mask_by_stem]
        if missing:
            raise ValueError(
                "Missing synthetic masks with matching stems for: "
                + ", ".join(missing[:10])
            )
        self.masks = [mask_by_stem[path.stem] for path in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        image_tensor = self.img_transform(image)
        resample_nearest = getattr(Image, "Resampling", Image).NEAREST
        mask = mask.resize(
            (self.scene_size_px, self.scene_size_px),
            resample=resample_nearest,
        )
        # Problem: synthetic masks may store raw ADE ids while training expects
        # zero-based CE targets. Solution: normalize ids here so every caller
        # shares the same ADE-compatible mask contract.
        mask_tensor = torch.from_numpy(
            remap_ade_mask_labels(np.asarray(mask)).astype(np.int64)
        )
        return image_tensor, mask_tensor


def infer_dataset_format(
    *,
    root: Path,
    split: str,
    requested: DatasetFormat = "auto",
) -> DatasetFormat:
    """Resolve auto dataset format from the common synthetic folder layouts."""
    if requested != "auto":
        return requested
    split_image_dir = root / "images" / split
    split_mask_dir = root / "masks" / split
    has_split_synthetic = split_image_dir.is_dir() and split_mask_dir.is_dir()
    has_flat_synthetic = (root / "images").is_dir() and (root / "masks").is_dir()
    return "synthetic" if has_split_synthetic or has_flat_synthetic else "ade20k"


def build_segmentation_dataset(
    *,
    root: Path,
    split: str,
    scene_size_px: int,
    img_transform,
    mask_transform,
    dataset_format: DatasetFormat = "auto",
    synthetic_image_dir: Path | None = None,
    synthetic_mask_dir: Path | None = None,
):
    """Build ADE20K or folder-based synthetic segmentation data."""
    resolved_format = infer_dataset_format(
        root=root,
        split=split,
        requested=dataset_format,
    )
    if resolved_format == "ade20k":
        return ADE20kDataset(
            root=root,
            split=split,
            img_transform=img_transform,
            mask_transform=mask_transform,
        )
    return SyntheticSegmentationDataset(
        root=root,
        split=split,
        image_dir=synthetic_image_dir,
        mask_dir=synthetic_mask_dir,
        scene_size_px=scene_size_px,
        img_transform=img_transform,
    )
