"""Generate padded-scene synthetic segmentation data from random ADE20K images.

Example:
    uv run python scripts/generate_synthetic_dataset.py \
        --ade-root datasets/ADE20k \
        --root synthetic_segmentation \
        --train-samples 7 \
        --val-samples 3 \
        --min-fraction 0.25 \
        --max-fraction 0.65 \
        --source-zoom-min 1.0 \
        --source-zoom-max 3.0

The output layout is:

    synthetic_segmentation/
    ├── images/
    │   ├── training/
    │   │   └── sample_00000.png
    │   └── validation/
    │       └── sample_00000.png
    ├── masks/
    │   ├── training/
    │   │   └── sample_00000.png
    │   └── validation/
    │       └── sample_00000.png
    ├── metadata_training.csv
    └── metadata_validation.csv

Masks preserve ADE20K class ids in the embedded region and use 255 outside the
embedded region so segmentation loss ignores the padded area.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from canvit_rl.ade_labels import remap_ade_mask_labels


IGNORE_LABEL = 255
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_pairs(*, ade_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Return matched ADE image/mask paths for a split."""
    image_dir = ade_root / "images" / split
    mask_dir = ade_root / "annotations" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"ADE image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"ADE annotation directory not found: {mask_dir}")
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    if not pairs:
        raise ValueError(f"No matched ADE image/mask pairs found under {ade_root}")
    return pairs


def _random_background(*, width: int, height: int) -> Image.Image:
    """Create a simple RGB padded scene background."""
    base_color = np.random.randint(30, 210, size=(1, 1, 3), dtype=np.uint8)
    noise = np.random.randint(0, 35, size=(height, width, 3), dtype=np.uint8)
    background = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(background, mode="RGB")


def _resize_to_fraction(
    *,
    image: Image.Image,
    mask: Image.Image,
    canvas_width: int,
    canvas_height: int,
    min_fraction: float,
    max_fraction: float,
) -> tuple[Image.Image, Image.Image, float, float]:
    """Resize source image/mask so the longest side occupies a random fraction."""
    source_width, source_height = image.size
    requested_fraction = random.uniform(min_fraction, max_fraction)
    canvas_side = min(canvas_width, canvas_height)
    target_long_side = max(1, int(round(canvas_side * requested_fraction)))
    target_fraction = target_long_side / canvas_side
    scale = target_long_side / max(source_width, source_height)
    target_width = max(1, min(canvas_width, int(round(source_width * scale))))
    target_height = max(1, min(canvas_height, int(round(source_height * scale))))
    image_resample = getattr(Image, "Resampling", Image).BICUBIC
    mask_resample = getattr(Image, "Resampling", Image).NEAREST
    return (
        image.resize((target_width, target_height), resample=image_resample),
        mask.resize((target_width, target_height), resample=mask_resample),
        target_fraction,
        requested_fraction,
    )


def _zoom_source(
    *,
    image: Image.Image,
    mask: Image.Image,
    source_zoom: float,
    zoom_position: str,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Crop into the source ADE image/mask before canvas embedding."""
    if source_zoom <= 1.0:
        return image, mask, (0, 0, image.size[0], image.size[1])
    source_width, source_height = image.size
    crop_width = max(1, int(round(source_width / source_zoom)))
    crop_height = max(1, int(round(source_height / source_zoom)))
    max_x = source_width - crop_width
    max_y = source_height - crop_height
    if zoom_position == "center":
        x0 = max_x // 2
        y0 = max_y // 2
    else:
        x0 = random.randint(0, max(max_x, 0))
        y0 = random.randint(0, max(max_y, 0))
    box = (x0, y0, x0 + crop_width, y0 + crop_height)
    return image.crop(box), mask.crop(box), box


def generate_embedded_sample(
    *,
    image_path: Path,
    mask_path: Path,
    width: int,
    height: int,
    min_fraction: float,
    max_fraction: float,
    source_zoom_min: float,
    source_zoom_max: float,
    source_zoom_position: str,
) -> tuple[Image.Image, Image.Image, dict]:
    """Embed one ADE image/mask pair into a larger padded scene."""
    source_image = Image.open(image_path).convert("RGB")
    source_mask = Image.fromarray(
        remap_ade_mask_labels(
            np.asarray(Image.open(mask_path).convert("L")),
            raw_ade=True,
        ).astype(np.uint8),
        mode="L",
    )
    source_zoom = random.uniform(source_zoom_min, source_zoom_max)
    source_image, source_mask, source_box = _zoom_source(
        image=source_image,
        mask=source_mask,
        source_zoom=source_zoom,
        zoom_position=source_zoom_position,
    )
    (
        embedded_image,
        embedded_mask,
        crop_fraction,
        requested_crop_fraction,
    ) = _resize_to_fraction(
        image=source_image,
        mask=source_mask,
        canvas_width=width,
        canvas_height=height,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
    )
    canvas = _random_background(width=width, height=height)
    mask_canvas = Image.new("L", (width, height), IGNORE_LABEL)
    max_x = width - embedded_image.size[0]
    max_y = height - embedded_image.size[1]
    x0 = random.randint(0, max(max_x, 0))
    y0 = random.randint(0, max(max_y, 0))
    canvas.paste(embedded_image, (x0, y0))
    mask_canvas.paste(embedded_mask, (x0, y0))
    metadata = {
        "crop_fraction": crop_fraction,
        "requested_crop_fraction": requested_crop_fraction,
        "source_zoom": source_zoom,
        "source_crop_x0": source_box[0],
        "source_crop_y0": source_box[1],
        "source_crop_x1": source_box[2],
        "source_crop_y1": source_box[3],
        "embed_x0": x0,
        "embed_y0": y0,
        "embed_x1": x0 + embedded_image.size[0],
        "embed_y1": y0 + embedded_image.size[1],
    }
    return canvas, mask_canvas, metadata


def _sample_pairs(
    *,
    pairs: list[tuple[Path, Path]],
    num_samples: int,
) -> list[tuple[Path, Path]]:
    """Sample source pairs, with replacement when more samples are requested."""
    if num_samples <= len(pairs):
        return random.sample(pairs, num_samples)
    return [random.choice(pairs) for _ in range(num_samples)]


def _generate_split(
    *,
    args: argparse.Namespace,
    split: str,
    num_samples: int,
) -> None:
    """Generate one synthetic split under images/<split> and masks/<split>."""
    pairs = _load_pairs(ade_root=args.ade_root, split=split)
    image_dir = args.root / "images" / split
    mask_dir = args.root / "masks" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    selected_pairs = _sample_pairs(pairs=pairs, num_samples=num_samples)

    for idx, (image_path, mask_path) in enumerate(selected_pairs):
        image, mask, metadata = generate_embedded_sample(
            image_path=image_path,
            mask_path=mask_path,
            width=args.width,
            height=args.height,
            min_fraction=args.min_fraction,
            max_fraction=args.max_fraction,
            source_zoom_min=args.source_zoom_min,
            source_zoom_max=args.source_zoom_max,
            source_zoom_position=args.source_zoom_position,
        )
        stem = f"sample_{idx:05d}_{image_path.stem}.png"
        image.save(image_dir / stem)
        mask.save(mask_dir / stem)
        metadata_rows.append(
            {
                "sample": stem,
                "split": split,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "source_zoom_position": args.source_zoom_position,
                **metadata,
            }
        )

    metadata_path = args.root / f"metadata_{split}.csv"
    with metadata_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Saved {num_samples} {split} samples")
    print(f"Images: {image_dir}")
    print(f"Masks:  {mask_dir}")
    print(f"Metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade-root", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument("--root", type=Path, default=Path("synthetic_segmentation"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Generate this many samples under images/training and masks/training.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Generate this many samples under images/validation and masks/validation.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--min-fraction", type=float, default=0.25)
    parser.add_argument("--max-fraction", type=float, default=0.65)
    parser.add_argument(
        "--source-zoom-min",
        type=float,
        default=1.0,
        help="Minimum crop zoom applied to ADE source before embedding.",
    )
    parser.add_argument(
        "--source-zoom-max",
        type=float,
        default=1.0,
        help="Maximum crop zoom applied to ADE source before embedding.",
    )
    parser.add_argument(
        "--source-zoom-position",
        choices=["center", "random"],
        default="random",
        help="Where to crop the zoomed ADE source region.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.train_samples is not None and args.train_samples < 1:
        raise ValueError("--train-samples must be positive.")
    if args.val_samples is not None and args.val_samples < 1:
        raise ValueError("--val-samples must be positive.")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be positive.")
    if not 0 < args.min_fraction <= args.max_fraction <= 1:
        raise ValueError("Require 0 < --min-fraction <= --max-fraction <= 1.")
    if not 1 <= args.source_zoom_min <= args.source_zoom_max:
        raise ValueError("Require 1 <= --source-zoom-min <= --source-zoom-max.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.train_samples is not None or args.val_samples is not None:
        if args.train_samples is not None:
            _generate_split(args=args, split="training", num_samples=args.train_samples)
        if args.val_samples is not None:
            _generate_split(args=args, split="validation", num_samples=args.val_samples)
    else:
        _generate_split(args=args, split=args.split, num_samples=args.num_samples)
    print(f"Saved ADE-embedded synthetic dataset to {args.root}")


if __name__ == "__main__":
    main()
