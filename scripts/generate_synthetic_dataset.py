"""Generate padded-scene synthetic segmentation data from random ADE20K images.

Example:
    uv run python scripts/generate_synthetic_dataset.py \
        --ade-root datasets/ADE20k \
        --split training \
        --root synthetic_segmentation \
        --num-samples 5 \
        --min-fraction 0.25 \
        --max-fraction 0.65

The output layout is:

    synthetic_segmentation/
    ├── images/
    │   └── sample_00000.png
    └── masks/
        └── sample_00000.png

Masks preserve ADE20K class ids in the embedded region and use 255 outside the
embedded region so segmentation loss ignores the padded area.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


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
) -> tuple[Image.Image, Image.Image]:
    """Resize source image/mask so the longest side occupies a random fraction."""
    source_width, source_height = image.size
    target_fraction = random.uniform(min_fraction, max_fraction)
    target_long_side = max(1, int(min(canvas_width, canvas_height) * target_fraction))
    scale = target_long_side / max(source_width, source_height)
    target_width = max(1, min(canvas_width, int(round(source_width * scale))))
    target_height = max(1, min(canvas_height, int(round(source_height * scale))))
    image_resample = getattr(Image, "Resampling", Image).BICUBIC
    mask_resample = getattr(Image, "Resampling", Image).NEAREST
    return (
        image.resize((target_width, target_height), resample=image_resample),
        mask.resize((target_width, target_height), resample=mask_resample),
    )


def generate_embedded_sample(
    *,
    image_path: Path,
    mask_path: Path,
    width: int,
    height: int,
    min_fraction: float,
    max_fraction: float,
) -> tuple[Image.Image, Image.Image]:
    """Embed one ADE image/mask pair into a larger padded scene."""
    source_image = Image.open(image_path).convert("RGB")
    source_mask = Image.open(mask_path).convert("L")
    embedded_image, embedded_mask = _resize_to_fraction(
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
    return canvas, mask_canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade-root", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument("--root", type=Path, default=Path("synthetic_segmentation"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--min-fraction", type=float, default=0.25)
    parser.add_argument("--max-fraction", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be positive.")
    if not 0 < args.min_fraction <= args.max_fraction <= 1:
        raise ValueError("Require 0 < --min-fraction <= --max-fraction <= 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    pairs = _load_pairs(ade_root=args.ade_root, split=args.split)
    image_dir = args.root / "images"
    mask_dir = args.root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if args.num_samples <= len(pairs):
        selected_pairs = random.sample(pairs, args.num_samples)
    else:
        selected_pairs = [random.choice(pairs) for _ in range(args.num_samples)]

    for idx, (image_path, mask_path) in enumerate(selected_pairs):
        # Fixed by Codex on 2026-06-23
        # Problem: Geometric synthetic masks used binary labels that do not
        # match the frozen ADE20K probe's semantic class space.
        # Solution: embed randomly sampled ADE image/mask pairs into padded
        # scenes and mark padding with IGNORE_LABEL.
        # Result: Canvas SAC can train on an active-vision toy task while CE
        # still compares against meaningful ADE class ids.
        image, mask = generate_embedded_sample(
            image_path=image_path,
            mask_path=mask_path,
            width=args.width,
            height=args.height,
            min_fraction=args.min_fraction,
            max_fraction=args.max_fraction,
        )
        stem = f"sample_{idx:05d}_{image_path.stem}.png"
        image.save(image_dir / stem)
        mask.save(mask_dir / stem)

    print(f"Saved {args.num_samples} ADE-embedded samples to {args.root}")
    print(f"Source split: {args.ade_root / 'images' / args.split}")
    print(f"Images: {image_dir}")
    print(f"Masks:  {mask_dir}")


if __name__ == "__main__":
    main()
