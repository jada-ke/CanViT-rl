"""Visualize a summary sheet of oracle t1 MNIST glimpse boxes and crops.

Example:
    uv run python scripts/synthetic_dataset/visualize_mnist_glimpse_t1.py \
        --dataset-root datasets/mnist_glimpse \
        --output-dir results/mnist_glimpse_t1_viz
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from _paths import repo_path


@dataclass(frozen=True)
class MnistGlimpseVizRecord:
    """One generated sample plus oracle t1 metadata."""

    image_path: Path
    split: str
    sample: str
    label: int
    center_x: float
    center_y: float
    scale: float
    sharp_box: tuple[int, int, int, int]


def _load_records(root: Path) -> list[MnistGlimpseVizRecord]:
    """Load all generated training and validation metadata rows."""
    records: list[MnistGlimpseVizRecord] = []
    for split in ("training", "validation"):
        metadata_path = root / f"metadata_{split}.csv"
        image_dir = root / "images" / split
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        with metadata_path.open(newline="") as file:
            for row in csv.DictReader(file):
                records.append(
                    MnistGlimpseVizRecord(
                        image_path=image_dir / row["sample"],
                        split=split,
                        sample=row["sample"],
                        label=int(row["label"]),
                        center_x=float(row["center_x"]),
                        center_y=float(row["center_y"]),
                        scale=float(row["scale"]),
                        sharp_box=(
                            int(row["sharp_x0"]),
                            int(row["sharp_y0"]),
                            int(row["sharp_x1"]),
                            int(row["sharp_y1"]),
                        ),
                    )
                )
    if not records:
        raise ValueError(f"No generated MNIST glimpse records found in {root}")
    return records


def _viewpoint_box(
    record: MnistGlimpseVizRecord,
    image: Image.Image,
) -> tuple[int, int, int, int]:
    """Convert normalized t1 center/scale metadata into image pixel box."""
    width, height = image.size
    side = record.scale * min(width, height)
    cx = record.center_x * width
    cy = record.center_y * height
    x0 = max(0, int(round(cx - side / 2.0)))
    y0 = max(0, int(round(cy - side / 2.0)))
    x1 = min(width, int(round(cx + side / 2.0)))
    y1 = min(height, int(round(cy + side / 2.0)))
    return x0, y0, x1, y1


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    """Draw readable text with a small dark backing rectangle."""
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def _make_overlay(record: MnistGlimpseVizRecord) -> tuple[Image.Image, Image.Image]:
    """Return overlay image and the exact oracle t1 crop."""
    image = Image.open(record.image_path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    t1_box = _viewpoint_box(record, image)
    # Problem: metadata has both the generated sharp patch bounds and the
    # normalized Viewpoint used as the t1 oracle. Solution: draw the actual
    # oracle Viewpoint in red and the sharp patch in cyan. Result: any scale or
    # location mismatch is visible immediately.
    draw.rectangle(t1_box, outline=(255, 40, 40), width=4)
    draw.rectangle(record.sharp_box, outline=(0, 220, 255), width=2)
    _draw_label(
        draw,
        (8, 8),
        (
            f"{record.split} {record.sample} label={record.label} "
            f"center=({record.center_x:.3f},{record.center_y:.3f}) "
            f"scale={record.scale:.3f}"
        ),
    )
    crop = image.crop(t1_box)
    return overlay, crop


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Create a padded thumbnail with stable dimensions."""
    thumb = image.copy()
    thumb.thumbnail(size, resample=getattr(Image, "Resampling", Image).BICUBIC)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - thumb.size[0]) // 2
    y = (size[1] - thumb.size[1]) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def _save_contact_sheet(
    *,
    rows: list[tuple[MnistGlimpseVizRecord, Image.Image, Image.Image]],
    output_path: Path,
    tile_size: int,
) -> None:
    """Save a contact sheet showing overlay and crop for every sample."""
    pair_width = tile_size * 2
    cols = 2
    row_count = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * pair_width, row_count * tile_size), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (record, overlay, crop) in enumerate(rows):
        col = idx % cols
        row = idx // cols
        x = col * pair_width
        y = row * tile_size
        sheet.paste(_thumbnail(overlay, (tile_size, tile_size)), (x, y))
        sheet.paste(_thumbnail(crop, (tile_size, tile_size)), (x + tile_size, y))
        _draw_label(
            draw,
            (x + 5, y + 5),
            f"{record.split[:3]} label={record.label} scale={record.scale:.3f}",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mnist_glimpse_t1_viz"),
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Also save per-sample overlay and crop PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    # Problem: visualization defaults should still target repo-root results
    # after moving under scripts/synthetic_dataset. Solution: resolve relative
    # paths against the repo root immediately after parsing. Result: outputs
    # do not drift into the nested scripts folder.
    args.dataset_root = repo_path(args.dataset_root)
    args.output_dir = repo_path(args.output_dir)
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive.")
    records = _load_records(args.dataset_root)
    overlay_dir = args.output_dir / "overlays"
    crop_dir = args.output_dir / "crops"
    if args.save_individual:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, record in enumerate(records):
        overlay, crop = _make_overlay(record)
        if args.save_individual:
            stem = f"{idx:04d}_{record.split}_{Path(record.sample).stem}"
            overlay.save(overlay_dir / f"{stem}_overlay.png")
            crop.save(crop_dir / f"{stem}_crop.png")
        rows.append((record, overlay, crop))
    sheet_path = args.output_dir / "contact_sheet.png"
    _save_contact_sheet(
        rows=rows,
        output_path=sheet_path,
        tile_size=args.tile_size,
    )
    if args.save_individual:
        print(f"Saved {len(records)} t1 overlays to {overlay_dir}")
        print(f"Saved {len(records)} t1 crops to {crop_dir}")
    print(f"Saved contact sheet to {sheet_path}")


if __name__ == "__main__":
    main()
