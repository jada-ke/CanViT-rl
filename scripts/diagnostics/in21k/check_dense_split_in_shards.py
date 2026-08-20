"""Check whether dense-feature shards and an image split root overlap.

Example:
    uv run python scripts/diagnostics/in21k/check_dense_split_in_shards.py \
        --feature-base-dir /features \
        --image-root /data/val \
        --teacher-name dinov3_vitb16 \
        --scene-resolution 512

    uv run python scripts/diagnostics/in21k/check_dense_split_in_shards.py \
        --feature-base-dir /features \
        --image-root /data/val \
        --from-images \
        --per-shard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse shard/image-root scanner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-base-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--teacher-name", type=str, default="dinov3_vitb16")
    parser.add_argument("--scene-resolution", type=int, default=512)
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="Maximum number of sorted shard files to scan. 0 scans all shards.",
    )
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument(
        "--per-shard",
        action="store_true",
        help="Print hit counts for each scanned shard.",
    )
    parser.add_argument(
        "--from-images",
        action="store_true",
        help=(
            "Reverse check: walk --image-root first, then count how many image "
            "relative paths appear in the scanned shards."
        ),
    )
    args = parser.parse_args()
    if args.max_shards < 0:
        raise ValueError("--max-shards must be non-negative.")
    if args.examples < 0:
        raise ValueError("--examples must be non-negative.")
    return args


def _shard_paths(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    """Return the dense shard directory and selected sorted shard files."""
    shards_dir = (
        args.feature_base_dir
        / args.teacher_name
        / str(args.scene_resolution)
        / "shards"
    )
    shard_paths = sorted(shards_dir.glob("*.pt"))
    if args.max_shards:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No dense-feature shards found in {shards_dir}")
    return shards_dir, shard_paths


def _scan_shards_to_image_root(
    args: argparse.Namespace,
    shards_dir: Path,
    shard_paths: list[Path],
) -> None:
    """Scan shard paths and count rows whose images exist under image_root."""

    total_rows = 0
    usable_rows = 0
    found_rows = 0
    missing_rows = 0
    examples_found: list[str] = []
    examples_missing: list[str] = []

    for shard_path in shard_paths:
        shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        failed = set(shard.get("failed_indices", []))
        shard_usable = 0
        shard_found = 0
        for idx, rel_path in enumerate(shard["paths"]):
            total_rows += 1
            if idx in failed:
                continue
            usable_rows += 1
            shard_usable += 1
            rel_path_str = str(rel_path)
            if (args.image_root / rel_path_str).is_file():
                found_rows += 1
                shard_found += 1
                if len(examples_found) < args.examples:
                    examples_found.append(rel_path_str)
            else:
                missing_rows += 1
                if len(examples_missing) < args.examples:
                    examples_missing.append(rel_path_str)
        del shard
        if args.per_shard:
            # Problem: mixed train/val shards are hard to diagnose from an
            # aggregate count. Solution: optionally print per-shard hit rates.
            # Result: you can see whether validation images live in later
            # shards and pick a better --subset-shards value.
            pct = 100.0 * shard_found / max(shard_usable, 1)
            print(
                f"{shard_path.name}: found={shard_found} "
                f"usable={shard_usable} pct={pct:.2f}%"
            )

    pct_found = 100.0 * found_rows / max(usable_rows, 1)
    print(f"shards_dir: {shards_dir}")
    print(f"image_root: {args.image_root}")
    print(f"scanned_shards: {len(shard_paths)}")
    print(f"total_rows: {total_rows}")
    print(f"usable_rows: {usable_rows}")
    print(f"rows_found_under_image_root: {found_rows}")
    print(f"rows_missing_under_image_root: {missing_rows}")
    print(f"found_pct_of_usable: {pct_found:.2f}%")
    if examples_found:
        print("found_examples:")
        for rel_path in examples_found:
            print(f"  {rel_path}")
    if examples_missing:
        print("missing_examples:")
        for rel_path in examples_missing:
            print(f"  {rel_path}")


def _iter_image_relpaths(image_root: Path) -> list[str]:
    """Return image paths relative to image_root in POSIX form."""
    # Problem: val image roots can have the right files even when shard rows do
    # not resolve from the shard side. Solution: walk the image split first and
    # compare relative paths directly with shard metadata. Result: the scanner
    # can answer whether validation images have corresponding feature rows.
    return sorted(
        str(path.relative_to(image_root))
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _scan_images_to_shards(
    args: argparse.Namespace,
    shards_dir: Path,
    shard_paths: list[Path],
) -> None:
    """Walk image_root first and report which images have matching shard rows."""
    image_relpaths = _iter_image_relpaths(args.image_root)
    image_set = set(image_relpaths)
    if not image_relpaths:
        raise FileNotFoundError(f"No image files found under {args.image_root}")

    shard_rel_to_name: dict[str, str] = {}
    shard_hits = {path.name: 0 for path in shard_paths}
    usable_rows = 0
    duplicate_rows = 0
    for shard_path in shard_paths:
        shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        failed = set(shard.get("failed_indices", []))
        for idx, rel_path in enumerate(shard["paths"]):
            if idx in failed:
                continue
            usable_rows += 1
            rel_path_str = str(rel_path)
            if rel_path_str in shard_rel_to_name:
                duplicate_rows += 1
            else:
                shard_rel_to_name[rel_path_str] = shard_path.name
            if rel_path_str in image_set:
                shard_hits[shard_path.name] += 1
        del shard

    matched_images = [rel for rel in image_relpaths if rel in shard_rel_to_name]
    missing_images = [rel for rel in image_relpaths if rel not in shard_rel_to_name]
    pct_matched = 100.0 * len(matched_images) / max(len(image_relpaths), 1)

    if args.per_shard:
        for shard_path in shard_paths:
            print(f"{shard_path.name}: val_image_matches={shard_hits[shard_path.name]}")

    print(f"shards_dir: {shards_dir}")
    print(f"image_root: {args.image_root}")
    print(f"mode: from_images")
    print(f"scanned_shards: {len(shard_paths)}")
    print(f"usable_shard_rows: {usable_rows}")
    print(f"duplicate_usable_shard_paths: {duplicate_rows}")
    print(f"image_files_under_root: {len(image_relpaths)}")
    print(f"images_found_in_shards: {len(matched_images)}")
    print(f"images_missing_from_shards: {len(missing_images)}")
    print(f"found_pct_of_images: {pct_matched:.2f}%")
    if matched_images and args.examples:
        print("matched_image_examples:")
        for rel_path in matched_images[: args.examples]:
            print(f"  {rel_path} -> {shard_rel_to_name[rel_path]}")
    if missing_images and args.examples:
        print("missing_image_examples:")
        for rel_path in missing_images[: args.examples]:
            print(f"  {rel_path}")


def main() -> None:
    """Run the selected dense shard/image-root overlap check."""
    args = parse_args()
    shards_dir, shard_paths = _shard_paths(args)
    if args.from_images:
        _scan_images_to_shards(args, shards_dir, shard_paths)
    else:
        _scan_shards_to_image_root(args, shards_dir, shard_paths)


if __name__ == "__main__":
    main()
