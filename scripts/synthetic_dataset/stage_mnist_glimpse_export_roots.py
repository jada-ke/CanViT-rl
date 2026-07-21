"""Stage MNIST glimpse images into ImageFolder-style export roots.

CanViT-pretrain's feature exporter indexes images with an ImageFolder-style
layout, so the generated ``images/{training,validation}`` and
``oracle_images/{training,validation}`` trees need to be rearranged by digit
label before feature export.

Example:
    uv run python scripts/synthetic_dataset/stage_mnist_glimpse_export_roots.py \
        --dataset-root datasets/mnist_glimpse \
        --output-root datasets/mnist_glimpse_export
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from _paths import repo_path


def _safe_link_or_copy(*, src: Path, dst: Path, copy: bool) -> None:
    """Create a stable staged file without overwriting unrelated files."""
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise FileExistsError(
            f"Refusing to overwrite existing staged file with different target: {dst}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        # Problem: duplicating generated images makes repeated export staging
        # heavier than needed. Solution: default to relative symlinks and keep
        # --copy available for filesystems/tools that dislike symlinks. Result:
        # hidden/oracle export roots have matching ImageFolder paths without
        # modifying the original dataset.
        dst.symlink_to(os.path.relpath(src, start=dst.parent))


def _stage_split(
    *,
    dataset_root: Path,
    output_root: Path,
    split: str,
    copy: bool,
) -> int:
    """Stage one generated split into hidden/oracle digit-label folders."""
    metadata_path = dataset_root / f"metadata_{split}.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    count = 0
    with metadata_path.open(newline="") as file:
        for row in csv.DictReader(file):
            label = int(row["label"])
            sample = row["sample"]
            staged_name = f"{split}_{sample}"
            hidden_src = dataset_root / "images" / split / sample
            oracle_src = dataset_root / "oracle_images" / split / sample
            if not hidden_src.is_file():
                raise FileNotFoundError(f"Missing hidden image: {hidden_src}")
            if not oracle_src.is_file():
                raise FileNotFoundError(f"Missing oracle image: {oracle_src}")
            hidden_dst = output_root / "hidden" / f"digit{label}" / staged_name
            oracle_dst = output_root / "oracle" / f"digit{label}" / staged_name
            _safe_link_or_copy(src=hidden_src, dst=hidden_dst, copy=copy)
            _safe_link_or_copy(src=oracle_src, dst=oracle_dst, copy=copy)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    """Parse staging arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mnist_glimpse"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/mnist_glimpse_export"),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating relative symlinks.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    # Problem: nested script execution otherwise interprets relative staging
    # paths from the caller's cwd. Solution: anchor dataset/output roots at the
    # repository root. Result: export staging works from this subfolder with
    # the same defaults used before the move.
    args.dataset_root = repo_path(args.dataset_root)
    args.output_root = repo_path(args.output_root)
    total = 0
    for split in ("training", "validation"):
        total += _stage_split(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            split=split,
            copy=args.copy,
        )
    print(f"Staged {total} samples")
    print(f"Hidden ImageFolder root: {args.output_root / 'hidden'}")
    print(f"Oracle ImageFolder root: {args.output_root / 'oracle'}")


if __name__ == "__main__":
    main()
