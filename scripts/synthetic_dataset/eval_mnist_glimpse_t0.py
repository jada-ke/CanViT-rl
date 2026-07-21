"""Evaluate whether CanViT t0/t1 features classify MNIST glimpse scenes.

This diagnostic compares the initial full-scene CanViT pass (t0) with an oracle
t1 pass that uses each sample's metadata center/scale for the second glimpse.
It extracts ``state.recurrent_cls``, builds one fixed non-learned centroid probe
from support examples inside the generated dataset, and reports only the active
view condition: t0 should be wrong and oracle t1 should be correct. If
``oracle_images/`` exists, t0 is sampled from ``images/`` and oracle t1 is
sampled from the paired reveal image.

Example:
    uv run python scripts/synthetic_dataset/eval_mnist_glimpse_t0.py \
        --dataset-root datasets/mnist_glimpse \
        --batch-size 16 \
        --support-shots-per-class 5

    uv run python scripts/synthetic_dataset/eval_mnist_glimpse_t0.py \
        --dataset-root datasets/mnist_glimpse \
        --batch-size 16 \
        --support-shots-per-class 5 \
        --t0-only

    uv run python scripts/synthetic_dataset/eval_mnist_glimpse_t0.py \
        --dataset-root datasets/mnist_glimpse \
        --batch-size 16 \
        --support-shots-per-class 5 \
        --no-oracle-images
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_pytorch.preprocess import preprocess
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _paths import repo_path
from canvit_rl.canvit_precision import resolve_canvit_dtype
from canvit_rl.env import get_device


DEFAULT_MODEL_REPO = (
    "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
)


@dataclass(frozen=True)
class MnistGlimpseRecord:
    """One generated MNIST glimpse sample with its digit label."""

    image_path: Path
    oracle_image_path: Path
    label: int
    center_y: float
    center_x: float
    scale: float


class MnistGlimpseClassificationDataset(Dataset):
    """Read all generated MNIST glimpse scenes and labels from metadata CSVs."""

    def __init__(
        self,
        *,
        root: Path,
        scene_size_px: int,
        use_oracle_images: bool = True,
    ) -> None:
        self.root = root
        self.transform = preprocess(scene_size_px)
        self.records = self._load_records(
            root=root,
            splits=("training", "validation"),
            use_oracle_images=use_oracle_images,
        )

    @staticmethod
    def _load_records(
        *,
        root: Path,
        splits: tuple[str, ...],
        use_oracle_images: bool,
    ) -> list[MnistGlimpseRecord]:
        """Load image paths and labels from every generated split."""
        records: list[MnistGlimpseRecord] = []
        for split in splits:
            metadata_path = root / f"metadata_{split}.csv"
            image_dir = root / "images" / split
            oracle_image_dir = root / "oracle_images" / split
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            with metadata_path.open(newline="") as file:
                for row in csv.DictReader(file):
                    image_path = image_dir / row["sample"]
                    oracle_image_path = (
                        oracle_image_dir / row["sample"]
                        if use_oracle_images and oracle_image_dir.is_dir()
                        else image_path
                    )
                    records.append(
                        MnistGlimpseRecord(
                            image_path=image_path,
                            oracle_image_path=oracle_image_path,
                            label=int(row["label"]),
                            center_y=float(row["center_y"]),
                            center_x=float(row["center_x"]),
                            scale=float(row["scale"]),
                        )
                    )
        if not records:
            raise ValueError(f"No generated MNIST glimpse records found in {root}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        record = self.records[index]
        with Image.open(record.image_path) as image_file:
            image = image_file.convert("RGB")
        if record.oracle_image_path == record.image_path:
            oracle_image = image.copy()
        else:
            with Image.open(record.oracle_image_path) as image_file:
                oracle_image = image_file.convert("RGB")
        center = torch.tensor(
            [
                2.0 * record.center_y - 1.0,
                2.0 * record.center_x - 1.0,
            ],
            dtype=torch.float32,
        )
        scale = torch.tensor(record.scale, dtype=torch.float32)
        return (
            self.transform(image),
            self.transform(oracle_image),
            torch.tensor(record.label, dtype=torch.long),
            center,
            scale,
        )


def _select_support_by_class(
    labels: Tensor,
    *,
    shots_per_class: int,
    seed: int,
) -> Tensor:
    """Select the support rows that define the one fixed centroid probe."""
    if shots_per_class <= 0:
        raise ValueError("--support-shots-per-class must be positive.")
    label_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(int(label), []).append(idx)
    rng = random.Random(seed)
    support: list[int] = []
    for label in sorted(label_to_indices):
        indices = label_to_indices[label]
        rng.shuffle(indices)
        if len(indices) < shots_per_class:
            raise ValueError(
                f"Digit {label} has only {len(indices)} samples, but "
                f"--support-shots-per-class={shots_per_class}."
            )
        support.extend(indices[:shots_per_class])
    if not support:
        raise ValueError("No support samples were selected.")
    return torch.tensor(support, dtype=torch.long)


def _dataloader(dataset: Dataset, *, batch_size: int, device: torch.device) -> DataLoader:
    """Create a simple eval dataloader with safe pinned-memory defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def load_frozen_model(args: argparse.Namespace, device: torch.device):
    """Load frozen CanViT and resolve the t0 full-scene glimpse size."""
    model = CanViTForPretrainingHFHub.from_pretrained(args.model_repo).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    canvit_dtype = resolve_canvit_dtype(args.canvit_dtype, device)
    model.to(device=device, dtype=canvit_dtype)
    for module in model.modules():
        if module.__class__.__name__ == "VPEEncoder":
            module.to(device=device, dtype=torch.float32)
    canvas_grid_size = model.canvas_patch_grid_sizes[0]
    glimpse_size_px = int(args.glimpse_grid_size * model.backbone.patch_size_px)
    return model, canvas_grid_size, glimpse_size_px, canvit_dtype


@torch.inference_mode()
def extract_t0_features(
    *,
    loader: DataLoader,
    model,
    canvas_grid_size: int,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the full-scene t0 pass and return recurrent_cls features."""
    features: list[Tensor] = []
    labels: list[Tensor] = []
    centers: list[Tensor] = []
    scales: list[Tensor] = []
    for images, _, batch_labels, batch_centers, batch_scales in tqdm(
        loader,
        desc="Extracting t0 features",
    ):
        images = images.to(device=device, non_blocking=True)
        batch_size = images.shape[0]
        state = model.init_state(
            batch_size=batch_size,
            canvas_grid_size=canvas_grid_size,
        )
        viewpoint = Viewpoint.full_scene(batch_size=batch_size, device=device)
        glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=viewpoint,
            glimpse_size_px=glimpse_size_px,
        ).to(dtype=canvit_dtype)
        out = model(glimpse=glimpse, state=state, viewpoint=viewpoint)
        features.append(out.state.recurrent_cls.squeeze(1).float().cpu())
        labels.append(batch_labels.cpu())
        centers.append(batch_centers.cpu())
        scales.append(batch_scales.cpu())
    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(centers, dim=0),
        torch.cat(scales, dim=0),
    )


@torch.inference_mode()
def extract_oracle_t1_features(
    *,
    loader: DataLoader,
    model,
    canvas_grid_size: int,
    glimpse_size_px: int,
    canvit_dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run t0, then an oracle t1 metadata Viewpoint, and return t1 features."""
    features: list[Tensor] = []
    labels: list[Tensor] = []
    centers: list[Tensor] = []
    scales: list[Tensor] = []
    for images, oracle_images, batch_labels, batch_centers, batch_scales in tqdm(
        loader,
        desc="Extracting oracle t1 features",
    ):
        images = images.to(device=device, non_blocking=True)
        oracle_images = oracle_images.to(device=device, non_blocking=True)
        batch_centers = batch_centers.to(device=device, non_blocking=True)
        batch_scales = batch_scales.to(device=device, non_blocking=True)
        batch_size = images.shape[0]
        state = model.init_state(
            batch_size=batch_size,
            canvas_grid_size=canvas_grid_size,
        )
        full_viewpoint = Viewpoint.full_scene(batch_size=batch_size, device=device)
        full_glimpse = sample_at_viewpoint(
            spatial=images,
            viewpoint=full_viewpoint,
            glimpse_size_px=glimpse_size_px,
        ).to(dtype=canvit_dtype)
        out = model(glimpse=full_glimpse, state=state, viewpoint=full_viewpoint)
        # Problem: we need to test whether the correct second glimpse supplies
        # enough class evidence. Solution: consume the metadata center/scale as
        # an oracle Viewpoint after the same full-scene t0 warmup used by SAC.
        # Result: oracle-t1 accuracy isolates task observability from policy
        # learning quality.
        oracle_viewpoint = Viewpoint(
            centers=batch_centers.float(),
            scales=batch_scales.float(),
        )
        # Problem: a same-image diagnostic leaks local t1 evidence into the
        # full-scene t0 pass. Solution: keep the t0 recurrent state from the
        # regular image, then sample only the oracle t1 crop from the paired
        # reveal image when present. Result: this tests the intended viewpoint.
        oracle_glimpse = sample_at_viewpoint(
            spatial=oracle_images,
            viewpoint=oracle_viewpoint,
            glimpse_size_px=glimpse_size_px,
        ).to(dtype=canvit_dtype)
        out = model(
            glimpse=oracle_glimpse,
            state=out.state,
            viewpoint=oracle_viewpoint,
        )
        features.append(out.state.recurrent_cls.squeeze(1).float().cpu())
        labels.append(batch_labels.cpu())
        centers.append(batch_centers.cpu())
        scales.append(batch_scales.cpu())
    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(centers, dim=0),
        torch.cat(scales, dim=0),
    )


def build_support_centroids(features: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Build normalized nearest-centroid prototypes for support digit classes."""
    features = F.normalize(features.float(), dim=1)
    classes = torch.unique(labels).sort().values
    centroids = []
    for label in classes.tolist():
        class_features = features[labels == label]
        # Problem: this diagnostic should not train a classifier. Solution:
        # build eval-only class prototypes from support examples and use cosine
        # nearest-centroid scoring. Result: high accuracy means t0 features are
        # already separable without any learned probe or later viewpoint.
        centroids.append(
            F.normalize(class_features.mean(dim=0, keepdim=True), dim=1)
        )
    return classes, torch.cat(centroids, dim=0)


def predict_with_centroids(
    *,
    support_features: Tensor,
    support_labels: Tensor,
    features: Tensor,
) -> Tensor:
    """Predict digit labels with one fixed nearest-centroid probe."""
    classes, centroids = build_support_centroids(support_features, support_labels)
    features = F.normalize(features.float(), dim=1)
    scores = features @ centroids.T
    return classes[scores.argmax(dim=1)]


def active_view_metrics(
    *,
    labels: Tensor,
    t0_predictions: Tensor,
    t1_predictions: Tensor,
) -> dict[str, float]:
    """Summarize whether t0 fails and oracle t1 succeeds under one probe."""
    t0_wrong = ~t0_predictions.eq(labels)
    t1_correct = t1_predictions.eq(labels)
    accepted = t0_wrong & t1_correct
    # Problem: accuracy-style outputs made this diagnostic look like several
    # classifiers were being compared. Solution: report the single condition
    # the dataset is meant to satisfy. Result: higher accepted_rate directly
    # means more examples where the right second viewpoint adds useful signal.
    metrics = {
        "t0_wrong_rate": float(t0_wrong.float().mean().item()),
        "oracle_t1_correct_rate": float(t1_correct.float().mean().item()),
        "accepted_rate": float(accepted.float().mean().item()),
    }
    for label in range(10):
        mask = labels == label
        if mask.any():
            metrics[f"t0_wrong_rate_digit_{label}"] = float(
                t0_wrong[mask].float().mean().item()
            )
            metrics[f"oracle_t1_correct_rate_digit_{label}"] = float(
                t1_correct[mask].float().mean().item()
            )
            metrics[f"accepted_rate_digit_{label}"] = float(
                accepted[mask].float().mean().item()
            )
    return metrics


def classification_metrics(
    *,
    prefix: str,
    labels: Tensor,
    predictions: Tensor,
) -> dict[str, float]:
    """Summarize top-1 classification accuracy for one fixed probe."""
    correct = predictions.eq(labels)
    # Problem: sometimes we only need to know whether the initial hidden image
    # is already classifiable, without involving oracle t1 at all. Solution:
    # report direct fixed-probe t0 accuracy/wrong-rate against labels. Result:
    # the diagnostic can isolate initial-image leakage from active-view gains.
    metrics = {
        f"{prefix}/accuracy": float(correct.float().mean().item()),
        f"{prefix}/wrong_rate": float((~correct).float().mean().item()),
    }
    for label in range(10):
        mask = labels == label
        if mask.any():
            metrics[f"{prefix}/accuracy_digit_{label}"] = float(
                correct[mask].float().mean().item()
            )
            metrics[f"{prefix}/wrong_rate_digit_{label}"] = float(
                (~correct[mask]).float().mean().item()
            )
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse t0 classification diagnostic arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO)
    parser.add_argument("--scene-size", type=int, default=512)
    parser.add_argument("--glimpse-grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--support-shots-per-class",
        type=int,
        default=1,
        help=(
            "Number of generated samples per digit used to define the one "
            "fixed non-learned centroid probe."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--canvit-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--t0-only",
        action="store_true",
        help="Evaluate only the initial hidden images and ignore oracle_images/.",
    )
    parser.add_argument(
        "--no-oracle-images",
        action="store_true",
        help=(
            "Ignore oracle_images/ but still run the metadata t1 viewpoint from "
            "the initial hidden image."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    # Problem: relative dataset roots should keep referring to repo-root data
    # after the script move. Solution: resolve the CLI path once at startup.
    # Result: diagnostics can be launched from the subfolder without changing
    # arguments or duplicating datasets beneath scripts/.
    args.dataset_root = repo_path(args.dataset_root)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.support_shots_per_class <= 0:
        raise ValueError("--support-shots-per-class must be positive.")
    device = get_device()
    eval_dataset: Dataset = MnistGlimpseClassificationDataset(
        root=args.dataset_root,
        scene_size_px=args.scene_size,
        use_oracle_images=not (args.t0_only or args.no_oracle_images),
    )
    model, canvas_grid_size, glimpse_size_px, canvit_dtype = load_frozen_model(
        args,
        device,
    )
    loader = _dataloader(eval_dataset, batch_size=args.batch_size, device=device)
    t0_features, labels, centers, scales = extract_t0_features(
        loader=_dataloader(eval_dataset, batch_size=args.batch_size, device=device),
        model=model,
        canvas_grid_size=canvas_grid_size,
        glimpse_size_px=glimpse_size_px,
        canvit_dtype=canvit_dtype,
        device=device,
    )
    support_idx = _select_support_by_class(
        labels,
        shots_per_class=args.support_shots_per_class,
        seed=args.seed,
    )
    if args.t0_only:
        support_features = t0_features.index_select(0, support_idx)
        support_labels = labels.index_select(0, support_idx)
        t0_predictions = predict_with_centroids(
            support_features=support_features,
            support_labels=support_labels,
            features=t0_features,
        )
        metrics = classification_metrics(
            prefix="fixed_probe/t0",
            labels=labels,
            predictions=t0_predictions,
        )
        support_labels = labels.index_select(0, support_idx)
        print("\nMNIST glimpse fixed-probe t0-only diagnostic")
        print(
            f"total_samples={len(eval_dataset)} "
            f"support_samples={len(support_idx)}"
        )
        print(f"support_classes={sorted(torch.unique(support_labels).tolist())}")
        print(f"fixed_probe/t0_accuracy={metrics['fixed_probe/t0/accuracy']:.4f}")
        print(f"fixed_probe/t0_wrong_rate={metrics['fixed_probe/t0/wrong_rate']:.4f}")
        for label in range(10):
            key = f"fixed_probe/t0/accuracy_digit_{label}"
            if key in metrics:
                print(f"fixed_probe/t0_accuracy_digit_{label}={metrics[key]:.4f}")
                print(
                    f"fixed_probe/t0_wrong_rate_digit_{label}="
                    f"{metrics[f'fixed_probe/t0/wrong_rate_digit_{label}']:.4f}"
                )
        return
    t1_features, t1_labels, _, _ = extract_oracle_t1_features(
        loader=loader,
        model=model,
        canvas_grid_size=canvas_grid_size,
        glimpse_size_px=glimpse_size_px,
        canvit_dtype=canvit_dtype,
        device=device,
    )
    if not torch.equal(labels, t1_labels):
        raise RuntimeError("t0 and oracle-t1 extraction returned different labels.")
    support_features = t1_features.index_select(0, support_idx)
    support_labels = labels.index_select(0, support_idx)
    t0_predictions = predict_with_centroids(
        support_features=support_features,
        support_labels=support_labels,
        features=t0_features,
    )
    t1_predictions = predict_with_centroids(
        support_features=support_features,
        support_labels=support_labels,
        features=t1_features,
    )
    metrics = active_view_metrics(
        labels=labels,
        t0_predictions=t0_predictions,
        t1_predictions=t1_predictions,
    )
    support_labels = labels.index_select(0, support_idx)
    print("\nMNIST glimpse fixed-probe active-view diagnostic")
    print(
        f"total_samples={len(eval_dataset)} "
        f"support_samples={len(support_idx)}"
    )
    print(f"t1_source={'images' if args.no_oracle_images else 'oracle_images'}")
    print(f"support_classes={sorted(torch.unique(support_labels).tolist())}")
    print(
        "oracle_viewpoint_scale_mean="
        f"{float(scales.mean().item()):.4f} "
        f"min={float(scales.min().item()):.4f} "
        f"max={float(scales.max().item()):.4f}"
    )
    print(f"fixed_probe/t0_wrong_rate={metrics['t0_wrong_rate']:.4f}")
    print(
        "fixed_probe/oracle_t1_correct_rate="
        f"{metrics['oracle_t1_correct_rate']:.4f}"
    )
    print(f"fixed_probe/accepted_rate={metrics['accepted_rate']:.4f}")
    for label in range(10):
        key = f"accepted_rate_digit_{label}"
        if key in metrics:
            print(
                f"fixed_probe/t0_wrong_rate_digit_{label}="
                f"{metrics[f't0_wrong_rate_digit_{label}']:.4f}"
            )
            print(
                f"fixed_probe/oracle_t1_correct_rate_digit_{label}="
                f"{metrics[f'oracle_t1_correct_rate_digit_{label}']:.4f}"
            )
            print(f"fixed_probe/{key}={metrics[key]:.4f}")


if __name__ == "__main__":
    main()
