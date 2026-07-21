"""Generate a synthetic MNIST active-glimpse diagnostic dataset.

The output images are large scenes with one digit placed at a known random
location. Splits are generated with a shuffled round-robin label schedule so
per-digit diagnostics are not dominated by random class imbalance. The regular
image contains only a blurred, low-contrast t0 scene. A matching oracle image
contains the sharp local patch used for t1 checks, so the full-scene warmup
cannot see the same evidence the correct later viewpoint receives.

Example:
    uv run python scripts/synthetic_dataset/generate_mnist_glimpse_dataset.py \
  --mnist-root datasets/mnist \
  --root datasets/mnist_glimpse \
  --train-samples 80 \
  --val-samples 20 \
  --min-digit-size 72 \
  --max-digit-size 104 \
  --blur-radius 160 \
  --global-digit-blur-radius 48 \
  --global-digit-alpha 0.001 \
  --sharp-digit-blur-radius 2.0 \
  --sharp-patch-fraction 1.15 \
  --canvit-filter \
  --filter-support-per-class 20 \
  --filter-max-attempts-per-sample 800 \
  --filter-progress-interval 25 \
  --adaptive-retry-oracle \
  --adaptive-retry-step 100 \
  --adaptive-blur-decay 0.5 \
  --adaptive-patch-growth 0.15 \
  --adaptive-max-sharp-patch-fraction 2.0
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_pytorch.preprocess import preprocess
from PIL import Image, ImageDraw, ImageFilter
from torch import Tensor
from torchvision.datasets import MNIST

from _paths import repo_path
from canvit_rl.canvit_precision import resolve_canvit_dtype
from canvit_rl.env import get_device


IGNORE_LABEL = 255
DEFAULT_MODEL_REPO = (
    "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
)


def _resampling(name: str) -> int:
    """Return Pillow resampling constants across Pillow versions."""
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def _make_background(*, width: int, height: int) -> Image.Image:
    """Create a quiet noisy background so the digit is not the only texture."""
    base = np.random.randint(118, 142, size=(1, 1, 3), dtype=np.uint8)
    noise = np.random.normal(0.0, 9.0, size=(height, width, 3))
    pixels = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _digit_alpha(digit: Image.Image) -> Image.Image:
    """Build an alpha mask from the white-on-black MNIST digit image."""
    alpha = digit.convert("L")
    alpha = alpha.point(lambda value: 255 if value > 12 else 0)
    return alpha.filter(ImageFilter.GaussianBlur(radius=0.7))


def _low_contrast_digit(
    digit_rgb: Image.Image,
    *,
    background_value: int,
    alpha: float,
    blur_radius: float,
) -> Image.Image:
    """Return a weak global copy that remains hard to classify after downsampling."""
    gray = Image.new("RGB", digit_rgb.size, (background_value,) * 3)
    # Problem: an ordinary full-scene warmup can leak the class immediately.
    # Solution: make the faint global trace configurable separately from the
    # local patch. Result: the diagnostic can be made harder when t=0 remains
    # too recognizable.
    return Image.blend(gray, digit_rgb, alpha=alpha).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )


def generate_sample(
    *,
    digit_image: Image.Image,
    label: int,
    width: int,
    height: int,
    digit_size: int,
    blur_radius: float,
    global_digit_alpha: float,
    global_digit_blur_radius: float,
    sharp_digit_blur_radius: float,
    sharp_patch_fraction: float,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, float | int]]:
    """Create paired t0/oracle-t1 digit scenes and a foreground mask."""
    background = _make_background(width=width, height=height)
    digit = digit_image.convert("L").resize(
        (digit_size, digit_size),
        resample=_resampling("LANCZOS"),
    )
    alpha = _digit_alpha(digit)
    digit_rgb = Image.merge("RGB", (digit, digit, digit))

    margin = max(1, digit_size // 2)
    center_x = random.randint(margin, width - margin)
    center_y = random.randint(margin, height - margin)
    x0 = center_x - digit_size // 2
    y0 = center_y - digit_size // 2

    low_contrast = _low_contrast_digit(
        digit_rgb,
        background_value=130,
        alpha=global_digit_alpha,
        blur_radius=global_digit_blur_radius,
    )
    blurred_scene = background.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_scene.paste(low_contrast, (x0, y0), alpha)

    patch_side = max(1, int(round(digit_size * sharp_patch_fraction)))
    patch_x0 = max(0, center_x - patch_side // 2)
    patch_y0 = max(0, center_y - patch_side // 2)
    patch_x1 = min(width, patch_x0 + patch_side)
    patch_y1 = min(height, patch_y0 + patch_side)
    if patch_x1 - patch_x0 < patch_side:
        patch_x0 = max(0, patch_x1 - patch_side)
    if patch_y1 - patch_y0 < patch_side:
        patch_y0 = max(0, patch_y1 - patch_side)

    sharp_scene = blurred_scene.copy()
    sharp_digit_rgb = (
        digit_rgb.filter(ImageFilter.GaussianBlur(radius=sharp_digit_blur_radius))
        if sharp_digit_blur_radius > 0.0
        else digit_rgb
    )
    sharp_scene.paste(sharp_digit_rgb, (x0, y0), alpha)
    image = blurred_scene.copy()
    oracle_image = blurred_scene.copy()
    patch = sharp_scene.crop((patch_x0, patch_y0, patch_x1, patch_y1))
    # Problem: when the sharp patch lives in the same image as t0, the
    # full-scene downsample can leak the digit class. Solution: save a separate
    # oracle scene that reveals the patch only for the t1 diagnostic. Result:
    # t0 and the correct later viewpoint can be evaluated as distinct inputs.
    oracle_image.paste(patch, (patch_x0, patch_y0))

    mask = Image.new("L", (width, height), IGNORE_LABEL)
    digit_mask = Image.new("L", (digit_size, digit_size), int(label))
    mask.paste(
        digit_mask,
        (x0, y0),
        alpha.point(lambda value: 255 if value > 64 else 0),
    )

    metadata = {
        "label": int(label),
        "digit_size": digit_size,
        "global_digit_alpha": global_digit_alpha,
        "global_digit_blur_radius": global_digit_blur_radius,
        "sharp_digit_blur_radius": sharp_digit_blur_radius,
        "center_x": center_x / width,
        "center_y": center_y / height,
        "scale": patch_side / min(width, height),
        "digit_x0": x0,
        "digit_y0": y0,
        "digit_x1": x0 + digit_size,
        "digit_y1": y0 + digit_size,
        "sharp_x0": patch_x0,
        "sharp_y0": patch_y0,
        "sharp_x1": patch_x1,
        "sharp_y1": patch_y1,
    }
    return image, oracle_image, mask, metadata


def _indices_by_label(dataset: MNIST) -> dict[int, list[int]]:
    """Return MNIST row indices grouped by digit label."""
    grouped: dict[int, list[int]] = {label: [] for label in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        grouped[int(label)].append(idx)
    return grouped


def _parse_labels(value: str) -> list[int]:
    """Parse a comma-separated set of MNIST labels to generate."""
    try:
        labels = [int(label.strip()) for label in value.split(",") if label.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--labels must be comma-separated digits.") from exc
    if not labels:
        raise argparse.ArgumentTypeError("--labels must contain at least one digit.")
    if len(set(labels)) != len(labels):
        raise argparse.ArgumentTypeError("--labels must not contain duplicates.")
    invalid = [label for label in labels if label < 0 or label > 9]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"--labels contains non-MNIST labels: {invalid}"
        )
    return labels


def _sample_digit_size(args: argparse.Namespace) -> int:
    """Sample the per-scene digit size, defaulting to the fixed legacy size."""
    min_size = (
        args.min_digit_size if args.min_digit_size is not None else args.digit_size
    )
    max_size = (
        args.max_digit_size if args.max_digit_size is not None else args.digit_size
    )
    # Problem: fixed digit scale can let policies overfit one target box size.
    # Solution: draw a per-sample size from an inclusive integer range while
    # preserving the old --digit-size fixed behavior when no range is provided.
    # Result: metadata records the actual size used for each sample.
    return random.randint(min_size, max_size)


def _generate_candidate(
    *,
    args: argparse.Namespace,
    dataset: MNIST,
    label_indices: dict[int, list[int]] | None = None,
    label: int | None = None,
    sharp_digit_blur_radius: float | None = None,
    sharp_patch_fraction: float | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, float | int], int, int]:
    """Generate one candidate sample, optionally constrained to a digit label."""
    if label is None:
        mnist_idx = random.randrange(len(dataset))
    else:
        if label_indices is None:
            raise ValueError("label_indices is required when label is provided.")
        mnist_idx = random.choice(label_indices[label])
    digit_image, sampled_label = dataset[mnist_idx]
    digit_size = _sample_digit_size(args)
    image, oracle_image, mask, metadata = generate_sample(
        digit_image=digit_image,
        label=int(sampled_label),
        width=args.width,
        height=args.height,
        digit_size=digit_size,
        blur_radius=args.blur_radius,
        global_digit_alpha=args.global_digit_alpha,
        global_digit_blur_radius=args.global_digit_blur_radius,
        sharp_digit_blur_radius=(
            args.sharp_digit_blur_radius
            if sharp_digit_blur_radius is None
            else sharp_digit_blur_radius
        ),
        sharp_patch_fraction=(
            args.sharp_patch_fraction
            if sharp_patch_fraction is None
            else sharp_patch_fraction
        ),
    )
    return image, oracle_image, mask, metadata, int(sampled_label), int(mnist_idx)


def _adaptive_oracle_params(
    *,
    args: argparse.Namespace,
    label_attempts: int,
    support: bool = False,
) -> tuple[float, float]:
    """Return oracle patch settings that get clearer after repeated failures."""
    if not args.adaptive_retry_oracle:
        return args.sharp_digit_blur_radius, args.sharp_patch_fraction
    stage = (
        args.adaptive_retry_support_stage
        if support
        else label_attempts // args.adaptive_retry_step
    )
    # Problem: some labels can be invisible to the oracle-t1 centroid probe
    # under one fixed patch recipe. Solution: after repeated failures, make
    # only the oracle reveal crop sharper and slightly larger while leaving the
    # t0 image unchanged. Result: difficult digits can remain in the task
    # without leaking extra evidence into the full-scene warmup.
    blur = max(
        args.adaptive_min_sharp_digit_blur_radius,
        args.sharp_digit_blur_radius * (args.adaptive_blur_decay**stage),
    )
    patch_fraction = min(
        args.adaptive_max_sharp_patch_fraction,
        args.sharp_patch_fraction + args.adaptive_patch_growth * stage,
    )
    return blur, patch_fraction


class CanViTOracleFilter:
    """Keep samples where t0 is wrong but oracle t1 is right."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        support_dataset: MNIST,
    ) -> None:
        self.args = args
        self.device = get_device()
        self.transform = preprocess(args.width)
        self.model = CanViTForPretrainingHFHub.from_pretrained(args.model_repo).to(
            self.device
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.canvit_dtype = resolve_canvit_dtype(args.canvit_dtype, self.device)
        self.model.to(device=self.device, dtype=self.canvit_dtype)
        for module in self.model.modules():
            if module.__class__.__name__ == "VPEEncoder":
                module.to(device=self.device, dtype=torch.float32)
        self.canvas_grid_size = self.model.canvas_patch_grid_sizes[0]
        self.glimpse_size_px = int(
            args.glimpse_grid_size * self.model.backbone.patch_size_px
        )
        self.classes, self.centroids = self._build_support_centroids(support_dataset)

    def _metadata_to_viewpoint_tensors(
        self,
        metadata_rows: list[dict[str, float | int]],
    ) -> tuple[Tensor, Tensor]:
        """Convert metadata center/scale rows to CanViT Viewpoint tensors."""
        centers = torch.tensor(
            [
                [
                    2.0 * float(row["center_y"]) - 1.0,
                    2.0 * float(row["center_x"]) - 1.0,
                ]
                for row in metadata_rows
            ],
            dtype=torch.float32,
            device=self.device,
        )
        scales = torch.tensor(
            [float(row["scale"]) for row in metadata_rows],
            dtype=torch.float32,
            device=self.device,
        )
        return centers, scales

    @torch.inference_mode()
    def _extract_features(
        self,
        *,
        images: list[Image.Image],
        oracle_images: list[Image.Image],
        metadata_rows: list[dict[str, float | int]],
        oracle_t1: bool,
    ) -> Tensor:
        """Extract t0 or oracle-t1 recurrent_cls features for PIL images."""
        batch = torch.stack([self.transform(image) for image in images]).to(
            device=self.device
        )
        oracle_batch = torch.stack(
            [self.transform(image) for image in oracle_images]
        ).to(device=self.device)
        batch_size = batch.shape[0]
        state = self.model.init_state(
            batch_size=batch_size,
            canvas_grid_size=self.canvas_grid_size,
        )
        full_viewpoint = Viewpoint.full_scene(batch_size=batch_size, device=self.device)
        full_glimpse = sample_at_viewpoint(
            spatial=batch,
            viewpoint=full_viewpoint,
            glimpse_size_px=self.glimpse_size_px,
        ).to(dtype=self.canvit_dtype)
        out = self.model(glimpse=full_glimpse, state=state, viewpoint=full_viewpoint)
        if not oracle_t1:
            return out.state.recurrent_cls.squeeze(1).float().cpu()
        centers, scales = self._metadata_to_viewpoint_tensors(metadata_rows)
        oracle_viewpoint = Viewpoint(centers=centers, scales=scales)
        # Problem: the diagnostic needs the full-scene state to come from the
        # hidden t0 scene but the oracle glimpse to reveal the local patch.
        # Solution: sample t1 from the paired oracle scene while preserving the
        # t0 state. Result: filtering tests the intended active-view condition.
        oracle_glimpse = sample_at_viewpoint(
            spatial=oracle_batch,
            viewpoint=oracle_viewpoint,
            glimpse_size_px=self.glimpse_size_px,
        ).to(dtype=self.canvit_dtype)
        out = self.model(
            glimpse=oracle_glimpse,
            state=out.state,
            viewpoint=oracle_viewpoint,
        )
        return out.state.recurrent_cls.squeeze(1).float().cpu()

    def _build_support_centroids(self, dataset: MNIST) -> tuple[Tensor, Tensor]:
        """Build internal oracle-t1 class prototypes used only for filtering."""
        label_indices = _indices_by_label(dataset)
        images: list[Image.Image] = []
        oracle_images: list[Image.Image] = []
        metadata_rows: list[dict[str, float | int]] = []
        labels: list[int] = []
        for label in self.args.labels:
            for _ in range(self.args.filter_support_per_class):
                sharp_blur, patch_fraction = _adaptive_oracle_params(
                    args=self.args,
                    label_attempts=0,
                    support=True,
                )
                image, oracle_image, _, metadata, sampled_label, _ = _generate_candidate(
                    args=self.args,
                    dataset=dataset,
                    label_indices=label_indices,
                    label=label,
                    sharp_digit_blur_radius=sharp_blur,
                    sharp_patch_fraction=patch_fraction,
                )
                images.append(image)
                oracle_images.append(oracle_image)
                metadata_rows.append(metadata)
                labels.append(sampled_label)
        features = F.normalize(
            self._extract_features(
                images=images,
                oracle_images=oracle_images,
                metadata_rows=metadata_rows,
                oracle_t1=True,
            ),
            dim=1,
        )
        label_t = torch.tensor(labels, dtype=torch.long)
        classes = torch.tensor(self.args.labels, dtype=torch.long)
        centroids = []
        for label in classes.tolist():
            class_features = features[label_t == label]
            centroids.append(
                F.normalize(class_features.mean(dim=0, keepdim=True), dim=1)
            )
        return classes, torch.cat(centroids, dim=0)

    def _predict(self, features: Tensor) -> Tensor:
        """Classify features by cosine similarity to internal centroids."""
        scores = F.normalize(features.float(), dim=1) @ self.centroids.T
        return self.classes[scores.argmax(dim=1)]

    def accept(
        self,
        *,
        image: Image.Image,
        oracle_image: Image.Image,
        metadata: dict[str, float | int],
        label: int,
    ) -> tuple[bool, int, int]:
        """Return whether t0 is wrong and oracle t1 is correct for one sample."""
        t0_feature = self._extract_features(
            images=[image],
            oracle_images=[oracle_image],
            metadata_rows=[metadata],
            oracle_t1=False,
        )
        t1_feature = self._extract_features(
            images=[image],
            oracle_images=[oracle_image],
            metadata_rows=[metadata],
            oracle_t1=True,
        )
        t0_pred = int(self._predict(t0_feature)[0].item())
        t1_pred = int(self._predict(t1_feature)[0].item())
        return t0_pred != label and t1_pred == label, t0_pred, t1_pred


def _write_preview(
    *,
    rows: list[dict[str, float | int | str]],
    image_dir: Path,
    output_path: Path,
    tile_size: int,
) -> None:
    """Save a small contact sheet for quick visual sanity checks."""
    preview_rows = rows[: min(16, len(rows))]
    if not preview_rows:
        return
    cols = 4
    rows_count = int(np.ceil(len(preview_rows) / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows_count * tile_size), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(preview_rows):
        image = Image.open(image_dir / str(row["sample"])).convert("RGB")
        image.thumbnail((tile_size, tile_size), resample=_resampling("BICUBIC"))
        x = (idx % cols) * tile_size
        y = (idx // cols) * tile_size
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + 4), str(row["label"]), fill=(255, 40, 40))
    canvas.save(output_path)


def _generate_split(
    *,
    args: argparse.Namespace,
    split: str,
    train: bool,
    count: int,
    canvit_filter: CanViTOracleFilter | None,
) -> None:
    """Generate one split under images, oracle_images, masks, and metadata."""
    dataset = MNIST(root=args.mnist_root, train=train, download=args.download)
    label_indices = _indices_by_label(dataset)
    label_schedule = list(args.labels)
    random.shuffle(label_schedule)
    image_dir = args.root / "images" / split
    oracle_image_dir = args.root / "oracle_images" / split
    mask_dir = args.root / "masks" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    oracle_image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    accepted_by_label: dict[int, int] = {label: 0 for label in label_schedule}
    attempted_by_label: dict[int, int] = {label: 0 for label in label_schedule}
    attempts = 0
    max_attempts = max(count * args.filter_max_attempts_per_sample, count)
    while len(rows) < count:
        attempts += 1
        if canvit_filter is not None and attempts > max_attempts:
            accepted_summary = ", ".join(
                f"{label}:{accepted_by_label[label]}/{attempted_by_label[label]}"
                for label in label_schedule
            )
            raise RuntimeError(
                f"Could only accept {len(rows)}/{count} {split} samples after "
                f"{max_attempts} attempts. Relax blur/size settings or raise "
                "--filter-max-attempts-per-sample. Per-digit accepted/attempted: "
                f"{accepted_summary}"
            )
        # Problem: purely random MNIST sampling can leave some digits
        # under-represented, making per-class t0/t1 diagnostics look unstable.
        # Solution: target labels with remaining quota, cycling by attempt
        # rather than by accepted row. Result: one hard digit no longer blocks
        # all other digits while the filter searches for valid examples.
        base_quota = count // len(label_schedule)
        extra_quota = count % len(label_schedule)
        target_quotas = {
            label: base_quota + (1 if idx < extra_quota else 0)
            for idx, label in enumerate(label_schedule)
        }
        remaining_labels = [
            label
            for label in label_schedule
            if accepted_by_label[label] < target_quotas[label]
        ]
        target_label = remaining_labels[(attempts - 1) % len(remaining_labels)]
        attempted_by_label[target_label] += 1
        sharp_blur, patch_fraction = _adaptive_oracle_params(
            args=args,
            label_attempts=attempted_by_label[target_label],
        )
        image, oracle_image, mask, metadata, label, mnist_idx = _generate_candidate(
            args=args,
            dataset=dataset,
            label_indices=label_indices,
            label=target_label,
            sharp_digit_blur_radius=sharp_blur,
            sharp_patch_fraction=patch_fraction,
        )
        if canvit_filter is not None:
            accepted, t0_pred, t1_pred = canvit_filter.accept(
                image=image,
                oracle_image=oracle_image,
                metadata=metadata,
                label=label,
            )
            metadata = {
                **metadata,
                "filter_t0_pred": t0_pred,
                "filter_t1_pred": t1_pred,
            }
            # Problem: a synthetic sample is only useful for this diagnostic if
            # the full-scene state fails but the correct second glimpse works.
            # Solution: reject candidates unless the CanViT support-centroid
            # probe sees t0 wrong and oracle-t1 right. Result: generated sets
            # are biased toward active-view-positive examples.
            if not accepted:
                if args.canvit_filter and attempts % args.filter_progress_interval == 0:
                    accepted_summary = ", ".join(
                        f"{digit}:{accepted_by_label[digit]}/"
                        f"{attempted_by_label[digit]}"
                        for digit in label_schedule
                    )
                    print(
                        f"{split} filter progress: attempts={attempts} "
                        f"accepted={len(rows)}/{count} "
                        f"per_digit={accepted_summary}"
                    )
                continue
        output_idx = len(rows)
        stem = f"sample_{output_idx:05d}_digit{int(label)}.png"
        image.save(image_dir / stem)
        oracle_image.save(oracle_image_dir / stem)
        mask.save(mask_dir / stem)
        rows.append(
            {
                "sample": stem,
                "split": split,
                "mnist_index": int(mnist_idx),
                **metadata,
            }
        )
        accepted_by_label[label] += 1
        if args.canvit_filter and attempts % args.filter_progress_interval == 0:
            accepted_summary = ", ".join(
                f"{digit}:{accepted_by_label[digit]}/{attempted_by_label[digit]}"
                for digit in label_schedule
            )
            print(
                f"{split} filter progress: attempts={attempts} "
                f"accepted={len(rows)}/{count} per_digit={accepted_summary}"
            )

    if args.canvit_filter:
        print(f"{split} CanViT filter attempts: {attempts} accepted: {count}")
    metadata_path = args.root / f"metadata_{split}.csv"
    with metadata_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_preview(
        rows=rows,
        image_dir=image_dir,
        output_path=args.root / f"preview_{split}.png",
        tile_size=160,
    )
    _write_preview(
        rows=rows,
        image_dir=oracle_image_dir,
        output_path=args.root / f"preview_oracle_{split}.png",
        tile_size=160,
    )
    print(f"Saved {count} {split} samples")
    print(f"Images:        {image_dir}")
    print(f"Oracle images: {oracle_image_dir}")
    print(f"Masks:         {mask_dir}")
    print(f"Metadata:      {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse synthetic MNIST glimpse dataset arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-root", type=Path, default=Path("datasets/mnist"))
    parser.add_argument("--root", type=Path, default=Path("datasets/mnist_glimpse"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument(
        "--labels",
        type=_parse_labels,
        default=list(range(10)),
        help="Comma-separated MNIST labels to generate, e.g. 0,1,2,3,4,6,7,8,9.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--digit-size", type=int, default=96)
    parser.add_argument(
        "--min-digit-size",
        type=int,
        default=None,
        help="Minimum randomized digit size in pixels; defaults to --digit-size.",
    )
    parser.add_argument(
        "--max-digit-size",
        type=int,
        default=None,
        help="Maximum randomized digit size in pixels; defaults to --digit-size.",
    )
    parser.add_argument("--blur-radius", type=float, default=80.0)
    parser.add_argument(
        "--global-digit-blur-radius",
        type=float,
        default=12.0,
        help="Blur radius for the faint full-scene digit trace.",
    )
    parser.add_argument(
        "--global-digit-alpha",
        type=float,
        default=0.06,
        help="Blend strength for the faint full-scene digit trace.",
    )
    parser.add_argument(
        "--sharp-digit-blur-radius",
        type=float,
        default=0.0,
        help="Optional blur applied to the local sharp digit patch.",
    )
    parser.add_argument(
        "--sharp-patch-fraction",
        type=float,
        default=1.15,
        help="Sharp crop side length as a multiple of the resized digit size.",
    )
    parser.add_argument(
        "--canvit-filter",
        action="store_true",
        help=(
            "Keep generating until samples satisfy t0 wrong and oracle-t1 "
            "correct under a frozen CanViT support-centroid probe."
        ),
    )
    parser.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO)
    parser.add_argument("--glimpse-grid-size", type=int, default=8)
    parser.add_argument(
        "--canvit-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--filter-support-per-class", type=int, default=5)
    parser.add_argument("--filter-max-attempts-per-sample", type=int, default=200)
    parser.add_argument(
        "--filter-progress-interval",
        type=int,
        default=100,
        help="Print CanViT filter accepted/attempted counts every N attempts.",
    )
    parser.add_argument(
        "--adaptive-retry-oracle",
        action="store_true",
        help=(
            "During filtered generation, make oracle patches sharper/larger for "
            "labels that repeatedly fail while leaving the t0 image unchanged."
        ),
    )
    parser.add_argument("--adaptive-retry-step", type=int, default=200)
    parser.add_argument("--adaptive-blur-decay", type=float, default=0.5)
    parser.add_argument("--adaptive-patch-growth", type=float, default=0.15)
    parser.add_argument("--adaptive-min-sharp-digit-blur-radius", type=float, default=0.0)
    parser.add_argument("--adaptive-max-sharp-patch-fraction", type=float, default=1.8)
    parser.add_argument("--adaptive-retry-support-stage", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    # Problem: moved nested scripts inherit the caller's cwd for relative
    # dataset/output paths, which can accidentally create duplicate trees under
    # scripts/synthetic_dataset. Solution: anchor relative paths at repo root.
    # Result: defaults and examples keep writing to the same project folders.
    args.mnist_root = repo_path(args.mnist_root)
    args.root = repo_path(args.root)
    if args.train_samples <= 0 or args.val_samples <= 0:
        raise ValueError("--train-samples and --val-samples must be positive.")
    if args.train_samples < len(args.labels) or args.val_samples < len(args.labels):
        raise ValueError(
            "--train-samples and --val-samples must each be at least the number "
            "of selected --labels so every class can appear."
        )
    min_digit_size = (
        args.min_digit_size if args.min_digit_size is not None else args.digit_size
    )
    max_digit_size = (
        args.max_digit_size if args.max_digit_size is not None else args.digit_size
    )
    if min_digit_size <= 0 or max_digit_size <= 0:
        raise ValueError("Digit sizes must be positive.")
    if min_digit_size > max_digit_size:
        raise ValueError("--min-digit-size must be <= --max-digit-size.")
    if max_digit_size > min(args.width, args.height):
        raise ValueError("The maximum digit size must fit inside the output canvas.")
    if args.blur_radius < 0.0 or args.global_digit_blur_radius < 0.0:
        raise ValueError("Blur radii must be non-negative.")
    if args.sharp_digit_blur_radius < 0.0:
        raise ValueError("--sharp-digit-blur-radius must be non-negative.")
    if not 0.0 <= args.global_digit_alpha <= 1.0:
        raise ValueError("--global-digit-alpha must be in [0, 1].")
    if args.sharp_patch_fraction <= 0.0:
        raise ValueError("--sharp-patch-fraction must be positive.")
    if args.canvit_filter and args.width != args.height:
        raise ValueError("--canvit-filter currently requires square output images.")
    if args.filter_support_per_class <= 0:
        raise ValueError("--filter-support-per-class must be positive.")
    if args.filter_max_attempts_per_sample <= 0:
        raise ValueError("--filter-max-attempts-per-sample must be positive.")
    if args.filter_progress_interval <= 0:
        raise ValueError("--filter-progress-interval must be positive.")
    if args.adaptive_retry_step <= 0:
        raise ValueError("--adaptive-retry-step must be positive.")
    if not 0.0 < args.adaptive_blur_decay <= 1.0:
        raise ValueError("--adaptive-blur-decay must be in (0, 1].")
    if args.adaptive_patch_growth < 0.0:
        raise ValueError("--adaptive-patch-growth must be non-negative.")
    if args.adaptive_min_sharp_digit_blur_radius < 0.0:
        raise ValueError("--adaptive-min-sharp-digit-blur-radius must be non-negative.")
    if args.adaptive_max_sharp_patch_fraction < args.sharp_patch_fraction:
        raise ValueError(
            "--adaptive-max-sharp-patch-fraction must be >= --sharp-patch-fraction."
        )
    if args.adaptive_retry_support_stage < 0:
        raise ValueError("--adaptive-retry-support-stage must be non-negative.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.root.mkdir(parents=True, exist_ok=True)
    canvit_filter = (
        CanViTOracleFilter(
            args=args,
            support_dataset=MNIST(
                root=args.mnist_root,
                train=True,
                download=args.download,
            ),
        )
        if args.canvit_filter
        else None
    )
    _generate_split(
        args=args,
        split="training",
        train=True,
        count=args.train_samples,
        canvit_filter=canvit_filter,
    )
    _generate_split(
        args=args,
        split="validation",
        train=False,
        count=args.val_samples,
        canvit_filter=canvit_filter,
    )


if __name__ == "__main__":
    main()
