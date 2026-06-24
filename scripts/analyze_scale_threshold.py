"""Analyze when embedded crop size stops making centered gazes useful.

This sweeps the **scale of the ADE crop pasted into a padded scene** and, if
requested, the zoom used to crop into the original ADE image before embedding.
It does not sweep the policy/viewpoint scale. For each sampled ADE image, it
creates several synthetic scenes with matched source content embedded at
different canvas fractions, runs a full-scene warm-up at t0, then commits one
centered follow-up glimpse at t1 and measures:

    CE gain = CE_before - CE_after

Positive gain means looking at the embedded area lowered CE. Negative gain
means the centered follow-up made segmentation worse.

Example:
    uv run python scripts/analyze_scale_threshold.py \
        --ade-root datasets/ADE20k \
        --split training \
        --source-index 1 \
        --crop-fractions 0.25,0.35,0.50,0.65,0.80,0.90 \
        --source-zooms 1.0,1.5,2.0,3.0,4.0 \
        --source-zoom-position center \
        --view-scale 0.5 \
        --output-dir results/crop_scale_threshold \
        --plot \
        --plot-source-scenes

    uv run python scripts/analyze_scale_threshold.py \
        --ade-root datasets/ADE20k \
        --split training \
        --max-sources 100 \
        --crop-fractions 0.05,0.10,0.15,0.25,0.35,0.50,0.65,0.80 \
        --source-zooms 1.0,1.5,2.0,3.0,4.0 \
        --view-scale 0.50 \
        --output-dir results/crop_scale_threshold \
        --plot
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_specialize.datasets.ade20k import IGNORE_LABEL, make_val_transforms
from PIL import Image
from tqdm import tqdm

from canvit_rl.ade_labels import remap_ade_mask_labels
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import _segmentation_cross_entropy_losses


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_pairs(*, ade_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Return matched ADE image/mask pairs for one split."""
    image_dir = ade_root / "images" / split
    mask_dir = ade_root / "annotations" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"ADE image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"ADE mask directory not found: {mask_dir}")
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


def _parse_fractions(value: str) -> list[float]:
    """Parse comma-separated crop fractions in (0, 1]."""
    fractions = [float(item) for item in value.split(",") if item.strip()]
    if not fractions or any(fraction <= 0 or fraction > 1 for fraction in fractions):
        raise ValueError("--crop-fractions must contain values in (0, 1].")
    return sorted(fractions)


def _parse_zooms(value: str) -> list[float]:
    """Parse comma-separated source zooms in [1, inf)."""
    zooms = [float(item) for item in value.split(",") if item.strip()]
    if not zooms or any(zoom < 1 for zoom in zooms):
        raise ValueError("--source-zooms must contain values >= 1.")
    return sorted(zooms)


def _image_for_plot(image: torch.Tensor) -> np.ndarray:
    """Convert one normalized CHW tensor to HWC numpy image."""
    image_cpu = (image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    return image_cpu.permute(1, 2, 0).numpy()


def _zoom_source(
    *,
    image: Image.Image,
    mask: Image.Image,
    source_zoom: float,
    zoom_position: str,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Crop into the source ADE image/mask before canvas embedding."""
    source_width, source_height = image.size
    if source_zoom <= 1.0:
        return image, mask, (0, 0, source_width, source_height)
    crop_width = max(1, int(round(source_width / source_zoom)))
    crop_height = max(1, int(round(source_height / source_zoom)))
    max_x = source_width - crop_width
    max_y = source_height - crop_height
    if zoom_position == "center":
        x0 = max_x // 2
        y0 = max_y // 2
    else:
        x0 = rng.randint(0, max(max_x, 0))
        y0 = rng.randint(0, max(max_y, 0))
    box = (x0, y0, x0 + crop_width, y0 + crop_height)
    return image.crop(box), mask.crop(box), box


def _load_zoomed_source(
    *,
    image_path: Path,
    mask_path: Path,
    source_zoom: float,
    zoom_position: str,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Load one ADE image/mask and crop it once for a zoom condition."""
    source_image = Image.open(image_path).convert("RGB")
    source_mask = Image.fromarray(
        remap_ade_mask_labels(
            np.asarray(Image.open(mask_path).convert("L")),
            raw_ade=True,
        ).astype(np.uint8),
        mode="L",
    )
    return _zoom_source(
        image=source_image,
        mask=source_mask,
        source_zoom=source_zoom,
        zoom_position=zoom_position,
        rng=rng,
    )


def _resize_to_fraction(
    *,
    image: Image.Image,
    mask: Image.Image,
    canvas_size: int,
    crop_fraction: float,
) -> tuple[Image.Image, Image.Image]:
    """Resize source image/mask so longest side equals crop_fraction of canvas."""
    source_width, source_height = image.size
    target_long_side = max(1, int(round(canvas_size * crop_fraction)))
    scale = target_long_side / max(source_width, source_height)
    target_width = max(1, min(canvas_size, int(round(source_width * scale))))
    target_height = max(1, min(canvas_size, int(round(source_height * scale))))
    resampling = getattr(Image, "Resampling", Image)
    return (
        image.resize((target_width, target_height), resample=resampling.BICUBIC),
        mask.resize((target_width, target_height), resample=resampling.NEAREST),
    )


def _make_scene(
    *,
    source_image: Image.Image,
    source_mask: Image.Image,
    canvas_size: int,
    crop_fraction: float,
    position: str,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Embed one ADE source at a requested crop fraction."""
    embedded_image, embedded_mask = _resize_to_fraction(
        image=source_image,
        mask=source_mask,
        canvas_size=canvas_size,
        crop_fraction=crop_fraction,
    )
    canvas = Image.new("RGB", (canvas_size, canvas_size), (127, 127, 127))
    mask_canvas = Image.new("L", (canvas_size, canvas_size), IGNORE_LABEL)
    max_x = canvas_size - embedded_image.size[0]
    max_y = canvas_size - embedded_image.size[1]
    if position == "center":
        x0 = max_x // 2
        y0 = max_y // 2
    else:
        x0 = rng.randint(0, max(max_x, 0))
        y0 = rng.randint(0, max(max_y, 0))
    canvas.paste(embedded_image, (x0, y0))
    mask_canvas.paste(embedded_mask, (x0, y0))
    bbox = (x0, y0, x0 + embedded_image.size[0], y0 + embedded_image.size[1])
    return canvas, mask_canvas, bbox


def _viewpoint_for_bbox(
    *,
    bbox: tuple[int, int, int, int],
    canvas_size: int,
    view_scale: float,
    device: torch.device,
) -> Viewpoint:
    """Create a centered viewpoint for an embedded crop bbox."""
    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    cx = center_x / canvas_size * 2.0 - 1.0
    cy = center_y / canvas_size * 2.0 - 1.0
    bound = max(1.0 - view_scale, 0.0)
    centers = torch.tensor(
        [[float(np.clip(cy, -bound, bound)), float(np.clip(cx, -bound, bound))]],
        device=device,
    )
    scales = torch.tensor([view_scale], device=device)
    return Viewpoint(centers=centers, scales=scales)


def _segmentation_ce(
    *,
    model,
    probe: torch.nn.Module,
    state,
    mask: torch.Tensor,
    cfg: CanViTEnvConfig,
) -> torch.Tensor:
    """Return per-image CE loss from a CanViT state."""
    return _segmentation_cross_entropy_losses(
        model=model,
        state=state,
        probe=probe,
        canvas_grid_size=cfg.canvas_grid_size,
        mask=mask,
        batch_size=mask.shape[0],
    )


def _plot_scene_grid(
    *,
    rows_by_zoom: dict[float, list[dict]],
    scene_sheets: dict[float, list[np.ndarray]],
    output: Path,
) -> None:
    """Plot padded whole-scene inputs in one zoom-by-fraction grid."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError("Install matplotlib or rerun without --plot.") from exc

    zooms = sorted(rows_by_zoom)
    n_rows = len(zooms)
    n_cols = max(len(rows_by_zoom[zoom]) for zoom in zooms)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 3.3 * n_rows),
        dpi=150,
        squeeze=False,
    )
    first_row = rows_by_zoom[zooms[0]][0]
    for row_idx, source_zoom in enumerate(zooms):
        rows = rows_by_zoom[source_zoom]
        scenes = scene_sheets[source_zoom]
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(rows):
                ax.set_axis_off()
                continue
            row = rows[col_idx]
            scene = scenes[col_idx]
            ax.imshow(scene)
            x0, y0 = row["bbox_x0"], row["bbox_y0"]
            width = row["bbox_x1"] - row["bbox_x0"]
            height = row["bbox_y1"] - row["bbox_y0"]
            canvas_height, canvas_width = scene.shape[:2]
            crop_center_x = (row["bbox_x0"] + row["bbox_x1"]) * 0.5
            crop_center_y = (row["bbox_y0"] + row["bbox_y1"]) * 0.5
            view_scale = row["view_scale"]
            view_bound = max(1.0 - view_scale, 0.0)
            view_cx_norm = crop_center_x / canvas_width * 2.0 - 1.0
            view_cy_norm = crop_center_y / canvas_height * 2.0 - 1.0
            view_cx = (
                float(np.clip(view_cx_norm, -view_bound, view_bound)) + 1.0
            ) * 0.5 * canvas_width
            view_cy = (
                float(np.clip(view_cy_norm, -view_bound, view_bound)) + 1.0
            ) * 0.5 * canvas_height
            view_width = canvas_width * view_scale
            view_height = canvas_height * view_scale
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    fill=False,
                    linewidth=1.6,
                    edgecolor="white",
                )
            )
            ax.add_patch(
                Rectangle(
                    (view_cx - view_width * 0.5, view_cy - view_height * 0.5),
                    view_width,
                    view_height,
                    fill=False,
                    linewidth=1.6,
                    linestyle="--",
                    edgecolor="cyan",
                )
            )
            # Fixed by Codex on 2026-06-24
            # Problem: Separate per-zoom PNGs made it hard to compare source
            # zoom, pasted crop fraction, and actual t1 viewpoint in one
            # glance.
            # Solution: render one grid per source image, with source zooms as
            # rows, canvas crop fractions as columns, and overlay the centered
            # t1 viewpoint using the same clipped center/scale convention as
            # the CanViT action.
            # Result: The visual context for every t0/t1 measurement is in a
            # single figure, including what the follow-up gaze sees.
            ax.set_title(
                f"zoom={row['source_zoom']:.1f}x crop={row['crop_fraction']:.2f}\n"
                f"t0={row['ce_t0']:.3f}  t1={row['ce_t1']:.3f}",
                fontsize=9,
            )
            ax.set_axis_off()
    fig.suptitle(
        f"source={first_row['source_index']} | white=crop bbox | cyan dashed=t1 viewpoint, scale={first_row['view_scale']:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _summarize_by_fraction(rows: list[dict]) -> list[dict]:
    """Aggregate t0/t1 CE and CE gain by source zoom and crop fraction."""
    summary = []
    keys = sorted({(row["source_zoom"], row["crop_fraction"]) for row in rows})
    for source_zoom, fraction in keys:
        group = [
            row
            for row in rows
            if row["source_zoom"] == source_zoom and row["crop_fraction"] == fraction
        ]
        ce_t0 = np.asarray([row["ce_t0"] for row in group], dtype=np.float64)
        ce_t1 = np.asarray([row["ce_t1"] for row in group], dtype=np.float64)
        ce_gain = np.asarray([row["ce_gain"] for row in group], dtype=np.float64)
        ddof = 1 if len(group) > 1 else 0
        summary.append(
            {
                "source_zoom": source_zoom,
                "crop_fraction": fraction,
                "n": len(group),
                "ce_t0_mean": float(ce_t0.mean()),
                "ce_t0_std": float(ce_t0.std(ddof=ddof)),
                "ce_t1_mean": float(ce_t1.mean()),
                "ce_t1_std": float(ce_t1.std(ddof=ddof)),
                "ce_gain_mean": float(ce_gain.mean()),
                "ce_gain_std": float(ce_gain.std(ddof=ddof)),
            }
        )
    return summary


def _plot_aggregate_results(
    *,
    summary_rows: list[dict],
    output: Path,
) -> None:
    """Plot mean/std CE at t0 and t1 across source images."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib or rerun without --plot.") from exc

    zooms = sorted({row["source_zoom"] for row in summary_rows})
    colors = dict(zip(zooms, plt.get_cmap("tab10").colors, strict=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    for source_zoom in zooms:
        zoom_rows = sorted(
            [row for row in summary_rows if row["source_zoom"] == source_zoom],
            key=lambda row: row["crop_fraction"],
        )
        fractions = np.asarray(
            [row["crop_fraction"] for row in zoom_rows], dtype=np.float64
        )
        t0_mean = np.asarray(
            [row["ce_t0_mean"] for row in zoom_rows], dtype=np.float64
        )
        t0_std = np.asarray([row["ce_t0_std"] for row in zoom_rows], dtype=np.float64)
        t1_mean = np.asarray(
            [row["ce_t1_mean"] for row in zoom_rows], dtype=np.float64
        )
        t1_std = np.asarray([row["ce_t1_std"] for row in zoom_rows], dtype=np.float64)
        gain_mean = np.asarray(
            [row["ce_gain_mean"] for row in zoom_rows], dtype=np.float64
        )
        gain_std = np.asarray(
            [row["ce_gain_std"] for row in zoom_rows], dtype=np.float64
        )
        color = colors[source_zoom]
        axes[0].plot(
            fractions,
            t0_mean,
            marker="o",
            linestyle="--",
            color=color,
            label=f"t0 zoom {source_zoom:g}x",
        )
        axes[0].fill_between(
            fractions,
            t0_mean - t0_std,
            t0_mean + t0_std,
            color=color,
            alpha=0.05,
        )
        axes[0].plot(
            fractions,
            t1_mean,
            marker="o",
            linestyle="-",
            color=color,
            label=f"t1 zoom {source_zoom:g}x",
        )
        axes[0].fill_between(
            fractions,
            t1_mean - t1_std,
            t1_mean + t1_std,
            color=color,
            alpha=0.05,
        )
        axes[1].plot(
            fractions,
            gain_mean,
            marker="o",
            color=color,
            label=f"zoom {source_zoom:g}x",
        )
        axes[1].fill_between(
            fractions,
            gain_mean - gain_std,
            gain_mean + gain_std,
            color=color,
            alpha=0.12,
        )
    axes[0].set_xlabel("embedded crop fraction of canvas")
    axes[0].set_ylabel("CE loss")
    axes[0].set_title("mean CE loss with std band")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xlabel("embedded crop fraction of canvas")
    axes[1].set_ylabel("CE gain = t0 CE - t1 CE")
    axes[1].set_title("positive means t1 gaze helped")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade-root", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument("--source-index", type=int, action="append", default=None)
    parser.add_argument("--max-sources", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--crop-fractions",
        type=str,
        default="0.10,0.15,0.20,0.25,0.35,0.50,0.65,0.80",
    )
    parser.add_argument(
        "--source-zooms",
        type=str,
        default="1.0",
        help="Comma-separated source crop zooms before embedding, e.g. 1,1.5,2,3.",
    )
    parser.add_argument(
        "--source-zoom-position",
        choices=["center", "random"],
        default="center",
        help="Where to crop the zoomed ADE source region.",
    )
    parser.add_argument(
        "--view-scale",
        type=float,
        default=0.50,
        help="Fixed follow-up viewpoint scale centered on the embedded crop.",
    )
    parser.add_argument(
        "--position",
        choices=["center", "random"],
        default="center",
        help="Where to paste the embedded crop on the padded canvas.",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/crop_scale_threshold"))
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save aggregate mean/std CE plots across selected source images.",
    )
    parser.add_argument(
        "--plot-source-scenes",
        action="store_true",
        help="Also save one diagnostic scene grid per source image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sources < 1:
        raise ValueError("--max-sources must be positive.")
    if not 0 < args.view_scale <= 1:
        raise ValueError("--view-scale must be in (0, 1].")
    crop_fractions = _parse_fractions(args.crop_fractions)
    source_zooms = _parse_zooms(args.source_zooms)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = CanViTEnvConfig()
    device = get_device()
    pairs = _load_pairs(ade_root=args.ade_root, split=args.split)
    if args.source_index:
        source_indices = args.source_index
    else:
        source_indices = rng.sample(range(len(pairs)), min(args.max_sources, len(pairs)))

    probe_repo = args.probe_repo or resolve_canvit_repo(
        f"probe-ade20k-40k-s512-c{cfg.canvas_grid_size}-in21k"
    )
    print(f"Loading CanViT segmentation model with probe: {probe_repo}")
    seg = (
        CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo=cfg.checkpoint,
            probe_repo=probe_repo,
        )
        .eval()
        .to(device)
    )
    model = seg.canvit
    probe = seg.head
    for module in (model, probe):
        for param in module.parameters():
            param.requires_grad_(False)

    img_tf, _ = make_val_transforms(cfg.scene_size_px, mode="squish")
    resampling = getattr(Image, "Resampling", Image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    with torch.inference_mode():
        for source_index in tqdm(source_indices, desc="Analyzing crop scale"):
            image_path, mask_path = pairs[source_index]
            source_rows = []
            scene_sheets: dict[float, list[np.ndarray]] = {}
            rows_by_zoom: dict[float, list[dict]] = {}
            for source_zoom in source_zooms:
                zoomed_image, zoomed_mask, source_box = _load_zoomed_source(
                    image_path=image_path,
                    mask_path=mask_path,
                    source_zoom=source_zoom,
                    zoom_position=args.source_zoom_position,
                    rng=rng,
                )
                zoom_rows = []
                zoom_scenes = []
                for crop_fraction in crop_fractions:
                    image_pil, mask_pil, bbox = _make_scene(
                        source_image=zoomed_image,
                        source_mask=zoomed_mask,
                        canvas_size=cfg.scene_size_px,
                        crop_fraction=crop_fraction,
                        position=args.position,
                        rng=rng,
                    )
                    if args.plot_source_scenes:
                        zoom_scenes.append(
                            np.asarray(image_pil).astype(np.float32) / 255.0
                        )
                    image = img_tf(image_pil)
                    mask_pil = mask_pil.resize(
                        (cfg.scene_size_px, cfg.scene_size_px),
                        resample=resampling.NEAREST,
                    )
                    mask = torch.from_numpy(
                        remap_ade_mask_labels(np.asarray(mask_pil)).astype(np.int64)
                    )
                    image_dev = image.unsqueeze(0).to(device)
                    mask_dev = mask.unsqueeze(0).to(device)
                    state = model.init_state(
                        batch_size=1,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                    full_vp = Viewpoint.full_scene(batch_size=1, device=device)
                    full_out = model(
                        glimpse=sample_at_viewpoint(
                            spatial=image_dev,
                            viewpoint=full_vp,
                            glimpse_size_px=cfg.glimpse_size_px,
                        ),
                        state=state,
                        viewpoint=full_vp,
                    )
                    ce_before = float(
                        _segmentation_ce(
                            model=model,
                            probe=probe,
                            state=full_out.state,
                            mask=mask_dev,
                            cfg=cfg,
                        ).item()
                    )
                    vp = _viewpoint_for_bbox(
                        bbox=bbox,
                        canvas_size=cfg.scene_size_px,
                        view_scale=args.view_scale,
                        device=device,
                    )
                    # Fixed by Codex on 2026-06-24
                    # Problem: The intended threshold is about embedded crop
                    # size and source detail, not action scale.
                    # Solution: keep the follow-up viewpoint scale fixed,
                    # sweep pasted crop fraction, and optionally crop-zoom the
                    # original ADE source before embedding.
                    # Result: CE gain separates "too small on canvas" from
                    # "source scene too compressed to recover segmentation."
                    out = model(
                        glimpse=sample_at_viewpoint(
                            spatial=image_dev,
                            viewpoint=vp,
                            glimpse_size_px=cfg.glimpse_size_px,
                        ),
                        state=full_out.state,
                        viewpoint=vp,
                    )
                    ce_after = float(
                        _segmentation_ce(
                            model=model,
                            probe=probe,
                            state=out.state,
                            mask=mask_dev,
                            cfg=cfg,
                        ).item()
                    )
                    x0, y0, x1, y1 = bbox
                    sx0, sy0, sx1, sy1 = source_box
                    row = {
                        "source_index": source_index,
                        "source_stem": image_path.stem,
                        "source_zoom": source_zoom,
                        "crop_fraction": crop_fraction,
                        "view_scale": args.view_scale,
                        "ce_t0": ce_before,
                        "ce_t1": ce_after,
                        "ce_before": ce_before,
                        "ce_after": ce_after,
                        "ce_gain": ce_before - ce_after,
                        "bbox_x0": x0,
                        "bbox_y0": y0,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "source_crop_x0": sx0,
                        "source_crop_y0": sy0,
                        "source_crop_x1": sx1,
                        "source_crop_y1": sy1,
                        "position": args.position,
                        "source_zoom_position": args.source_zoom_position,
                    }
                    zoom_rows.append(row)
                    source_rows.append(row)
                    all_rows.append(row)
                rows_by_zoom[source_zoom] = zoom_rows
                if args.plot_source_scenes:
                    scene_sheets[source_zoom] = zoom_scenes

            best = max(source_rows, key=lambda row: row["ce_gain"])
            positives = [
                row["crop_fraction"] for row in source_rows if row["ce_gain"] > 0
            ]
            min_positive = min(positives) if positives else float("nan")
            print(
                f"source={source_index} {image_path.stem} "
                f"best_zoom={best['source_zoom']:.3f} "
                f"best_fraction={best['crop_fraction']:.3f} "
                f"best_gain={best['ce_gain']:+.4f} "
                f"min_positive_fraction={min_positive}"
            )
            if args.plot_source_scenes:
                _plot_scene_grid(
                    rows_by_zoom=rows_by_zoom,
                    scene_sheets=scene_sheets,
                    output=args.output_dir / f"crop_scale_scenes_{source_index:05d}.png",
                )

    csv_path = args.output_dir / "crop_scale_threshold.csv"
    fieldnames = [
        "source_index",
        "source_stem",
        "source_zoom",
        "crop_fraction",
        "view_scale",
        "ce_t0",
        "ce_t1",
        "ce_before",
        "ce_after",
        "ce_gain",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "source_crop_x0",
        "source_crop_y0",
        "source_crop_x1",
        "source_crop_y1",
        "position",
        "source_zoom_position",
    ]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {csv_path}")

    summary_rows = _summarize_by_fraction(all_rows)
    summary_path = args.output_dir / "crop_scale_threshold_summary.csv"
    summary_fieldnames = [
        "source_zoom",
        "crop_fraction",
        "n",
        "ce_t0_mean",
        "ce_t0_std",
        "ce_t1_mean",
        "ce_t1_std",
        "ce_gain_mean",
        "ce_gain_std",
    ]
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {summary_path}")
    if args.plot:
        _plot_aggregate_results(
            summary_rows=summary_rows,
            output=args.output_dir / "crop_scale_threshold_summary.png",
        )


if __name__ == "__main__":
    main()
