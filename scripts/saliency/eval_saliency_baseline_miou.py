"""
Evaluate cached saliency-map viewpoint heuristics on ADE20K mIoU.

The rollout starts with a full-scene glimpse, then selects --t saliency-guided
glimpses using average saliency inside candidate windows with simple NMS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if "--comet" in sys.argv:
    try:
        from comet_ml import Experiment
    except ImportError:
        Experiment = None
else:
    Experiment = None

import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    NUM_CLASSES,
    ADE20kDataset,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _paths import repo_path
from canvit_rl.environment.canvit_env import CanViTEnvConfig, get_device
from canvit_rl.ade20k.greedy import miou_from_state


class IndexedDataset(Dataset):
    """Return dataset indices alongside ADE tensors so cache lookups are stable."""

    def __init__(self, dataset: ADE20kDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        image, mask = self.dataset[idx]
        return image, mask, idx


def _make_comet_experiment(args: argparse.Namespace):
    """Create an opt-in Comet experiment for saliency baseline eval outputs."""
    if not args.comet:
        return None
    if Experiment is None:
        raise RuntimeError("Install comet-ml or rerun without --comet.")

    # Problem: Comet can warn or miss framework setup if imported after torch.
    # Solution: import it above torch when --comet is present, then construct
    # the experiment here. Result: normal offline runs avoid Comet import cost,
    # while Comet runs follow the repo's training-script import order.
    experiment = Experiment(
        project_name=args.comet_project,
        workspace=args.comet_workspace,
        auto_param_logging=False,
        auto_metric_logging=False,
    )
    if args.experiment_name:
        experiment.set_name(args.experiment_name)
    if args.comet_tags:
        experiment.add_tags(
            [tag.strip() for tag in args.comet_tags.split(",") if tag.strip()]
        )
    experiment.log_parameters(vars(args))
    return experiment


def _parse_scales(value: str) -> list[float]:
    scales = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not scales:
        raise ValueError("--scales must include at least one value.")
    for scale in scales:
        if scale <= 0.0 or scale >= 1.0:
            raise ValueError(
                "Require every saliency scale to satisfy 0 < scale < 1; "
                "the evaluator already logs a full-scene t0 step."
            )
    return scales


def _update_miou(
    acc: mIoUAccumulator,
    probe: torch.nn.Module,
    features: Tensor,
    masks: Tensor,
) -> None:
    """Update one timestep's dataset-level mIoU accumulator."""
    with torch.autocast(device_type=features.device.type, enabled=False):
        logits = probe(features.float())
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    acc.update(logits.argmax(dim=1), masks)


def _load_saliency(cache_dir: Path, image_id: str, device: torch.device) -> Tensor:
    path = cache_dir / f"{image_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing saliency cache file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saliency = payload["saliency"] if isinstance(payload, dict) else payload
    saliency = torch.as_tensor(saliency, dtype=torch.float32, device=device)
    saliency = torch.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    saliency = saliency - saliency.min()
    return saliency / saliency.max().clamp_min(1e-6)


def _valid_center_mask(height: int, width: int, scale: float, device: torch.device) -> Tensor:
    half_h = max(1, round(height * scale)) // 2
    half_w = max(1, round(width * scale)) // 2
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    mask[half_h : height - half_h, half_w : width - half_w] = True
    return mask


def _select_one_viewpoint(
    saliency: Tensor,
    scales: list[float],
) -> tuple[Tensor, Tensor, Tensor]:
    height, width = saliency.shape
    best_score: Tensor | None = None
    best_yx: Tensor | None = None
    best_scale: float | None = None
    for scale in scales:
        kernel_h = max(1, round(height * scale))
        kernel_w = max(1, round(width * scale))
        # Problem: selecting the brightest pixel over-favors tiny highlights.
        # Solution: score each legal crop by average saliency in the glimpse
        # footprint. Result: selected Viewpoints target salient regions with
        # roughly the same support CanViT will observe.
        scores = F.avg_pool2d(
            saliency[None, None],
            kernel_size=(kernel_h, kernel_w),
            stride=1,
            padding=(kernel_h // 2, kernel_w // 2),
        ).squeeze()
        scores = scores[:height, :width]
        scores = scores.masked_fill(
            ~_valid_center_mask(height, width, scale, saliency.device),
            -torch.inf,
        )
        score = scores.max()
        if best_score is None or score > best_score:
            flat_idx = scores.argmax()
            best_score = score
            best_yx = torch.stack((flat_idx // width, flat_idx % width))
            best_scale = scale

    assert best_yx is not None and best_scale is not None
    y, x = best_yx.float()
    scale_t = torch.tensor(best_scale, dtype=torch.float32, device=saliency.device)
    center_x = (2.0 * (x + 0.5) / width) - 1.0
    center_y = (2.0 * (y + 0.5) / height) - 1.0
    bound = (1.0 - scale_t).clamp_min(0.0)
    center = torch.stack((center_x.clamp(-bound, bound), center_y.clamp(-bound, bound)))
    return center, scale_t, best_yx


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if (
                        ny < 0
                        or ny >= height
                        or nx < 0
                        or nx >= width
                        or visited[ny, nx]
                        or not mask[ny, nx]
                    ):
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(np.asarray(pixels, dtype=np.int64))
    return components


def _scale_for_blob(
    *,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    height: int,
    width: int,
    scales: list[float],
    margin: float,
) -> float:
    needed = max(
        ((y_max - y_min + 1) / height) * margin,
        ((x_max - x_min + 1) / width) * margin,
    )
    for scale in sorted(scales):
        if scale >= needed:
            return scale
    return max(scales)


def _select_blob_viewpoint(
    saliency: Tensor,
    scales: list[float],
    *,
    threshold_quantile: float,
    min_area_px: int,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    height, width = saliency.shape
    saliency_cpu = saliency.detach().cpu()
    positive = saliency_cpu[saliency_cpu > 0]
    if positive.numel() == 0:
        return _select_one_viewpoint(saliency, scales)

    threshold = float(torch.quantile(positive, threshold_quantile).item())
    mask = saliency_cpu.numpy() >= threshold
    components = [
        component
        for component in _connected_components(mask)
        if len(component) >= min_area_px
    ]
    if not components:
        return _select_one_viewpoint(saliency, scales)

    saliency_np = saliency_cpu.numpy()
    best_component = max(
        components,
        key=lambda component: float(saliency_np[component[:, 0], component[:, 1]].sum()),
    )
    y_values = best_component[:, 0]
    x_values = best_component[:, 1]
    weights = saliency_np[y_values, x_values]
    weight_sum = max(float(weights.sum()), 1e-6)
    y = float((y_values * weights).sum() / weight_sum)
    x = float((x_values * weights).sum() / weight_sum)
    scale = _scale_for_blob(
        y_min=int(y_values.min()),
        y_max=int(y_values.max()),
        x_min=int(x_values.min()),
        x_max=int(x_values.max()),
        height=height,
        width=width,
        scales=scales,
        margin=margin,
    )
    # Problem: raw blob extents can produce arbitrary zooms that make saliency
    # comparisons hard to sweep. Solution: use the blob bbox only to pick from
    # the allowed scale set, e.g. 0.25 or 0.5. Result: zoom is blob-aware but
    # still controlled by explicit hyperparameters.
    scale_t = torch.tensor(scale, dtype=torch.float32, device=saliency.device)
    center_x = (2.0 * (x + 0.5) / width) - 1.0
    center_y = (2.0 * (y + 0.5) / height) - 1.0
    bound = (1.0 - scale_t).clamp_min(0.0)
    center = torch.stack(
        (
            torch.tensor(center_x, dtype=torch.float32, device=saliency.device).clamp(
                -bound,
                bound,
            ),
            torch.tensor(center_y, dtype=torch.float32, device=saliency.device).clamp(
                -bound,
                bound,
            ),
        )
    )
    yx = torch.tensor([round(y), round(x)], dtype=torch.long, device=saliency.device)
    return center, scale_t, yx


def _suppress_region(saliency: Tensor, yx: Tensor, scale: float, nms_scale: float) -> None:
    height, width = saliency.shape
    radius_y = max(1, round(height * scale * nms_scale / 2.0))
    radius_x = max(1, round(width * scale * nms_scale / 2.0))
    y, x = int(yx[0].item()), int(yx[1].item())
    saliency[
        max(0, y - radius_y) : min(height, y + radius_y + 1),
        max(0, x - radius_x) : min(width, x + radius_x + 1),
    ] = 0.0


def _saliency_schedule(
    saliency_maps: list[Tensor],
    *,
    n_glimpses: int,
    scales: list[float],
    nms_scale: float,
    selection_mode: str,
    blob_threshold_quantile: float,
    blob_min_area_px: int,
    blob_margin: float,
) -> tuple[Tensor, Tensor]:
    if n_glimpses == 0:
        batch_size = len(saliency_maps)
        device = saliency_maps[0].device
        return (
            torch.empty((0, batch_size, 2), dtype=torch.float32, device=device),
            torch.empty((0, batch_size), dtype=torch.float32, device=device),
        )

    centers_by_t: list[Tensor] = []
    scales_by_t: list[Tensor] = []
    working = [m.clone() for m in saliency_maps]
    for _ in range(n_glimpses):
        step_centers = []
        step_scales = []
        for saliency in working:
            if selection_mode == "blob":
                center, scale, yx = _select_blob_viewpoint(
                    saliency,
                    scales,
                    threshold_quantile=blob_threshold_quantile,
                    min_area_px=blob_min_area_px,
                    margin=blob_margin,
                )
            else:
                center, scale, yx = _select_one_viewpoint(saliency, scales)
            step_centers.append(center)
            step_scales.append(scale)
            _suppress_region(saliency, yx, float(scale.item()), nms_scale)
        centers_by_t.append(torch.stack(step_centers))
        scales_by_t.append(torch.stack(step_scales))
    return torch.stack(centers_by_t), torch.stack(scales_by_t)


def _viewpoint_box(
    center: Tensor,
    scale: Tensor,
    *,
    image_size: int,
) -> tuple[int, int, int, int]:
    center = center.detach().cpu().float()
    scale_value = float(scale.detach().cpu().item())
    half = max(1.0, image_size * scale_value / 2.0)
    x = float((center[0].item() + 1.0) * 0.5 * image_size)
    y = float((center[1].item() + 1.0) * 0.5 * image_size)
    return (
        max(0, round(x - half)),
        max(0, round(y - half)),
        min(image_size - 1, round(x + half)),
        min(image_size - 1, round(y + half)),
    )


def _load_visualization_image(image_path: Path, scene_size_px: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(
        (scene_size_px, scene_size_px),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.uint8)


def _saliency_overlay(image_np: np.ndarray, saliency: Tensor) -> np.ndarray:
    sal = saliency.detach().cpu().float()
    sal = sal - sal.min()
    sal = sal / sal.max().clamp_min(1e-6)
    sal_np = sal.numpy()
    heat = np.zeros_like(image_np)
    heat[..., 0] = (sal_np * 255.0).astype(np.uint8)
    heat[..., 2] = (sal_np * 255.0).astype(np.uint8)
    return np.clip(image_np.astype(np.float32) * 0.72 + heat.astype(np.float32) * 0.28, 0, 255).astype(
        np.uint8
    )


def _save_visualization_figure(
    *,
    rows: list[dict[str, object]],
    method: str,
    output_dir: Path,
    scene_size_px: int,
) -> Path | None:
    if not rows:
        return None
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to save saliency timestep figures.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    n_steps = int(rows[0]["centers_by_t"].shape[0]) + 1
    n_cols = n_steps + 1
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n_steps))
    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(4.0 * n_cols, max(3.2 * len(rows), 4.0)),
        dpi=150,
        squeeze=False,
    )
    for sample_idx, row in enumerate(rows):
        image_np = _load_visualization_image(
            Path(row["image_path"]),
            scene_size_px,
        )
        saliency = row["saliency"]
        centers_by_t = row["centers_by_t"]
        scales_by_t = row["scales_by_t"]
        image_id = str(row["image_id"])
        overlay_np = _saliency_overlay(image_np, saliency)

        overview_ax = axes[sample_idx, 0]
        overview_ax.imshow(image_np)
        overview_ax.set_title(f"{image_id}\nall viewpoints")
        overview_ax.axis("off")

        full_box = (0, 0, scene_size_px - 1, scene_size_px - 1)
        all_boxes: list[tuple[int, int, int, int]] = [full_box]
        for step_idx in range(1, n_steps):
            all_boxes.append(
                _viewpoint_box(
                    centers_by_t[step_idx - 1, 0],
                    scales_by_t[step_idx - 1, 0],
                    image_size=scene_size_px,
                )
            )

        for step_idx, box in enumerate(all_boxes):
            color = colors[step_idx]
            # Problem: per-sample strip images made it harder to compare many
            # saliency rollouts at once. Solution: mirror the IN21k SAC figure
            # layout: an overview column plus one focused column per timestep.
            # Result: a single PNG shows all requested samples and their gaze
            # sequence with consistent colors and titles.
            overview_ax.add_patch(
                patches.Rectangle(
                    (box[0], box[1]),
                    max(box[2] - box[0], 1),
                    max(box[3] - box[1], 1),
                    linewidth=2.5,
                    edgecolor=color,
                    facecolor="none",
                )
            )
            overview_ax.text(
                box[0] + 3,
                box[1] + 12,
                f"t{step_idx}",
                color=color,
                fontsize=9,
                weight="bold",
            )

        for step_idx, box in enumerate(all_boxes):
            ax = axes[sample_idx, step_idx + 1]
            ax.imshow(image_np if step_idx == 0 else overlay_np)
            ax.add_patch(
                patches.Rectangle(
                    (box[0], box[1]),
                    max(box[2] - box[0], 1),
                    max(box[3] - box[1], 1),
                    linewidth=3.0,
                    edgecolor=colors[step_idx],
                    facecolor="none",
                )
            )
            if step_idx == 0:
                title = "t0 full\nscale=1.00"
            else:
                center = centers_by_t[step_idx - 1, 0].detach().cpu().tolist()
                scale = float(scales_by_t[step_idx - 1, 0].detach().cpu())
                title = f"t{step_idx} {method}\ns={scale:.2f} c=({center[0]:+.2f},{center[1]:+.2f})"
            ax.set_title(title)
            ax.axis("off")

    fig.suptitle(f"ADE20K saliency heuristic glimpses: {method}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output = output_dir / f"{method}_saliency_timesteps.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def _log_to_comet(
    *,
    comet_exp,
    metrics: dict[str, float],
    output: Path,
    visualization_path: Path | None,
) -> None:
    """Log final saliency mIoU metrics and generated artifacts to Comet."""
    if comet_exp is None:
        return
    comet_exp.log_metrics(metrics)
    if visualization_path is not None:
        # Problem: saliency crop sanity checks were only local files. Solution:
        # log the combined timestep figure alongside scalar mIoU. Result:
        # Comet runs show both quantitative outcome and the exact gaze pattern.
        comet_exp.log_image(
            str(visualization_path),
            name=f"saliency/{visualization_path.name}",
        )
    comet_exp.log_asset(str(output), file_name=output.name)
    comet_exp.end()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t",
        type=int,
        default=5,
        help="Saliency glimpses after t0 full scene",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.25",
        help=(
            "Comma-separated allowed glimpse scales. In blob mode, the blob "
            "bbox is quantized to the smallest allowed scale that covers it."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=["fixed_fovea", "blob"],
        default="fixed_fovea",
        help="How saliency maps choose each post-t0 Viewpoint.",
    )
    parser.add_argument(
        "--blob-threshold-quantile",
        type=float,
        default=0.85,
        help="Saliency quantile used to threshold connected blobs.",
    )
    parser.add_argument(
        "--blob-min-area-px",
        type=int,
        default=16,
        help="Ignore connected saliency blobs smaller than this many pixels.",
    )
    parser.add_argument(
        "--blob-margin",
        type=float,
        default=1.25,
        help="Multiplier applied to blob bbox before quantizing to --scales.",
    )
    parser.add_argument("--nms-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--saliency-cache-root", type=Path, default=Path("cache/saliency"))
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/saliency_baseline_miou.json"),
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=0,
        help="Save timestep-strip overlays for the first N evaluated images.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=Path("results/saliency_visualizations"),
        help="Directory for --visualize-samples PNG strips.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--miou-mode",
        choices=["accumulator", "mean"],
        default="accumulator",
        help="Match random baseline modes for dataset-level or per-image mIoU.",
    )
    parser.add_argument(
        "--comet",
        action="store_true",
        help="Log final metrics, result JSON, and optional visualization to Comet.",
    )
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="canvit-rl")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--comet-tags", type=str, default="saliency,ade20k")
    args = parser.parse_args()

    if args.t < 0:
        raise ValueError("--t must be non-negative.")
    if args.nms_scale <= 0:
        raise ValueError("--nms-scale must be positive.")
    if not 0.0 < args.blob_threshold_quantile < 1.0:
        raise ValueError("--blob-threshold-quantile must satisfy 0 < q < 1.")
    if args.blob_min_area_px < 1:
        raise ValueError("--blob-min-area-px must be positive.")
    if args.blob_margin <= 0:
        raise ValueError("--blob-margin must be positive.")
    if args.visualize_samples < 0:
        raise ValueError("--visualize-samples must be non-negative.")
    scales = _parse_scales(args.scales)
    comet_exp = _make_comet_experiment(args)

    cfg = CanViTEnvConfig()
    device = get_device()
    amp = not args.no_amp
    amp_dtype = torch.bfloat16 if amp else torch.float32
    print(f"Device: {device}")

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=repo_path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    effective_batch_size = 1 if args.miou_mode == "mean" else args.batch_size
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    cache_dir = repo_path(args.saliency_cache_root) / f"ade20k_{args.split}" / args.method
    print(f"Dataset: {len(dataset)} {args.split} images")
    print(f"Saliency cache: {cache_dir}")

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
    for p in model.parameters():
        p.requires_grad_(False)
    for p in probe.parameters():
        p.requires_grad_(False)

    n_steps = args.t + 1
    accs = (
        [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(n_steps)]
        if args.miou_mode == "accumulator"
        else None
    )
    miou_sums = [0.0 for _ in range(n_steps)]
    scale_sums = [0.0 for _ in range(n_steps)]
    count_sums = [0 for _ in range(n_steps)]
    n_images = 0
    n_visualized = 0
    visualization_rows: list[dict[str, object]] = []
    visualization_dir = repo_path(args.visualization_dir)
    t_start = time.monotonic()

    with torch.inference_mode():
        for batch_idx, (images, masks, indices) in enumerate(
            tqdm(loader, desc="Evaluating")
        ):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.shape[0]
            n_images += batch_size
            image_ids = [dataset.images[int(idx)].stem for idx in indices]
            saliency_maps = [
                _load_saliency(cache_dir, image_id, device) for image_id in image_ids
            ]
            centers_by_t, scales_by_t = _saliency_schedule(
                saliency_maps,
                n_glimpses=args.t,
                scales=scales,
                nms_scale=args.nms_scale,
                selection_mode=args.selection_mode,
                blob_threshold_quantile=args.blob_threshold_quantile,
                blob_min_area_px=args.blob_min_area_px,
                blob_margin=args.blob_margin,
            )
            if n_visualized < args.visualize_samples:
                for local_idx, image_id in enumerate(image_ids):
                    if n_visualized >= args.visualize_samples:
                        break
                    visualization_rows.append(
                        {
                            "image_path": dataset.images[int(indices[local_idx])],
                            "image_id": image_id,
                            "saliency": saliency_maps[local_idx].detach().cpu(),
                            "centers_by_t": centers_by_t[
                                :,
                                local_idx : local_idx + 1,
                            ]
                            .detach()
                            .cpu(),
                            "scales_by_t": scales_by_t[
                                :,
                                local_idx : local_idx + 1,
                            ]
                            .detach()
                            .cpu(),
                        }
                    )
                    n_visualized += 1
            state = model.init_state(
                batch_size=batch_size,
                canvas_grid_size=cfg.canvas_grid_size,
            )

            for step_idx in range(n_steps):
                if step_idx == 0:
                    vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
                else:
                    vp = Viewpoint(
                        centers=centers_by_t[step_idx - 1].to(device),
                        scales=scales_by_t[step_idx - 1].to(device),
                    )

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp,
                ):
                    glimpse = sample_at_viewpoint(
                        spatial=images,
                        viewpoint=vp,
                        glimpse_size_px=cfg.glimpse_size_px,
                    )
                    out = model(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state

                if args.miou_mode == "accumulator":
                    assert accs is not None
                    spatial = model.get_spatial(state.canvas).view(
                        batch_size,
                        cfg.canvas_grid_size,
                        cfg.canvas_grid_size,
                        -1,
                    )
                    _update_miou(accs[step_idx], probe, spatial, masks)
                else:
                    miou_sums[step_idx] += (
                        miou_from_state(
                            model=model,
                            state=state,
                            probe=probe,
                            mask=masks,
                            canvas_grid_size=cfg.canvas_grid_size,
                        )
                        * batch_size
                    )
                scale_sums[step_idx] += float(vp.scales.detach().sum().item())
                count_sums[step_idx] += batch_size

    if args.miou_mode == "accumulator":
        assert accs is not None
        mious = {f"t{t}": float(acc.compute()) for t, acc in enumerate(accs)}
    else:
        mious = {f"t{t}": miou_sums[t] / count_sums[t] for t in range(n_steps)}
    mean_scales = {f"t{t}": scale_sums[t] / count_sums[t] for t in range(n_steps)}
    wall_time = time.monotonic() - t_start
    visualization_path = _save_visualization_figure(
        rows=visualization_rows,
        method=args.method,
        output_dir=visualization_dir,
        scene_size_px=cfg.scene_size_px,
    )

    print("\n--- Saliency Baseline mIoU ---")
    for t in range(n_steps):
        label = "full_scene" if t == 0 else args.method
        print(
            f"  t={t} ({label}): "
            f"scale={mean_scales[f't{t}']:.3f}  "
            f"miou={mious[f't{t}']:.4f}"
        )
    final_metrics = {
        **{f"miou/t{t}": mious[f"t{t}"] for t in range(n_steps)},
        **{f"scale/t{t}": mean_scales[f"t{t}"] for t in range(n_steps)},
        "miou/final": mious[f"t{n_steps - 1}"],
        "eval/n_images": float(n_images),
        "eval/wall_time_seconds": wall_time,
    }

    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "mious": mious,
        "mean_scales": mean_scales,
        "metadata": {
            "policy": "saliency_nms_after_full_scene",
            "method": args.method,
            "dataset": str(repo_path(args.dataset)),
            "split": args.split,
            "n_images": n_images,
            "canvas_grid_size": cfg.canvas_grid_size,
            "glimpse_size_px": cfg.glimpse_size_px,
            "scene_size_px": cfg.scene_size_px,
            "n_saliency_glimpses": args.t,
            "n_logged_steps": n_steps,
            "scales": scales,
            "nms_scale": args.nms_scale,
            "selection_mode": args.selection_mode,
            "blob_threshold_quantile": args.blob_threshold_quantile,
            "blob_min_area_px": args.blob_min_area_px,
            "blob_margin": args.blob_margin,
            "requested_batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "probe_repo": probe_repo,
            "model_repo": cfg.checkpoint,
            "amp": amp,
            "miou_mode": args.miou_mode,
            "wall_time_seconds": wall_time,
            "visualized_samples": n_visualized,
            "visualization_dir": str(visualization_dir),
            "visualization_path": (
                None if visualization_path is None else str(visualization_path)
            ),
        },
    }
    # Problem: saliency eval results were saved as a PyTorch payload even
    # though the contents are scalar metrics and metadata. Solution: write a
    # plain JSON result file. Result: Comet, plotting scripts, and humans can
    # inspect the output without torch.load.
    with output.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, sort_keys=True)
    _log_to_comet(
        comet_exp=comet_exp,
        metrics=final_metrics,
        output=output,
        visualization_path=visualization_path,
    )
    print(f"\nSaved {output} after {wall_time:.1f}s")
    if visualization_path is not None:
        print(
            f"Saved one visualization figure with {n_visualized} sample(s) to "
            f"{visualization_path}"
        )


if __name__ == "__main__":
    main()
