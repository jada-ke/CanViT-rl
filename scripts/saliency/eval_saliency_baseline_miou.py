"""
Evaluate cached saliency-map viewpoint heuristics on ADE20K mIoU.

The rollout starts with a full-scene glimpse, then selects --t saliency-guided
glimpses using average saliency inside candidate windows with simple NMS.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _paths import repo_path
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import miou_from_state


class IndexedDataset(Dataset):
    """Return dataset indices alongside ADE tensors so cache lookups are stable."""

    def __init__(self, dataset: ADE20kDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        image, mask = self.dataset[idx]
        return image, mask, idx


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
            center, scale, yx = _select_one_viewpoint(saliency, scales)
            step_centers.append(center)
            step_scales.append(scale)
            _suppress_region(saliency, yx, float(scale.item()), nms_scale)
        centers_by_t.append(torch.stack(step_centers))
        scales_by_t.append(torch.stack(step_scales))
    return torch.stack(centers_by_t), torch.stack(scales_by_t)


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
        help="Comma-separated glimpse scales",
    )
    parser.add_argument("--nms-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--saliency-cache-root", type=Path, default=Path("cache/saliency"))
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/saliency_baseline_miou.pt"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--miou-mode",
        choices=["accumulator", "mean"],
        default="accumulator",
        help="Match random baseline modes for dataset-level or per-image mIoU.",
    )
    args = parser.parse_args()

    if args.t < 0:
        raise ValueError("--t must be non-negative.")
    if args.nms_scale <= 0:
        raise ValueError("--nms-scale must be positive.")
    scales = _parse_scales(args.scales)

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
            )
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

    print("\n--- Saliency Baseline mIoU ---")
    for t in range(n_steps):
        label = "full_scene" if t == 0 else args.method
        print(
            f"  t={t} ({label}): "
            f"scale={mean_scales[f't{t}']:.3f}  "
            f"miou={mious[f't{t}']:.4f}"
        )

    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
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
                "requested_batch_size": args.batch_size,
                "effective_batch_size": effective_batch_size,
                "probe_repo": probe_repo,
                "model_repo": cfg.checkpoint,
                "amp": amp,
                "miou_mode": args.miou_mode,
                "wall_time_seconds": wall_time,
            },
        },
        output,
    )
    print(f"\nSaved {output} after {wall_time:.1f}s")


if __name__ == "__main__":
    main()
