"""
Precompute ADE20K saliency maps for heuristic active-view baselines.

Native methods are intentionally cheap and dependency-light. GBVS/AWS/DeepGaze
can be converted into the same cache format by passing their exported maps via
--external-map-dir.

Examples:
    uv run python scripts/saliency/precompute_saliency_maps.py --method itti
    uv run python scripts/saliency/precompute_saliency_maps.py \
        --method gbvs --external-map-dir results/gbvs_maps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from _paths import repo_path

NATIVE_METHODS = {"edge", "itti", "spectral_residual"}
EXTERNAL_METHODS = {"aws", "gbvs", "deepgaze", "itti_reference"}
ALL_METHODS = sorted(NATIVE_METHODS | EXTERNAL_METHODS)


def _normalize_map(saliency: Tensor) -> Tensor:
    """Normalize arbitrary finite saliency scores into a stable [0, 1] map."""
    saliency = saliency.float()
    saliency = torch.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    saliency = saliency - saliency.min()
    denom = saliency.max().clamp_min(1e-6)
    return saliency / denom


def _pil_to_tensor(image: Image.Image, scene_size_px: int) -> Tensor:
    image = image.convert("RGB").resize(
        (scene_size_px, scene_size_px),
        resample=Image.Resampling.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _blur2d(image: Tensor, kernel_size: int) -> Tensor:
    pad = kernel_size // 2
    weight = torch.ones(
        image.shape[0],
        1,
        kernel_size,
        kernel_size,
        dtype=image.dtype,
        device=image.device,
    ) / float(kernel_size * kernel_size)
    return F.conv2d(
        image[None],
        weight,
        padding=pad,
        groups=image.shape[0],
    ).squeeze(0)


def _edge_saliency(rgb: Tensor) -> Tensor:
    gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])[None, None]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=rgb.dtype,
    )[None, None]
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return _normalize_map(torch.sqrt(gx.square() + gy.square()).squeeze())


def _spectral_residual_saliency(rgb: Tensor) -> Tensor:
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    # Problem: full-resolution FFT saliency is slow and noisy for high-res ADE
    # images. Solution: compute the spectral residual at a compact resolution
    # and upsample. Result: a cheap global-contrast map suitable for caching.
    small = F.interpolate(
        gray[None, None],
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    spectrum = torch.fft.fft2(small)
    log_amp = torch.log(torch.abs(spectrum).clamp_min(1e-6))
    phase = torch.angle(spectrum)
    smooth = F.avg_pool2d(log_amp[None, None], kernel_size=3, stride=1, padding=1)
    residual = log_amp - smooth.squeeze()
    saliency = torch.abs(torch.fft.ifft2(torch.exp(residual + 1j * phase))).square()
    saliency = F.avg_pool2d(saliency[None, None], kernel_size=5, stride=1, padding=2)
    saliency = F.interpolate(
        saliency,
        size=gray.shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return _normalize_map(saliency)


def _center_surround(channel: Tensor) -> Tensor:
    responses = []
    for small_kernel, large_kernel in ((3, 15), (5, 31), (9, 63)):
        small = _blur2d(channel[None], small_kernel).squeeze(0)
        large = _blur2d(channel[None], large_kernel).squeeze(0)
        responses.append((small - large).abs())
    return torch.stack(responses).mean(dim=0)


def _itti_saliency(rgb: Tensor) -> Tensor:
    r, g, b = rgb
    intensity = rgb.mean(dim=0)
    rg = (r - g).abs()
    by = (b - 0.5 * (r + g)).abs()
    edges = _edge_saliency(rgb)
    # Problem: the original Itti-Koch-Niebur model is a larger multi-scale
    # feature pipeline. Solution: combine center-surround intensity,
    # opponent-color, and orientation/edge conspicuity. Result: a deterministic
    # Itti-style bottom-up saliency heuristic without MATLAB dependencies.
    saliency = (
        _center_surround(intensity)
        + _center_surround(rg)
        + _center_surround(by)
        + edges
    )
    return _normalize_map(saliency)


def _native_saliency(method: str, rgb: Tensor) -> Tensor:
    if method == "edge":
        return _edge_saliency(rgb)
    if method == "itti":
        return _itti_saliency(rgb)
    if method == "spectral_residual":
        return _spectral_residual_saliency(rgb)
    raise ValueError(f"Unsupported native saliency method: {method}")


def _load_external_map(path: Path, scene_size_px: int) -> Tensor:
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        saliency = payload["saliency"] if isinstance(payload, dict) else payload
        saliency = torch.as_tensor(saliency).float()
    elif path.suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError(
                "Reading MATLAB .mat saliency maps requires scipy. "
                "Install scipy or export maps as .png/.npy instead."
            ) from exc
        mat = loadmat(path)
        if "salmap" in mat:
            saliency_np = mat["salmap"]
        else:
            candidates = [
                value
                for key, value in mat.items()
                if not key.startswith("__") and np.asarray(value).ndim >= 2
            ]
            if len(candidates) != 1:
                raise KeyError(
                    f"{path} must contain a saliency variable named 'salmap' "
                    "or exactly one non-metadata array."
                )
            saliency_np = candidates[0]
        # Problem: MATLAB reference implementations preserve the most faithful
        # saliency values in .mat files. Solution: load numeric arrays directly
        # before normalization. Result: no intermediate 8-bit image quantizing.
        saliency = torch.from_numpy(np.asarray(saliency_np)).float()
    elif path.suffix == ".npy":
        saliency = torch.from_numpy(np.load(path)).float()
    else:
        saliency = torch.from_numpy(np.asarray(Image.open(path), dtype=np.float32))
        if saliency.ndim == 3:
            saliency = saliency.mean(dim=2)
    if saliency.ndim == 3:
        saliency = saliency.squeeze()
    saliency = F.interpolate(
        saliency[None, None],
        size=(scene_size_px, scene_size_px),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return _normalize_map(saliency)


def _find_external_map(external_dir: Path, stem: str) -> Path:
    for suffix in (".mat", ".pt", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = external_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No external saliency map found for {stem} in {external_dir}")


def _save_preview(saliency: Tensor, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = (saliency.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(output)


def _iter_images(dataset: Path, split: str) -> list[Path]:
    img_dir = dataset / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    paths = sorted(
        [*img_dir.glob("*.jpg"), *img_dir.glob("*.png"), *img_dir.glob("*.jpeg")]
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--method", choices=ALL_METHODS, required=True)
    parser.add_argument("--external-map-dir", type=Path, default=None)
    parser.add_argument("--scene-size-px", type=int, default=512)
    parser.add_argument("--output-root", type=Path, default=Path("cache/saliency"))
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = repo_path(args.dataset)
    output_dir = repo_path(args.output_root) / f"ade20k_{args.split}" / args.method
    external_dir = (
        repo_path(args.external_map_dir) if args.external_map_dir is not None else None
    )
    if args.method in EXTERNAL_METHODS and external_dir is None:
        raise ValueError(
            f"--method {args.method} needs --external-map-dir with precomputed "
            ".pt/.npy/.png maps from that model."
        )

    images = _iter_images(dataset, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(tqdm(images, desc=f"Precomputing {args.method}")):
        out_path = output_dir / f"{image_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            continue

        with Image.open(image_path) as image:
            original_size = image.size
            if args.method in NATIVE_METHODS:
                rgb = _pil_to_tensor(image, args.scene_size_px)
                saliency = _native_saliency(args.method, rgb)
            else:
                assert external_dir is not None
                saliency = _load_external_map(
                    _find_external_map(external_dir, image_path.stem),
                    args.scene_size_px,
                )

        torch.save(
            {
                "saliency": saliency.cpu(),
                "image_id": image_path.stem,
                "method": args.method,
                "source_path": str(image_path),
                "source_size": original_size,
                "saliency_size": tuple(saliency.shape),
            },
            out_path,
        )
        if idx < args.preview_samples:
            _save_preview(
                saliency.cpu(),
                output_dir / "previews" / f"{image_path.stem}.png",
            )

    print(f"Saved saliency cache to {output_dir}")


if __name__ == "__main__":
    main()
