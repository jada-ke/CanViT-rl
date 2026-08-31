"""Export ADE20K saliency maps with the official DeepGaze PyTorch models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from _paths import repo_path


def _iter_images(dataset: Path, split: str) -> list[Path]:
    image_dir = dataset / "images" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted(
        [*image_dir.glob("*.jpg"), *image_dir.glob("*.png"), *image_dir.glob("*.jpeg")]
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resize_for_deepgaze(image: Image.Image, max_long_side: int) -> Image.Image:
    if max_long_side <= 0:
        return image.convert("RGB")
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image.convert("RGB")
    scale = max_long_side / float(long_side)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.convert("RGB").resize(size, resample=Image.Resampling.BILINEAR)


def _image_to_tensor(image: Image.Image, device: torch.device) -> Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1))[None].to(device)


def _uniform_centerbias(height: int, width: int, device: torch.device) -> Tensor:
    # Problem: DeepGaze expects a log-density centerbias input, but ADE20K does
    # not provide human fixation data for fitting one. Solution: use a uniform
    # log density by default. Result: the exported map is driven by DeepGaze's
    # image model rather than an assumed dataset-specific center bias.
    value = -math.log(float(height * width))
    return torch.full((1, height, width), value, dtype=torch.float32, device=device)


def _log_density_to_map(log_density: Tensor) -> np.ndarray:
    log_density = log_density.detach().cpu().float().squeeze()
    log_density = log_density - torch.logsumexp(log_density.flatten(), dim=0)
    density = torch.exp(log_density)
    saliency = density - density.min()
    denom = saliency.max().clamp_min(1e-12)
    return (saliency / denom).numpy().astype(np.float32)


def _save_preview(saliency: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = (np.clip(saliency, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(output)


def _load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    try:
        import deepgaze_pytorch
    except ImportError as exc:
        raise ImportError(
            "DeepGaze export requires deepgaze_pytorch and the dependencies "
            "imported by the current package. Install them with:\n"
            "  uv pip install einops "
            "'git+https://github.com/openai/CLIP.git' "
            "'git+https://github.com/matthias-k/DeepGaze.git'"
        ) from exc

    if model_name == "iie":
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
    else:
        raise ValueError(f"Unsupported DeepGaze model: {model_name}")
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--output-dir", type=Path, default=Path("results/deepgaze_maps"))
    parser.add_argument("--model", choices=["iie"], default="iie")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-long-side", type=int, default=1024)
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    dataset = repo_path(args.dataset)
    output_dir = repo_path(args.output_dir)
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)
    model = _load_model(args.model, device)
    images = _iter_images(dataset, args.split)
    if args.max_images is not None:
        images = images[: args.max_images]

    for idx, image_path in enumerate(tqdm(images, desc=f"DeepGaze {args.model}")):
        out_path = output_dir / f"{image_path.stem}.npy"
        if out_path.exists() and not args.overwrite:
            continue

        with Image.open(image_path) as image:
            resized = _resize_for_deepgaze(image, args.max_long_side)
        width, height = resized.size
        image_tensor = _image_to_tensor(resized, device)
        centerbias = _uniform_centerbias(height, width, device)

        with torch.inference_mode():
            log_density = model(image_tensor, centerbias)
        saliency = _log_density_to_map(log_density)
        np.save(out_path, saliency)

        if idx < args.preview_samples:
            _save_preview(saliency, preview_dir / f"{image_path.stem}.png")

    print(f"Saved DeepGaze maps to {output_dir}")


if __name__ == "__main__":
    main()
