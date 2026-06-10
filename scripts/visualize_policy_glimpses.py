"""
Visualize where the k-greedy CE policy looks over a CanViT episode.

Usage:
    python scripts/visualize_policy_glimpses.py
    python scripts/visualize_policy_glimpses.py --image-index 12 --t 5 --k 50
    python scripts/visualize_policy_glimpses.py \
        --episodes 8 --split validation --output-dir results/greedy_glimpses
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForSemanticSegmentation, Viewpoint, resolve_canvit_repo
from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import run_greedy_episode


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _image_for_plot(image: torch.Tensor):
    """Convert one normalized CHW tensor to an HWC numpy image for plotting."""
    image_cpu = image.detach().cpu()
    image_cpu = (image_cpu * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    return image_cpu.permute(1, 2, 0).numpy()


def _ade_palette(num_classes: int = 150) -> torch.Tensor:
    """Build a deterministic categorical palette for ADE20K predictions."""
    labels = torch.arange(num_classes, dtype=torch.float32)
    palette = torch.stack(
        [
            (labels * 37 + 17) % 255,
            (labels * 67 + 71) % 255,
            (labels * 97 + 149) % 255,
        ],
        dim=1,
    ) / 255.0
    palette[0] = torch.tensor([0.0, 0.0, 0.0])
    return palette


def _segmentation_for_plot(
    *,
    model,
    probe,
    state,
    canvas_grid_size: int,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Decode one committed CanViT recurrent state into an RGB label image."""
    spatial = model.get_spatial(state.canvas).view(
        1,
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float()).float()
    if logits.shape[-2:] != output_size:
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
    pred = logits.argmax(dim=1)[0].detach().cpu()
    palette = _ade_palette(max(int(pred.max().item()) + 1, 150))
    return palette[pred.clamp_min(0)]


def _viewpoint_rect(
    viewpoint: Viewpoint,
    *,
    image_size: int,
    index: int = 0,
) -> tuple[float, float, float, float]:
    """Convert a CanViT [-1, 1] center plus scale into pixel rectangle bounds."""
    cx, cy = viewpoint.centers[index].detach().cpu().tolist()
    scale = float(viewpoint.scales[index].detach().cpu().item())
    center_x = (float(cx) + 1.0) * 0.5 * image_size
    center_y = (float(cy) + 1.0) * 0.5 * image_size
    side = scale * image_size
    x0 = max(0.0, center_x - side * 0.5)
    y0 = max(0.0, center_y - side * 0.5)
    x1 = min(float(image_size), center_x + side * 0.5)
    y1 = min(float(image_size), center_y + side * 0.5)
    return x0, y0, x1, y1


def _save_visualization(
    *,
    image: torch.Tensor,
    viewpoints: list[Viewpoint],
    states: list,
    model,
    probe,
    canvas_grid_size: int,
    scores: list[float],
    loss_reductions: list[float],
    mious: list[float],
    output: Path,
    title: str,
) -> None:
    """Save a contact sheet with timestep-aligned glimpse and segmentation rows."""
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires matplotlib. Install it in this environment "
            "or add it to the dev dependencies."
        ) from exc

    image_np = _image_for_plot(image)
    image_size = image.shape[-1]
    n_steps = len(viewpoints)
    fig, axes = plt.subplots(2, n_steps, figsize=(4 * n_steps, 8), dpi=150)
    axes_grid = axes.reshape(2, n_steps)
    colors = plt.cm.viridis(torch.linspace(0.05, 0.95, n_steps).numpy())
    fig.suptitle(title)

    for step_idx, vp in enumerate(viewpoints):
        ax = axes_grid[0, step_idx]
        ax.imshow(image_np)
        x0, y0, x1, y1 = _viewpoint_rect(vp, image_size=image_size)
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=3.0,
            edgecolor=colors[step_idx],
            facecolor="none",
        )
        ax.add_patch(rect)
        scale = float(vp.scales[0].detach().cpu().item())
        center = vp.centers[0].detach().cpu().tolist()
        ax.set_title(
            f"t{step_idx} image scale={scale:.3f}\n"
            f"ce={scores[step_idx]:.3f} d={loss_reductions[step_idx]:+.3f}\n"
            f"miou={mious[step_idx]:.3f} center=({center[0]:+.2f}, {center[1]:+.2f})"
        )
        ax.axis("off")

        seg_ax = axes_grid[1, step_idx]
        seg_np = _segmentation_for_plot(
            model=model,
            probe=probe,
            state=states[step_idx],
            canvas_grid_size=canvas_grid_size,
            output_size=image.shape[-2:],
        ).numpy()
        seg_ax.imshow(seg_np)
        seg_rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2.0,
            edgecolor=colors[step_idx],
            facecolor="none",
        )
        seg_ax.add_patch(seg_rect)
        seg_ax.set_title(
            f"t{step_idx} segmentation\n"
            f"ce={scores[step_idx]:.3f} miou={mious[step_idx]:.3f}"
        )
        seg_ax.axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=5, help="Timesteps per episode")
    parser.add_argument("--k", type=int, default=50, help="Candidates per greedy step")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--image-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="validation",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/greedy_glimpses"),
    )
    parser.add_argument(
        "--no-full-scene-start",
        action="store_true",
        help="Use greedy random-candidate search at t=0 instead of full scene.",
    )
    args = parser.parse_args()

    if args.t < 1:
        raise ValueError("--t must be positive.")
    if args.k < 1:
        raise ValueError("--k must be positive.")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = CanViTEnvConfig()
    device = get_device()
    print(f"Device: {device}")

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    if args.image_index is not None:
        indices = [args.image_index]
    else:
        indices = random.sample(range(len(dataset)), min(args.episodes, len(dataset)))
    print(f"Dataset: {len(dataset)} {args.split} images, visualizing {len(indices)}")

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
    for param in model.parameters():
        param.requires_grad_(False)
    for param in probe.parameters():
        param.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for idx in indices:
            image, mask = dataset[idx]
            image_dev = image.unsqueeze(0).to(device)
            mask_dev = mask.unsqueeze(0).to(device)
            init_state = model.init_state(
                batch_size=1,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            result = run_greedy_episode(
                model=model,
                image=image_dev,
                init_state=init_state,
                t=args.t,
                k=args.k,
                device=device,
                seed=args.seed + idx,
                mask=mask_dev,
                probe=probe,
                canvas_grid_size=cfg.canvas_grid_size,
                start_with_full_scene=not args.no_full_scene_start,
                compute_miou=True,
                keep_states=True,
            )
            name = dataset.images[idx].stem
            output = args.output_dir / f"{args.split}_{idx:05d}_{name}_greedy.png"
            _save_visualization(
                image=image,
                viewpoints=result["viewpoints"],
                states=result["states"],
                model=model,
                probe=probe,
                canvas_grid_size=cfg.canvas_grid_size,
                scores=result["scores"],
                loss_reductions=result["rewards"],
                mious=result["mious"],
                output=output,
                title=f"k-greedy k={args.k} idx={idx}",
            )
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
