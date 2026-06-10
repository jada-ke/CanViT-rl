"""
Run canvit-eval's entropy-guided coarse-to-fine policy and print diagnostics.

Usage:
    python scripts/run_entropy_c2f.py --episodes 5
    python scripts/run_entropy_c2f.py --episodes 5 --t 5 --dataset datasets/ADE20k
    python scripts/run_entropy_c2f.py --all --t 21
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

EVAL_REPO = Path(__file__).resolve().parents[1] / "CanViT-eval"
if EVAL_REPO.is_dir() and str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))

from canvit_eval.episode import run_episode  # noqa: E402
from canvit_eval.policies import make_policy  # noqa: E402
from canvit_pytorch import CanViTForSemanticSegmentation, resolve_canvit_repo
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    ADE20kDataset,
    make_val_transforms,
)

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import miou_from_state


def _segmentation_ce_loss(
    model,
    probe: torch.nn.Module,
    state,
    mask: Tensor,
    canvas_grid_size: int,
) -> float:
    """Compute unweighted segmentation CE for one committed canvas state."""
    spatial = model.get_spatial(state.canvas).view(
        mask.shape[0],
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float())
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pixel_loss = F.cross_entropy(
        logits,
        mask.long(),
        ignore_index=IGNORE_LABEL,
        reduction="none",
    )
    valid = mask != IGNORE_LABEL
    loss_sum = pixel_loss.flatten(1).sum(dim=1)
    denom = valid.flatten(1).sum(dim=1).clamp_min(1)
    return float((loss_sum / denom).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every image in the selected split instead of sampling --episodes",
    )
    parser.add_argument("--t", type=int, default=21, help="Timesteps per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ADE20k",
        help="Path to ADE20K root directory",
    )
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="validation",
        help="ADE20K split to sample",
    )
    parser.add_argument(
        "--probe-repo",
        type=str,
        default=None,
        help="ADE20K probe repo; defaults to the probe matching canvas_grid_size",
    )
    args = parser.parse_args()

    if args.t > 21:
        raise ValueError("entropy_coarse_to_fine has 21 built-in C2F timesteps.")

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
    if args.all:
        indices = list(range(len(dataset)))
        print(f"Dataset: {len(dataset)} {args.split} images, running all")
    else:
        indices = random.sample(range(len(dataset)), min(args.episodes, len(dataset)))
        print(f"Dataset: {len(dataset)} {args.split} images, sampling {len(indices)}")

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

    all_scales = [[] for _ in range(args.t)]
    all_ce_losses = [[] for _ in range(args.t)]
    all_loss_reductions = [[] for _ in range(args.t)]
    all_mious = [[] for _ in range(args.t)]

    for ep, idx in enumerate(indices):
        image, mask = dataset[idx]
        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        img_name = dataset.images[idx].name

        with torch.inference_mode():
            policy = make_policy(
                "entropy_coarse_to_fine",
                batch_size=1,
                device=device,
                n_viewpoints=args.t,
                canvas_grid=cfg.canvas_grid_size,
                probe=probe,
                get_spatial_fn=model.get_spatial,
            )
            steps = run_episode(
                model=model,
                images=image,
                policy=policy,
                n_timesteps=args.t,
                canvas_grid=cfg.canvas_grid_size,
                glimpse_px=cfg.glimpse_size_px,
            )

        print(f"\n--- Episode {ep + 1} ({img_name}) ---")
        prev_ce_loss = None
        for step in steps:
            ce_loss = _segmentation_ce_loss(
                model=model,
                probe=probe,
                state=step.state,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            loss_reduction = 0.0 if prev_ce_loss is None else prev_ce_loss - ce_loss
            prev_ce_loss = ce_loss
            scale = float(step.viewpoint.scales[0].item())
            center = step.viewpoint.centers[0].detach().cpu().tolist()
            miou = miou_from_state(
                model=model,
                state=step.state,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )

            all_scales[step.t].append(scale)
            all_ce_losses[step.t].append(ce_loss)
            all_loss_reductions[step.t].append(loss_reduction)
            all_mious[step.t].append(miou)

            print(
                f"  step {step.t + 1}: "
                f"scale={scale:.3f}  "
                f"center=({center[0]:+.3f}, {center[1]:+.3f})  "
                f"seg_ce={ce_loss:.4f}  "
                f"loss_reduction={loss_reduction:+.4f}  "
                f"miou={miou:.4f}"
            )

    print("\n--- Mean metrics per timestep ---")
    for step in range(args.t):
        mean_scale = sum(all_scales[step]) / len(all_scales[step])
        mean_ce_loss = sum(all_ce_losses[step]) / len(all_ce_losses[step])
        mean_loss_reduction = (
            sum(all_loss_reductions[step]) / len(all_loss_reductions[step])
        )
        mean_miou = sum(all_mious[step]) / len(all_mious[step])
        bar = "*" * int(mean_scale * 20)
        print(
            f"  step {step + 1}: "
            f"scale={mean_scale:.3f} {bar}  "
            f"seg_ce={mean_ce_loss:.4f}  "
            f"loss_reduction={mean_loss_reduction:+.4f}  "
            f"miou={mean_miou:.4f}"
        )


if __name__ == "__main__":
    main()
