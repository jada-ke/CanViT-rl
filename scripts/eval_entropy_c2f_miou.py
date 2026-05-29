"""
Evaluate canvit-eval's entropy-guided C2F policy with dataset-level mIoU.

Unlike scripts/run_entropy_c2f.py, this script uses an mIoUAccumulator per
timestep, so reported mIoU is accumulated over the dataset rather than averaged
from per-image mIoUs.

Usage:
    python scripts/eval_entropy_c2f_miou.py
    python scripts/eval_entropy_c2f_miou.py --batch-size 8 --t 21
    python scripts/eval_entropy_c2f_miou.py --t 21 --batch-size 8 --max-batches 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

EVAL_REPO = Path(__file__).resolve().parents[1] / "CanViT-eval"
if EVAL_REPO.is_dir() and str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))

from canvit_eval.episode import run_episode  # noqa: E402
from canvit_eval.policies import make_policy  # noqa: E402
from canvit_pytorch import CanViTForSemanticSegmentation, resolve_canvit_repo
from canvit_pytorch.teacher import load_teacher
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    NUM_CLASSES,
    ADE20kDataset,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator

from canvit_rl.env import CanViTEnvConfig, get_device


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=21, help="Timesteps per episode")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/entropy_c2f_miou.pt"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.t > 21:
        raise ValueError("entropy_coarse_to_fine has 21 built-in C2F timesteps.")

    cfg = CanViTEnvConfig()
    device = get_device()
    amp = not args.no_amp
    amp_dtype = torch.bfloat16 if amp else torch.float32
    print(f"Device: {device}")

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} {args.split} images")

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

    print("Loading teacher...")
    teacher = load_teacher(cfg.teacher_repo, device)

    accs = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(args.t)]
    scale_sums = [0.0 for _ in range(args.t)]
    sim_sums = [0.0 for _ in range(args.t)]
    reward_sums = [0.0 for _ in range(args.t)]
    count_sums = [0 for _ in range(args.t)]
    n_images = 0
    t_start = time.monotonic()

    with torch.inference_mode():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.shape[0]
            n_images += batch_size

            teacher_cls = teacher.forward_norm_features(images).cls
            policy = make_policy(
                "entropy_coarse_to_fine",
                batch_size=batch_size,
                device=device,
                n_viewpoints=args.t,
                canvas_grid=cfg.canvas_grid_size,
                probe=probe,
                get_spatial_fn=model.get_spatial,
            )

            # Fixed by Codex on 2026-05-27
            # Problem: The diagnostic entropy-C2F runner recomputed per-image
            # mIoU at every timestep, which is useful for inspection but not the
            # same as validation-set mIoU.
            # Solution: Run canvit-eval's entropy_coarse_to_fine policy in
            # batches and update one mIoUAccumulator per timestep.
            # Result: Final mIoU is computed from accumulated intersections and
            # unions across the evaluated split, matching eval_miou.py's pattern.
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                steps = run_episode(
                    model=model,
                    images=images,
                    policy=policy,
                    n_timesteps=args.t,
                    canvas_grid=cfg.canvas_grid_size,
                    glimpse_px=cfg.glimpse_size_px,
                )

            prev_sim = torch.zeros(batch_size, device=device)
            for step in steps:
                spatial = model.get_spatial(step.state.canvas).view(
                    batch_size,
                    cfg.canvas_grid_size,
                    cfg.canvas_grid_size,
                    -1,
                )
                _update_miou(accs[step.t], probe, spatial, masks)

                # Fixed by Codex on 2026-05-27
                # Problem: seg.canvit is the backbone inside
                # CanViTForSemanticSegmentation and does not expose the
                # pretraining wrapper's predict_scene_teacher_cls method.
                # Solution: Match greedy.py's diagnostic similarity by comparing
                # the recurrent CLS state directly against the frozen teacher CLS.
                # Result: Entropy-C2F accumulator eval reports similarity/reward
                # without requiring pretraining-only wrapper methods.
                predicted_cls = step.state.recurrent_cls.squeeze(1)
                sim = F.cosine_similarity(
                    predicted_cls.float(),
                    teacher_cls.float(),
                    dim=-1,
                )
                reward = sim - prev_sim
                prev_sim = sim

                scale_sums[step.t] += float(step.viewpoint.scales.detach().sum().item())
                sim_sums[step.t] += float(sim.detach().sum().item())
                reward_sums[step.t] += float(reward.detach().sum().item())
                count_sums[step.t] += batch_size

    mious = {f"t{t}": acc.compute() for t, acc in enumerate(accs)}
    mean_scales = {
        f"t{t}": scale_sums[t] / count_sums[t] for t in range(args.t)
    }
    mean_sims = {f"t{t}": sim_sums[t] / count_sums[t] for t in range(args.t)}
    mean_rewards = {
        f"t{t}": reward_sums[t] / count_sums[t] for t in range(args.t)
    }
    wall_time = time.monotonic() - t_start

    print("\n--- Dataset-Level Entropy-C2F Metrics ---")
    for t in range(args.t):
        key = f"t{t}"
        print(
            f"  step {t + 1}: "
            f"scale={mean_scales[key]:.3f}  "
            f"sim={mean_sims[key]:.4f}  "
            f"reward={mean_rewards[key]:+.4f}  "
            f"miou={mious[key]:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mious": mious,
            "mean_scales": mean_scales,
            "mean_sims": mean_sims,
            "mean_rewards": mean_rewards,
            "metadata": {
                "policy": "entropy_coarse_to_fine",
                "dataset": args.dataset,
                "split": args.split,
                "n_images": n_images,
                "canvas_grid_size": cfg.canvas_grid_size,
                "glimpse_size_px": cfg.glimpse_size_px,
                "scene_size_px": cfg.scene_size_px,
                "n_timesteps": args.t,
                "batch_size": args.batch_size,
                "probe_repo": probe_repo,
                "model_repo": cfg.checkpoint,
                "amp": amp,
                "wall_time_seconds": wall_time,
            },
        },
        args.output,
    )
    print(f"\nSaved {args.output} after {wall_time:.1f}s")


if __name__ == "__main__":
    main()
