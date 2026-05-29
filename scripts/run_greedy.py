"""
scripts/run_greedy.py

Run a greedy episode and print per-step diagnostics.

Usage:
    python scripts/run_greedy.py
    python scripts/run_greedy.py --episodes 10 --seed 42 --dataset datasets/ADE20k --verbose
    python scripts/run_greedy.py --miou --episodes 5 --t 5 --k 10 --dataset datasets/ADE20k
    python scripts/run_greedy.py --miou --all --split validation --t 5 --k 10
    python scripts/run_greedy.py --miou --all --split validation --t 5 --k 10 --objective kl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_pytorch import (
    CanViTForPretrainingHFHub,
    CanViTForSemanticSegmentation,
    resolve_canvit_repo,
)
from canvit_pytorch.teacher import load_teacher
from canvit_rl.greedy import run_greedy_episode
from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every image in the selected split instead of sampling --episodes",
    )
    parser.add_argument("--t", type=int, default=5, help="Timesteps per episode")
    parser.add_argument("--k", type=int, default=5, help="Candidates per step")
    parser.add_argument(
        "--objective",
        choices=["cosine", "kl"],
        default="cosine",
        help="Greedy candidate-selection objective",
    )
    parser.add_argument(
        "--kl-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for --objective kl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--miou", action="store_true", help="Print mIoU after each greedy step")
    parser.add_argument("--probe-repo", type=str, default=None, help="ADE20K probe repo for --miou")
    parser.add_argument(
        "--no-full-scene-start",
        action="store_true",
        help="Use greedy random-candidate search at t=0 instead of a full-scene glimpse",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ADE20k",
        help="Path to ADE20K root directory",
    )
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
        help="ADE20K split to sample or run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode step metrics; default is quiet for --all",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = CanViTEnvConfig()
    device = get_device()
    print(f"Device: {device}")

    # Dataset — use squish mode and ImageNet normalization
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    # Fixed by Codex on 2026-05-28
    # Problem: Greedy diagnostics could only sample a small number of images,
    # making it awkward to inspect behavior across the full validation split.
    # Solution: Add --all plus a configurable split, while preserving the
    # previous --episodes random-sampling flow for quick checks.
    # Result: The same greedy metrics can be computed over every validation
    # image with `--all --split validation`.
    if args.all:
        indices = list(range(len(dataset)))
        print(f"Dataset: {len(dataset)} {args.split} images, running all")
    else:
        indices = random.sample(range(len(dataset)), min(args.episodes, len(dataset)))
        print(f"Dataset: {len(dataset)} {args.split} images, sampling {len(indices)}")

    probe = None
    if args.miou:
        probe_repo = args.probe_repo or resolve_canvit_repo(
            f"probe-ade20k-40k-s512-c{cfg.canvas_grid_size}-in21k"
        )
        print(f"Loading CanViT segmentation model with probe: {probe_repo}")
        # Fixed by Codex on 2026-05-25
        # Problem: Greedy diagnostics could only report teacher-CLS similarity,
        # not segmentation quality at the committed canvas states.
        # Solution: When --miou is enabled, load the semantic-segmentation wrapper
        # and pass its probe into run_greedy_episode for per-timestep mIoU.
        # Result: The runner prints mIoU beside scale, center, sim, and reward.
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
    else:
        print("Loading CanViT...")
        model = (
            CanViTForPretrainingHFHub.from_pretrained(cfg.checkpoint)
            .eval()
            .to(device)
        )
    for p in model.parameters():
        p.requires_grad_(False)
    if probe is not None:
        for p in probe.parameters():
            p.requires_grad_(False)

    print("Loading teacher...")
    teacher = load_teacher(cfg.teacher_repo, device)

    all_scales = [[] for _ in range(args.t)]
    all_sims = [[] for _ in range(args.t)]
    all_scores = [[] for _ in range(args.t)]
    all_objective_rewards = [[] for _ in range(args.t)]
    all_mious = [[] for _ in range(args.t)] if args.miou else None
    show_episode_logs = args.verbose or not args.all

    # Fixed by Codex on 2026-05-28
    # Problem: Full validation greedy runs printed every episode and had no ETA,
    # making long runs noisy and hard to monitor.
    # Solution: Wrap the episode loop in tqdm and suppress per-episode logs for
    # --all unless --verbose is requested.
    # Result: Full validation runs show progress/ETA and only print aggregate
    # metrics at the end by default.
    for ep, idx in enumerate(tqdm(indices, desc="Evaluating greedy", unit="img")):
        image, mask = dataset[idx]
        image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
        mask = mask.unsqueeze(0).to(device)
        img_name = dataset.images[idx].name

        with torch.inference_mode():
            teacher_cls = teacher.forward_norm_features(image).cls

        init_state = model.init_state(
            batch_size=1, canvas_grid_size=cfg.canvas_grid_size
        )

        result = run_greedy_episode(
            model=model,
            image=image,
            teacher_cls=teacher_cls,
            init_state=init_state,
            t=args.t,
            k=args.k,
            device=device,
            # Fixed by Codex on 2026-05-29
            # Problem: Greedy candidate sampling was seeded by episode order,
            # so the same image could get different timestep candidates if the
            # selected image order changed between runs.
            # Solution: Derive the episode seed from the stable dataset index.
            # Result: A given image index reproduces its t=1, t=2, ... candidate
            # pools across reruns with the same base --seed.
            seed=args.seed + idx,
            mask=mask if args.miou else None,
            probe=probe,
            canvas_grid_size=cfg.canvas_grid_size if args.miou else None,
            start_with_full_scene=not args.no_full_scene_start,
            objective=args.objective,
            kl_temperature=args.kl_temperature,
        )

        if show_episode_logs:
            print(f"\n--- Episode {ep + 1} ({img_name}) ---")
        prev_score = 0.0
        for step, (sim, score, reward, scale, center) in enumerate(
            zip(
                result["sims"],
                result["scores"],
                result["rewards"],
                result["scales"],
                result["centers"],
            )
        ):
            objective_reward = score - prev_score
            prev_score = score
            miou_text = ""
            if args.miou:
                miou = result["mious"][step]
                miou_text = f"  miou={miou:.4f}"
                assert all_mious is not None
                all_mious[step].append(miou)
            if show_episode_logs:
                print(
                    f"  step {step + 1}: "
                    f"scale={scale:.3f}  "
                    f"center=({center[0]:+.3f}, {center[1]:+.3f})  "
                    f"sim={sim:.4f}  "
                    f"score={score:+.4f}  "
                    f"reward={reward:+.4f}"
                    f"{miou_text}"
                )
            all_scales[step].append(scale)
            all_sims[step].append(sim)
            all_scores[step].append(score)
            all_objective_rewards[step].append(objective_reward)

    # Fixed by Codex on 2026-05-28
    # Problem: Quiet full-run KL experiments only reported scale/mIoU averages,
    # hiding the mean KL score and per-step objective gain.
    # Solution: Aggregate sim, objective score, and score deltas per timestep
    # across the selected dataset.
    # Result: KL runs report mean KL and KL reduction without requiring
    # per-episode verbose logs.
    print("\n--- Mean metrics per timestep ---")
    for step in range(args.t):
        step_scales = all_scales[step]
        mean_scale = sum(step_scales) / len(step_scales)
        mean_sim = sum(all_sims[step]) / len(all_sims[step])
        mean_score = sum(all_scores[step]) / len(all_scores[step])
        mean_objective_reward = (
            sum(all_objective_rewards[step]) / len(all_objective_rewards[step])
        )
        bar = "█" * int(mean_scale * 20)
        if args.objective == "kl":
            score_text = (
                f"kl={-mean_score:.4f}  "
                f"kl_reduction={mean_objective_reward:+.4f}"
            )
        else:
            score_text = (
                f"score={mean_score:+.4f}  "
                f"score_delta={mean_objective_reward:+.4f}"
            )
        print(
            f"  step {step + 1}: "
            f"scale={mean_scale:.3f} {bar}  "
            f"sim={mean_sim:.4f}  "
            f"{score_text}"
        )

    if all_mious is not None:
        print("\n--- Mean mIoU per timestep ---")
        for step, step_mious in enumerate(all_mious):
            mean_miou = sum(step_mious) / len(step_mious)
            print(f"  step {step + 1}: {mean_miou:.4f}")


if __name__ == "__main__":
    main()
