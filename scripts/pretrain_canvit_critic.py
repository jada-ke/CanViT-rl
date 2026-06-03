"""
Pretrain continuous SAC critics from k-candidate delta-mIoU labels.

Usage:
    uv run python scripts/pretrain_canvit_critic.py --episodes 500 --t 5 --k 16
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_pytorch.policies import random_viewpoints
from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import miou_from_state
from canvit_rl.sac_models import CanViTSequenceEncoder, ContinuousCritic
from canvit_rl.sac_state import (
    append_glimpse,
    batch_from_sequence,
    empty_sequence,
    extract_local_patches,
)


def _viewpoint_to_action(viewpoint: Viewpoint, *, min_scale: float) -> torch.Tensor:
    """Map an upstream Viewpoint back to the SAC tanh action range."""
    centers = viewpoint.centers.float()
    scale_action = 2.0 * (viewpoint.scales.float() - min_scale) / (1.0 - min_scale)
    scale_action = scale_action - 1.0
    return torch.cat([centers, scale_action[:, None]], dim=-1).clamp(-1.0, 1.0)


def _repeat_batch(
    batch: dict[str, torch.Tensor],
    repeats: int,
) -> dict[str, torch.Tensor]:
    """Repeat a one-state actor/critic batch for K candidate actions."""
    return {
        key: value.repeat_interleave(repeats, dim=0)
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-patches", type=int, default=64)
    parser.add_argument("--patch-dim", type=int, default=768)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--rollout-policy",
        choices=["best", "random"],
        default="best",
        help="How to advance the state after labeling each K-candidate set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/canvit_critic/pretrained.pt"),
    )
    args = parser.parse_args()

    if args.t < 0 or args.k <= 0:
        raise ValueError("Require --t >= 0 and --k > 0.")
    if args.min_scale <= 0 or args.min_scale >= 1:
        raise ValueError("Require 0 < --min-scale < 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    cfg = CanViTEnvConfig()
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=RandomSampler(dataset, replacement=True),
        num_workers=0,
    )

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

    encoder = CanViTSequenceEncoder(
        patch_dim=args.patch_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_steps=args.t,
        n_patches=args.n_patches,
    ).to(device)
    q1 = ContinuousCritic(copy.deepcopy(encoder), args.d_model).to(device)
    q2 = ContinuousCritic(copy.deepcopy(encoder), args.d_model).to(device)
    opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)

    data_iter = iter(loader)
    losses: list[float] = []
    n_labels = 0

    for episode in tqdm(range(1, args.episodes + 1), desc="Pretraining critic"):
        try:
            image, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            image, mask = next(data_iter)
        image = image.to(device)
        mask = mask.to(device)
        state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)
        seq = empty_sequence(n_patches=args.n_patches, patch_dim=args.patch_dim)

        full_vp = Viewpoint.full_scene(batch_size=1, device=device)
        full_glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=full_vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        with torch.inference_mode():
            full_out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
        state = full_out.state
        patches = extract_local_patches(full_out)
        seq = append_glimpse(
            seq=seq,
            patches=patches,
            viewpoint=full_vp,
        )
        current_miou = miou_from_state(
            model=model,
            state=state,
            probe=probe,
            mask=mask,
            canvas_grid_size=cfg.canvas_grid_size,
        )

        for _ in range(args.t):
            batch = batch_from_sequence(
                seq,
                max_steps=args.t,
                n_patches=args.n_patches,
                patch_dim=args.patch_dim,
                device=device,
            )
            candidates = random_viewpoints(
                batch_size=1,
                device=device,
                n_viewpoints=args.k,
                min_scale=args.min_scale,
                max_scale=1.0,
                start_with_full_scene=False,
            )
            actions = []
            targets = []
            candidate_records = []
            with torch.inference_mode():
                for candidate_vp in candidates:
                    glimpse = sample_at_viewpoint(
                        spatial=image,
                        viewpoint=candidate_vp,
                        glimpse_size_px=cfg.glimpse_size_px,
                    )
                    out = model(glimpse=glimpse, state=state, viewpoint=candidate_vp)
                    miou = miou_from_state(
                        model=model,
                        state=out.state,
                        probe=probe,
                        mask=mask,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                    action = _viewpoint_to_action(
                        candidate_vp,
                        min_scale=args.min_scale,
                    )
                    target = (miou - current_miou) * args.reward_scale
                    actions.append(action.squeeze(0))
                    targets.append(target)
                    candidate_records.append((target, candidate_vp, out, miou))

            action_batch = torch.stack(actions).to(device)
            target_batch = torch.as_tensor(targets, device=device, dtype=torch.float32)
            obs_batch = _repeat_batch(batch, args.k)
            loss = F.mse_loss(q1(obs_batch, action_batch), target_batch)
            loss = loss + F.mse_loss(q2(obs_batch, action_batch), target_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
            n_labels += args.k

            if args.rollout_policy == "best":
                _, next_vp, next_out, next_miou = max(
                    candidate_records,
                    key=lambda item: item[0],
                )
            else:
                _, next_vp, next_out, next_miou = random.choice(candidate_records)
            state = next_out.state
            patches = extract_local_patches(next_out)
            seq = append_glimpse(
                seq=seq,
                patches=patches,
                viewpoint=next_vp,
            )
            current_miou = next_miou

        if episode % 50 == 0:
            recent = losses[-50 * max(args.t, 1) :]
            print(
                f"episode={episode} labels={n_labels} "
                f"mean_q_mse={sum(recent) / len(recent):.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "args": vars(args),
            "metadata": {
                "probe_repo": probe_repo,
                "model_repo": cfg.checkpoint,
                "canvas_grid_size": cfg.canvas_grid_size,
                "glimpse_size_px": cfg.glimpse_size_px,
                "n_labels": n_labels,
            },
        },
        args.output,
    )
    print(f"Saved pretrained critic to {args.output}")


if __name__ == "__main__":
    main()
