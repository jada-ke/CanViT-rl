"""
Evaluate a pretrained continuous CanViT critic on validation candidates.

Usage:
    uv run python scripts/eval_canvit_critic.py \
        --checkpoint checkpoints/canvit_critic/pretrained.pt \
        --split validation --episodes 100 --t 5 --k 50
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

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
from torch.utils.data import DataLoader
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
from scripts.pretrain_canvit_critic import _viewpoint_to_action


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values with average ranks for ties; lowest value gets rank 1."""
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or nan for constant inputs."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman correlation without scipy."""
    return _pearson(_average_ranks(x), _average_ranks(y))


def _mean_valid(values: list[float]) -> float:
    """Mean ignoring nan values."""
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if valid.size else float("nan")


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a critic checkpoint with architecture args."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "q1" not in checkpoint:
        raise ValueError(f"Expected critic checkpoint with q1/q2 keys: {path}")
    return checkpoint, dict(checkpoint.get("args", {}))


def _checkpoint_value(
    checkpoint_args: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    """Read a pretraining arg from checkpoint metadata with CLI fallback."""
    return checkpoint_args.get(key.replace("-", "_"), fallback)


def _build_critic_pair(
    *,
    checkpoint: dict[str, Any],
    checkpoint_args: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ContinuousCritic, ContinuousCritic, dict[str, int]]:
    """Reconstruct q1/q2 and return architecture dimensions."""
    d_model = int(_checkpoint_value(checkpoint_args, "d-model", args.d_model))
    n_heads = int(_checkpoint_value(checkpoint_args, "n-heads", args.n_heads))
    n_layers = int(_checkpoint_value(checkpoint_args, "n-layers", args.n_layers))
    n_patches = int(_checkpoint_value(checkpoint_args, "n-patches", args.n_patches))
    patch_dim = int(_checkpoint_value(checkpoint_args, "patch-dim", args.patch_dim))
    max_steps = int(_checkpoint_value(checkpoint_args, "t", args.t))

    def make_critic() -> ContinuousCritic:
        encoder = CanViTSequenceEncoder(
            patch_dim=patch_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_steps=max_steps,
            n_patches=n_patches,
        )
        return ContinuousCritic(encoder, d_model).to(device).eval()

    q1 = make_critic()
    q2 = make_critic()
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint.get("q2", checkpoint["q1"]))
    for critic in (q1, q2):
        for param in critic.parameters():
            param.requires_grad_(False)
    dims = {"n_patches": n_patches, "patch_dim": patch_dim, "max_steps": max_steps}
    return q1, q2, dims


def _score_candidates(
    *,
    q1: ContinuousCritic,
    q2: ContinuousCritic,
    batch: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> torch.Tensor:
    """Use clipped double-Q scores for candidate ranking."""
    repeated = {
        key: value.repeat_interleave(actions.shape[0], dim=0)
        for key, value in batch.items()
    }
    return torch.minimum(q1(repeated, actions), q2(repeated, actions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/canvit_critic/pretrained.pt"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="validation",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-patches", type=int, default=64)
    parser.add_argument("--patch-dim", type=int, default=768)
    parser.add_argument("--min-scale", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/critic_eval.pt"))
    args = parser.parse_args()

    if args.t < 0 or args.k <= 1:
        raise ValueError("Require --t >= 0 and --k > 1.")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    cfg = CanViTEnvConfig()
    checkpoint, checkpoint_args = _load_checkpoint(args.checkpoint)
    q1, q2, dims = _build_critic_pair(
        checkpoint=checkpoint,
        checkpoint_args=checkpoint_args,
        args=args,
        device=device,
    )
    min_scale = float(_checkpoint_value(checkpoint_args, "min-scale", 0.05))
    if args.min_scale is not None:
        min_scale = args.min_scale
    reward_scale = float(_checkpoint_value(checkpoint_args, "reward-scale", 100.0))
    if args.reward_scale is not None:
        reward_scale = args.reward_scale

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=Path(args.dataset),
        split=args.split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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

    mse_values: list[float] = []
    mae_values: list[float] = []
    pearsons: list[float] = []
    spearmans: list[float] = []
    oracle_deltas: list[float] = []
    critic_deltas: list[float] = []
    random_deltas: list[float] = []
    regrets: list[float] = []
    top10_hits = 0
    top25_hits = 0
    n_states = 0
    miou_sums = [0.0 for _ in range(args.t + 1)]
    scale_sums = [0.0 for _ in range(args.t + 1)]
    n_images = 0
    t_start = time.monotonic()
    total_eval_images = min(args.episodes, len(dataset))
    pbar = tqdm(
        total=total_eval_images,
        desc="Evaluating critic",
        miniters=args.progress_interval,
        maxinterval=float("inf"),
    )

    with torch.inference_mode():
        for image_idx, (image, mask) in enumerate(loader):
            if image_idx >= args.episodes:
                break
            image = image.to(device)
            mask = mask.to(device)
            n_images += 1
            state = model.init_state(
                batch_size=1,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            seq = empty_sequence(
                n_patches=dims["n_patches"],
                patch_dim=dims["patch_dim"],
            )

            full_vp = Viewpoint.full_scene(batch_size=1, device=device)
            full_glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            )
            out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
            state = out.state
            patches = extract_local_patches(out)
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
            miou_sums[0] += current_miou
            scale_sums[0] += 1.0

            for step_idx in range(args.t):
                batch = batch_from_sequence(
                    seq,
                    max_steps=dims["max_steps"],
                    n_patches=dims["n_patches"],
                    patch_dim=dims["patch_dim"],
                    device=device,
                )
                candidates = random_viewpoints(
                    batch_size=1,
                    device=device,
                    n_viewpoints=args.k,
                    min_scale=min_scale,
                    max_scale=1.0,
                    start_with_full_scene=False,
                )
                actions = []
                deltas = []
                records = []
                for vp in candidates:
                    glimpse = sample_at_viewpoint(
                        spatial=image,
                        viewpoint=vp,
                        glimpse_size_px=cfg.glimpse_size_px,
                    )
                    out = model(glimpse=glimpse, state=state, viewpoint=vp)
                    miou = miou_from_state(
                        model=model,
                        state=out.state,
                        probe=probe,
                        mask=mask,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                    delta = (miou - current_miou) * reward_scale
                    actions.append(_viewpoint_to_action(vp, min_scale=min_scale)[0])
                    deltas.append(delta)
                    records.append((vp, out, miou, delta))

                action_batch = torch.stack(actions).to(device)
                true = np.asarray(deltas, dtype=np.float64)
                pred = _score_candidates(
                    q1=q1,
                    q2=q2,
                    batch=batch,
                    actions=action_batch,
                )
                pred_np = pred.detach().cpu().numpy().astype(np.float64)
                mse_values.append(float(np.mean((pred_np - true) ** 2)))
                mae_values.append(float(np.mean(np.abs(pred_np - true))))
                pearsons.append(_pearson(pred_np, true))
                spearmans.append(_spearman(pred_np, true))

                oracle_idx = int(np.argmax(true))
                critic_idx = int(np.argmax(pred_np))
                random_idx = random.randrange(args.k)
                oracle_delta = float(true[oracle_idx])
                critic_delta = float(true[critic_idx])
                random_delta = float(true[random_idx])
                oracle_deltas.append(oracle_delta)
                critic_deltas.append(critic_delta)
                random_deltas.append(random_delta)
                regrets.append(oracle_delta - critic_delta)

                rank_desc = np.argsort(true)[::-1]
                top10 = set(rank_desc[: max(1, int(np.ceil(0.10 * args.k)))])
                top25 = set(rank_desc[: max(1, int(np.ceil(0.25 * args.k)))])
                top10_hits += int(critic_idx in top10)
                top25_hits += int(critic_idx in top25)
                n_states += 1

                vp, out, next_miou, _ = records[critic_idx]
                state = out.state
                patches = extract_local_patches(out)
                seq = append_glimpse(
                    seq=seq,
                    patches=patches,
                    viewpoint=vp,
                )
                current_miou = next_miou
                miou_sums[step_idx + 1] += current_miou
                scale_sums[step_idx + 1] += float(vp.scales[0].cpu().item())

            if n_images % args.progress_interval == 0:
                pbar.update(args.progress_interval)

    remainder = n_images % args.progress_interval
    if remainder:
        pbar.update(remainder)
    pbar.close()

    if n_states == 0 or n_images == 0:
        raise RuntimeError("No critic evaluation states were processed.")

    summary = {
        "critic_mse": float(np.mean(mse_values)),
        "critic_mae": float(np.mean(mae_values)),
        "mean_pearson": _mean_valid(pearsons),
        "mean_spearman": _mean_valid(spearmans),
        "mean_oracle_delta": float(np.mean(oracle_deltas)),
        "mean_critic_delta": float(np.mean(critic_deltas)),
        "mean_random_delta": float(np.mean(random_deltas)),
        "mean_regret": float(np.mean(regrets)),
        "top10_hit_rate": top10_hits / n_states,
        "top25_hit_rate": top25_hits / n_states,
    }
    mious = {f"t{idx}": value / n_images for idx, value in enumerate(miou_sums)}
    mean_scales = {
        f"t{idx}": value / n_images for idx, value in enumerate(scale_sums)
    }
    wall_time = time.monotonic() - t_start

    print("\n--- Critic Candidate Ranking ---")
    for key, value in summary.items():
        print(f"  {key}: {value:+.6f}")
    print("\n--- Critic-Greedy Mean mIoU ---")
    for step_idx in range(args.t + 1):
        key = f"t{step_idx}"
        label = "full_scene" if step_idx == 0 else "critic"
        print(
            f"  t={step_idx} ({label}): "
            f"scale={mean_scales[key]:.3f}  "
            f"miou={mious[key]:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summary": summary,
            "mious": mious,
            "mean_scales": mean_scales,
            "metadata": {
                "checkpoint": str(args.checkpoint),
                "dataset": args.dataset,
                "split": args.split,
                "n_images": n_images,
                "n_states": n_states,
                "k": args.k,
                "t": args.t,
                "min_scale": min_scale,
                "reward_scale": reward_scale,
                "probe_repo": probe_repo,
                "wall_time_seconds": wall_time,
            },
        },
        args.output,
    )
    print(f"\nSaved {args.output} after {wall_time:.1f}s")


if __name__ == "__main__":
    main()
