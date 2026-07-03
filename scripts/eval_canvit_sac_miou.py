"""
Evaluate a trained continuous CanViT SAC actor on ADE20K mIoU.

Usage:
    uv run python scripts/eval_canvit_sac_miou.py
    uv run python scripts/eval_canvit_sac_miou.py \
        --checkpoint checkpoints/canvit_sac/latest.pt \
        --t 5 --miou-mode mean
    uv run python scripts/eval_canvit_sac_miou.py \
        --checkpoint checkpoints/canvit_sac/actor_final.pt \
        --t 5 --miou-mode accumulator
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_specialize.datasets.ade20k import (
    ADE20kDataset,
    IGNORE_LABEL,
    NUM_CLASSES,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.canvas.state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    empty_viewpoint_history,
)
from canvit_rl.greedy import miou_from_state
from canvit_rl.sac_models import (
    CanViTSequenceEncoder,
    CanvasStateActor,
    GaussianActor,
)
from canvit_rl.sac_state import (
    append_glimpse,
    batch_from_sequence,
    empty_sequence,
    extract_local_patches,
)
from scripts.train_canvit_sac import _action_to_viewpoint


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


def _load_checkpoint(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Load either latest.pt dict checkpoints or bare actor_final.pt weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        return checkpoint["actor"], dict(checkpoint.get("args", {})), checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, {}, {}
    raise TypeError(f"Unsupported checkpoint format at {path}")


def _checkpoint_value(
    checkpoint_args: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    """Read a training arg from checkpoint metadata with CLI fallback."""
    return checkpoint_args.get(key.replace("-", "_"), fallback)


def _is_canvas_checkpoint(payload: dict[str, Any]) -> bool:
    """Detect image-dependent Canvas SAC checkpoints from saved metadata."""
    state_representation = str(
        payload.get(
            "state_representation",
            payload.get("metadata", {}).get("state_representation", ""),
        )
    )
    return state_representation == "current_canvas_layernorm_with_viewpoint_history"


def _build_actor(
    *,
    actor_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    checkpoint_args: dict[str, Any],
    device: torch.device,
) -> GaussianActor:
    """Reconstruct the SAC actor architecture and load checkpoint weights."""
    d_model = int(_checkpoint_value(checkpoint_args, "d-model", args.d_model))
    n_heads = int(_checkpoint_value(checkpoint_args, "n-heads", args.n_heads))
    n_layers = int(_checkpoint_value(checkpoint_args, "n-layers", args.n_layers))
    n_patches = int(_checkpoint_value(checkpoint_args, "n-patches", args.n_patches))
    patch_dim = int(_checkpoint_value(checkpoint_args, "patch-dim", args.patch_dim))
    max_steps = int(_checkpoint_value(checkpoint_args, "t", args.t))
    encoder = CanViTSequenceEncoder(
        patch_dim=patch_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_steps=max_steps,
        n_patches=n_patches,
    )
    actor = GaussianActor(encoder, d_model).to(device).eval()
    actor.load_state_dict(actor_state)
    for param in actor.parameters():
        param.requires_grad_(False)
    return actor


def _build_canvas_actor(
    *,
    actor_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    checkpoint_args: dict[str, Any],
    payload: dict[str, Any],
    device: torch.device,
) -> CanvasStateActor:
    """Reconstruct a current-canvas SAC actor and load checkpoint weights."""
    d_model = int(_checkpoint_value(checkpoint_args, "d-model", args.d_model))
    rff_dim = int(_checkpoint_value(checkpoint_args, "rff-dim", args.rff_dim))
    rff_seed = int(_checkpoint_value(checkpoint_args, "rff-seed", args.rff_seed))
    canvas_feature_dim = payload.get(
        "canvas_feature_dim",
        payload.get("metadata", {}).get("canvas_feature_dim", args.canvas_feature_dim),
    )
    if canvas_feature_dim is None:
        raise ValueError(
            "Canvas SAC checkpoint is missing canvas_feature_dim; pass "
            "--canvas-feature-dim or use a train_canvas_sac.py checkpoint."
        )
    actor = CanvasStateActor(
        canvas_feature_dim=int(canvas_feature_dim),
        d_model=d_model,
        rff_dim=rff_dim,
        rff_seed=rff_seed,
    ).to(device).eval()
    actor.load_state_dict(actor_state)
    for param in actor.parameters():
        param.requires_grad_(False)
    return actor


def _deterministic_action(
    actor: GaussianActor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return tanh(mean) action for deterministic policy evaluation."""
    mean, _ = actor(batch)
    return torch.tanh(mean)


def _deterministic_canvas_action(
    actor: CanvasStateActor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return deterministic action for image-dependent Canvas SAC actors."""
    return actor.deterministic_action(batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/canvit_sac/latest.pt"),
    )
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="validation",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/sac_miou.pt"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-patches", type=int, default=64)
    parser.add_argument("--patch-dim", type=int, default=768)
    parser.add_argument("--rff-dim", type=int, default=128)
    parser.add_argument("--rff-seed", type=int, default=42)
    parser.add_argument("--max-history", type=int, default=6)
    parser.add_argument("--canvas-feature-dim", type=int, default=None)
    parser.add_argument(
        "--miou-mode",
        choices=["accumulator", "mean"],
        default="accumulator",
        help=(
            "accumulator uses dataset-level mIoUAccumulator; mean averages "
            "per-image mIoU values without dataset-level integration"
        ),
    )
    args = parser.parse_args()

    if args.t < 0:
        raise ValueError("--t must be non-negative.")
    if args.min_scale <= 0 or args.min_scale > 1:
        raise ValueError("Require 0 < --min-scale <= 1.")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive.")

    torch.manual_seed(args.seed)
    cfg = CanViTEnvConfig()
    device = get_device()
    print(f"Device: {device}")

    actor_state, checkpoint_args, checkpoint_payload = _load_checkpoint(args.checkpoint)
    is_canvas_policy = _is_canvas_checkpoint(checkpoint_payload)
    checkpoint_t = int(_checkpoint_value(checkpoint_args, "t", args.t))
    if args.t != checkpoint_t:
        print(
            f"Using --t {args.t} for rollout; checkpoint actor was trained "
            f"with max_steps={checkpoint_t}"
        )
    n_patches = int(_checkpoint_value(checkpoint_args, "n-patches", args.n_patches))
    patch_dim = int(_checkpoint_value(checkpoint_args, "patch-dim", args.patch_dim))
    min_scale = float(_checkpoint_value(checkpoint_args, "min-scale", args.min_scale))
    max_history = int(_checkpoint_value(checkpoint_args, "max-history", args.max_history))
    if is_canvas_policy and args.t + 1 > max_history:
        raise ValueError(
            f"Canvas SAC rollout needs max_history >= t+1, got "
            f"max_history={max_history} and t={args.t}."
        )

    if is_canvas_policy:
        actor = _build_canvas_actor(
            actor_state=actor_state,
            args=args,
            checkpoint_args=checkpoint_args,
            payload=checkpoint_payload,
            device=device,
        )
        policy_name = "canvas_sac"
        print("Policy: canvas-dependent SAC")
    else:
        actor = _build_actor(
            actor_state=actor_state,
            args=args,
            checkpoint_args=checkpoint_args,
            device=device,
        )
        policy_name = "canvit_sac"
        print("Policy: viewpoint-history SAC")

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
    for param in model.parameters():
        param.requires_grad_(False)
    for param in probe.parameters():
        param.requires_grad_(False)

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
    total_eval_images = len(dataset)
    if args.max_images is not None:
        total_eval_images = min(total_eval_images, args.max_images)
    pbar = tqdm(
        total=total_eval_images,
        desc="Evaluating SAC",
        miniters=args.progress_interval,
        maxinterval=float("inf"),
    )

    with torch.inference_mode():
        for image_idx, (image, mask) in enumerate(loader):
            if args.max_images is not None and image_idx >= args.max_images:
                break

            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            n_images += 1
            state = model.init_state(
                batch_size=1,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            seq = empty_sequence(n_patches=n_patches, patch_dim=patch_dim)
            coords, lengths = empty_viewpoint_history(
                batch_size=1,
                max_steps=max_history,
                device=device,
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
            coords, lengths = append_viewpoint_history(
                coords=coords,
                lengths=lengths,
                viewpoint=full_vp,
                step=0,
            )
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=cfg.canvas_grid_size,
            )

            step_states = [state]
            step_scales = [1.0]
            for step_idx in range(args.t):
                if is_canvas_policy:
                    batch = {
                        "canvas": canvas_summary,
                        "coords": coords,
                        "lengths": lengths,
                    }
                    if args.stochastic:
                        action, _ = actor.sample(batch)
                    else:
                        action = _deterministic_canvas_action(actor, batch)
                else:
                    batch = batch_from_sequence(
                        seq,
                        max_steps=checkpoint_t,
                        n_patches=n_patches,
                        patch_dim=patch_dim,
                        device=device,
                    )
                    if args.stochastic:
                        action, _ = actor.sample(batch)
                    else:
                        action = _deterministic_action(actor, batch)
                vp = _action_to_viewpoint(action, min_scale=min_scale)
                glimpse = sample_at_viewpoint(
                    spatial=image,
                    viewpoint=vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                )
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state
                patches = extract_local_patches(out)
                seq = append_glimpse(
                    seq=seq,
                    patches=patches,
                    viewpoint=vp,
                )
                coords, lengths = append_viewpoint_history(
                    coords=coords,
                    lengths=lengths,
                    viewpoint=vp,
                    step=step_idx + 1,
                )
                if is_canvas_policy:
                    canvas_summary = canvas_layernorm_spatial(
                        model=model,
                        state=state,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                step_states.append(state)
                step_scales.append(float(vp.scales[0].detach().cpu().item()))

            for step_idx, step_state in enumerate(step_states):
                if args.miou_mode == "accumulator":
                    assert accs is not None
                    spatial = model.get_spatial(step_state.canvas).view(
                        1,
                        cfg.canvas_grid_size,
                        cfg.canvas_grid_size,
                        -1,
                    )
                    _update_miou(accs[step_idx], probe, spatial, mask)
                else:
                    miou_sums[step_idx] += miou_from_state(
                        model=model,
                        state=step_state,
                        probe=probe,
                        mask=mask,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                scale_sums[step_idx] += step_scales[step_idx]
                count_sums[step_idx] += 1

            if n_images % args.progress_interval == 0:
                pbar.update(args.progress_interval)

    remainder = n_images % args.progress_interval
    if remainder:
        pbar.update(remainder)
    pbar.close()

    if args.miou_mode == "accumulator":
        assert accs is not None
        mious = {f"t{t}": acc.compute() for t, acc in enumerate(accs)}
    else:
        mious = {f"t{t}": miou_sums[t] / count_sums[t] for t in range(n_steps)}
    mean_scales = {
        f"t{t}": scale_sums[t] / count_sums[t] for t in range(n_steps)
    }
    wall_time = time.monotonic() - t_start

    title = (
        "Dataset-Level SAC Actor Metrics"
        if args.miou_mode == "accumulator"
        else "Mean SAC Actor Metrics"
    )
    print(f"\n--- {title} ---")
    for step_idx in range(n_steps):
        key = f"t{step_idx}"
        label = "full_scene" if step_idx == 0 else "actor"
        print(
            f"  t={step_idx} ({label}): "
            f"scale={mean_scales[key]:.3f}  "
            f"miou={mious[key]:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mious": mious,
            "mean_scales": mean_scales,
            "metadata": {
                "policy": policy_name,
                "checkpoint": str(args.checkpoint),
                "dataset": args.dataset,
                "split": args.split,
                "n_images": n_images,
                "canvas_grid_size": cfg.canvas_grid_size,
                "glimpse_size_px": cfg.glimpse_size_px,
                "scene_size_px": cfg.scene_size_px,
                "n_actor_glimpses": args.t,
                "n_logged_steps": n_steps,
                "min_scale": min_scale,
                "probe_repo": probe_repo,
                "model_repo": cfg.checkpoint,
                "seed": args.seed,
                "stochastic": args.stochastic,
                "miou_mode": args.miou_mode,
                "wall_time_seconds": wall_time,
            },
        },
        args.output,
    )
    print(f"\nSaved {args.output} after {wall_time:.1f}s")


if __name__ == "__main__":
    main()
