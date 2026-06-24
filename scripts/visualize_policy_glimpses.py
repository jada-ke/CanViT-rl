"""
Visualize where active-view policies look over a CanViT episode.

Usage:
    python scripts/visualize_policy_glimpses.py
    python scripts/visualize_policy_glimpses.py --image-index 0 --t 5 --k 32
    python scripts/visualize_policy_glimpses.py --policy eg-c2f --t 5 --image-index 3648 --split training
    python scripts/visualize_policy_glimpses.py \
        --policy viewpoint-bc \
        --policy-checkpoint checkpoints/viewpoint_bc/im32_k16_t1/latest.pt \
        --t 1 --image-index 0
    python scripts/visualize_policy_glimpses.py \
        --policy viewpoint-sac \
        --policy-checkpoint checkpoints/viewpoint_sac/full-im32-t1/best.pt \
        --t 1 --image-index 0
    python scripts/visualize_policy_glimpses.py \
        --policy canvas-sac \
        --policy-checkpoint checkpoints/canvas_sac/im1-t1/best.pt \
        --t 1 --image-index 0 --image-index 1
    python scripts/visualize_policy_glimpses.py \
        --episodes 8 --split validation --output-dir results/greedy_glimpses
    uv run python scripts/visualize_policy_glimpses.py \
        --policy canvas-sac \
        --policy-checkpoint checkpoints/canvas_sac/im1-t1/best.pt \
        --t 1 \
        --image-index 0 --image-index 1 \
        --split training \
        --output-dir results/canvas_policy_glimpses

  uv run python scripts/visualize_policy_glimpses.py \
        --policy canvas-sac  \
        --policy-checkpoint checkpoints/canvas_sac/synthetic-im1-t1_2000/best.pt \
        --dataset synthetic_segmentation \
        --t 1 \
        --image-index 0  \
        --output-dir results/canvas_policy_glimpses/synthetic-im1-t1_2000
"""

from __future__ import annotations

import argparse
import random
import sys
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
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    ADE20kDataset,
    make_val_transforms,
)
from PIL import Image

EVAL_REPO = Path(__file__).resolve().parents[1] / "CanViT-eval"
if EVAL_REPO.is_dir() and str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))

from canvit_eval.episode import run_episode  # noqa: E402
from canvit_eval.policies import make_policy  # noqa: E402

from canvit_rl.ade_labels import remap_ade_mask_labels
from canvit_rl.canvas_state import canvas_layernorm_spatial
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import miou_from_state, run_greedy_episode
from canvit_rl.sac_models import CanvasStateActor
from canvit_rl.viewpoint_policy import (
    ViewpointGaussianActor,
    action_to_viewpoint,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class SyntheticSegmentationDataset(torch.utils.data.Dataset):
    """Folder dataset for synthetic images/ and masks/ segmentation roots."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        *,
        root: Path,
        split: str,
        scene_size_px: int,
        img_transform,
    ) -> None:
        split_image_dir = root / "images" / split
        split_mask_dir = root / "masks" / split
        image_dir = split_image_dir if split_image_dir.is_dir() else root / "images"
        mask_dir = split_mask_dir if split_mask_dir.is_dir() else root / "masks"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
        self.images = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        mask_by_stem = {
            path.stem: path
            for path in mask_dir.iterdir()
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        }
        self.masks = [mask_by_stem[path.stem] for path in self.images]
        self.scene_size_px = scene_size_px
        self.img_transform = img_transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        image_tensor = self.img_transform(image)
        resample_nearest = getattr(Image, "Resampling", Image).NEAREST
        mask = mask.resize(
            (self.scene_size_px, self.scene_size_px),
            resample=resample_nearest,
        )
        # Fixed by Codex on 2026-06-24
        # Problem: Older synthetic masks may contain raw ADE ids 1..150, and
        # label 150 is out of bounds for visualization CE metrics.
        # Solution: preserve semantic labels but normalize raw ADE ids to
        # zero-based 0..149 while keeping IGNORE_LABEL=255 padding.
        # Result: Synthetic roots can be visualized without ADE-style split
        # subfolders while retaining valid semantic targets.
        mask_tensor = torch.from_numpy(
            remap_ade_mask_labels(np.asarray(mask)).astype(np.int64)
        )
        return image_tensor, mask_tensor


def _build_dataset(*, root: Path, split: str, cfg: CanViTEnvConfig, img_tf, mask_tf):
    """Auto-detect synthetic roots; otherwise use ADE20K split folders."""
    if (
        ((root / "images" / split).is_dir() and (root / "masks" / split).is_dir())
        or ((root / "images").is_dir() and (root / "masks").is_dir())
    ):
        # Fixed by Codex on 2026-06-24
        # Problem: Policy-glimpse visualization needed to read the same
        # synthetic training/validation split layout produced by the generator.
        # Solution: detect images/<split> and masks/<split> first, falling back
        # to the old flat images/ and masks/ layout.
        # Result: --dataset synthetic_segmentation --split validation reads the
        # validation synthetic split without custom paths.
        return SyntheticSegmentationDataset(
            root=root,
            split=split,
            scene_size_px=cfg.scene_size_px,
            img_transform=img_tf,
        )
    return ADE20kDataset(
        root=root,
        split=split,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )


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


def _segmentation_ce_loss(
    *,
    model,
    probe: torch.nn.Module,
    state,
    mask: torch.Tensor,
    canvas_grid_size: int,
) -> float:
    """Compute unweighted ADE20K cross-entropy for one canvas state."""
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


def _mask_for_plot(mask: torch.Tensor, num_classes: int = 150) -> torch.Tensor:
    """Convert one ADE20K target mask to an RGB label image."""
    mask_cpu = mask.detach().cpu().long()
    if mask_cpu.ndim == 3:
        mask_cpu = mask_cpu.squeeze(0)
    valid = mask_cpu != IGNORE_LABEL
    safe_mask = mask_cpu.clamp(0, num_classes - 1)
    palette = _ade_palette(num_classes)
    rgb = palette[safe_mask]
    rgb[~valid] = torch.tensor([0.2, 0.2, 0.2])
    return rgb


def _viewpoint_rect(
    viewpoint: Viewpoint,
    *,
    image_size: int,
    index: int = 0,
) -> tuple[float, float, float, float]:
    """Convert a CanViT [-1, 1] center plus scale into pixel rectangle bounds."""
    # Fixed by Codex on 2026-06-12
    # Problem: Viewpoint centers use CanViT's matrix-indexing order (y,x), but
    # the visualizer previously treated them as Cartesian (x,y).
    # Solution: unpack row/col as y/x before converting to image-space pixels.
    # Result: boxes align with the same convention used by sample_at_viewpoint.
    cy, cx = viewpoint.centers[index].detach().cpu().tolist()
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
    mask: torch.Tensor,
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
    mask_np = _mask_for_plot(mask).numpy()
    image_size = image.shape[-1]
    n_steps = len(viewpoints)
    fig, axes = plt.subplots(3, n_steps, figsize=(4 * n_steps, 12), dpi=150)
    axes_grid = axes.reshape(3, n_steps)
    colors = plt.cm.viridis(torch.linspace(0.05, 0.95, n_steps).numpy())
    fig.suptitle(f"{title}\nΔCE gain = previous CE loss - current CE loss; positive is better")

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
            f"CE loss={scores[step_idx]:.3f} "
            f"ΔCE gain={loss_reductions[step_idx]:+.3f}\n"
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
            f"CE loss={scores[step_idx]:.3f} miou={mious[step_idx]:.3f}"
        )
        seg_ax.axis("off")

        expected_ax = axes_grid[2, step_idx]
        expected_ax.imshow(mask_np)
        expected_rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2.0,
            edgecolor=colors[step_idx],
            facecolor="none",
        )
        expected_ax.add_patch(expected_rect)
        expected_ax.set_title(f"t{step_idx} expected segmentation")
        expected_ax.axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _save_multi_image_visualization(
    *,
    rows: list[dict],
    output: Path,
    title: str,
) -> None:
    """Save one contact sheet with each selected image on its own row."""
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires matplotlib. Install it in this environment "
            "or add it to the dev dependencies."
        ) from exc

    if not rows:
        return
    n_images = len(rows)
    n_steps = len(rows[0]["result"]["viewpoints"])
    fig, axes = plt.subplots(
        n_images,
        n_steps,
        figsize=(4 * n_steps, max(3.2 * n_images, 3.6)),
        dpi=150,
        squeeze=False,
    )
    colors = plt.cm.viridis(torch.linspace(0.05, 0.95, n_steps).numpy())

    for row_idx, row in enumerate(rows):
        image = row["image"]
        result = row["result"]
        image_np = _image_for_plot(image)
        image_size = image.shape[-1]
        for step_idx, vp in enumerate(result["viewpoints"]):
            ax = axes[row_idx, step_idx]
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
                f"{row['label']} t{step_idx}\n"
                f"s={scale:.2f} c=({center[0]:+.2f},{center[1]:+.2f})\n"
                f"CE={result['scores'][step_idx]:.3f} "
                f"ΔCE={result['rewards'][step_idx]:+.3f} "
                f"miou={result['mious'][step_idx]:.3f}"
            )
            ax.axis("off")

    fig.suptitle(f"{title}\nΔCE gain = previous CE loss - current CE loss; positive is better")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _run_eg_c2f_episode(
    *,
    model,
    probe: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    t: int,
    cfg: CanViTEnvConfig,
    device: torch.device,
) -> dict:
    """Run canvit-eval's entropy-guided coarse-to-fine policy for visualization."""
    # Fixed by Codex on 2026-06-12
    # Problem: the visualizer only supported the local k-greedy rollout path.
    # Solution: call canvit-eval's policy/episode API directly and adapt its
    # returned steps into the same dict shape used by the renderer.
    # Result: k-greedy and EG-C2F visualizations share one plotting pipeline.
    policy = make_policy(
        "entropy_coarse_to_fine",
        batch_size=1,
        device=device,
        n_viewpoints=t,
        canvas_grid=cfg.canvas_grid_size,
        probe=probe,
        get_spatial_fn=model.get_spatial,
    )
    steps = run_episode(
        model=model,
        images=image,
        policy=policy,
        n_timesteps=t,
        canvas_grid=cfg.canvas_grid_size,
        glimpse_px=cfg.glimpse_size_px,
    )

    viewpoints, states, scores, rewards, mious = [], [], [], [], []
    prev_score = None
    for step in steps:
        score = _segmentation_ce_loss(
            model=model,
            probe=probe,
            state=step.state,
            mask=mask,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        reward = 0.0 if prev_score is None else prev_score - score
        prev_score = score
        viewpoints.append(step.viewpoint)
        states.append(step.state)
        scores.append(score)
        rewards.append(reward)
        mious.append(
            miou_from_state(
                model=model,
                state=step.state,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
        )
    return {
        "viewpoints": viewpoints,
        "states": states,
        "scores": scores,
        "rewards": rewards,
        "mious": mious,
    }


def _checkpoint_arg(checkpoint_args: dict, key: str, fallback):
    """Read a saved training arg with CLI fallback for bare actor checkpoints."""
    return checkpoint_args.get(key.replace("-", "_"), fallback)


def _build_viewpoint_actor_from_checkpoint(
    *,
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ViewpointGaussianActor, dict]:
    """Load a viewpoint-history actor from actor-only or full SAC checkpoints."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = {}
    if isinstance(payload, dict) and "actor" in payload:
        # Fixed by Codex on 2026-06-18
        # Problem: SAC best.pt stores actor and critics together, while the
        # visualizer previously documented only BC actor checkpoints.
        # Solution: Treat any checkpoint with an "actor" key as a compatible
        # viewpoint-history actor source and ignore critic weights for rollout.
        # Result: Full SAC checkpoints can be visualized directly.
        actor_state = payload["actor"]
        checkpoint_args = dict(payload.get("args", {}))
    elif isinstance(payload, dict):
        actor_state = payload
    else:
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")

    max_history = int(_checkpoint_arg(checkpoint_args, "max-history", args.t + 1))
    d_model = int(_checkpoint_arg(checkpoint_args, "d-model", args.d_model))
    rff_dim = int(_checkpoint_arg(checkpoint_args, "rff-dim", args.rff_dim))
    rff_seed = int(_checkpoint_arg(checkpoint_args, "rff-seed", args.rff_seed))
    actor = ViewpointGaussianActor(
        d_model=d_model,
        max_steps=max_history,
        rff_dim=rff_dim,
        rff_seed=rff_seed,
    ).to(device).eval()
    actor.load_state_dict(actor_state)
    for param in actor.parameters():
        param.requires_grad_(False)
    return actor, checkpoint_args


def _build_learned_actor_from_checkpoint(
    *,
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, str, int]:
    """Load either a viewpoint-history actor or an image-dependent canvas actor."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = dict(payload.get("args", {})) if isinstance(payload, dict) else {}
    state_representation = (
        str(payload.get("state_representation", "viewpoint_history"))
        if isinstance(payload, dict)
        else "viewpoint_history"
    )
    if state_representation == "current_canvas_layernorm_with_viewpoint_history":
        if not isinstance(payload, dict) or "actor" not in payload:
            raise ValueError(f"Expected canvas SAC checkpoint with actor: {checkpoint}")
        # Fixed by Codex on 2026-06-23
        # Problem: policy-glimpse visualization only loaded image-independent
        # viewpoint actors, so canvas SAC checkpoints could not be rolled out.
        # Solution: detect train_canvas_sac.py checkpoint metadata and rebuild
        # CanvasStateActor with the saved canvas feature dimension.
        # Result: Image-dependent SAC policies can be visualized with the same
        # command-line tool as greedy, EG-C2F, and viewpoint-history policies.
        d_model = int(_checkpoint_arg(checkpoint_args, "d-model", args.d_model))
        rff_dim = int(_checkpoint_arg(checkpoint_args, "rff-dim", args.rff_dim))
        rff_seed = int(_checkpoint_arg(checkpoint_args, "rff-seed", args.rff_seed))
        max_history = int(_checkpoint_arg(checkpoint_args, "max-history", args.t + 1))
        actor = CanvasStateActor(
            canvas_feature_dim=int(payload["canvas_feature_dim"]),
            d_model=d_model,
            rff_dim=rff_dim,
            rff_seed=rff_seed,
        ).to(device).eval()
        actor.load_state_dict(payload["actor"])
        policy_kind = "canvas-sac"
    else:
        actor, checkpoint_args = _build_viewpoint_actor_from_checkpoint(
            checkpoint=checkpoint,
            args=args,
            device=device,
        )
        max_history = actor.max_steps
        policy_kind = "viewpoint-sac"
    for param in actor.parameters():
        param.requires_grad_(False)
    return actor, checkpoint_args, policy_kind, max_history


def _append_viewpoint_history(
    coords: torch.Tensor,
    lengths: torch.Tensor,
    viewpoint: Viewpoint,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write one CanViT Viewpoint into the actor history at a fixed slot."""
    if step >= coords.shape[1]:
        raise ValueError(
            f"History slot {step} is out of range for loaded actor "
            f"(max_history={coords.shape[1]})."
        )
    coords[:, step, :2] = viewpoint.centers.detach().float()
    coords[:, step, 2] = viewpoint.scales.detach().float()
    return coords, lengths + 1


def _record_state_metrics(
    *,
    model,
    probe: torch.nn.Module,
    state,
    mask: torch.Tensor,
    canvas_grid_size: int,
) -> tuple[float, float]:
    """Return CE and mIoU for one committed recurrent state."""
    score = _segmentation_ce_loss(
        model=model,
        probe=probe,
        state=state,
        mask=mask,
        canvas_grid_size=canvas_grid_size,
    )
    miou = miou_from_state(
        model=model,
        state=state,
        probe=probe,
        mask=mask,
        canvas_grid_size=canvas_grid_size,
    )
    return score, miou


def _run_viewpoint_bc_episode(
    *,
    actor: ViewpointGaussianActor,
    model,
    probe: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    t: int,
    min_scale: float,
    cfg: CanViTEnvConfig,
    device: torch.device,
) -> dict:
    """Roll out a loaded image-independent viewpoint BC actor."""
    batch_size = image.shape[0]
    max_history = actor.max_steps
    if t + 1 > max_history:
        raise ValueError(
            f"Requested --t {t} learned glimpses, but checkpoint max_history "
            f"is {max_history}. Use --t <= {max_history - 1}."
        )
    state = model.init_state(
        batch_size=batch_size,
        canvas_grid_size=cfg.canvas_grid_size,
    )
    coords = torch.zeros(batch_size, max_history, 3, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    viewpoints, states, scores, rewards, mious = [], [], [], [], []

    full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
    full_glimpse = sample_at_viewpoint(
        spatial=image,
        viewpoint=full_vp,
        glimpse_size_px=cfg.glimpse_size_px,
    )
    out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
    state = out.state
    coords, lengths = _append_viewpoint_history(coords, lengths, full_vp, step=0)
    score, miou = _record_state_metrics(
        model=model,
        probe=probe,
        state=state,
        mask=mask,
        canvas_grid_size=cfg.canvas_grid_size,
    )
    viewpoints.append(full_vp)
    states.append(state)
    scores.append(score)
    rewards.append(0.0)
    mious.append(miou)
    prev_score = score

    for step in range(1, t + 1):
        action = actor.deterministic_action({"coords": coords, "lengths": lengths})
        vp = action_to_viewpoint(action, min_scale=min_scale)
        glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state
        coords, lengths = _append_viewpoint_history(coords, lengths, vp, step=step)
        score, miou = _record_state_metrics(
            model=model,
            probe=probe,
            state=state,
            mask=mask,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        viewpoints.append(vp)
        states.append(state)
        scores.append(score)
        rewards.append(prev_score - score)
        mious.append(miou)
        prev_score = score

    return {
        "viewpoints": viewpoints,
        "states": states,
        "scores": scores,
        "rewards": rewards,
        "mious": mious,
    }


def _run_canvas_sac_episode(
    *,
    actor: CanvasStateActor,
    model,
    probe: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    t: int,
    max_history: int,
    min_scale: float,
    cfg: CanViTEnvConfig,
    device: torch.device,
) -> dict:
    """Roll out a loaded image-dependent canvas SAC actor."""
    batch_size = image.shape[0]
    if t + 1 > max_history:
        raise ValueError(
            f"Requested --t {t} learned glimpses, but checkpoint max_history "
            f"is {max_history}. Use --t <= {max_history - 1}."
        )
    state = model.init_state(
        batch_size=batch_size,
        canvas_grid_size=cfg.canvas_grid_size,
    )
    coords = torch.zeros(batch_size, max_history, 3, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    viewpoints, states, scores, rewards, mious = [], [], [], [], []

    full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
    full_glimpse = sample_at_viewpoint(
        spatial=image,
        viewpoint=full_vp,
        glimpse_size_px=cfg.glimpse_size_px,
    )
    out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
    state = out.state
    coords, lengths = _append_viewpoint_history(coords, lengths, full_vp, step=0)
    score, miou = _record_state_metrics(
        model=model,
        probe=probe,
        state=state,
        mask=mask,
        canvas_grid_size=cfg.canvas_grid_size,
    )
    viewpoints.append(full_vp)
    states.append(state)
    scores.append(score)
    rewards.append(0.0)
    mious.append(miou)
    prev_score = score

    for step in range(1, t + 1):
        # Fixed by Codex on 2026-06-23
        # Problem: The canvas actor's observation includes the current CanViT
        # canvas, not just the fixed-slot viewpoint history.
        # Solution: reconstruct the layernorm-spatial canvas summary at each
        # committed state before asking the actor for the next viewpoint.
        # Result: The rollout matches train_canvas_sac.py's observation
        # contract for image-dependent policies.
        canvas_summary = canvas_layernorm_spatial(
            model=model,
            state=state,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        action = actor.deterministic_action(
            {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
        )
        vp = action_to_viewpoint(action, min_scale=min_scale)
        glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=vp,
            glimpse_size_px=cfg.glimpse_size_px,
        )
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state
        coords, lengths = _append_viewpoint_history(coords, lengths, vp, step=step)
        score, miou = _record_state_metrics(
            model=model,
            probe=probe,
            state=state,
            mask=mask,
            canvas_grid_size=cfg.canvas_grid_size,
        )
        viewpoints.append(vp)
        states.append(state)
        scores.append(score)
        rewards.append(prev_score - score)
        mious.append(miou)
        prev_score = score

    return {
        "viewpoints": viewpoints,
        "states": states,
        "scores": scores,
        "rewards": rewards,
        "mious": mious,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["k-greedy", "eg-c2f", "viewpoint-bc", "viewpoint-sac", "canvas-sac"],
        default="k-greedy",
        help="Policy rollout to visualize.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint for --policy viewpoint-bc, viewpoint-sac, or canvas-sac."
        ),
    )
    parser.add_argument("--t", type=int, default=5, help="Timesteps per episode")
    parser.add_argument("--k", type=int, default=50, help="Candidates per greedy step")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--image-index", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ADE20k",
        help=(
            "ADE20K root, or synthetic root containing images/<split> and "
            "masks/<split> folders."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
    )
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--rff-dim", type=int, default=128)
    parser.add_argument("--rff-seed", type=int, default=42)
    parser.add_argument("--min-scale", type=float, default=0.05)
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
    if args.policy == "eg-c2f" and args.t > 21:
        raise ValueError("eg-c2f has 21 built-in coarse-to-fine timesteps.")
    if args.policy != "k-greedy" and args.no_full_scene_start:
        raise ValueError("--no-full-scene-start only applies to --policy k-greedy.")
    if (
        args.policy in {"viewpoint-bc", "viewpoint-sac", "canvas-sac"}
        and args.policy_checkpoint is None
    ):
        raise ValueError(
            f"--policy {args.policy} requires --policy-checkpoint."
        )
    if args.min_scale <= 0 or args.min_scale >= 1:
        raise ValueError("Require 0 < --min-scale < 1.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = CanViTEnvConfig()
    device = get_device()
    print(f"Device: {device}")

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = _build_dataset(
        root=Path(args.dataset),
        split=args.split,
        cfg=cfg,
        img_tf=img_tf,
        mask_tf=mask_tf,
    )
    if args.image_index is not None:
        indices = args.image_index
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

    actor = None
    ckpt_min_scale = args.min_scale
    loaded_policy_kind = args.policy
    actor_max_history = args.t + 1
    if args.policy in {"viewpoint-bc", "viewpoint-sac", "canvas-sac"}:
        assert args.policy_checkpoint is not None
        actor, checkpoint_args, loaded_policy_kind, actor_max_history = (
            _build_learned_actor_from_checkpoint(
                checkpoint=args.policy_checkpoint,
                args=args,
                device=device,
            )
        )
        if args.policy == "canvas-sac" and loaded_policy_kind != "canvas-sac":
            raise ValueError(
                "--policy canvas-sac requires a canvas SAC checkpoint from "
                "scripts/train_canvas_sac.py."
            )
        if (
            args.policy in {"viewpoint-bc", "viewpoint-sac"}
            and loaded_policy_kind == "canvas-sac"
        ):
            raise ValueError(
                f"--policy {args.policy} received a canvas SAC checkpoint; "
                "use --policy canvas-sac."
            )
        if loaded_policy_kind != "canvas-sac":
            loaded_policy_kind = args.policy
        ckpt_t = int(_checkpoint_arg(checkpoint_args, "t", args.t))
        ckpt_min_scale = float(
            _checkpoint_arg(checkpoint_args, "min-scale", args.min_scale)
        )
        if args.t != ckpt_t:
            print(
                f"Using --t {args.t} for rollout; checkpoint was trained "
                f"with t={ckpt_t}"
            )
        print(f"Loaded {loaded_policy_kind} policy: {args.policy_checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with torch.inference_mode():
        for idx in indices:
            image, mask = dataset[idx]
            image_dev = image.unsqueeze(0).to(device)
            mask_dev = mask.unsqueeze(0).to(device)
            if args.policy == "k-greedy":
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
                policy_label = f"k-greedy-k{args.k}"
                title = f"k-greedy k={args.k} idx={idx}"
            elif args.policy == "eg-c2f":
                result = _run_eg_c2f_episode(
                    model=model,
                    probe=probe,
                    image=image_dev,
                    mask=mask_dev,
                    t=args.t,
                    cfg=cfg,
                    device=device,
                )
                policy_label = "eg-c2f"
                title = f"EG-C2F idx={idx}"
            else:
                assert actor is not None
                if loaded_policy_kind == "canvas-sac":
                    result = _run_canvas_sac_episode(
                        actor=actor,
                        model=model,
                        probe=probe,
                        image=image_dev,
                        mask=mask_dev,
                        t=args.t,
                        max_history=actor_max_history,
                        min_scale=ckpt_min_scale,
                        cfg=cfg,
                        device=device,
                    )
                else:
                    result = _run_viewpoint_bc_episode(
                        actor=actor,
                        model=model,
                        probe=probe,
                        image=image_dev,
                        mask=mask_dev,
                        t=args.t,
                        min_scale=ckpt_min_scale,
                        cfg=cfg,
                        device=device,
                    )
                policy_label = loaded_policy_kind
                title = f"{loaded_policy_kind} idx={idx}"
            name = dataset.images[idx].stem
            rows.append(
                {
                    "image": image,
                    "mask": mask,
                    "result": result,
                    "idx": idx,
                    "name": name,
                    "label": f"{args.split} {idx:05d}",
                    "policy_label": policy_label,
                    "title": title,
                }
            )

    if len(rows) == 1:
        row = rows[0]
        output = (
            args.output_dir
            / f"{args.split}_{row['idx']:05d}_{row['name']}_{row['policy_label']}.png"
        )
        _save_visualization(
            image=row["image"],
            mask=row["mask"],
            viewpoints=row["result"]["viewpoints"],
            states=row["result"]["states"],
            model=model,
            probe=probe,
            canvas_grid_size=cfg.canvas_grid_size,
            scores=row["result"]["scores"],
            loss_reductions=row["result"]["rewards"],
            mious=row["result"]["mious"],
            output=output,
            title=row["title"],
        )
        print(f"Saved {output}")
    else:
        policy_label = rows[0]["policy_label"]
        index_label = "-".join(f"{row['idx']:05d}" for row in rows)
        output = args.output_dir / (
            f"{args.split}_{policy_label}_rows_{index_label}.png"
        )
        _save_multi_image_visualization(
            rows=rows,
            output=output,
            title=f"{policy_label} {args.split} policy glimpses",
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
