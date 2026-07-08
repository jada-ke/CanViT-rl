"""
Visualize true and predicted SAC reward landscapes for t1 viewpoints.

For each selected image, the script:
1. Runs the full-scene warm-up.
2. Sweeps a grid of candidate centers for each requested scale.
3. Computes true one-step reward: CE_before - CE_after.
4. Computes predicted Q(state, action) from a full SAC checkpoint.
5. Saves side-by-side heatmaps for true reward and critic prediction.

Example:

    python scripts/visualize_sac_reward_maps.py \
        --checkpoint checkpoints/viewpoint_sac/full-im32-t1/best.pt \
        --image-index 0 --image-index 1 \
        --grid-size 21 \
        --scales 0.25,0.50 \
        --output-dir results/sac_reward_maps

    python scripts/visualize_sac_reward_maps.py \
        --checkpoint checkpoints/canvas_sac/im1-t1/best.pt \
        --image-index 0 --split training \
        --grid-size 21 \
        --scales 0.25,0.50 \
        --output-dir results/sac_canvas_reward_maps

    uv run python scripts/visualize_sac_reward_maps.py \
        --checkpoint checkpoints/canvas_sac/synthetic-im7-t1-10_000-critlr_3/best.pt \
        --dataset synthetic_segmentation \
        --split training \
        --image-index 0,1,2,3,4,5,6 \
        --state-steps 0,1 \
        --grid-size 21 \
        --scales 0.25,0.50 \
        --chunk-size 16 \
        --output-dir results/synthetic-im7-t1-10_000-critlr_3

    uv run python scripts/visualize_sac_reward_maps.py \
        --checkpoint checkpoints/canvas_critic/canvas_synthetic_im7_t1_k32_5000/best.pt \
        --dataset synthetic_segmentation \
        --split validation \
        --image-index 0,1,2 \
        --grid-size 21 \
        --scales 0.25,0.50 \
        --state-step 0 \
        --output-dir results/canvas_critic_reward_maps
"""

from __future__ import annotations

import argparse
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
from canvit_specialize.datasets.ade20k import (
    ADE20kDataset,
    IGNORE_LABEL,
    make_val_transforms,
)
from PIL import Image
from tqdm import tqdm

from canvit_rl.ade_labels import remap_ade_mask_labels
from canvit_rl.canvas.state import canvas_layernorm_spatial, canvas_segmentation_entropy
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.greedy import _repeat_state_chunks, _segmentation_cross_entropy_losses
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic
from canvit_rl.viewpoint_policy import (
    ViewpointGaussianActor,
    ViewpointHistoryCritic,
    action_to_viewpoint,
    viewpoint_to_action,
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


def _image_for_plot(image: torch.Tensor) -> np.ndarray:
    """Convert one normalized CHW image to HWC for matplotlib."""
    image_cpu = image.detach().cpu()
    image_cpu = (image_cpu * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    return image_cpu.permute(1, 2, 0).numpy()


def _segmentation_ce(
    *,
    model,
    probe: torch.nn.Module,
    state,
    mask: torch.Tensor,
    cfg: CanViTEnvConfig,
) -> torch.Tensor:
    """Return per-image segmentation CE from a CanViT state."""
    return _segmentation_cross_entropy_losses(
        model=model,
        state=state,
        probe=probe,
        canvas_grid_size=cfg.canvas_grid_size,
        mask=mask,
        batch_size=mask.shape[0],
    )


def _segmentation_for_plot(
    *,
    model,
    probe: torch.nn.Module,
    state,
    mask: torch.Tensor,
    cfg: CanViTEnvConfig,
) -> np.ndarray:
    """Decode one state to an ADE-like RGB segmentation prediction."""
    spatial = model.get_spatial(state.canvas).view(
        mask.shape[0],
        cfg.canvas_grid_size,
        cfg.canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float()).float()
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pred = logits.argmax(dim=1)[0].detach().cpu()
    labels = torch.arange(max(int(pred.max().item()) + 1, 150), dtype=torch.float32)
    palette = torch.stack(
        [
            (labels * 37 + 17) % 255,
            (labels * 67 + 71) % 255,
            (labels * 97 + 149) % 255,
        ],
        dim=1,
    ) / 255.0
    palette[0] = torch.tensor([0.0, 0.0, 0.0])
    return palette[pred.clamp_min(0)].numpy()


def _canvit_glimpse_dtype(model) -> torch.dtype:
    """Return the dtype expected by CanViT patch embedding inputs."""
    try:
        return next(model.backbone.patch_embed.parameters()).dtype
    except (AttributeError, StopIteration):
        return next(model.parameters()).dtype


def _sample_canvit_glimpse(
    *,
    image: torch.Tensor,
    viewpoint: Viewpoint,
    cfg: CanViTEnvConfig,
    canvit_dtype: torch.dtype,
) -> torch.Tensor:
    """Sample one CanViT glimpse and match the frozen backbone precision."""
    return sample_at_viewpoint(
        spatial=image,
        viewpoint=viewpoint,
        glimpse_size_px=cfg.glimpse_size_px,
    ).to(dtype=canvit_dtype)


def _append_history(
    *,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    viewpoint: Viewpoint,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write a Viewpoint into a fixed-slot history batch."""
    coords[:, step, :2] = viewpoint.centers.detach().float()
    coords[:, step, 2] = viewpoint.scales.detach().float()
    return coords, lengths + 1


def _build_actor_and_critics(
    *,
    checkpoint: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module | None,
    torch.nn.Module,
    torch.nn.Module,
    dict,
    str,
    str,
]:
    """Load actor and critics from a full SAC or critic-only checkpoint."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "q1" not in payload:
        raise ValueError(f"Expected checkpoint with q1/q2 critic keys: {checkpoint}")
    saved_args = dict(payload.get("args", {}))
    d_model = int(saved_args.get("d_model", 256))
    max_history = int(saved_args.get("max_history", 16))
    rff_dim = int(saved_args.get("rff_dim", 128))
    rff_seed = int(saved_args.get("rff_seed", 42))
    state_representation = str(
        payload.get(
            "state_representation",
            payload.get("metadata", {}).get("state_representation", "viewpoint_history"),
        )
    )
    target = str(
        payload.get("target", payload.get("metadata", {}).get("target", "raw_ce_gain"))
    )
    if state_representation in {
        "current_canvas_layernorm_with_viewpoint_history",
        "current_canvas_layernorm_entropy_with_viewpoint_history",
    }:
        canvas_feature_dim = int(
            payload.get(
                "canvas_feature_dim",
                payload.get("metadata", {}).get("canvas_feature_dim"),
            )
        )
        kwargs = dict(
            canvas_feature_dim=canvas_feature_dim,
            d_model=d_model,
            rff_dim=rff_dim,
            rff_seed=rff_seed,
            use_entropy_state=bool(
                saved_args.get("canvas_entropy_state", False)
                or state_representation
                == "current_canvas_layernorm_entropy_with_viewpoint_history"
            ),
            use_canvas_avg_pool=not bool(
                saved_args.get("disable_canvas_avg_pool", False)
            ),
            use_canvas_max_pool=not bool(
                saved_args.get("disable_canvas_max_pool", False)
            ),
        )
        actor = CanvasStateActor(**kwargs).to(device).eval() if "actor" in payload else None
        critic_kwargs = dict(
            kwargs,
            use_action_location_features=bool(
                saved_args.get("critic_local_action_features", False)
            ),
        )
        # Problem: reward-map reconstruction must match the saved critic
        # architecture. Solution: replay the train_canvas_sac.py flag stored in
        # checkpoint args. Result: local-feature critics load without manual
        # visualization-script edits.
        q1 = CanvasStateCritic(**critic_kwargs).to(device).eval()
        q2 = CanvasStateCritic(**critic_kwargs).to(device).eval()
        policy_kind = "canvas"
    else:
        actor = (
            ViewpointGaussianActor(
                d_model=d_model,
                max_steps=max_history,
                rff_dim=rff_dim,
                rff_seed=rff_seed,
            ).to(device).eval()
            if "actor" in payload
            else None
        )
        critic_kwargs = dict(
            d_model=d_model,
            max_steps=max_history,
            rff_dim=rff_dim,
            rff_seed=rff_seed,
        )
        q1 = ViewpointHistoryCritic(**critic_kwargs).to(device).eval()
        q2 = ViewpointHistoryCritic(**critic_kwargs).to(device).eval()
        policy_kind = "viewpoint_history"
    if actor is not None:
        actor.load_state_dict(payload["actor"])
    q1.load_state_dict(payload["q1"])
    q2.load_state_dict(payload.get("q2", payload["q1"]))
    for module in (q1, q2) if actor is None else (actor, q1, q2):
        for param in module.parameters():
            param.requires_grad_(False)
    return actor, q1, q2, saved_args, policy_kind, target


def _candidate_grid(*, scale: float, grid_size: int, device: torch.device) -> Viewpoint:
    """Build an in-bounds y/x center grid for one scale."""
    bound = max(1.0 - scale, 0.0)
    values = torch.linspace(-bound, bound, grid_size, device=device)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    centers = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    scales = torch.full((centers.shape[0],), float(scale), device=device)
    return Viewpoint(centers=centers, scales=scales)


def _evaluate_grid(
    *,
    image: torch.Tensor,
    mask: torch.Tensor,
    state,
    coords: torch.Tensor,
    lengths: torch.Tensor,
    canvas_summary: torch.Tensor | None,
    canvas_entropy: torch.Tensor | None,
    current_ce: torch.Tensor,
    reward_target: str,
    q1: torch.nn.Module,
    q2: torch.nn.Module,
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    min_scale: float,
    scale: float,
    grid_size: int,
    chunk_size: int,
    device: torch.device,
    canvit_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Return true reward and predicted Q maps for one candidate scale."""
    vp_all = _candidate_grid(scale=scale, grid_size=grid_size, device=device)
    rewards = []
    q_values = []
    total = vp_all.centers.shape[0]
    with torch.inference_mode():
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            vp = Viewpoint(
                centers=vp_all.centers[start:stop],
                scales=vp_all.scales[start:stop],
            )
            repeats = stop - start
            candidate_images = image.repeat((repeats,) + (1,) * (image.ndim - 1))
            candidate_masks = mask.repeat((repeats,) + (1,) * (mask.ndim - 1))
            candidate_state = _repeat_state_chunks(state, repeats)
            out = model(
                glimpse=_sample_canvit_glimpse(
                    image=candidate_images,
                    viewpoint=vp,
                    cfg=cfg,
                    canvit_dtype=canvit_dtype,
                ),
                state=candidate_state,
                viewpoint=vp,
            )
            ce_after = _segmentation_ce(
                model=model,
                probe=probe,
                state=out.state,
                mask=candidate_masks,
                cfg=cfg,
            )
            raw_gain = current_ce.expand_as(ce_after) - ce_after
            if reward_target == "relative_ce_reduction":
                reward = raw_gain / current_ce.expand_as(ce_after).clamp_min(1e-6)
            else:
                reward = raw_gain
            history_batch = {
                "coords": coords.repeat(repeats, 1, 1),
                "lengths": lengths.repeat(repeats),
            }
            if canvas_summary is not None:
                history_batch["canvas"] = canvas_summary.repeat(
                    repeats,
                    1,
                    1,
                    1,
                )
            if canvas_entropy is not None:
                history_batch["entropy"] = canvas_entropy.repeat(
                    repeats,
                    1,
                    1,
                    1,
                )
            action = viewpoint_to_action(vp, min_scale=min_scale)
            q_pred = torch.minimum(q1(history_batch, action), q2(history_batch, action))
            rewards.append(reward.detach().cpu())
            q_values.append(q_pred.detach().cpu())
    reward_map = torch.cat(rewards).view(grid_size, grid_size).numpy()
    q_map = torch.cat(q_values).view(grid_size, grid_size).numpy()
    return reward_map, q_map


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation for flattened maps, nan for constant maps."""
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    if np.std(x_flat) == 0 or np.std(y_flat) == 0:
        return float("nan")
    return float(np.corrcoef(x_flat, y_flat)[0, 1])


def _show_image_background(ax, image_np: np.ndarray) -> None:
    """Show an image in normalized viewpoint-center coordinates."""
    ax.imshow(image_np, extent=[-1.0, 1.0, 1.0, -1.0])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(1.0, -1.0)


def _map_argmax_center(values: np.ndarray, *, bound: float) -> tuple[float, float] | None:
    """Return (x, y) center coordinates for the finite max of a grid map."""
    finite = np.isfinite(values)
    if not finite.any():
        return None
    row, col = np.unravel_index(np.nanargmax(values), values.shape)
    y_centers = np.linspace(-bound, bound, values.shape[0])
    x_centers = np.linspace(-bound, bound, values.shape[1])
    return float(x_centers[col]), float(y_centers[row])


def _save_combined_reward_maps(
    *,
    rows: list[dict],
    output: Path,
    title: str,
) -> None:
    """Save one figure with one image row and per-scale true/Q overlays."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to save reward-map figures.") from exc

    if not rows:
        return
    n_images = len(rows)
    n_scales = len(rows[0]["maps"])
    n_cols = 1 + 2 * n_scales
    fig, axes = plt.subplots(
        n_images,
        n_cols,
        figsize=(3.4 * n_cols, max(3.2 * n_images, 4.0)),
        dpi=150,
        squeeze=False,
    )

    for row_idx, row_data in enumerate(rows):
        image_np = _image_for_plot(row_data["image"])
        actor_vp = row_data.get("actor_vp")
        actor_center = (
            actor_vp.centers[0].detach().cpu().numpy()
            if actor_vp is not None
            else None
        )
        actor_scale = (
            float(actor_vp.scales[0].detach().cpu().item())
            if actor_vp is not None
            else None
        )
        _show_image_background(axes[row_idx, 0], image_np)
        axes[row_idx, 0].imshow(
            row_data["seg_rgb"],
            alpha=0.35,
            extent=[-1.0, 1.0, 1.0, -1.0],
        )
        axes[row_idx, 0].set_title(
            f"{row_data['label']}\nCE at state t{row_data['state_step']}="
            f"{row_data['current_ce']:.4f}"
        )
        axes[row_idx, 0].axis("off")

        for scale_idx, (scale, reward_map, q_map) in enumerate(row_data["maps"]):
            bound = max(1.0 - scale, 0.0)
            extent = [-bound, bound, bound, -bound]
            reward_ax = axes[row_idx, 1 + 2 * scale_idx]
            q_ax = axes[row_idx, 2 + 2 * scale_idx]
            reward_max_center = _map_argmax_center(reward_map, bound=bound)
            q_max_center = _map_argmax_center(q_map, bound=bound)

            _show_image_background(reward_ax, image_np)
            reward_im = reward_ax.imshow(
                reward_map,
                origin="upper",
                extent=extent,
                cmap="coolwarm",
                alpha=0.58,
            )
            reward_ax.set_title(
                f"scale={scale:.2f} true reward\nmax={np.nanmax(reward_map):+.4f}"
            )
            reward_ax.set_xlabel("x center")
            reward_ax.set_ylabel("y center")
            fig.colorbar(reward_im, ax=reward_ax, fraction=0.046, pad=0.04)
            if reward_max_center is not None:
                reward_ax.scatter(
                    [reward_max_center[0]],
                    [reward_max_center[1]],
                    c="lime",
                    s=34,
                    marker="x",
                    linewidths=1.6,
                )

            _show_image_background(q_ax, image_np)
            q_im = q_ax.imshow(
                q_map,
                origin="upper",
                extent=extent,
                cmap="coolwarm",
                alpha=0.58,
            )
            q_ax.set_title(
                f"scale={scale:.2f} predicted Q\ncorr={_corr(reward_map, q_map):+.3f}"
            )
            q_ax.set_xlabel("x center")
            q_ax.set_ylabel("y center")
            fig.colorbar(q_im, ax=q_ax, fraction=0.046, pad=0.04)
            if q_max_center is not None:
                q_ax.scatter(
                    [q_max_center[0]],
                    [q_max_center[1]],
                    c="yellow",
                    s=34,
                    marker="x",
                    linewidths=1.6,
                )

            if actor_scale is not None and abs(actor_scale - scale) <= 0.5 * max(scale, 1e-6):
                for ax in (reward_ax, q_ax):
                    ax.scatter(
                        [actor_center[1]],
                        [actor_center[0]],
                        c="black",
                        s=40,
                        marker="x",
                        linewidths=2.0,
            )

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _dataset_stem(dataset, index: int) -> str:
    """Return an image stem for ADE datasets and Subset-wrapped ADE datasets."""
    base = dataset
    real_index = index
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        real_index = int(dataset.indices[index])
    if hasattr(base, "images"):
        return base.images[real_index].stem
    return f"image_{index:05d}"


def visualize_reward_maps_for_indices(
    *,
    actor: torch.nn.Module | None,
    q1: torch.nn.Module,
    q2: torch.nn.Module,
    dataset,
    indices: list[int],
    model,
    probe: torch.nn.Module,
    cfg: CanViTEnvConfig,
    device: torch.device,
    min_scale: float,
    scales: list[float],
    grid_size: int,
    chunk_size: int,
    output_dir: Path,
    split_label: str,
    title_prefix: str,
    policy_kind: str = "viewpoint_history",
    reward_target: str = "raw_ce_gain",
    max_history: int | None = None,
    state_step: int = 0,
    state_steps: list[int] | None = None,
    output_name_suffix: str | None = None,
) -> list[Path]:
    """Generate reward/Q landscape figures using live SAC networks."""
    rows: list[dict] = []
    if max_history is None:
        if actor is None:
            raise ValueError("max_history is required when visualizing a critic-only checkpoint.")
        max_history = actor.max_steps
    state_steps = [state_step] if state_steps is None else state_steps
    if actor is None and any(step != 0 for step in state_steps):
        raise ValueError(
            "Critic-only checkpoints have no actor to roll out; use --state-step 0."
        )
    for step in state_steps:
        if step < 0:
            raise ValueError("--state-step/--state-steps must be non-negative.")
        if step + 1 >= max_history:
            raise ValueError(
                f"State step {step} leaves no room for the next action "
                f"with max_history={max_history}."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    canvit_dtype = _canvit_glimpse_dtype(model)
    with torch.inference_mode():
        for idx in indices:
            for current_state_step in state_steps:
                image, mask = dataset[idx]
                image_dev = image.unsqueeze(0).to(device)
                mask_dev = mask.unsqueeze(0).to(device)
                state = model.init_state(
                    batch_size=1,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
                coords = torch.zeros(1, max_history, 3, device=device)
                lengths = torch.zeros(1, dtype=torch.long, device=device)
                full_vp = Viewpoint.full_scene(batch_size=1, device=device)
                full_out = model(
                    glimpse=_sample_canvit_glimpse(
                        image=image_dev,
                        viewpoint=full_vp,
                        cfg=cfg,
                        canvit_dtype=canvit_dtype,
                    ),
                    state=state,
                    viewpoint=full_vp,
                )
                state = full_out.state
                coords, lengths = _append_history(
                    coords=coords,
                    lengths=lengths,
                    viewpoint=full_vp,
                    step=0,
                )
                current_ce = _segmentation_ce(
                    model=model,
                    probe=probe,
                    state=state,
                    mask=mask_dev,
                    cfg=cfg,
                )
                for rollout_step in range(1, current_state_step + 1):
                    if actor is None:
                        raise RuntimeError("Cannot roll out state steps without an actor.")
                    canvas_summary = None
                    actor_batch = {"coords": coords, "lengths": lengths}
                    if policy_kind == "canvas":
                        canvas_summary = canvas_layernorm_spatial(
                            model=model,
                            state=state,
                            canvas_grid_size=cfg.canvas_grid_size,
                        )
                        actor_batch["canvas"] = canvas_summary
                        if getattr(actor.encoder, "use_entropy_state", False):
                            # Problem: entropy-state actors require the same
                            # uncertainty input at visualization time as they
                            # saw during training. Solution: derive the
                            # normalized probe-entropy map from the current
                            # canvas before each rollout action.
                            actor_batch["entropy"] = canvas_segmentation_entropy(
                                model=model,
                                probe=probe,
                                state=state,
                                canvas_grid_size=cfg.canvas_grid_size,
                            )
                    action = actor.deterministic_action(actor_batch)
                    vp = action_to_viewpoint(action, min_scale=min_scale)
                    out = model(
                        glimpse=_sample_canvit_glimpse(
                            image=image_dev,
                            viewpoint=vp,
                            cfg=cfg,
                            canvit_dtype=canvit_dtype,
                        ),
                        state=state,
                        viewpoint=vp,
                    )
                    state = out.state
                    coords, lengths = _append_history(
                        coords=coords,
                        lengths=lengths,
                        viewpoint=vp,
                        step=rollout_step,
                    )
                current_ce = _segmentation_ce(
                    model=model,
                    probe=probe,
                    state=state,
                    mask=mask_dev,
                    cfg=cfg,
                )
                canvas_summary = None
                canvas_entropy = None
                actor_vp = None
                actor_batch = {"coords": coords, "lengths": lengths}
                if policy_kind == "canvas":
                    canvas_summary = canvas_layernorm_spatial(
                        model=model,
                        state=state,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                    actor_batch["canvas"] = canvas_summary
                    needs_entropy = (
                        actor is not None
                        and getattr(actor.encoder, "use_entropy_state", False)
                    ) or any(
                        getattr(critic.encoder, "use_entropy_state", False)
                        for critic in (q1, q2)
                    )
                    if needs_entropy:
                        canvas_entropy = canvas_segmentation_entropy(
                            model=model,
                            probe=probe,
                            state=state,
                            canvas_grid_size=cfg.canvas_grid_size,
                        )
                        actor_batch["entropy"] = canvas_entropy
                if actor is not None:
                    actor_action = actor.deterministic_action(actor_batch)
                    actor_vp = action_to_viewpoint(actor_action, min_scale=min_scale)
                seg_rgb = _segmentation_for_plot(
                    model=model,
                    probe=probe,
                    state=state,
                    mask=mask_dev,
                    cfg=cfg,
                )

                maps = []
                for scale in tqdm(
                    scales,
                    desc=f"image {idx} t{current_state_step} reward maps",
                    leave=False,
                ):
                    reward_map, q_map = _evaluate_grid(
                        image=image_dev,
                        mask=mask_dev,
                        state=state,
                        coords=coords,
                        lengths=lengths,
                        canvas_summary=canvas_summary,
                        canvas_entropy=canvas_entropy,
                        current_ce=current_ce,
                        reward_target=reward_target,
                        q1=q1,
                        q2=q2,
                        model=model,
                        probe=probe,
                        cfg=cfg,
                        min_scale=min_scale,
                        scale=scale,
                        grid_size=grid_size,
                        chunk_size=chunk_size,
                        device=device,
                        canvit_dtype=canvit_dtype,
                    )
                    maps.append((scale, reward_map, q_map))

                name = _dataset_stem(dataset, idx)
                rows.append(
                    {
                        "image": image,
                        "seg_rgb": seg_rgb,
                        "maps": maps,
                        "actor_vp": actor_vp,
                        "label": f"{split_label} {idx:05d} {name}",
                        "current_ce": float(current_ce.item()),
                        "state_step": current_state_step,
                    }
                )
    state_suffix = (
        f"t{state_steps[0]}"
        if len(state_steps) == 1
        else "t" + "-".join(str(step) for step in state_steps)
    )
    suffix = f"_{output_name_suffix}" if output_name_suffix else ""
    output = output_dir / f"{split_label}_reward_maps_{state_suffix}{suffix}.png"
    _save_combined_reward_maps(
        rows=rows,
        output=output,
        title=title_prefix,
    )
    return [output]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ADE20k",
        help=(
            "ADE20K root, or synthetic root containing images/<split> and "
            "masks/<split> folders."
        ),
    )
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument(
        "--image-index",
        type=str,
        action="append",
        default=None,
        help=(
            "Image index to visualize. May be repeated, or passed as a "
            "comma-separated list such as 0,1,2,3."
        ),
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--scales", type=str, default="0.25,0.50")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--state-step",
        type=int,
        default=0,
        help=(
            "Number of learned policy glimpses to roll out before drawing the "
            "next-action reward map. 0 means immediately after full-scene warmup."
        ),
    )
    parser.add_argument(
        "--state-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated state steps to render in one figure, or 'all'. "
            "Overrides --state-step."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/sac_reward_maps"))
    return parser.parse_args()


def _parse_state_steps(value: str | None, *, saved_t: int, max_history: int) -> list[int] | None:
    """Parse multi-state reward-map selection."""
    if value is None:
        return None
    if value.strip().lower() == "all":
        max_step = min(saved_t, max_history - 2)
        return list(range(max_step + 1))
    steps = [int(item) for item in value.split(",") if item.strip()]
    if not steps:
        raise ValueError("--state-steps must contain at least one step or 'all'.")
    return steps


def _parse_image_indices(values: list[str] | None) -> list[int] | None:
    """Parse repeated and comma-separated --image-index values."""
    if values is None:
        return None
    indices: list[int] = []
    for value in values:
        indices.extend(int(item) for item in value.split(",") if item.strip())
    if not indices:
        raise ValueError("--image-index must include at least one integer.")
    return indices


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be >= 2.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")
    if args.state_step < 0:
        raise ValueError("--state-step must be non-negative.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    scales = [float(item) for item in args.scales.split(",") if item.strip()]
    if any(scale <= 0 or scale > 1 for scale in scales):
        raise ValueError("--scales must be in (0, 1].")

    device = get_device()
    cfg = CanViTEnvConfig()
    actor, q1, q2, ckpt_args, policy_kind, reward_target = _build_actor_and_critics(
        checkpoint=args.checkpoint,
        device=device,
    )
    min_scale = float(ckpt_args.get("min_scale", 0.05))
    max_history = int(ckpt_args.get("max_history", 16))
    saved_t = int(ckpt_args.get("t", max_history - 1))
    state_steps = _parse_state_steps(
        args.state_steps,
        saved_t=saved_t,
        max_history=max_history,
    )
    checkpoint_kind = "SAC" if actor is not None else "critic-only"
    print(
        f"Loaded {policy_kind} {checkpoint_kind} checkpoint "
        f"(target={reward_target}): {args.checkpoint}"
    )
    print(f"Reward-map grid: {args.grid_size}x{args.grid_size} scales={scales}")

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = _build_dataset(
        root=Path(args.dataset),
        split=args.split,
        cfg=cfg,
        img_tf=img_tf,
        mask_tf=mask_tf,
    )
    parsed_indices = _parse_image_indices(args.image_index)
    if parsed_indices is not None:
        indices = parsed_indices
    else:
        indices = random.sample(range(len(dataset)), min(args.episodes, len(dataset)))

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
    for module in (model, probe):
        for param in module.parameters():
            param.requires_grad_(False)

    output_paths = visualize_reward_maps_for_indices(
        actor=actor,
        q1=q1,
        q2=q2,
        dataset=dataset,
        indices=indices,
        model=model,
        probe=probe,
        cfg=cfg,
        device=device,
        min_scale=min_scale,
        scales=scales,
        grid_size=args.grid_size,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        split_label=args.split,
        title_prefix=(
            f"SAC reward landscape ({policy_kind}) "
            f"state_steps={state_steps if state_steps is not None else [args.state_step]} "
            f"checkpoint={args.checkpoint.name}"
        ),
        policy_kind=policy_kind,
        reward_target=reward_target,
        max_history=max_history,
        state_step=args.state_step,
        state_steps=state_steps,
    )
    for output in output_paths:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
