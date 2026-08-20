"""
Analyze whether tile entropy predicts immediate reward.

For each validation image, this script commits the full-scene tile, then scores
each EG-F2C candidate tile independently from that same state:

    corr(entropy(tile), mIoU_after_tile - mIoU_full_scene)

Optionally, it also computes teacher-CLS cosine-similarity deltas as a
representation proxy and segmentation-KL deltas to a full-scene teacher
distribution.

Usage:
    uv run python scripts/analysis/ade20k/analyze_entropy_delta_correlation.py
    uv run python scripts/analysis/ade20k/analyze_entropy_delta_correlation.py --max-images 100
    uv run python scripts/analysis/ade20k/analyze_entropy_delta_correlation.py --teacher-corr --kl-corr
    uv run python scripts/analysis/ade20k/analyze_entropy_delta_correlation.py --print-per-image
"""

from __future__ import annotations

import argparse
import time
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
from canvit_pytorch.teacher import load_teacher
from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms
from tqdm import tqdm

from canvit_rl.environment import CanViTEnvConfig, get_device
from canvit_rl.ade20k.greedy import miou_from_state
from canvit_rl.tiles import TileSpec, eg_f2c_tiles, tile_to_viewpoint


def _mean_entropy_for_tile(
    entropy_map: torch.Tensor,
    tile: TileSpec,
) -> torch.Tensor:
    """Average a grid entropy map inside a normalized Viewpoint tile."""
    h, w = entropy_map.shape[-2:]
    cx, cy = tile.center
    size_x = max(1, int(round(tile.scale * w)))
    size_y = max(1, int(round(tile.scale * h)))
    center_x = int(round((cx + 1.0) * 0.5 * w))
    center_y = int(round((cy + 1.0) * 0.5 * h))
    x0 = min(max(center_x - size_x // 2, 0), w - 1)
    y0 = min(max(center_y - size_y // 2, 0), h - 1)
    x1 = min(max(x0 + size_x, x0 + 1), w)
    y1 = min(max(y0 + size_y, y0 + 1), h)
    return entropy_map[y0:y1, x0:x1].mean()


def _tile_entropies(
    *,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    state,
    tiles: tuple[TileSpec, ...],
    canvas_grid_size: int,
    entropy_eps: float,
) -> np.ndarray:
    """Compute normalized segmentation entropy for each EG-F2C tile."""
    spatial = model.get_spatial(state.canvas).view(
        1,
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        logits = probe(spatial.float()).float()
    prob = F.softmax(logits, dim=1)
    entropy = -(prob * torch.log(prob.clamp_min(entropy_eps))).sum(dim=1)
    entropy = entropy / float(np.log(logits.shape[1]))
    entropy_map = entropy[0]
    return np.asarray(
        [float(_mean_entropy_for_tile(entropy_map, tile).item()) for tile in tiles],
        dtype=np.float32,
    )


def _seg_logits_from_state(
    *,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    state,
    canvas_grid_size: int,
) -> torch.Tensor:
    """Decode segmentation logits from one recurrent canvas state."""
    spatial = model.get_spatial(state.canvas).view(
        1,
        canvas_grid_size,
        canvas_grid_size,
        -1,
    )
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        return probe(spatial.float()).float()


def _negative_segmentation_kl(
    *,
    logits: torch.Tensor,
    target_prob: torch.Tensor,
    temperature: float,
) -> float:
    """Return negative KL(target || student) for segmentation distributions."""
    log_prob = F.log_softmax(logits / temperature, dim=1)
    kl = F.kl_div(log_prob, target_prob, reduction="batchmean")
    return -float(kl.item())


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or nan for constant inputs."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman rank correlation without requiring scipy."""
    return _pearson(_average_ranks(x), _average_ranks(y))


def _summarize(name: str, x: list[float], y: list[float]) -> dict[str, float]:
    """Compute correlation summary for two collected scalar lists."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    summary = {
        "n": float(x_arr.size),
        "pearson": _pearson(x_arr, y_arr),
        "spearman": _spearman(x_arr, y_arr),
        "x_mean": float(np.mean(x_arr)) if x_arr.size else float("nan"),
        "y_mean": float(np.mean(y_arr)) if y_arr.size else float("nan"),
    }
    print(
        f"{name}: n={int(summary['n'])} "
        f"pearson={summary['pearson']:+.4f} "
        f"spearman={summary['spearman']:+.4f} "
        f"x_mean={summary['x_mean']:.4f} "
        f"y_mean={summary['y_mean']:+.6f}"
    )
    return summary


def _summarize_values(name: str, values: list[float]) -> dict[str, float]:
    """Summarize a list of scalar per-image statistics, ignoring nans."""
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    summary = {
        "n": float(valid.size),
        "mean": float(np.mean(valid)) if valid.size else float("nan"),
        "median": float(np.median(valid)) if valid.size else float("nan"),
        "std": float(np.std(valid)) if valid.size else float("nan"),
        "positive_frac": float(np.mean(valid > 0)) if valid.size else float("nan"),
    }
    print(
        f"{name}: n={int(summary['n'])} "
        f"mean={summary['mean']:+.4f} "
        f"median={summary['median']:+.4f} "
        f"std={summary['std']:.4f} "
        f"positive_frac={summary['positive_frac']:.2%}"
    )
    return summary


def _plot_scatter(
    *,
    rows: list[dict],
    x_key: str,
    y_key: str,
    title: str,
    output: Path,
) -> None:
    """Save a tile-level scatter plot colored by EG-F2C tile level."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it or rerun without --plot."
        ) from exc

    x = np.asarray([row[x_key] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_key] for row in rows], dtype=np.float64)
    levels = np.asarray([row["tile_level"] for row in rows], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    # Fixed by Codex on 2026-06-01
    # Problem: Correlation coefficients hide whether the relationship is
    # monotonic, noisy, outlier-driven, or different for coarse/fine tiles.
    # Solution: Save scatter plots colored by EG-F2C level whenever --plot is
    # requested, keeping plotting optional so headless metric runs stay light.
    # Result: The analysis produces both numerical correlations and visual
    # structure for entropy-vs-reward debugging.
    scatter = ax.scatter(
        x,
        y,
        c=levels,
        cmap="viridis",
        s=16,
        alpha=0.55,
        edgecolors="none",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("tile_level")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved plot: {output}")


def _plot_outputs(
    base_output: Path,
    *,
    teacher_corr: bool,
    kl_corr: bool,
) -> dict[str, Path]:
    """Return deterministic plot paths derived from the .pt output path."""
    stem = base_output.with_suffix("")
    paths = {"entropy_vs_delta_miou": stem.with_name(f"{stem.name}.png")}
    if teacher_corr:
        paths["entropy_vs_delta_teacher_sim"] = stem.with_name(
            f"{stem.name}_teacher.png"
        )
        paths["delta_miou_vs_delta_teacher_sim"] = stem.with_name(
            f"{stem.name}_miou_teacher.png"
        )
    if kl_corr:
        paths["entropy_vs_delta_neg_seg_kl"] = stem.with_name(
            f"{stem.name}_seg_kl.png"
        )
        paths["delta_miou_vs_delta_neg_seg_kl"] = stem.with_name(
            f"{stem.name}_miou_seg_kl.png"
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/ADE20k")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="validation",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--probe-repo", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/entropy_delta_corr.pt"),
    )
    parser.add_argument("--include-full-scene", action="store_true")
    parser.add_argument("--teacher-corr", action="store_true")
    parser.add_argument("--kl-corr", action="store_true")
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--print-per-image",
        action="store_true",
        help="Print each image's entropy/delta correlation row to the terminal.",
    )
    parser.add_argument("--entropy-eps", type=float, default=1e-8)
    args = parser.parse_args()

    if args.max_images <= 0:
        raise ValueError("--max-images must be positive.")
    if args.kl_temperature <= 0:
        raise ValueError("--kl-temperature must be positive.")

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
    n_images = min(args.max_images, len(dataset))
    print(f"Dataset: {len(dataset)} {args.split} images, analyzing {n_images}")

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

    teacher = None
    if args.teacher_corr:
        print("Loading teacher...")
        teacher = load_teacher(cfg.teacher_repo, device)

    tiles = eg_f2c_tiles()
    candidate_indices = (
        range(len(tiles)) if args.include_full_scene else range(1, len(tiles))
    )
    entropies_all: list[float] = []
    deltas_all: list[float] = []
    teacher_deltas_all: list[float] = []
    kl_deltas_all: list[float] = []
    rows: list[dict] = []
    per_image_rows: list[dict] = []
    t_start = time.monotonic()

    with torch.inference_mode():
        for image_index in tqdm(range(n_images), desc="Analyzing"):
            image, mask = dataset[image_index]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            state0 = model.init_state(
                batch_size=1,
                canvas_grid_size=cfg.canvas_grid_size,
            )

            # Fixed by Codex on 2026-06-01
            # Problem: To test whether entropy is a useful SAC state variable,
            # every candidate must be scored from the same full-scene context.
            # Solution: Commit full scene once, then evaluate each candidate
            # tile independently from that recurrent state.
            # Result: entropy(tile) and immediate delta_mIoU(tile) are aligned
            # to the exact decision state used by the tile policy.
            full_vp = Viewpoint.full_scene(batch_size=1, device=device)
            full_glimpse = sample_at_viewpoint(
                spatial=image,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            )
            full_out = model(glimpse=full_glimpse, state=state0, viewpoint=full_vp)
            full_state = full_out.state
            base_miou = miou_from_state(
                model=model,
                state=full_state,
                probe=probe,
                mask=mask,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            entropies = _tile_entropies(
                model=model,
                probe=probe,
                state=full_state,
                tiles=tiles,
                canvas_grid_size=cfg.canvas_grid_size,
                entropy_eps=args.entropy_eps,
            )
            target_prob = None
            base_neg_seg_kl = None
            if args.kl_corr:
                full_logits = _seg_logits_from_state(
                    model=model,
                    probe=probe,
                    state=full_state,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
                # Fixed by Codex on 2026-06-01
                # Problem: mIoU correlation says whether entropy predicts task
                # reward, but not whether it predicts the AdaGlimpse-like
                # student-vs-teacher KL objective.
                # Solution: Use the full-scene segmentation distribution as
                # the teacher target and score each candidate by negative KL.
                # Result: The script can compare entropy against immediate
                # delta negative-KL while keeping larger-is-better semantics.
                target_prob = F.softmax(
                    full_logits / args.kl_temperature,
                    dim=1,
                ).detach()
                base_neg_seg_kl = _negative_segmentation_kl(
                    logits=full_logits,
                    target_prob=target_prob,
                    temperature=args.kl_temperature,
                )

            base_teacher_sim = None
            teacher_cls = None
            if teacher is not None:
                teacher_cls = teacher.forward_norm_features(image).cls
                base_teacher_sim = float(
                    F.cosine_similarity(
                        full_state.recurrent_cls.squeeze(1).float(),
                        teacher_cls.float(),
                        dim=-1,
                    ).item()
                )

            image_entropies: list[float] = []
            image_deltas: list[float] = []
            image_teacher_deltas: list[float] = []
            image_kl_deltas: list[float] = []
            for tile_idx in candidate_indices:
                tile = tiles[tile_idx]
                vp = tile_to_viewpoint(tile, batch_size=1, device=device)
                glimpse = sample_at_viewpoint(
                    spatial=image,
                    viewpoint=vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                )
                out = model(glimpse=glimpse, state=full_state, viewpoint=vp)
                miou = miou_from_state(
                    model=model,
                    state=out.state,
                    probe=probe,
                    mask=mask,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
                delta_miou = miou - base_miou
                entropy = float(entropies[tile_idx])
                entropies_all.append(entropy)
                deltas_all.append(delta_miou)
                image_entropies.append(entropy)
                image_deltas.append(delta_miou)

                row = {
                    "image_index": image_index,
                    "tile_index": tile_idx,
                    "tile_name": tile.name,
                    "tile_level": tile.level,
                    "entropy": entropy,
                    "base_miou": base_miou,
                    "miou": miou,
                    "delta_miou": delta_miou,
                }
                if target_prob is not None and base_neg_seg_kl is not None:
                    candidate_logits = _seg_logits_from_state(
                        model=model,
                        probe=probe,
                        state=out.state,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                    neg_seg_kl = _negative_segmentation_kl(
                        logits=candidate_logits,
                        target_prob=target_prob,
                        temperature=args.kl_temperature,
                    )
                    delta_neg_seg_kl = neg_seg_kl - base_neg_seg_kl
                    kl_deltas_all.append(delta_neg_seg_kl)
                    image_kl_deltas.append(delta_neg_seg_kl)
                    row["base_neg_seg_kl"] = base_neg_seg_kl
                    row["neg_seg_kl"] = neg_seg_kl
                    row["delta_neg_seg_kl"] = delta_neg_seg_kl

                if teacher_cls is not None and base_teacher_sim is not None:
                    teacher_sim = float(
                        F.cosine_similarity(
                            out.state.recurrent_cls.squeeze(1).float(),
                            teacher_cls.float(),
                            dim=-1,
                        ).item()
                    )
                    delta_teacher_sim = teacher_sim - base_teacher_sim
                    teacher_deltas_all.append(delta_teacher_sim)
                    image_teacher_deltas.append(delta_teacher_sim)
                    row["base_teacher_sim"] = base_teacher_sim
                    row["teacher_sim"] = teacher_sim
                    row["delta_teacher_sim"] = delta_teacher_sim

                rows.append(row)

            image_entropy_arr = np.asarray(image_entropies, dtype=np.float64)
            image_delta_arr = np.asarray(image_deltas, dtype=np.float64)
            per_image = {
                "image_index": image_index,
                "n_candidates": len(image_entropies),
                "base_miou": base_miou,
                "entropy_mean": float(np.mean(image_entropy_arr)),
                "delta_miou_mean": float(np.mean(image_delta_arr)),
                # Fixed by Codex on 2026-06-01
                # Problem: Pooled all-region correlation can be dominated by
                # cross-image differences and miss whether entropy helps within
                # each image's actual candidate ranking problem.
                # Solution: Compute Pearson/Spearman per image across candidate
                # tiles, then summarize those per-image correlations.
                # Result: The analysis directly tests the Path A question:
                # does higher entropy usually mean higher immediate delta mIoU?
                "entropy_delta_miou_pearson": _pearson(
                    image_entropy_arr,
                    image_delta_arr,
                ),
                "entropy_delta_miou_spearman": _spearman(
                    image_entropy_arr,
                    image_delta_arr,
                ),
            }
            if image_teacher_deltas:
                image_teacher_arr = np.asarray(image_teacher_deltas, dtype=np.float64)
                per_image["entropy_delta_teacher_pearson"] = _pearson(
                    image_entropy_arr,
                    image_teacher_arr,
                )
                per_image["entropy_delta_teacher_spearman"] = _spearman(
                    image_entropy_arr,
                    image_teacher_arr,
                )
                per_image["delta_miou_delta_teacher_pearson"] = _pearson(
                    image_delta_arr,
                    image_teacher_arr,
                )
                per_image["delta_miou_delta_teacher_spearman"] = _spearman(
                    image_delta_arr,
                    image_teacher_arr,
                )
            if image_kl_deltas:
                image_kl_arr = np.asarray(image_kl_deltas, dtype=np.float64)
                per_image["entropy_delta_neg_seg_kl_pearson"] = _pearson(
                    image_entropy_arr,
                    image_kl_arr,
                )
                per_image["entropy_delta_neg_seg_kl_spearman"] = _spearman(
                    image_entropy_arr,
                    image_kl_arr,
                )
                per_image["delta_miou_delta_neg_seg_kl_pearson"] = _pearson(
                    image_delta_arr,
                    image_kl_arr,
                )
                per_image["delta_miou_delta_neg_seg_kl_spearman"] = _spearman(
                    image_delta_arr,
                    image_kl_arr,
                )
            per_image_rows.append(per_image)
            if args.print_per_image:
                # Fixed by Codex on 2026-06-01
                # Problem: Per-image correlations were saved for later, but
                # quick Path A inspection benefits from opt-in live terminal
                # output without making normal runs noisy.
                # Solution: Add --print-per-image to emit one compact row per
                # analyzed image.
                # Result: Users can watch whether entropy correlation is
                # positive image-by-image while preserving quiet defaults.
                message = (
                    "per_image_corr "
                    f"image={image_index} "
                    f"pearson={per_image['entropy_delta_miou_pearson']:+.4f} "
                    f"spearman={per_image['entropy_delta_miou_spearman']:+.4f} "
                    f"delta_mean={per_image['delta_miou_mean']:+.6f}"
                )
                if args.teacher_corr:
                    message += (
                        " "
                        f"teacher_pearson="
                        f"{per_image['entropy_delta_teacher_pearson']:+.4f}"
                    )
                if args.kl_corr:
                    message += (
                        " "
                        f"kl_pearson="
                        f"{per_image['entropy_delta_neg_seg_kl_pearson']:+.4f}"
                    )
                print(message)

    print("\n--- Entropy/Immediate Reward Correlations ---")
    summaries = {
        "entropy_vs_delta_miou": _summarize(
            "entropy_vs_delta_miou",
            entropies_all,
            deltas_all,
        )
    }
    if args.teacher_corr:
        summaries["entropy_vs_delta_teacher_sim"] = _summarize(
            "entropy_vs_delta_teacher_sim",
            entropies_all,
            teacher_deltas_all,
        )
        summaries["delta_miou_vs_delta_teacher_sim"] = _summarize(
            "delta_miou_vs_delta_teacher_sim",
            deltas_all,
            teacher_deltas_all,
        )
    if args.kl_corr:
        summaries["entropy_vs_delta_neg_seg_kl"] = _summarize(
            "entropy_vs_delta_neg_seg_kl",
            entropies_all,
            kl_deltas_all,
        )
        summaries["delta_miou_vs_delta_neg_seg_kl"] = _summarize(
            "delta_miou_vs_delta_neg_seg_kl",
            deltas_all,
            kl_deltas_all,
        )

    print("\n--- Mean Per-Image Correlations ---")
    per_image_summaries = {
        "entropy_delta_miou_pearson": _summarize_values(
            "per_image_entropy_delta_miou_pearson",
            [row["entropy_delta_miou_pearson"] for row in per_image_rows],
        ),
        "entropy_delta_miou_spearman": _summarize_values(
            "per_image_entropy_delta_miou_spearman",
            [row["entropy_delta_miou_spearman"] for row in per_image_rows],
        ),
    }
    if args.teacher_corr:
        per_image_summaries["entropy_delta_teacher_pearson"] = _summarize_values(
            "per_image_entropy_delta_teacher_pearson",
            [row["entropy_delta_teacher_pearson"] for row in per_image_rows],
        )
        per_image_summaries["entropy_delta_teacher_spearman"] = _summarize_values(
            "per_image_entropy_delta_teacher_spearman",
            [row["entropy_delta_teacher_spearman"] for row in per_image_rows],
        )
        per_image_summaries["delta_miou_delta_teacher_pearson"] = _summarize_values(
            "per_image_delta_miou_delta_teacher_pearson",
            [row["delta_miou_delta_teacher_pearson"] for row in per_image_rows],
        )
        per_image_summaries["delta_miou_delta_teacher_spearman"] = (
            _summarize_values(
                "per_image_delta_miou_delta_teacher_spearman",
                [row["delta_miou_delta_teacher_spearman"] for row in per_image_rows],
            )
        )
    if args.kl_corr:
        per_image_summaries["entropy_delta_neg_seg_kl_pearson"] = (
            _summarize_values(
                "per_image_entropy_delta_neg_seg_kl_pearson",
                [row["entropy_delta_neg_seg_kl_pearson"] for row in per_image_rows],
            )
        )
        per_image_summaries["entropy_delta_neg_seg_kl_spearman"] = (
            _summarize_values(
                "per_image_entropy_delta_neg_seg_kl_spearman",
                [row["entropy_delta_neg_seg_kl_spearman"] for row in per_image_rows],
            )
        )
        per_image_summaries["delta_miou_delta_neg_seg_kl_pearson"] = (
            _summarize_values(
                "per_image_delta_miou_delta_neg_seg_kl_pearson",
                [row["delta_miou_delta_neg_seg_kl_pearson"] for row in per_image_rows],
            )
        )
        per_image_summaries["delta_miou_delta_neg_seg_kl_spearman"] = (
            _summarize_values(
                "per_image_delta_miou_delta_neg_seg_kl_spearman",
                [
                    row["delta_miou_delta_neg_seg_kl_spearman"]
                    for row in per_image_rows
                ],
            )
        )

    plot_paths: dict[str, Path] = {}
    if args.plot:
        plot_paths = _plot_outputs(
            args.output,
            teacher_corr=args.teacher_corr,
            kl_corr=args.kl_corr,
        )
        _plot_scatter(
            rows=rows,
            x_key="entropy",
            y_key="delta_miou",
            title="Entropy vs Immediate Delta mIoU",
            output=plot_paths["entropy_vs_delta_miou"],
        )
        if args.teacher_corr:
            _plot_scatter(
                rows=rows,
                x_key="entropy",
                y_key="delta_teacher_sim",
                title="Entropy vs Immediate Delta Teacher Similarity",
                output=plot_paths["entropy_vs_delta_teacher_sim"],
            )
            _plot_scatter(
                rows=rows,
                x_key="delta_miou",
                y_key="delta_teacher_sim",
                title="Delta mIoU vs Delta Teacher Similarity",
                output=plot_paths["delta_miou_vs_delta_teacher_sim"],
            )
        if args.kl_corr:
            _plot_scatter(
                rows=rows,
                x_key="entropy",
                y_key="delta_neg_seg_kl",
                title="Entropy vs Immediate Delta Negative Segmentation KL",
                output=plot_paths["entropy_vs_delta_neg_seg_kl"],
            )
            _plot_scatter(
                rows=rows,
                x_key="delta_miou",
                y_key="delta_neg_seg_kl",
                title="Delta mIoU vs Delta Negative Segmentation KL",
                output=plot_paths["delta_miou_vs_delta_neg_seg_kl"],
            )

    wall_time = time.monotonic() - t_start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summaries": summaries,
            "per_image_summaries": per_image_summaries,
            "rows": rows,
            "per_image_rows": per_image_rows,
            "metadata": {
                "dataset": args.dataset,
                "split": args.split,
                "n_images": n_images,
                "candidate_tile_indices": list(candidate_indices),
                "include_full_scene": args.include_full_scene,
                "teacher_corr": args.teacher_corr,
                "kl_corr": args.kl_corr,
                "kl_temperature": args.kl_temperature,
                "plot": args.plot,
                "print_per_image": args.print_per_image,
                "plot_paths": {key: str(path) for key, path in plot_paths.items()},
                "canvas_grid_size": cfg.canvas_grid_size,
                "glimpse_size_px": cfg.glimpse_size_px,
                "scene_size_px": cfg.scene_size_px,
                "probe_repo": probe_repo,
                "model_repo": cfg.checkpoint,
                "wall_time_seconds": wall_time,
            },
        },
        args.output,
    )
    print(f"\nSaved {args.output} after {wall_time:.1f}s")


if __name__ == "__main__":
    main()
