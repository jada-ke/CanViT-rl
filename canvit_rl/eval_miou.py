"""
Evaluate a trained canvit-rl policy on ADE20K mIoU using canvit-eval.

This script intentionally reuses canvit-eval's episode runner and ADE20K
segmentation components, but supplies a custom policy wrapper backed by the
MLPPolicy checkpoint trained in canvit_rl.train.

Example:
    python -m canvit_rl.eval_miou --checkpoint checkpoints/policy_final.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForSemanticSegmentation, RecurrentState, Viewpoint
from canvit_pytorch.policies import random_viewpoints
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL,
    NUM_CLASSES,
    ADE20kDataset,
    ResizeMode,
    make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_rl.policy import MLPPolicy

EVAL_REPO = Path(__file__).resolve().parents[1] / "CanViT-eval"
if EVAL_REPO.is_dir() and str(EVAL_REPO) not in sys.path:
    sys.path.insert(0, str(EVAL_REPO))

from canvit_eval.config import DEFAULT_PRETRAINED_REPO, require_existing_dir  # noqa: E402
from canvit_eval.episode import run_episode  # noqa: E402
from canvit_eval.tasks.ade20k_obj.iou import CANVIT_PROBE_REPOS  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    output: Path
    ade20k_root: Path
    model_repo: str
    probe_repo: str | None
    device: str
    batch_size: int
    num_workers: int
    scene_size_px: int
    canvas_grid: int
    glimpse_px: int
    n_timesteps: int
    hidden_dim: int
    cls_dim: int
    resize_mode: ResizeMode
    amp: bool
    warmup: str
    max_batches: int | None


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def action_to_viewpoint(action: Tensor) -> Viewpoint:
    """
    Map trained policy output to the upstream Viewpoint interface.

    """
    centers = action[:, :2].float()
    scales = ((action[:, 2] + 1.0) / 2.0 * 0.95 + 0.05).float()
    return Viewpoint(centers=centers, scales=scales)


class TrainedPolicy:
    """canvit-eval Policy adapter for a canvit-rl MLPPolicy checkpoint."""

    def __init__(
        self,
        policy: MLPPolicy,
        *,
        batch_size: int,
        device: torch.device,
        warmup: str,
    ) -> None:
        self.policy = policy
        self.batch_size = batch_size
        self.device = device
        self.warmup = warmup

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        if t == 0 and self.warmup == "full_scene":
            return Viewpoint.full_scene(batch_size=self.batch_size, device=self.device)
        if t == 0 and self.warmup == "random":
            return random_viewpoints(
                batch_size=self.batch_size,
                device=self.device,
                n_viewpoints=1,
                min_scale=0.3,
                max_scale=0.8,
                start_with_full_scene=False,
            ).pop()

        obs = state.recurrent_cls.squeeze(1).detach()
        action = self.policy(obs)
        return action_to_viewpoint(action)


def load_policy(cfg: EvalConfig, device: torch.device) -> MLPPolicy:
    policy = MLPPolicy(cls_dim=cfg.cls_dim, hidden_dim=cfg.hidden_dim).to(device).eval()
    state_dict = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    return policy


def make_loader(cfg: EvalConfig) -> DataLoader:
    require_existing_dir(cfg.ade20k_root, description="ADE20K root", env_var="ADE20K_ROOT")
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root,
        split="validation",
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def update_miou(acc: mIoUAccumulator, probe: torch.nn.Module, features: Tensor, masks: Tensor) -> None:
    with torch.autocast(device_type=features.device.type, enabled=False):
        logits = probe(features.float())
    if logits.shape[-1] != masks.shape[-1]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    acc.update(logits.argmax(dim=1), masks)


@torch.inference_mode()
def evaluate(cfg: EvalConfig) -> Path:
    device = get_device(cfg.device)
    if cfg.probe_repo is None and cfg.canvas_grid not in CANVIT_PROBE_REPOS:
        raise ValueError(
            "No default ADE20K probe for canvas_grid="
            f"{cfg.canvas_grid}. Pass --probe-repo or choose one of "
            f"{sorted(CANVIT_PROBE_REPOS)}."
        )
    probe_repo = cfg.probe_repo or CANVIT_PROBE_REPOS[cfg.canvas_grid]

    log.info("Loading CanViT segmentation model: %s", cfg.model_repo)
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=cfg.model_repo,
        probe_repo=probe_repo,
    ).to(device).eval()
    probe = seg.head
    policy = load_policy(cfg, device)
    loader = make_loader(cfg)

    accs = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(cfg.n_timesteps)]
    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    n_images = 0
    t_start = time.monotonic()

    log.info(
        "Evaluating checkpoint=%s canvas_grid=%d T=%d warmup=%s probe=%s",
        cfg.checkpoint,
        cfg.canvas_grid,
        cfg.n_timesteps,
        cfg.warmup,
        probe_repo,
    )

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating")):
        if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]
        n_images += batch_size

        episode_policy = TrainedPolicy(
            policy,
            batch_size=batch_size,
            device=device,
            warmup=cfg.warmup,
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
            steps = run_episode(
                model=seg.canvit,
                images=images,
                policy=episode_policy,
                n_timesteps=cfg.n_timesteps,
                canvas_grid=cfg.canvas_grid,
                glimpse_px=cfg.glimpse_px,
            )

        for step in steps:
            spatial = seg.canvit.get_spatial(step.state.canvas).view(
                batch_size,
                cfg.canvas_grid,
                cfg.canvas_grid,
                -1,
            )
            update_miou(accs[step.t], probe, spatial, masks)

    mious = {f"t{t}": acc.compute() for t, acc in enumerate(accs)}
    wall_time = time.monotonic() - t_start
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mious": mious,
            "metadata": {
                **asdict(cfg),
                "checkpoint": str(cfg.checkpoint),
                "output": str(cfg.output),
                "ade20k_root": str(cfg.ade20k_root),
                "model_repo": cfg.model_repo,
                "probe_repo": probe_repo,
                "device": str(device),
                "n_images": n_images,
                "wall_time_seconds": wall_time,
            },
        },
        cfg.output,
    )

    for key, value in mious.items():
        log.info("%s mIoU: %.2f%%", key, 100 * value)
    log.info("Saved %s after %.1fs", cfg.output, wall_time)
    return cfg.output


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/policy_final.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/rl_policy_ade20k_miou.pt"))
    parser.add_argument("--ade20k-root", type=Path, default=Path("datasets/ADE20k"))
    parser.add_argument("--model-repo", default=DEFAULT_PRETRAINED_REPO)
    parser.add_argument("--probe-repo", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scene-size-px", type=int, default=512)
    parser.add_argument("--canvas-grid", type=int, default=32)
    parser.add_argument("--glimpse-px", type=int, default=128)
    parser.add_argument("--n-timesteps", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cls-dim", type=int, default=768)
    parser.add_argument("--resize-mode", choices=["squish", "crop"], default="squish")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--warmup", choices=["full_scene", "random", "none"], default="full-scene")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    return EvalConfig(
        checkpoint=args.checkpoint,
        output=args.output,
        ade20k_root=args.ade20k_root,
        model_repo=args.model_repo,
        probe_repo=args.probe_repo,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scene_size_px=args.scene_size_px,
        canvas_grid=args.canvas_grid,
        glimpse_px=args.glimpse_px,
        n_timesteps=args.n_timesteps,
        hidden_dim=args.hidden_dim,
        cls_dim=args.cls_dim,
        resize_mode=args.resize_mode,
        amp=not args.no_amp,
        warmup=args.warmup,
        max_batches=args.max_batches,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args()
    evaluate(cfg)


if __name__ == "__main__":
    main()
