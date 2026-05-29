"""
canvit_rl/train.py

Differentiable policy gradient training for CanViT glimpse selection.

Since sample_at_viewpoint uses bilinear interpolation (F.grid_sample) and
CanViTForPretraining has no gradient blocking, we can backprop directly
through the entire pipeline:

    policy(obs) → viewpoint → sample_at_viewpoint → CanViT
                → predict_scene_teacher_cls → cosine_sim → loss

Usage:
    python -m canvit_rl.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, RandomSampler

from canvit_pytorch import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit_pytorch.teacher import load_teacher
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.policy import MLPPolicy
from canvit_specialize.datasets.ade20k import ADE20kDataset, make_val_transforms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action → Viewpoint
# ---------------------------------------------------------------------------

def action_to_viewpoint(action: torch.Tensor) -> Viewpoint:
    """
    Map policy output [B, 3] → Viewpoint.
    action[:, 0:2] → centers (cx, cy) in [-1, 1]
    action[:, 2]   → scale, remapped from [-1, 1] to [0.05, 1.0]
    """
    centers = action[:, :2].float()                        # [B, 2]
    scale_raw = action[:, 2]
    scales = (scale_raw + 1.0) / 2.0 * 0.95 + 0.05       # [0.05, 1.0]
    scales = scales.float()
    return Viewpoint(centers=centers, scales=scales)


# ---------------------------------------------------------------------------
# Episode rollout (differentiable)
# ---------------------------------------------------------------------------

def run_episode(
    model: CanViTForPretrainingHFHub,
    policy: MLPPolicy,
    image: torch.Tensor,
    teacher_cls: torch.Tensor,
    cfg: CanViTEnvConfig,
    n_steps: int,
    device: torch.device,
    lambda_scale: float,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """
    Roll out one episode differentiably.

    Returns mean loss across steps plus per-step diagnostics. Here s_t is the
    teacher-CLS cosine similarity after a glimpse, which is the available
    train-time proxy for representation quality; 

    Objective: reward_t = (sim_t - sim_{t-1}) - lambda_scale * scale_t, so the
    next glimpse maximizes marginal information gain while paying a cost for
    larger glimpses.

    Gradients flow: policy → viewpoint → sample_at_viewpoint → CanViT
                    → predict_scene_teacher_cls → cosine_sim → delta loss
    """
    state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid_size)

    # Initial full-scene glimpse (no policy action, no gradient)
    with torch.no_grad():
        vp0 = Viewpoint.full_scene(batch_size=1, device=device)
        glimpse0 = sample_at_viewpoint(
            spatial=image, viewpoint=vp0, glimpse_size_px=cfg.glimpse_size_px
        )
        out = model(glimpse=glimpse0, state=state, viewpoint=vp0)
        state = out.state

    step_losses = []
    diagnostics = {"s": [], "delta_s": [], "reward": [], "scales": []}

    with torch.no_grad():
        predicted_cls0 = model.predict_scene_teacher_cls(state.recurrent_cls)
        prev_sim = F.cosine_similarity(
            predicted_cls0.float(), teacher_cls.float(), dim=-1
        ).detach()
        prev_s = float(prev_sim.mean().item())

    for _ in range(n_steps):
        # Observation: recurrent_cls — detach to treat as input, not backprop through state
        obs = state.recurrent_cls.squeeze(1).detach()   # [1, 768]

        # Policy forward (differentiable)
        action = policy(obs)                             # [1, 3]
        vp = action_to_viewpoint(action)

        # Sample glimpse and step CanViT (differentiable w.r.t. vp/action)
        glimpse = sample_at_viewpoint(
            spatial=image, viewpoint=vp, glimpse_size_px=cfg.glimpse_size_px
        )
        out = model(glimpse=glimpse, state=state, viewpoint=vp)

        # Project canvas CLS into teacher space and compute similarity
        predicted_cls = model.predict_scene_teacher_cls(out.state.recurrent_cls)  # [1, 768]
        sim = F.cosine_similarity(predicted_cls.float(), teacher_cls.float(), dim=-1)

        delta_sim = sim - prev_sim
        reward = delta_sim - lambda_scale * vp.scales
        step_losses.append(-reward.mean())

        current_s = float(sim.detach().mean().item())
        diagnostics["s"].append(current_s)
        diagnostics["delta_s"].append(current_s - prev_s)
        diagnostics["reward"].append(float(reward.detach().mean().item()))
        diagnostics["scales"].append(float(vp.scales.detach().mean().item()))
        prev_s = current_s
        prev_sim = sim.detach()

        # Update state (detached — each step is independent, myopic reward)
        state = out.state

    return torch.stack(step_losses).mean(), diagnostics


def _format_step_means(values: list[float]) -> str:
    """Format per-timestep means compactly for training logs."""
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _mean_by_step(step_values: list[list[float]], n_steps: int) -> list[float]:
    """
    Fixed by Codex on 2026-05-22
    Problem: Per-glimpse diagnostics need to be averaged across episodes before
    they are interpretable.
    Solution: Accumulate episode-local vectors and compute one mean per timestep.
    Result: logs show Δs_t, s_t, and scale trends over the same episode window
    as avg_loss.
    """
    return [
        sum(values[step] for values in step_values) / len(step_values)
        for step in range(n_steps)
    ]


def _summarize_scale_trend(mean_scales: list[float]) -> tuple[float, float, float]:
    """Return early mean, late mean, and late-minus-early scale trend."""
    midpoint = max(1, len(mean_scales) // 2)
    early = sum(mean_scales[:midpoint]) / midpoint
    late_values = mean_scales[midpoint:] or mean_scales[:midpoint]
    late = sum(late_values) / len(late_values)
    return early, late, late - early


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # --- Device ---
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = get_device()
    else:
        device = torch.device(device_str)
    log.info(f"Device: {device}")

    # --- Environment config ---
    cfg = CanViTEnvConfig(
        max_steps=config.get("max_steps", 5),
        canvas_grid_size=config.get("canvas_grid_size", 32),
    )

    # --- Dataset ---
    dataset_root = Path(config.get("dataset", "datasets/ADE20k"))
    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, mode="squish")
    dataset = ADE20kDataset(
        root=dataset_root,
        split="training",
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=RandomSampler(dataset, replacement=True),
        num_workers=0,
    )
    log.info(f"Dataset: {len(dataset)} images")

    # --- CanViT (frozen) ---
    log.info("Loading CanViT...")
    model = (
        CanViTForPretrainingHFHub.from_pretrained(cfg.checkpoint)
        .eval()
        .to(device)
    )
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Teacher (frozen) ---
    log.info("Loading teacher...")
    teacher = load_teacher(cfg.teacher_repo, device)

    # --- Policy ---
    policy = MLPPolicy(
        cls_dim=cfg.cls_dim,
        hidden_dim=config.get("hidden_dim", 256),
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=config.get("lr", 3e-4))

    n_episodes = config.get("n_episodes", 1000)
    n_steps = config.get("max_steps", 5)
    lambda_scale = config.get("lambda_scale", 0.0)
    log_interval = config.get("log_interval", 50)
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Training for %s episodes, %s steps/episode, lambda_scale=%.4f",
        n_episodes,
        n_steps,
        lambda_scale,
    )

    data_iter = iter(loader)
    running_loss = 0.0
    running_s: list[list[float]] = []
    running_delta_s: list[list[float]] = []
    running_reward: list[list[float]] = []
    running_scales: list[list[float]] = []

    for episode in range(1, n_episodes + 1):
        # Sample image
        try:
            image, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            image, _ = next(data_iter)

        image = image.to(device)

        # Teacher CLS (no grad needed)
        with torch.no_grad():
            teacher_cls = teacher.forward_norm_features(image).cls  # [1, 768]

        # Differentiable episode rollout
        optimizer.zero_grad()
        loss, diagnostics = run_episode(
            model=model,
            policy=policy,
            image=image,
            teacher_cls=teacher_cls,
            cfg=cfg,
            n_steps=n_steps,
            device=device,
            lambda_scale=lambda_scale,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_s.append(diagnostics["s"])
        running_delta_s.append(diagnostics["delta_s"])
        running_reward.append(diagnostics["reward"])
        running_scales.append(diagnostics["scales"])

        if episode % log_interval == 0:
            avg_loss = running_loss / log_interval
            mean_s = _mean_by_step(running_s, n_steps)
            mean_delta_s = _mean_by_step(running_delta_s, n_steps)
            mean_reward = _mean_by_step(running_reward, n_steps)
            mean_scales = _mean_by_step(running_scales, n_steps)
            early_scale, late_scale, scale_trend = _summarize_scale_trend(
                mean_scales
            )

            log.info(
                "Episode %s/%s | avg_loss=%.4f | "
                "mean_s_by_t=%s | mean_delta_s_by_t=%s | mean_reward_by_t=%s | "
                "mean_scale_by_t=%s | early_scale=%.4f | late_scale=%.4f | "
                "late_minus_early_scale=%.4f",
                episode,
                n_episodes,
                avg_loss,
                _format_step_means(mean_s),
                _format_step_means(mean_delta_s),
                _format_step_means(mean_reward),
                _format_step_means(mean_scales),
                early_scale,
                late_scale,
                scale_trend,
            )
            running_loss = 0.0
            running_s.clear()
            running_delta_s.clear()
            running_reward.clear()
            running_scales.clear()

        if episode % config.get("checkpoint_interval", 500) == 0:
            ckpt_path = checkpoint_dir / f"policy_ep{episode}.pt"
            torch.save(policy.state_dict(), ckpt_path)
            log.info(f"Saved checkpoint: {ckpt_path}")

    torch.save(policy.state_dict(), checkpoint_dir / "policy_final.pt")
    log.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
