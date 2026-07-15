"""
Train image-dependent PPO over the current layernorm-pooled CanViT canvas.

This mirrors ``scripts/train_canvas_sac.py`` for ADE20K segmentation while
replacing replay/SAC updates with on-policy clipped PPO updates.

python scripts/train_canvas_ppo.py \
  --dataset synthetic_segmentation \
  --dataset-format synthetic \
  --batches 5001 \
  --batch-size 4 \
  --max-samples 7 \
  --t 1 \
  --eval-images 3 \
  --eval-batch-size 1 \
  --eval-split training \
  --skip-final-full-validation-miou \
  --skip-eval-random \
  --checkpoint-dir checkpoints/canvas_ppo/synthetic-smoke \
  --critic-local-action-features \
  --canvas-entropy-state \
  --disable-canvas-max-pool \
  --reward-map-images 3 \
  --reward-map-interval 500 \
  --experiment-name synthetic-ppo \
  --ppo-epochs 2 \
  --ppo-entropy-coef 0.05 \
  --ppo-target-kl 0.02 \
  --actor-lr 1e-4
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Problem: Comet's framework integrations warn when comet_ml is imported after
# torch. Solution: pre-scan argv and import Comet before torch only for enabled
# Comet runs. Result: --comet runs get clean integration ordering while
# --no-comet/help paths do not pay the import cost.
if "--no-comet" not in sys.argv:
    try:
        import comet_ml as _comet_ml  # noqa: F401
    except ImportError:
        pass

import numpy as np
import torch
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from tqdm import tqdm

from canvit_rl.canvas.args import add_canvas_sac_args, validate_canvas_sac_args
from canvit_rl.canvas.checkpoints import checkpoint_module_state
from canvit_rl.canvas.eval import (
    evaluate_best_full_validation_miou,
    evaluate_canvas_sac,
    evaluate_egc2f_full_validation_miou,
    evaluate_full_validation_miou,
    segmentation_metrics,
    should_run_final_full_validation_miou,
    viewpoint_entropy,
)
from canvit_rl.canvas.logging import (
    log_canvas_sac_final_metrics,
    log_final_full_validation_miou_curve,
    make_comet_experiment,
)
from canvit_rl.canvas.ppo import CanvasPPO, CanvasPPORollout
from canvit_rl.canvas.state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    canvas_segmentation_entropy,
    empty_viewpoint_history,
)
from canvit_rl.canvas.training import (
    build_canvas_sac_data,
    build_canvas_sac_networks,
    combine_eval_metrics,
    sync_for_timing,
)
from canvit_rl.canvas.visualization import (
    maybe_visualize_canvas_sac_reward_maps,
    parse_reward_map_scales,
)
from canvit_rl.canvit_precision import configure_frozen_canvit_precision
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.reward import relative_ce_reduction
from canvit_rl.viewpoint_policy import action_to_viewpoint

try:
    from canvas_ppo_optuna import add_canvas_ppo_optuna_args, run_canvas_ppo_optuna
except ImportError:
    from scripts.canvas_ppo_optuna import (
        add_canvas_ppo_optuna_args,
        run_canvas_ppo_optuna,
    )

IMPORTANT_TRAIN_METRICS = {
    "actor/loss",
    "actor/entropy",
    "actor/std_mean",
    "critic/value_loss",
    "ppo/approx_kl",
    "ppo/clip_fraction",
    "train/online_reward/mean",
    "throughput/glimpses_per_sec",
    "train/viewpoint_entropy",
}

IMPORTANT_EVAL_SUFFIXES = {
    "reward",
    "ce_gain",
    "sac_miou",
    "final_ce",
    "egc2f_miou",
    "random_miou",
    "sac_viewpoint_entropy",
}


def _capture_rng_state() -> dict[str, object]:
    """Capture RNG streams before diagnostic-only eval work."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    """Restore RNG streams after diagnostic-only eval work."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _important_train_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Return the compact train metric set logged to Comet."""
    filtered = {
        key: value for key, value in metrics.items() if key in IMPORTANT_TRAIN_METRICS
    }
    filtered.update(
        {
            key: value
            for key, value in metrics.items()
            if key.startswith("train/mean_scale_by_t")
        }
    )
    return filtered


def _important_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Return the compact eval metric set logged to Comet."""
    filtered: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("eval/"):
            suffix = key.rsplit("/", 1)[-1]
            if suffix in IMPORTANT_EVAL_SUFFIXES:
                filtered[key] = value
            continue
        if key in {"reward", "ce_gain", "final_ce", "viewpoint_entropy"}:
            filtered[key] = value
    return filtered


def _normal_autograd_input(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Detach frozen-model outputs and clone them out of inference tensor mode."""
    if tensor is None:
        return None
    return tensor.detach().clone()


def validate_canvas_ppo_args(args: argparse.Namespace) -> None:
    """Validate PPO-specific constraints after shared Canvas arg checks."""
    validate_canvas_sac_args(args)
    if args.t < 1:
        raise ValueError("Canvas PPO requires --t >= 1 so each rollout has actions.")
    if args.ppo_epochs < 1:
        raise ValueError("--ppo-epochs must be positive.")
    if args.ppo_minibatch_size < 1:
        raise ValueError("--ppo-minibatch-size must be positive.")
    if not 0.0 <= args.ppo_clip_coef <= 1.0:
        raise ValueError("--ppo-clip-coef must be in [0, 1].")
    if args.ppo_target_kl < 0.0:
        raise ValueError("--ppo-target-kl must be non-negative.")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1].")
    if args.max_grad_norm < 0.0:
        raise ValueError("--max-grad-norm must be non-negative.")
    if args.progress_log_interval < 0:
        raise ValueError("--progress-log-interval must be non-negative.")


def save_canvas_ppo_checkpoint(
    *,
    path: Path,
    actor,
    critic,
    agent: CanvasPPO,
    args: argparse.Namespace,
    canvas_feature_dim: int,
    batch: int,
    updates: int,
    best_eval_final_ce: float,
    eval_metrics: dict[str, float] | None,
) -> None:
    """Save PPO state with the same selection contract as Canvas SAC."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": agent.actor_opt.state_dict(),
            "critic_opt": agent.critic_opt.state_dict(),
            "args": vars(args),
            "algorithm": "ppo",
            "canvas_feature_dim": canvas_feature_dim,
            "batch": batch,
            "updates": updates,
            "best_eval_final_ce": best_eval_final_ce,
            "selection_metric": "eval/final_ce",
            "selection_mode": "min",
            "eval_metrics": eval_metrics or {},
            "state_representation": (
                "current_canvas_layernorm_entropy_with_viewpoint_history"
                if getattr(args, "canvas_entropy_state", False)
                else "current_canvas_layernorm_with_viewpoint_history"
            ),
        },
        path,
    )


def load_canvas_ppo_resume(
    *,
    args: argparse.Namespace,
    actor,
    critic,
    agent: CanvasPPO,
) -> tuple[int, int, float]:
    """Resume a Canvas PPO checkpoint without touching unrelated SAC state."""
    if args.resume is None:
        return 1, 0, float("inf")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint.get("critic", checkpoint.get("q1")))
    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "critic_opt" in checkpoint:
        agent.critic_opt.load_state_dict(checkpoint["critic_opt"])
    best_eval_final_ce = checkpoint.get("best_eval_final_ce")
    if best_eval_final_ce is None:
        best_eval_final_ce = checkpoint.get("eval_metrics", {}).get(
            "eval/final_ce",
            float("inf"),
        )
    return (
        int(checkpoint.get("batch", 0)) + 1,
        int(checkpoint.get("updates", 0)),
        float(best_eval_final_ce),
    )


def load_canvas_ppo_pretrained_initializers(
    *,
    args: argparse.Namespace,
    actor,
    critic,
) -> None:
    """Initialize PPO actor/critic from the same checkpoint shapes SAC uses."""
    if args.resume is not None:
        return
    if args.init_actor_checkpoint is not None:
        checkpoint = torch.load(
            args.init_actor_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        actor.load_state_dict(
            checkpoint_module_state(
                checkpoint,
                "actor",
                path=args.init_actor_checkpoint,
            )
        )
        print(f"Initialized canvas PPO actor from {args.init_actor_checkpoint}")
    if args.init_critic_checkpoint is not None:
        checkpoint = torch.load(
            args.init_critic_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        # Problem: existing critic initializers are SAC checkpoints with q1/q2
        # keys. Solution: load q1 into PPO's single action-conditioned critic.
        # Result: PPO can warm-start from the same critic artifacts as SAC.
        critic.load_state_dict(
            checkpoint_module_state(
                checkpoint,
                "q1",
                path=args.init_critic_checkpoint,
            )
        )
        print(f"Initialized canvas PPO critic from {args.init_critic_checkpoint}")


def train_once(args: argparse.Namespace) -> float:
    """Run full current-canvas PPO and return best relative eval CE gain."""
    validate_canvas_ppo_args(args)
    parse_reward_map_scales(args.reward_map_scales)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    cfg = CanViTEnvConfig()
    data = build_canvas_sac_data(args=args, cfg=cfg, device=device)

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
    canvit_dtype = configure_frozen_canvit_precision(
        model=model,
        probe=probe,
        requested=args.canvit_dtype,
        device=device,
    )
    print(f"CanViT inference dtype: {canvit_dtype}")
    for param in model.parameters():
        param.requires_grad_(False)
    for param in probe.parameters():
        param.requires_grad_(False)

    canvas_feature_dim = int(model.canvas_dim)
    actor, critic, _q2, _target_q1, _target_q2 = build_canvas_sac_networks(
        args=args,
        canvas_feature_dim=canvas_feature_dim,
        device=device,
    )
    agent = CanvasPPO(
        actor=actor,
        critic=critic,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_coef=args.ppo_clip_coef,
        value_coef=args.ppo_value_coef,
        entropy_coef=args.ppo_entropy_coef,
        max_grad_norm=args.max_grad_norm,
        epochs=args.ppo_epochs,
        minibatch_size=args.ppo_minibatch_size,
        target_kl=args.ppo_target_kl,
    )
    start_batch, update_count, best_eval_final_ce = load_canvas_ppo_resume(
        args=args,
        actor=actor,
        critic=critic,
        agent=agent,
    )
    load_canvas_ppo_pretrained_initializers(
        args=args,
        actor=actor,
        critic=critic,
    )

    comet_exp = make_comet_experiment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_full_validation_timesteps: list[int] | None = None
    initial_full_validation_mious: list[float] | None = None
    if args.eval_init_full_validation_miou:
        rng_state = _capture_rng_state()
        try:
            # Problem: optional initial full-validation eval can advance RNG
            # before training. Solution: bracket it with RNG save/restore.
            # Result: enabling the diagnostic does not perturb PPO rollouts.
            (
                _initial_full_validation_metrics,
                initial_full_validation_timesteps,
                initial_full_validation_mious,
            ) = evaluate_full_validation_miou(
                actor=actor,
                model=model,
                probe=probe,
                cfg=cfg,
                args=args,
                canvit_dtype=canvit_dtype,
                device=device,
                metric_prefix="initial_full_validation",
                description="Initialized actor",
            )
        finally:
            _restore_rng_state(rng_state)

    egc2f_eval_cache: dict[str, dict[str, float]] = {}

    def evaluate_canvas_ppo_with_cached_baselines(split_label: str, eval_loader):
        """Evaluate PPO actor while reusing deterministic baseline metrics."""
        metrics = evaluate_canvas_sac(
            actor=actor,
            eval_loader=eval_loader,
            model=model,
            probe=probe,
            cfg=cfg,
            args=args,
            canvas_feature_dim=canvas_feature_dim,
            canvit_dtype=canvit_dtype,
            device=device,
            fixed_baseline_metrics=egc2f_eval_cache.get(split_label),
        )
        if (
            not args.skip_eval_egc2f
            and split_label not in egc2f_eval_cache
            and "eval/egc2f_miou" in metrics
        ):
            egc2f_eval_cache[split_label] = {
                "eval/egc2f_miou": metrics["eval/egc2f_miou"]
            }
        return metrics

    train_iter = iter(data.train_loader)
    train_windows: dict[str, list[float]] = defaultdict(list)
    reward_window: list[float] = []
    entropy_points: list[np.ndarray] = []
    scale_sums = [0.0 for _ in range(args.t)]
    scale_counts = [0 for _ in range(args.t)]
    latest_metrics: dict[str, float] | None = None
    next_eval_update = max(args.eval_interval, 1)
    reward_map_interval = max(args.reward_map_interval or args.eval_interval, 1)
    next_reward_map_update = reward_map_interval
    last_reward_map_update: int | None = None
    elapsed_seconds = 0.0
    committed_glimpses = 0

    def maybe_log_canvas_ppo_reward_maps(update: int) -> None:
        """Save/log PPO reward maps using the single critic as both Q inputs."""
        # Problem: PPO reuses the Canvas SAC args, but the initial PPO trainer
        # never consumed --reward-map-* flags. Solution: route PPO through the
        # shared visualization helper with q1=q2=critic. Result: Comet receives
        # the same true-reward/prediction and policy-glimpse images, labelled
        # as PPO, without requiring twin critics.
        maybe_visualize_canvas_sac_reward_maps(
            actor=actor,
            q1=critic,
            q2=critic,
            eval_dataset=data.eval_dataset,
            model=model,
            probe=probe,
            cfg=cfg,
            args=args,
            device=device,
            canvit_dtype=canvit_dtype,
            update_count=update,
            comet_exp=comet_exp,
            algorithm_label="PPO",
            policy_label="ppo",
        )

    pbar = tqdm(
        range(start_batch, args.batches + 1),
        desc="Training canvas PPO",
        dynamic_ncols=True,
        # Problem: on clusters, tqdm writes carriage-return updates into the
        # SLURM .out file as one line per batch. Solution: show the live bar
        # only for interactive terminals. Result: batch-level progress stays
        # useful locally without flooding non-TTY job logs.
        disable=not sys.stderr.isatty(),
    )
    for batch_idx in pbar:
        sync_for_timing(device)
        batch_start = time.perf_counter()
        batch_reward_values: list[float] = []
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(data.train_loader)
            images, masks = next(train_iter)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]

        rollout = CanvasPPORollout(gamma=args.gamma, gae_lambda=args.gae_lambda)
        state = model.init_state(batch_size=batch_size, canvas_grid_size=cfg.canvas_grid_size)
        coords, lengths = empty_viewpoint_history(
            batch_size=batch_size,
            max_steps=args.max_history,
            device=device,
        )
        full_vp = Viewpoint.full_scene(batch_size=batch_size, device=device)
        with torch.inference_mode():
            full_glimpse = sample_at_viewpoint(
                spatial=images,
                viewpoint=full_vp,
                glimpse_size_px=cfg.glimpse_size_px,
            ).to(dtype=canvit_dtype)
            full_out = model(glimpse=full_glimpse, state=state, viewpoint=full_vp)
            state = full_out.state
            current_ce, _ = segmentation_metrics(
                model=model,
                probe=probe,
                state=state,
                masks=masks,
                cfg=cfg,
            )
            canvas_summary = canvas_layernorm_spatial(
                model=model,
                state=state,
                canvas_grid_size=cfg.canvas_grid_size,
            )
            canvas_entropy = (
                canvas_segmentation_entropy(
                    model=model,
                    probe=probe,
                    state=state,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
                if args.canvas_entropy_state
                else None
            )
        # Problem: frozen CanViT outputs are created under inference_mode, but
        # PPO feeds canvas state into trainable actor/critic conv layers.
        # Solution: clone detached state tensors after leaving inference_mode.
        # Result: autograd can save actor/critic intermediates normally while
        # CanViT stays frozen and untracked.
        canvas_summary = _normal_autograd_input(canvas_summary)
        canvas_entropy = _normal_autograd_input(canvas_entropy)
        coords, lengths = append_viewpoint_history(
            coords=coords,
            lengths=lengths,
            viewpoint=full_vp,
            step=0,
        )

        for step_idx in range(args.t):
            obs = {"canvas": canvas_summary, "coords": coords, "lengths": lengths}
            if canvas_entropy is not None:
                obs["entropy"] = canvas_entropy
            action, log_prob = actor.sample(obs)
            with torch.no_grad():
                value = critic(obs, action)
            train_windows["actor/log_prob"].append(float(log_prob.mean().item()))
            train_windows["actor/entropy"].append(float((-log_prob).mean().item()))
            train_windows["actor/action_std"].append(
                float(action.std(unbiased=False).item())
            )

            vp = action_to_viewpoint(action, min_scale=args.min_scale)
            entropy_points.append(
                torch.cat([vp.centers, vp.scales[:, None]], dim=1).detach().cpu().numpy()
            )
            scale_sums[step_idx] += float(vp.scales.detach().sum().item())
            scale_counts[step_idx] += batch_size
            prev_canvas = canvas_summary.clone()
            prev_entropy = canvas_entropy.clone() if canvas_entropy is not None else None
            prev_coords = coords.clone()
            prev_lengths = lengths.clone()
            with torch.inference_mode():
                glimpse = sample_at_viewpoint(
                    spatial=images,
                    viewpoint=vp,
                    glimpse_size_px=cfg.glimpse_size_px,
                ).to(dtype=canvit_dtype)
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
                next_ce, _ = segmentation_metrics(
                    model=model,
                    probe=probe,
                    state=out.state,
                    masks=masks,
                    cfg=cfg,
                )
                next_canvas_summary = canvas_layernorm_spatial(
                    model=model,
                    state=out.state,
                    canvas_grid_size=cfg.canvas_grid_size,
                )
                next_canvas_entropy = (
                    canvas_segmentation_entropy(
                        model=model,
                        probe=probe,
                        state=out.state,
                        canvas_grid_size=cfg.canvas_grid_size,
                    )
                    if args.canvas_entropy_state
                    else None
                )
            next_canvas_summary = _normal_autograd_input(next_canvas_summary)
            next_canvas_entropy = _normal_autograd_input(next_canvas_entropy)
            reward = relative_ce_reduction(current_ce, next_ce)
            coords, lengths = append_viewpoint_history(
                coords=coords,
                lengths=lengths,
                viewpoint=vp,
                step=step_idx + 1,
            )
            done = torch.full(
                (batch_size,),
                float(step_idx == args.t - 1),
                device=device,
            )
            rollout.add_batch(
                canvas=prev_canvas,
                coords=prev_coords,
                lengths=prev_lengths,
                actions=action,
                old_log_probs=log_prob,
                rewards=reward,
                dones=done,
                values=value,
                entropy=prev_entropy,
            )
            reward_values = reward.detach().cpu().numpy().astype(float).tolist()
            reward_window.extend(reward_values)
            batch_reward_values.extend(reward_values)
            state = out.state
            current_ce = next_ce
            canvas_summary = next_canvas_summary
            canvas_entropy = next_canvas_entropy

        metrics = agent.update(rollout)
        update_count += 1
        for key, value in metrics.items():
            train_windows[key].append(value)

        sync_for_timing(device)
        elapsed_seconds += time.perf_counter() - batch_start
        committed_glimpses += batch_size * (args.t + 1)
        glimpses_per_sec = committed_glimpses / max(elapsed_seconds, 1e-12)

        if (
            not sys.stderr.isatty()
            and args.progress_log_interval > 0
            and (
                batch_idx == start_batch
                or batch_idx == args.batches
                or batch_idx % args.progress_log_interval == 0
            )
        ):
            batch_reward_mean = (
                float(np.mean(batch_reward_values)) if batch_reward_values else float("nan")
            )
            # Problem: disabling tqdm keeps SLURM logs clean but can make long
            # runs look silent between evals. Solution: emit one compact plain
            # progress line every N batches in non-TTY runs. Result: logs show
            # liveness and core PPO loss/exploration signals without a line per
            # batch.
            print(
                "train_progress "
                f"batch={batch_idx}/{args.batches} "
                f"update={update_count} "
                f"reward={batch_reward_mean:+.4f} "
                f"actor_loss={metrics.get('actor/loss', float('nan')):+.4f} "
                f"value_loss={metrics.get('critic/value_loss', float('nan')):.4f} "
                f"entropy={metrics.get('actor/entropy', float('nan')):.4f} "
                f"std={metrics.get('actor/std_mean', float('nan')):.4f} "
                f"kl={metrics.get('ppo/approx_kl', float('nan')):.5f} "
                f"glimpses_per_sec={glimpses_per_sec:.1f}",
                flush=True,
            )

        if update_count % args.comet_log_interval == 0:
            train_metrics = {
                key: float(np.mean(values))
                for key, values in train_windows.items()
                if values
            }
            if reward_window:
                rewards_np = np.asarray(reward_window, dtype=np.float64)
                train_metrics.update(
                    {
                        "train/online_reward/mean": float(np.mean(rewards_np)),
                        "train/online_reward/std": float(np.std(rewards_np)),
                        "train/online_reward/max": float(np.max(rewards_np)),
                        "train/online_reward/min": float(np.min(rewards_np)),
                    }
                )
            train_metrics["throughput/glimpses_per_sec"] = glimpses_per_sec
            train_metrics["train/viewpoint_entropy"] = viewpoint_entropy(
                entropy_points,
                bins=args.viewpoint_entropy_bins,
            )
            for step in range(args.t):
                train_metrics[f"train/mean_scale_by_t{step + 1}"] = (
                    scale_sums[step] / max(scale_counts[step], 1)
                )
            if comet_exp is not None:
                comet_exp.log_metrics(
                    _important_train_metrics(train_metrics),
                    step=update_count,
                )
            train_windows.clear()
            reward_window.clear()
            entropy_points.clear()
            scale_sums = [0.0 for _ in range(args.t)]
            scale_counts = [0 for _ in range(args.t)]

        if update_count >= next_eval_update:
            train_eval_metrics = evaluate_canvas_ppo_with_cached_baselines(
                args.split,
                data.train_eval_loader,
            )
            if args.eval_split == args.split:
                selected_eval_metrics = train_eval_metrics
            else:
                selected_eval_metrics = evaluate_canvas_ppo_with_cached_baselines(
                    args.eval_split,
                    data.eval_loader,
                )
            eval_metrics = combine_eval_metrics(
                selected_metrics=selected_eval_metrics,
                train_metrics=train_eval_metrics,
                selected_split=args.eval_split,
                train_split=args.split,
            )
            latest_metrics = eval_metrics
            if comet_exp is not None:
                comet_exp.log_metrics(
                    _important_eval_metrics(eval_metrics),
                    step=update_count,
                )
            current_eval_final_ce = eval_metrics["eval/final_ce"]
            save_canvas_ppo_checkpoint(
                path=args.checkpoint_dir / "latest.pt",
                actor=actor,
                critic=critic,
                agent=agent,
                args=args,
                canvas_feature_dim=canvas_feature_dim,
                batch=batch_idx,
                updates=update_count,
                best_eval_final_ce=best_eval_final_ce,
                eval_metrics=eval_metrics,
            )
            # Problem: reward is a noisy proxy for segmentation quality.
            # Solution: select best.pt by lowest selected-split final CE, just
            # like Canvas SAC. Result: PPO/SAC checkpoints are comparable.
            if current_eval_final_ce < best_eval_final_ce:
                best_eval_final_ce = current_eval_final_ce
                save_canvas_ppo_checkpoint(
                    path=args.checkpoint_dir / "best.pt",
                    actor=actor,
                    critic=critic,
                    agent=agent,
                    args=args,
                    canvas_feature_dim=canvas_feature_dim,
                    batch=batch_idx,
                    updates=update_count,
                    best_eval_final_ce=best_eval_final_ce,
                    eval_metrics=eval_metrics,
                )
            pbar.write(
                f"update={update_count} batch={batch_idx} "
                f"reward={eval_metrics['eval/reward']:+.4f} "
                f"ce_gain={eval_metrics['eval/ce_gain']:+.4f} "
                f"ppo_miou={eval_metrics['eval/sac_miou']:.4f}"
            )
            next_eval_update += args.eval_interval

        if args.reward_map_images > 0 and update_count >= next_reward_map_update:
            maybe_log_canvas_ppo_reward_maps(update_count)
            last_reward_map_update = update_count
            while next_reward_map_update <= update_count:
                next_reward_map_update += reward_map_interval

        pbar.set_postfix(
            {
                "upd": update_count,
                "gl/s": f"{glimpses_per_sec:.1f}",
                "best_ce": f"{best_eval_final_ce:.4f}"
                if best_eval_final_ce != float("inf")
                else "nan",
            }
        )

    if latest_metrics is None:
        train_eval_metrics = evaluate_canvas_ppo_with_cached_baselines(
            args.split,
            data.train_eval_loader,
        )
        if args.eval_split == args.split:
            selected_eval_metrics = train_eval_metrics
        else:
            selected_eval_metrics = evaluate_canvas_ppo_with_cached_baselines(
                args.eval_split,
                data.eval_loader,
            )
        latest_metrics = combine_eval_metrics(
            selected_metrics=selected_eval_metrics,
            train_metrics=train_eval_metrics,
            selected_split=args.eval_split,
            train_split=args.split,
        )
        if comet_exp is not None:
            comet_exp.log_metrics(
                _important_eval_metrics(latest_metrics),
                step=update_count,
            )
        if latest_metrics["eval/final_ce"] < best_eval_final_ce:
            best_eval_final_ce = latest_metrics["eval/final_ce"]
            save_canvas_ppo_checkpoint(
                path=args.checkpoint_dir / "best.pt",
                actor=actor,
                critic=critic,
                agent=agent,
                args=args,
                canvas_feature_dim=canvas_feature_dim,
                batch=args.batches,
                updates=update_count,
                best_eval_final_ce=best_eval_final_ce,
                eval_metrics=latest_metrics,
            )

    if args.reward_map_images > 0 and last_reward_map_update != update_count:
        maybe_log_canvas_ppo_reward_maps(update_count)

    save_canvas_ppo_checkpoint(
        path=args.checkpoint_dir / "latest.pt",
        actor=actor,
        critic=critic,
        agent=agent,
        args=args,
        canvas_feature_dim=canvas_feature_dim,
        batch=args.batches,
        updates=update_count,
        best_eval_final_ce=best_eval_final_ce,
        eval_metrics=latest_metrics,
    )
    torch.save(actor.state_dict(), args.checkpoint_dir / "actor_final.pt")
    log_canvas_sac_final_metrics(
        comet_exp=comet_exp,
        metrics=latest_metrics,
        step=update_count,
    )

    if should_run_final_full_validation_miou(args):
        egc2f_full_validation_timesteps: list[int] | None = None
        egc2f_full_validation_mious: list[float] | None = None
        if not args.skip_eval_egc2f:
            (
                _egc2f_full_validation_metrics,
                egc2f_full_validation_timesteps,
                egc2f_full_validation_mious,
            ) = evaluate_egc2f_full_validation_miou(
                model=model,
                probe=probe,
                cfg=cfg,
                args=args,
                canvit_dtype=canvit_dtype,
                device=device,
            )
        (
            _full_validation_metrics,
            full_validation_timesteps,
            full_validation_mious,
        ) = evaluate_best_full_validation_miou(
            actor=actor,
            model=model,
            probe=probe,
            cfg=cfg,
            args=args,
            canvit_dtype=canvit_dtype,
            device=device,
        )
        log_final_full_validation_miou_curve(
            comet_exp=comet_exp,
            timesteps=full_validation_timesteps,
            miou_values=full_validation_mious,
            step=update_count,
            initial_miou_values=initial_full_validation_mious
            if initial_full_validation_timesteps == full_validation_timesteps
            else None,
            egc2f_miou_values=egc2f_full_validation_mious
            if egc2f_full_validation_timesteps == full_validation_timesteps
            else None,
            comparison_output=(
                args.checkpoint_dir
                / "final_full_validation_miou_by_timestep_overlay.png"
            ),
        )
    else:
        print("Skipped final full validation mIoU evaluation.")

    if comet_exp is not None:
        comet_exp.end()
    print(f"Saved canvas PPO latest checkpoint to {args.checkpoint_dir / 'latest.pt'}")
    print(f"Best eval/final_ce: {best_eval_final_ce:.4f}")
    return best_eval_final_ce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_canvas_sac_args(parser)
    parser.set_defaults(
        checkpoint_dir=Path("checkpoints/canvas_ppo"),
        comet_project="canvas-ppo",
        comet_tags="canvas-ppo",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=4)
    parser.add_argument("--ppo-clip-coef", type=float, default=0.2)
    parser.add_argument("--ppo-value-coef", type=float, default=0.5)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.03)
    parser.add_argument(
        "--ppo-target-kl",
        type=float,
        default=0.03,
        help="Stop PPO epochs early when mean approximate KL exceeds this value.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--progress-log-interval",
        type=int,
        default=100,
        help=(
            "For non-interactive runs with tqdm disabled, print one compact "
            "training progress line every N batches. Use 0 to disable."
        ),
    )
    add_canvas_ppo_optuna_args(parser)
    args = parser.parse_args()
    validate_canvas_ppo_args(args)
    return args


def main() -> None:
    args = parse_args()
    if args.optuna_trials:
        run_canvas_ppo_optuna(args, train_once)
        return
    train_once(args)


if __name__ == "__main__":
    main()
