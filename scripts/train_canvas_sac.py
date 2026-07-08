"""
Train image-dependent SAC over the current layernorm-pooled CanViT canvas.

This is the canvas-state analogue of scripts/train_viewpoint_sac.py:

    state  = current CanViT canvas summary + compact viewpoint history
    action = next Viewpoint
    reward = (CE_before - CE_after) / CE_before

Example:
    python scripts/train_canvas_sac.py \
    --dataset synthetic_segmentation \
    --batches 100 --batch-size 1 --max-samples 1 --t 2 \
    --eval-images 1 --eval-batch-size 1 --eval-split training\
    --replay-batch-size 4 \
    --checkpoint-dir checkpoints/canvas_sac/synthetic-im1-t2 \
    --experiment-name synthetic-im1-t2_no_max \
    --comet-project synthetic-tests \
    --reward-map-output-dir results/synthetic_sac_test\
    --reward-map-images 1 \
    --reward-map-interval 100 \
    --skip-eval-random  \
    --eval-init-full-validation-miou 

"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

from canvit_rl.canvas.logging import (
    log_canvas_sac_final_metrics,
    log_final_full_validation_miou_curve,
    make_comet_experiment,
)

import numpy as np
import torch
from canvit_pytorch import (
    CanViTForSemanticSegmentation,
    Viewpoint,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from tqdm import tqdm

from canvit_rl.canvas.state import (
    append_viewpoint_history,
    canvas_layernorm_spatial,
    canvas_segmentation_entropy,
    empty_viewpoint_history,
)
from canvit_rl.canvas.sac import (
    REPLAY_STORAGE_DTYPE,
    CanvasReplayBuffer,
    CanvasSAC,
    replay_canvas_bytes,
    resolve_replay_device,
    validate_replay_memory,
)
from canvit_rl.canvit_precision import configure_frozen_canvit_precision
from canvit_rl.canvas.args import add_canvas_sac_args, validate_canvas_sac_args
from canvit_rl.canvas.checkpoints import (
    load_canvas_sac_pretrained_initializers,
    load_canvas_sac_resume,
    save_canvas_sac_checkpoint,
)
from canvit_rl.canvas.eval import (
    evaluate_best_full_validation_miou,
    evaluate_canvas_sac,
    evaluate_egc2f_full_validation_miou,
    evaluate_full_validation_miou,
    segmentation_metrics,
    should_run_final_full_validation_miou,
    viewpoint_entropy,
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
from canvit_rl.env import CanViTEnvConfig, get_device
from canvit_rl.reward import relative_ce_reduction
from canvit_rl.viewpoint_policy import action_to_viewpoint

try:
    from canvas_sac_optuna import add_canvas_sac_optuna_args, run_canvas_sac_optuna
except ImportError:
    from scripts.canvas_sac_optuna import (
        add_canvas_sac_optuna_args,
        run_canvas_sac_optuna,
    )

IMPORTANT_TRAIN_METRICS = {
    "actor/loss",
    "actor/entropy",
    "critic/q1_loss",
    "critic/q2_loss",
    "sac/alpha",
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
    """Capture all RNG streams that can affect subsequent SAC training."""
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


def train_once(args: argparse.Namespace) -> float:
    """Run full current-canvas SAC and return best relative eval CE gain."""
    validate_canvas_sac_args(args)
    parse_reward_map_scales(args.reward_map_scales)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    cfg = CanViTEnvConfig()
    data = build_canvas_sac_data(
        args=args,
        cfg=cfg,
        device=device,
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
    actor, q1, q2, target_q1, target_q2 = build_canvas_sac_networks(
        args=args,
        canvas_feature_dim=canvas_feature_dim,
        device=device,
    )
    agent = CanvasSAC(
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        init_alpha=args.init_alpha,
        target_entropy=args.target_entropy,
    )
    start_batch, update_count, best_relative_ce_gain = load_canvas_sac_resume(
        args=args,
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        agent=agent,
    )
    load_canvas_sac_pretrained_initializers(
        args=args,
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
    )
    replay_bytes = replay_canvas_bytes(
        capacity=args.buffer_size,
        canvas_feature_dim=canvas_feature_dim,
        canvas_grid_size=cfg.canvas_grid_size,
        include_entropy=args.canvas_entropy_state,
    )
    replay_device = resolve_replay_device(
        train_device=device,
        replay_bytes=replay_bytes,
    )
    validate_replay_memory(
        storage_device=replay_device,
        replay_bytes=replay_bytes,
    )
    print(
        "Replay storage: "
        f"device={replay_device}, dtype={REPLAY_STORAGE_DTYPE}, "
        f"canvas_bytes={replay_bytes / 1024**3:.2f} GiB"
    )
    replay = CanvasReplayBuffer(
        capacity=args.buffer_size,
        max_history=args.max_history,
        canvas_feature_dim=canvas_feature_dim,
        canvas_grid_size=cfg.canvas_grid_size,
        storage_device=replay_device,
        store_entropy=args.canvas_entropy_state,
    )
    comet_exp = make_comet_experiment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_full_validation_timesteps: list[int] | None = None
    initial_full_validation_mious: list[float] | None = None
    if args.eval_init_full_validation_miou:
        rng_state = _capture_rng_state()
        try:
            # Problem: final full-validation mIoU has no untrained reference
            # curve, but running that diagnostic can advance RNG streams before
            # SAC training starts. Solution: evaluate the initialized actor
            # inside an RNG save/restore bracket. Result: enabling
            # --eval-init-full-validation-miou does not change later sampler,
            # warmup-action, actor-sampling, or replay-sampling randomness.
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

    def evaluate_canvas_sac_with_cached_baselines(
        split_label: str,
        eval_loader,
    ) -> dict[str, float]:
        """Evaluate SAC while reusing deterministic baseline metrics per split."""
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
            # Problem: EG-C2F is deterministic for a fixed loader, but it was
            # recomputed at every SAC eval interval. Solution: cache the mIoU
            # the first time each split/loader is evaluated. Result: later
            # evals update SAC metrics only while reusing the constant baseline.
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

    pbar = tqdm(
        range(start_batch, args.batches + 1),
        desc="Training canvas SAC",
        dynamic_ncols=True,
    )
    for batch_idx in pbar:
        sync_for_timing(device)
        batch_start = time.perf_counter()
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(data.train_loader)
            images, masks = next(train_iter)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.shape[0]

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
            full_out = model(
                glimpse=full_glimpse,
                state=state,
                viewpoint=full_vp,
            )
            state = full_out.state
            current_ce, _ = segmentation_metrics(
                model=model, probe=probe, state=state, masks=masks, cfg=cfg
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
            if replay.size < args.learning_starts:
                action = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)
            else:
                with torch.no_grad():
                    action, log_prob = actor.sample(obs)
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
                out = model(
                    glimpse=glimpse,
                    state=state,
                    viewpoint=vp,
                )
                next_ce, _ = segmentation_metrics(
                    model=model, probe=probe, state=out.state, masks=masks, cfg=cfg
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
            replay.add_batch(
                canvas=prev_canvas,
                coords=prev_coords,
                lengths=prev_lengths,
                actions=action.detach().clone(),
                rewards=reward.detach().clone(),
                next_canvas=next_canvas_summary,
                next_coords=coords,
                next_lengths=lengths,
                dones=done,
                entropy=prev_entropy,
                next_entropy=next_canvas_entropy,
            )
            reward_window.extend(reward.detach().cpu().numpy().astype(float).tolist())
            state = out.state
            current_ce = next_ce
            canvas_summary = next_canvas_summary
            canvas_entropy = next_canvas_entropy

        sync_for_timing(device)
        elapsed_seconds += time.perf_counter() - batch_start
        committed_glimpses += batch_size * (args.t + 1)
        glimpses_per_sec = committed_glimpses / max(elapsed_seconds, 1e-12)

        if replay.size >= args.learning_starts:
            for _ in range(args.updates_per_batch):
                metrics = agent.update(replay.sample(args.replay_batch_size, device))
                update_count += 1
                for key, value in metrics.items():
                    train_windows[key].append(value)

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
                                "train/online_reward/mean": float(
                                    np.mean(rewards_np)
                                ),
                                "train/online_reward/std": float(
                                    np.std(rewards_np)
                                ),
                                "train/online_reward/max": float(
                                    np.max(rewards_np)
                                ),
                                "train/online_reward/min": float(
                                    np.min(rewards_np)
                                ),
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
                    train_eval_metrics = evaluate_canvas_sac_with_cached_baselines(
                        args.split,
                        data.train_eval_loader,
                    )
                    if args.eval_split == args.split:
                        selected_eval_metrics = train_eval_metrics
                    else:
                        selected_eval_metrics = evaluate_canvas_sac_with_cached_baselines(
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
                    current_eval_reward = eval_metrics["eval/reward"]
                    current_train_reward = eval_metrics[f"eval/{args.split}/reward"]
                    current_selected_reward = eval_metrics[
                        f"eval/{args.eval_split}/reward"
                    ]
                    save_canvas_sac_checkpoint(
                        path=args.checkpoint_dir / "latest.pt",
                        actor=actor,
                        q1=q1,
                        q2=q2,
                        target_q1=target_q1,
                        target_q2=target_q2,
                        agent=agent,
                        args=args,
                        canvas_feature_dim=canvas_feature_dim,
                        batch=batch_idx,
                        updates=update_count,
                        best_relative_ce_gain=best_relative_ce_gain,
                        eval_metrics=eval_metrics,
                    )
                    if current_eval_reward > best_relative_ce_gain:
                        best_relative_ce_gain = current_eval_reward
                        save_canvas_sac_checkpoint(
                            path=args.checkpoint_dir / "best.pt",
                            actor=actor,
                            q1=q1,
                            q2=q2,
                            target_q1=target_q1,
                            target_q2=target_q2,
                            agent=agent,
                            args=args,
                            canvas_feature_dim=canvas_feature_dim,
                            batch=batch_idx,
                            updates=update_count,
                            best_relative_ce_gain=best_relative_ce_gain,
                            eval_metrics=eval_metrics,
                        )
                    reward_text = (
                        f"{args.split}_reward={current_train_reward:+.4f}"
                        if args.eval_split == args.split
                        else (
                            f"{args.split}_reward={current_train_reward:+.4f} "
                            f"{args.eval_split}_reward={current_selected_reward:+.4f}"
                        )
                    )
                    pbar.write(
                        f"update={update_count} batch={batch_idx} "
                        f"{reward_text} "
                        f"ce_gain={eval_metrics['eval/ce_gain']:+.4f} "
                        f"sac_miou={eval_metrics['eval/sac_miou']:.4f}"
                    )
                    next_eval_update += args.eval_interval

                if (
                    args.reward_map_images > 0
                    and update_count >= next_reward_map_update
                ):
                    maybe_visualize_canvas_sac_reward_maps(
                        actor=actor,
                        q1=q1,
                        q2=q2,
                        eval_dataset=data.eval_dataset,
                        model=model,
                        probe=probe,
                        cfg=cfg,
                        args=args,
                        device=device,
                        canvit_dtype=canvit_dtype,
                        update_count=update_count,
                        comet_exp=comet_exp,
                    )
                    last_reward_map_update = update_count
                    while next_reward_map_update <= update_count:
                        next_reward_map_update += reward_map_interval

        pbar.set_postfix(
            {
                "upd": update_count,
                "replay": replay.size,
                "gl/s": f"{glimpses_per_sec:.1f}",
                "best": f"{best_relative_ce_gain:+.4f}"
                if best_relative_ce_gain != float("-inf")
                else "nan",
            }
        )

    if latest_metrics is None:
        train_eval_metrics = evaluate_canvas_sac_with_cached_baselines(
            args.split,
            data.train_eval_loader,
        )
        if args.eval_split == args.split:
            selected_eval_metrics = train_eval_metrics
        else:
            selected_eval_metrics = evaluate_canvas_sac_with_cached_baselines(
                args.eval_split,
                data.eval_loader,
            )
        latest_metrics = combine_eval_metrics(
            selected_metrics=selected_eval_metrics,
            train_metrics=train_eval_metrics,
            selected_split=args.eval_split,
            train_split=args.split,
        )
        current_eval_reward = latest_metrics["eval/reward"]
        if current_eval_reward > best_relative_ce_gain:
            best_relative_ce_gain = current_eval_reward
            save_canvas_sac_checkpoint(
                path=args.checkpoint_dir / "best.pt",
                actor=actor,
                q1=q1,
                q2=q2,
                target_q1=target_q1,
                target_q2=target_q2,
                agent=agent,
                args=args,
                canvas_feature_dim=canvas_feature_dim,
                batch=args.batches,
                updates=update_count,
                best_relative_ce_gain=best_relative_ce_gain,
                eval_metrics=latest_metrics,
            )
        if comet_exp is not None:
            comet_exp.log_metrics(
                _important_eval_metrics(latest_metrics),
                step=update_count,
            )
        if args.reward_map_images > 0 and last_reward_map_update != update_count:
            maybe_visualize_canvas_sac_reward_maps(
                actor=actor,
                q1=q1,
                q2=q2,
                eval_dataset=data.eval_dataset,
                model=model,
                probe=probe,
                cfg=cfg,
                args=args,
                device=device,
                canvit_dtype=canvit_dtype,
                update_count=update_count,
                comet_exp=comet_exp,
            )
    # Problem: end-of-training weights can differ from the last evaluated
    # checkpoint. Solution: only periodic/fallback eval blocks write best.pt,
    # so best.pt always corresponds to the metrics stored with it.
    save_canvas_sac_checkpoint(
        path=args.checkpoint_dir / "latest.pt",
        actor=actor,
        q1=q1,
        q2=q2,
        target_q1=target_q1,
        target_q2=target_q2,
        agent=agent,
        args=args,
        canvas_feature_dim=canvas_feature_dim,
        batch=args.batches,
        updates=update_count,
        best_relative_ce_gain=best_relative_ce_gain,
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
    print(f"Saved canvas SAC latest checkpoint to {args.checkpoint_dir / 'latest.pt'}")
    print(f"Best eval/reward: {best_relative_ce_gain:+.4f}")
    return best_relative_ce_gain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_canvas_sac_args(parser)
    add_canvas_sac_optuna_args(parser)
    args = parser.parse_args()
    validate_canvas_sac_args(args)
    return args


def main() -> None:
    args = parse_args()
    if args.optuna_trials:
        run_canvas_sac_optuna(args, train_once)
        return
    train_once(args)


if __name__ == "__main__":
    main()
