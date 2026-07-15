import torch

from canvit_rl.canvas.ppo import CanvasPPO, CanvasPPORollout, canvas_actor_log_prob
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic


def _modules():
    actor = CanvasStateActor(
        canvas_feature_dim=4,
        d_model=16,
        rff_dim=8,
        rff_seed=0,
    )
    critic = CanvasStateCritic(
        canvas_feature_dim=4,
        d_model=16,
        rff_dim=8,
        rff_seed=0,
    )
    return actor, critic


def _obs(batch_size: int = 3):
    return {
        "canvas": torch.randn(batch_size, 4, 5, 5),
        "coords": torch.zeros(batch_size, 2, 3),
        "lengths": torch.ones(batch_size, dtype=torch.long),
    }


def test_canvas_actor_log_prob_replays_sampled_actions():
    actor, _critic = _modules()
    obs = _obs()
    action, old_log_prob = actor.sample(obs)

    log_prob = canvas_actor_log_prob(actor, obs, action)

    assert log_prob.shape == old_log_prob.shape
    assert torch.isfinite(log_prob).all()
    assert torch.allclose(log_prob, old_log_prob, atol=1e-4, rtol=1e-4)


def test_canvas_ppo_update_uses_canvas_actor_and_critic():
    actor, critic = _modules()
    agent = CanvasPPO(
        actor=actor,
        critic=critic,
        actor_lr=1e-3,
        critic_lr=1e-3,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        epochs=2,
        minibatch_size=3,
    )
    rollout = CanvasPPORollout(gamma=0.9, gae_lambda=0.95)
    for step in range(2):
        obs = _obs()
        action, log_prob = actor.sample(obs)
        with torch.no_grad():
            value = critic(obs, action)
        # Problem: PPO is on-policy, so the unit test needs stored action
        # log-probs and action-conditioned values from the exact rollout
        # modules. Solution: create a tiny synthetic rollout through the real
        # actor/critic. Result: update() exercises the production tensor paths.
        rollout.add_batch(
            canvas=obs["canvas"],
            coords=obs["coords"],
            lengths=obs["lengths"],
            actions=action,
            old_log_probs=log_prob,
            rewards=torch.full((3,), 0.1 * (step + 1)),
            dones=torch.full((3,), float(step == 1)),
            values=value,
        )

    metrics = agent.update(rollout)

    assert "actor/loss" in metrics
    assert "critic/value_loss" in metrics
    assert "ppo/clip_fraction" in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
