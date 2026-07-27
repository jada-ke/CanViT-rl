import torch

from canvit_rl.canvas.critic_losses import (
    candidate_critic_loss,
    pairwise_best_margin_loss,
    topk_reward_weights,
)


def test_topk_reward_weights_emphasize_best_candidates_per_sample():
    rewards = torch.tensor(
        [
            0.1,
            0.4,
            0.2,
            0.3,
            0.9,
            0.5,
            0.8,
            0.7,
        ]
    )

    weights = topk_reward_weights(
        rewards,
        batch_size=2,
        k=4,
        top_frac=0.25,
        top_weight=5.0,
    ).view(4, 2)

    assert torch.equal(weights[:, 0], torch.tensor([1.0, 1.0, 5.0, 1.0]))
    assert torch.equal(weights[:, 1], torch.tensor([1.0, 1.0, 1.0, 5.0]))


def test_pairwise_loss_penalizes_wrong_best_ordering():
    rewards = torch.tensor([0.1, 0.2, 0.5, 0.1])
    bad_q = torch.tensor([0.3, 0.4, 0.2, 0.1])
    good_q = torch.tensor([0.1, 0.2, 0.7, 0.0])

    bad_loss = pairwise_best_margin_loss(
        bad_q,
        rewards,
        batch_size=1,
        k=4,
        margin=0.05,
    )
    good_loss = pairwise_best_margin_loss(
        good_q,
        rewards,
        batch_size=1,
        k=4,
        margin=0.05,
    )

    assert bad_loss > good_loss
    assert good_loss == torch.tensor(0.0)


def test_hybrid_listwise_loss_keeps_mse_component_for_logging():
    rewards = torch.tensor([0.1, 0.3, 0.5, 0.2])
    q = torch.tensor([0.1, 0.2, 0.4, 0.2])

    loss, components = candidate_critic_loss(
        q,
        rewards,
        batch_size=1,
        k=4,
        mode="mse_listwise",
        top_frac=0.25,
        top_weight=5.0,
        aux_weight=0.3,
        pairwise_margin=0.02,
        listwise_temperature=0.05,
    )

    assert loss > components["mse"]
    assert "listwise" in components
    assert components["mse"] >= 0.0
