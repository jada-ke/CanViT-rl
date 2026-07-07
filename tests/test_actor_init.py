import pytest
import torch
import torch.nn as nn

from canvit_rl.viewpoint_policy import randomize_actor_mean_viewpoint_prior


class DummyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(4, 6))


def test_random_actor_prior_sets_mean_rows_and_preserves_log_std_rows():
    torch.manual_seed(0)
    actor = DummyActor()
    original_log_std_weight = actor.head[-1].weight[3:].detach().clone()
    original_log_std_bias = actor.head[-1].bias[3:].detach().clone()

    prior = randomize_actor_mean_viewpoint_prior(
        actor,
        min_scale=0.25,
        center_radius=0.2,
    )

    mean_action = torch.tanh(actor.head[-1].bias[:3])
    decoded_scale = (mean_action[2] + 1.0) * 0.5 * (1.0 - 0.25) + 0.25

    assert actor.head[-1].weight[:3].abs().sum().item() == pytest.approx(0.0)
    assert float(mean_action[0]) == pytest.approx(prior["center_y"])
    assert float(mean_action[1]) == pytest.approx(prior["center_x"])
    assert float(decoded_scale) == pytest.approx(prior["scale"])
    assert -0.2 <= prior["center_y"] <= 0.2
    assert -0.2 <= prior["center_x"] <= 0.2
    assert 0.25 <= prior["scale"] <= 1.0
    assert torch.equal(actor.head[-1].weight[3:], original_log_std_weight)
    assert torch.equal(actor.head[-1].bias[3:], original_log_std_bias)
