import pytest
import torch
from canvit_pytorch import Viewpoint

from canvit_rl.policies.image_independent.viewpoint import (
    action_to_viewpoint,
    parse_discrete_scales,
    viewpoint_to_action,
)


def test_parse_discrete_scales():
    assert parse_discrete_scales(None) is None
    assert parse_discrete_scales("") is None
    assert parse_discrete_scales("0.25,0.5") == [0.25, 0.5]
    with pytest.raises(ValueError):
        parse_discrete_scales("0.25,1.5")


def test_action_to_viewpoint_bins_scale_and_keeps_centers_continuous():
    action = torch.tensor(
        [
            [0.1, -0.3, -1.0],
            [0.2, 0.4, -0.01],
            [-0.5, 0.7, 0.0],
            [0.9, -0.9, 1.0],
        ]
    )
    vp = action_to_viewpoint(action, min_scale=0.25, discrete_scales=[0.25, 0.5])
    assert torch.allclose(vp.centers, action[:, :2])
    assert vp.scales.tolist() == [0.25, 0.25, 0.5, 0.5]


def test_action_to_viewpoint_continuous_unchanged_without_discrete_scales():
    action = torch.tensor([[0.0, 0.0, 0.0]])
    vp = action_to_viewpoint(action, min_scale=0.25)
    assert float(vp.scales[0]) == pytest.approx(0.625)


def test_viewpoint_to_action_round_trips_discrete_scales():
    discrete = [0.25, 0.5]
    vp = Viewpoint(
        centers=torch.tensor([[0.1, 0.2], [-0.3, 0.4]]),
        scales=torch.tensor([0.25, 0.5]),
    )
    action = viewpoint_to_action(vp, min_scale=0.25, discrete_scales=discrete)
    assert action[:, 2].tolist() == [-0.5, 0.5]
    back = action_to_viewpoint(action, min_scale=0.25, discrete_scales=discrete)
    assert torch.allclose(back.centers, vp.centers)
    assert torch.allclose(back.scales, vp.scales)
