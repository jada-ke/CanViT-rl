import torch

from canvit_rl.sac_models import CanvasStateCritic


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    coords = torch.zeros(batch_size, 3, 3)
    coords[:, 0, 2] = 1.0
    return {
        "canvas": torch.randn(batch_size, 4, 5, 5),
        "coords": coords,
        "lengths": torch.ones(batch_size, dtype=torch.long),
    }


def test_canvas_critic_local_action_features_are_opt_in():
    base = CanvasStateCritic(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
    )
    local = CanvasStateCritic(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
        use_action_location_features=True,
    )

    assert base.q[0].normalized_shape == (19,)
    assert local.q[0].normalized_shape == (27,)


def test_canvas_critic_samples_action_location_features_forward_shape():
    critic = CanvasStateCritic(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
        use_action_location_features=True,
    )
    action = torch.tensor(
        [
            [0.0, 0.0, -0.5],
            [0.5, -0.5, 0.25],
        ],
        dtype=torch.float32,
    )

    q = critic(_batch(), action)

    assert q.shape == (2,)
