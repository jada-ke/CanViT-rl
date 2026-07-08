import torch

from canvit_rl.canvas.sac import CanvasReplayBuffer
from canvit_rl.sac_models import CanvasStateActor, CanvasStateCritic


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


def test_canvas_entropy_state_is_opt_in_for_actor_and_critic():
    actor = CanvasStateActor(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
        use_entropy_state=True,
    )
    critic = CanvasStateCritic(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
        use_entropy_state=True,
    )
    batch = _batch()
    batch["entropy"] = torch.rand(2, 1, 5, 5)

    mean, log_std = actor(batch)
    q = critic(batch, torch.zeros(2, 3))

    assert actor.encoder.output_dim == 24
    assert critic.q[0].normalized_shape == (27,)
    assert mean.shape == (2, 3)
    assert log_std.shape == (2, 3)
    assert q.shape == (2,)


def test_canvas_pooling_branches_are_optional_but_not_both_disabled():
    base = CanvasStateActor(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
    )
    avg_only = CanvasStateActor(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
        use_canvas_max_pool=False,
    )
    max_only = CanvasStateActor(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
        use_canvas_avg_pool=False,
    )

    assert base.encoder.canvas_proj[1].normalized_shape == (256,)
    assert avg_only.encoder.canvas_proj[1].normalized_shape == (128,)
    assert max_only.encoder.canvas_proj[1].normalized_shape == (128,)
    assert avg_only(_batch())[0].shape == (2, 3)
    assert max_only(_batch())[0].shape == (2, 3)

    try:
        CanvasStateActor(
            canvas_feature_dim=4,
            d_model=8,
            rff_dim=4,
            rff_seed=0,
            use_canvas_avg_pool=False,
            use_canvas_max_pool=False,
        )
    except ValueError as exc:
        assert "At least one canvas pooling branch" in str(exc)
    else:
        raise AssertionError("Disabling both canvas pooling branches should fail.")


def test_canvas_replay_buffer_samples_optional_entropy_state():
    replay = CanvasReplayBuffer(
        capacity=4,
        max_history=3,
        canvas_feature_dim=4,
        canvas_grid_size=5,
        storage_device=torch.device("cpu"),
        store_entropy=True,
    )
    batch = _batch()

    replay.add_batch(
        canvas=batch["canvas"],
        coords=batch["coords"],
        lengths=batch["lengths"],
        actions=torch.zeros(2, 3),
        rewards=torch.ones(2),
        next_canvas=batch["canvas"] + 1.0,
        next_coords=batch["coords"],
        next_lengths=batch["lengths"],
        dones=torch.zeros(2),
        entropy=torch.rand(2, 1, 5, 5),
        next_entropy=torch.rand(2, 1, 5, 5),
    )
    sample = replay.sample(2, torch.device("cpu"))

    assert sample["entropy"].shape == (2, 1, 5, 5)
    assert sample["next_entropy"].shape == (2, 1, 5, 5)
