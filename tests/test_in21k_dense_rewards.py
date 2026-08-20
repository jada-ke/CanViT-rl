import torch

from canvit_rl.in21k.dense.rewards import (
    DenseDistillationMetrics,
    dense_loss_eps_reduction_reward,
    dense_loss_reduction_reward,
    dense_loss_tanh_reduction_reward,
    dense_reward,
)


def _metrics(*, norm: list[float], raw: list[float] | None = None) -> DenseDistillationMetrics:
    values = torch.tensor(norm, dtype=torch.float32)
    raw_values = torch.tensor(raw if raw is not None else norm, dtype=torch.float32)
    return DenseDistillationMetrics(
        scene_loss_norm=values,
        cls_loss_norm=values,
        loss_norm=values,
        scene_loss_raw=raw_values,
        cls_loss_raw=raw_values,
        loss_raw=raw_values,
    )


def test_dense_loss_tanh_reduction_bounds_step_relative_reward() -> None:
    before = _metrics(norm=[2.0, 0.5])
    after = _metrics(norm=[1.0, 0.75])

    # Problem: norm_loss_reduction is useful for gamma=0 myopic training but can
    # expose SAC to large proportional rewards. Solution: tanh-bound the same
    # step-relative reduction with the existing reward scale. Result: the mode
    # preserves the current-state denominator while keeping rewards bounded.
    expected = torch.tanh(torch.tensor([1.0, 1.0]) * torch.tensor([0.5, -0.5]))

    actual = dense_loss_tanh_reduction_reward(before, after, scale=1.0)

    torch.testing.assert_close(actual, expected)


def test_dense_reward_dispatches_norm_loss_tanh_reduction() -> None:
    before = _metrics(norm=[4.0])
    after = _metrics(norm=[3.0])

    actual = dense_reward(
        mode="norm_loss_tanh_reduction",
        before=before,
        after=after,
        tanh_scale=2.0,
    )

    torch.testing.assert_close(actual, torch.tanh(torch.tensor([0.5])))


def test_dense_loss_eps_reduction_adds_tunable_denominator_epsilon() -> None:
    before = _metrics(norm=[2.0, 0.5])
    after = _metrics(norm=[1.0, 0.75])

    expected = torch.tensor([1.0, -0.5]) / torch.tensor([2.5, 1.0])

    actual = dense_loss_eps_reduction_reward(
        before,
        after,
        reduction_eps=0.5,
    )

    torch.testing.assert_close(actual, expected)


def test_dense_loss_eps_reduction_matches_reduction_when_regularizer_is_zero() -> None:
    before = _metrics(norm=[2.0, 0.5])
    after = _metrics(norm=[1.0, 0.75])

    actual = dense_loss_eps_reduction_reward(
        before,
        after,
        reduction_eps=0.0,
    )

    torch.testing.assert_close(actual, dense_loss_reduction_reward(before, after))


def test_dense_reward_dispatches_norm_loss_eps_reduction() -> None:
    before = _metrics(norm=[4.0])
    after = _metrics(norm=[3.0])

    actual = dense_reward(
        mode="norm_loss_eps_reduction",
        before=before,
        after=after,
        reduction_eps=1.0,
    )

    torch.testing.assert_close(actual, torch.tensor([0.2]))
