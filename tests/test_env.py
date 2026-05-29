"""
tests/test_env.py
 
Unit tests for CanViTEnv.
 
Most tests use synthetic random images so no dataset or HuggingFace
download is required. Tests that need the real checkpoint are marked
with @pytest.mark.network and skipped by default (see pyproject.toml).
"""
 
from __future__ import annotations
 
import numpy as np
import pytest
import torch
 
from canvit_rl.env import CanViTEnv, CanViTEnvConfig, get_device
 
 
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
 
@pytest.fixture(scope="module")
def env():
    """Shared env instance — model is loaded once per test session."""
    cfg = CanViTEnvConfig(max_steps=4)
    return CanViTEnv(config=cfg)
 
 
@pytest.fixture
def reset_env(env):
    """Returns an env that has been reset with a random image."""
    obs, info = env.reset()
    return env, obs, info
 
 
# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------
 
def test_action_space_shape(env):
    assert env.action_space.shape == (3,)
 
 
def test_observation_space_shape(env):
    assert env.observation_space.shape == (768,)
 
 
def test_action_space_bounds(env):
    assert (env.action_space.low == -1.0).all()
    assert (env.action_space.high == 1.0).all()
 
 
# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
 
def test_reset_returns_correct_obs_shape(reset_env):
    _, obs, _ = reset_env
    assert obs.shape == (768,)
    assert obs.dtype == np.float32
 
 
def test_reset_returns_empty_info(reset_env):
    _, _, info = reset_env
    assert isinstance(info, dict)
 
 
def test_reset_with_explicit_image(env):
    image = torch.rand(1, 3, 512, 512)
    obs, _ = env.reset(options={"image": image})
    assert obs.shape == (768,)
 
 
def test_reset_with_unbatched_image(env):
    """reset() should accept [3, H, W] and add the batch dim automatically."""
    image = torch.rand(3, 512, 512)
    obs, _ = env.reset(options={"image": image})
    assert obs.shape == (768,)
 
 
def test_reset_clears_step_count(env):
    env.reset()
    for _ in range(3):
        env.step(env.action_space.sample())
    env.reset()
    assert env._step_count == 0
 
 
# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------
 
def test_step_returns_correct_shapes(reset_env):
    env, _, _ = reset_env
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (768,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cosine_sim" in info
 
 
def test_step_not_terminated_before_max_steps(reset_env):
    env, _, _ = reset_env
    cfg_steps = env.cfg.max_steps
    for i in range(cfg_steps - 1):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert not terminated, f"Terminated early at step {i + 1}"
 
 
def test_step_terminates_at_max_steps(reset_env):
    env, _, _ = reset_env
    terminated = False
    for _ in range(env.cfg.max_steps):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert terminated
 
 
def test_step_without_reset_raises(env):
    """Calling step() before reset() should raise AssertionError."""
    fresh_env = CanViTEnv(config=CanViTEnvConfig(max_steps=2))
    fresh_env._image = None  # simulate pre-reset state
    with pytest.raises(AssertionError):
        fresh_env.step(fresh_env.action_space.sample())
 
 
def test_cosine_sim_in_info_is_float(reset_env):
    env, _, _ = reset_env
    _, _, _, _, info = env.step(env.action_space.sample())
    assert isinstance(info["cosine_sim"], float)
    assert 0.0 <= info["cosine_sim"]
 
 
# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------
 
def test_action_to_viewpoint_scale_minimum(env):
    """Scale should never go below the minimum threshold."""
    env.reset()
    action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)  # worst case
    vp = env._action_to_viewpoint(action)
    assert float(vp.scales[0]) >= 0.05
 
 
def test_action_to_viewpoint_scale_maximum(env):
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    vp = env._action_to_viewpoint(action)
    assert float(vp.scales[0]) <= 1.0
 
 
def test_action_to_viewpoint_centers_passthrough(env):
    action = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    vp = env._action_to_viewpoint(action)
    assert pytest.approx(float(vp.centers[0, 0]), abs=1e-5) == 0.5
    assert pytest.approx(float(vp.centers[0, 1]), abs=1e-5) == -0.5
 
 
# ---------------------------------------------------------------------------
# Backbone is frozen
# ---------------------------------------------------------------------------
 
def test_backbone_is_frozen(env):
    for p in env._model.parameters():
        assert not p.requires_grad, "Backbone parameter has requires_grad=True"
 
 
# ---------------------------------------------------------------------------
# Gymnasium compliance
# ---------------------------------------------------------------------------
 
def test_obs_in_observation_space(reset_env):
    env, obs, _ = reset_env
    assert env.observation_space.contains(obs), \
        "Initial obs is outside observation_space"
 
 
def test_step_obs_in_observation_space(reset_env):
    env, _, _ = reset_env
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs), \
        "Step obs is outside observation_space"
 
 
def test_sampled_action_in_action_space(env):
    for _ in range(20):
        action = env.action_space.sample()
        assert env.action_space.contains(action)