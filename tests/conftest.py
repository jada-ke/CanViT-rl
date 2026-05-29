# tests/conftest.py
import pytest
from canvit_rl.env import CanViTEnv, CanViTEnvConfig

@pytest.fixture(scope="module")
def env():
    cfg = CanViTEnvConfig(max_steps=4)
    return CanViTEnv(config=cfg)

@pytest.fixture
def reset_env(env):
    obs, info = env.reset()
    return env, obs, info