"""CanViT reinforcement-learning package.

The stable public surface is intentionally small: environment integration,
baseline policy/reward helpers, and subpackages for larger experiment families.
See `canvit_rl/README.md` for the newcomer-oriented package map.
"""

from canvit_rl.environment import CanViTEnv, CanViTEnvConfig, get_device
from canvit_rl.policies import MLPPolicy
from canvit_rl.ade20k.rewards import (
    delta_reward,
    reconstruction_reward,
    relative_ce_reduction,
)

__all__ = [
    "CanViTEnv",
    "CanViTEnvConfig",
    "MLPPolicy",
    "delta_reward",
    "get_device",
    "reconstruction_reward",
    "relative_ce_reduction",
]
