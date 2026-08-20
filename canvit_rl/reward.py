"""Compatibility wrapper for ADE20K reward helpers.

New code should import from `canvit_rl.ade20k.rewards`.
"""

from canvit_rl.ade20k.rewards import (
    delta_reward,
    reconstruction_reward,
    relative_ce_reduction,
)

__all__ = ["delta_reward", "reconstruction_reward", "relative_ce_reduction"]
