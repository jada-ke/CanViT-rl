"""Compatibility wrapper for the CanViT Gymnasium environment.

New code should import from `canvit_rl.environment.canvit_env`.
"""

from canvit_rl.environment.canvit_env import CanViTEnv, CanViTEnvConfig, get_device

__all__ = ["CanViTEnv", "CanViTEnvConfig", "get_device"]
