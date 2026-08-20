"""Compatibility wrapper for CanViT precision helpers.

New code should import from `canvit_rl.vision.precision`.
"""

from canvit_rl.vision.precision import (
    configure_frozen_canvit_precision,
    resolve_canvit_dtype,
)

__all__ = ["configure_frozen_canvit_precision", "resolve_canvit_dtype"]
