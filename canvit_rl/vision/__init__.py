"""Image/model utilities that depend on CanViT vision outputs."""

from canvit_rl.vision.precision import (
    configure_frozen_canvit_precision,
    resolve_canvit_dtype,
)

__all__ = ["configure_frozen_canvit_precision", "resolve_canvit_dtype"]
