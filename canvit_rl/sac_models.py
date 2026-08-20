"""Compatibility wrapper for SAC model classes.

New code should import image-independent sequence models from
`canvit_rl.policies.sac_sequence` and image-dependent Canvas models from
`canvit_rl.canvas.models`.
"""

from canvit_rl.canvas.models import CanvasStateActor, CanvasStateCritic, CanvasStateEncoder
from canvit_rl.policies.sac_sequence import (
    CanViTSequenceEncoder,
    ContinuousCritic,
    GaussianActor,
)

__all__ = [
    "CanViTSequenceEncoder",
    "CanvasStateActor",
    "CanvasStateCritic",
    "CanvasStateEncoder",
    "ContinuousCritic",
    "GaussianActor",
]
