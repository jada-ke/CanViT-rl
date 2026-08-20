"""Compatibility wrapper for image-independent viewpoint policy utilities.

New code should import from `canvit_rl.policies.viewpoint`.
"""

from canvit_rl.policies.viewpoint import (
    ViewpointGaussianActor,
    ViewpointHistoryCritic,
    action_to_viewpoint,
    randomize_actor_mean_viewpoint_prior,
    viewpoint_to_action,
)

__all__ = [
    "ViewpointGaussianActor",
    "ViewpointHistoryCritic",
    "action_to_viewpoint",
    "randomize_actor_mean_viewpoint_prior",
    "viewpoint_to_action",
]
