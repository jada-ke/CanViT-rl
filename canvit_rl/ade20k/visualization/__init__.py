"""Reusable ADE20K visualization entry points."""

__all__ = [
    "visualize_canvas_policy_for_indices",
    "visualize_reward_maps_for_indices",
]


def __getattr__(name: str):
    """Load heavy visualization modules only when a specific helper is requested."""
    if name == "visualize_canvas_policy_for_indices":
        from canvit_rl.ade20k.visualization.policy_glimpses import (
            visualize_canvas_policy_for_indices,
        )

        return visualize_canvas_policy_for_indices
    if name == "visualize_reward_maps_for_indices":
        from canvit_rl.ade20k.visualization.reward_maps import (
            visualize_reward_maps_for_indices,
        )

        return visualize_reward_maps_for_indices
    raise AttributeError(name)
