"""Compatibility wrapper for segmentation-scored greedy baselines.

New code should import from `canvit_rl.ade20k.greedy`.
"""

from canvit_rl.ade20k.greedy import (
    _index_state_batch,
    _repeat_state_chunks,
    _segmentation_cross_entropy_losses,
    full_scene_step,
    full_scene_step_batch,
    greedy_step,
    greedy_step_batch,
    miou_from_state,
    run_greedy_batch,
    run_greedy_episode,
)

__all__ = [
    "_index_state_batch",
    "_repeat_state_chunks",
    "_segmentation_cross_entropy_losses",
    "full_scene_step",
    "full_scene_step_batch",
    "greedy_step",
    "greedy_step_batch",
    "miou_from_state",
    "run_greedy_batch",
    "run_greedy_episode",
]
