"""Compatibility wrapper for ADE label utilities.

New code should import from `canvit_rl.ade20k.labels`.
"""

from canvit_rl.ade20k.labels import remap_ade_mask_labels

__all__ = ["remap_ade_mask_labels"]
