"""ADE20K label helpers."""

from __future__ import annotations

import numpy as np


ADE_NUM_CLASSES = 150
ADE_IGNORE_LABEL = 255


def remap_ade_mask_labels(
    mask: np.ndarray,
    *,
    raw_ade: bool = False,
    num_classes: int = ADE_NUM_CLASSES,
    ignore_label: int = ADE_IGNORE_LABEL,
) -> np.ndarray:
    """Return ADE labels in the zero-based class range expected by CE.

    ADE annotation PNGs are commonly stored as 1..150 class ids with 0 as
    unlabeled/ignore, while the segmentation probe and torch CE expect targets
    in 0..149 with 255 ignored. Pass raw_ade=True when reading directly from
    ADE annotations. Without it, already-remapped masks are returned unchanged
    unless an out-of-range raw label is detected.
    """
    labels = np.asarray(mask).astype(np.int64, copy=True)
    valid = labels != ignore_label
    if raw_ade or (np.any(valid) and int(labels[valid].max()) >= num_classes):
        # Fixed by Codex on 2026-06-24
        # Problem: Raw ADE masks are 1-based even when a crop does not contain
        # class 150, so max-label detection alone can leave silent off-by-one
        # CE targets.
        # Solution: allow callers that read ADE annotations directly to force
        # conversion: 0 becomes ignore, and valid 1..150 ids become 0..149.
        # Result: CE compares against the intended ADE class ids.
        raw_ignore = labels == 0
        labels[raw_ignore] = ignore_label
        valid = labels != ignore_label
        labels[valid] -= 1
    return labels
