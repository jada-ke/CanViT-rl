"""Path helpers for saliency scripts launched from any working directory."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def repo_path(path: str | Path) -> Path:
    """Resolve relative CLI paths against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
