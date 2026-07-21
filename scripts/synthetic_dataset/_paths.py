"""Path helpers for scripts that live below ``scripts/synthetic_dataset``."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Problem: moved scripts run with scripts/synthetic_dataset on sys.path,
    # so imports of canvit_rl can fail outside repo-root cwd. Solution: add the
    # repository root before local package imports. Result: the scripts work
    # both from the repo root and from their nested subfolder.
    sys.path.insert(0, str(REPO_ROOT))


def repo_path(path: Path) -> Path:
    """Resolve relative CLI paths against the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path
