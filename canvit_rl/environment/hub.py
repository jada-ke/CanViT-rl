"""Pin CanViT Hugging Face checkpoints to a revision this repo can load.

Problem: the ``canvit/*`` Hub repos moved ``main`` to the canvit-pytorch 0.2
checkpoint format (renamed config keys and weights), which the installed
canvit-pytorch 0.1.x cannot parse. Upstream tags the last 0.1-format commit of
each repo as ``canvit-pytorch-0.1``.

Solution: resolve ``canvit/*`` repo ids to a local snapshot of that tag before
handing them to ``from_pretrained`` / ``from_pretrained_with_probe`` (neither
of which takes a ``revision`` argument here). Local paths and non-``canvit/``
repos pass through unchanged.

Result: every load site gets 0.1-format checkpoints regardless of what the
Hub's ``main`` points at. Override with ``CANVIT_HUB_REVISION`` (set it to an
empty string to follow ``main`` again, e.g. after migrating to 0.2).
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_CANVIT_HUB_REVISION = "canvit-pytorch-0.1"


def canvit_hub_revision() -> str | None:
    """Return the pinned revision for ``canvit/*`` repos, or None for ``main``."""
    revision = os.environ.get("CANVIT_HUB_REVISION", DEFAULT_CANVIT_HUB_REVISION)
    return revision.strip() or None


def pinned_repo(repo: str) -> str:
    """Map a ``canvit/*`` repo id to a local snapshot at the pinned revision."""
    revision = canvit_hub_revision()
    if revision is None or Path(repo).expanduser().exists():
        return repo
    if not repo.startswith("canvit/"):
        return repo
    return snapshot_download(repo_id=repo, revision=revision)
