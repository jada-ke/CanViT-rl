"""Narrow imports for CanViT-pretrain modules used by IN21k RL pretraining."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import NamedTuple

import canvit_pretrain


class PretrainModules(NamedTuple):
    """CanViT-pretrain symbols needed without importing train package __init__."""

    Config: type
    ShardedFeatureLoader: type


def install_pretrain_train_shim() -> None:
    """Install a lightweight package shim for ``canvit_pretrain.train``.

    Problem: importing ``canvit_pretrain.train`` executes its package
    ``__init__``, which pulls optional plotting/probe modules into even
    headless RL jobs. Solution: register a package object with the same
    ``__path__`` so Python can import the specific submodules we need. Result:
    dense SAC can use Config and data helpers without importing the
    plotting-heavy package initializer.
    """
    existing = sys.modules.get("canvit_pretrain.train")
    if existing is not None and hasattr(existing, "__path__"):
        return
    train_dir = Path(canvit_pretrain.__file__).parent / "train"
    module = types.ModuleType("canvit_pretrain.train")
    module.__path__ = [str(train_dir)]  # type: ignore[attr-defined]
    sys.modules["canvit_pretrain.train"] = module


def load_pretrain_modules() -> PretrainModules:
    """Load the CanViT-pretrain submodules required by the dense SAC script."""
    install_pretrain_train_shim()
    from canvit_pretrain.train.config import Config
    from canvit_pretrain.train.data.shards import ShardedFeatureLoader

    return PretrainModules(
        Config=Config,
        ShardedFeatureLoader=ShardedFeatureLoader,
    )
