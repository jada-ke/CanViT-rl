"""Resume continuation for IN21k dense SAC: args, RNG, and loader position."""

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import torch

from canvit_rl.in21k.checkpoints import (
    capture_rng_state,
    inherit_resume_defaults,
    restore_rng_state,
)
from canvit_rl.in21k.dense_train_batch import FixedDenseSubsetSampler, PairedDenseShardLoader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-base-dir", type=Path, required=True)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def test_resume_inherits_saved_args_and_cli_overrides() -> None:
    saved = {
        "feature_base_dir": Path("feats"),
        "t": 4,
        "seed": 44,
        "batches": 25_000,
        "checkpoint_dir": Path("old/dir"),
    }
    parser = _parser()
    inherited = inherit_resume_defaults(parser, saved)
    args = parser.parse_args(["--resume", "ckpt.pt", "--batches", "50000", "--seed", "7"])
    # Omitted flags (including a required one) come from the checkpoint.
    assert args.feature_base_dir == Path("feats")
    assert args.t == 4
    # Explicit CLI flags override; launch-specific args are never inherited.
    assert args.seed == 7
    assert args.batches == 50_000
    assert args.checkpoint_dir is None
    assert "batches" not in inherited and "checkpoint_dir" not in inherited


def _sampler(seed: int = 0, size: int = 7) -> FixedDenseSubsetSampler:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator)
    return FixedDenseSubsetSampler(subset_size=size, initial_order=order, generator=generator)


def test_subset_sampler_skip_matches_uninterrupted_stream() -> None:
    for skipped in (0, 3, 7, 10, 21, 30):
        reference = list(itertools.islice(iter(_sampler()), skipped + 20))
        resumed = _sampler()
        resumed.skip(skipped)
        assert list(itertools.islice(iter(resumed), 20)) == reference[skipped:]


def _paired_loader(num_rows: int = 5, batch_size: int = 2) -> PairedDenseShardLoader:
    # Bypass shard loading: only the index bookkeeping matters for skipping.
    loader = object.__new__(PairedDenseShardLoader)
    loader.batch_size = batch_size
    loader.generator = torch.Generator().manual_seed(0)
    loader.order = torch.randperm(num_rows, generator=loader.generator)
    loader.initial_order = loader.order.clone()
    loader.pos = 0
    return loader


def test_paired_loader_skip_matches_uninterrupted_stream() -> None:
    reference = _paired_loader()
    expected = [reference._next_indices().tolist() for _ in range(12)]
    resumed = _paired_loader()
    resumed.skip_batches(7)
    assert [resumed._next_indices().tolist() for _ in range(5)] == expected[7:]


def test_rng_state_roundtrip_continues_stream() -> None:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
