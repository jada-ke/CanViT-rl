from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from canvit_rl.in21k.shard_staging import (
    ShardStager,
    StagedShardedFeatureLoader,
    build_link_dir,
    resolve_stage_dir,
)


def _make_sources(root: Path, n: int, size: int = 64) -> list[Path]:
    root.mkdir()
    files = []
    for i in range(n):
        f = root / f"{i:05d}.pt"
        f.write_bytes(bytes([i]) * size)
        files.append(f)
    return files


def _is_local(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _wait(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def _stager(tmp_path: Path, n: int = 6, **kwargs) -> tuple[ShardStager, list[Path], list[Path]]:
    sources = _make_sources(tmp_path / "src", n)
    link_dir = tmp_path / "stage"
    by_name = build_link_dir(sources, link_dir)
    links = [link_dir / f.name for f in sources]
    kwargs.setdefault("min_free_bytes", 0)
    return ShardStager(links, by_name, **kwargs), sources, links


def test_link_dir_points_at_sources(tmp_path) -> None:
    sources = _make_sources(tmp_path / "src", 3)
    build_link_dir(sources, tmp_path / "stage")
    for src in sources:
        link = tmp_path / "stage" / src.name
        assert link.is_symlink()
        assert Path(os.readlink(link)) == src.resolve()
        assert link.read_bytes() == src.read_bytes()


def test_stager_keeps_window_local_and_evicts_behind(tmp_path) -> None:
    # Problem: staging must stay ahead of training and keep disk use bounded.
    # Expect shards [pos, pos+ahead] local, pos-1 kept, older ones back to
    # symlinks, with file contents always identical to the originals.
    stager, sources, links = _stager(tmp_path, ahead=2, behind=1)
    stager.stage_blocking(0)
    assert _is_local(links[0])
    stager.start()
    try:
        _wait(lambda: all(_is_local(links[i]) for i in (0, 1, 2)))
        assert all(links[i].is_symlink() for i in (3, 4, 5))

        stager.set_position(3)
        _wait(lambda: all(_is_local(links[i]) for i in (3, 4, 5)))
        _wait(lambda: links[0].is_symlink() and links[1].is_symlink())
        assert _is_local(links[2])  # behind=1 keeps the previous shard
        for src, link in zip(sources, links):
            assert link.read_bytes() == src.read_bytes()
    finally:
        stager.close()


def test_note_opened_counts_each_shard_once(tmp_path) -> None:
    # Problem: LOCAL/REMOTE must reflect the shard workers open, counted once
    # per shard even though the caller reports it on every batch.
    stager, _, _ = _stager(tmp_path, ahead=1)
    stager.stage_blocking(0)
    stager.note_opened(0)
    stager.note_opened(0)
    stager.note_opened(1)  # never staged (no thread running) -> remote
    stager.note_opened(1)
    assert (stager.entered_local, stager.entered_remote) == (1, 1)
    assert stager.metrics()["staging/local_shard_frac"] == 0.5


def test_staging_thread_survives_loop_error(tmp_path, monkeypatch) -> None:
    stager, _, links = _stager(tmp_path, ahead=2)
    stager.stage_blocking(0)
    calls = {"n": 0}
    real_evict = stager._evict

    def flaky_evict(keep):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        real_evict(keep)

    monkeypatch.setattr(stager, "_evict", flaky_evict)
    stager.start()
    try:
        stager.set_position(1)
        _wait(lambda: all(_is_local(links[i]) for i in (1, 2, 3)))
        assert calls["n"] >= 2
    finally:
        stager.close()


def test_stager_wraps_around_dataset_end(tmp_path) -> None:
    stager, _, links = _stager(tmp_path, n=4, ahead=2, behind=0)
    stager.stage_blocking(3)
    stager.start()
    try:
        # Counter 3 with 4 shards needs shards 3, 0, 1 local.
        _wait(lambda: all(_is_local(links[i]) for i in (3, 0, 1)))
        assert links[2].is_symlink()
    finally:
        stager.close()


def test_failed_copy_falls_back_to_source(tmp_path) -> None:
    def broken_copy(src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("disk full")

    stager, sources, links = _stager(tmp_path, ahead=1, copy_fn=broken_copy)
    stager.stage_blocking(0)
    assert links[0].is_symlink()
    assert links[0].read_bytes() == sources[0].read_bytes()
    assert not (links[0].parent / f".{links[0].name}.partial").exists()


def test_low_disk_skips_staging(tmp_path) -> None:
    stager, _, links = _stager(tmp_path, ahead=1, min_free_bytes=1 << 62)
    stager.stage_blocking(0)
    assert links[0].is_symlink()


def test_resolve_stage_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    assert resolve_stage_dir(None) is None
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    assert resolve_stage_dir(None) == tmp_path / "canvit_shards"
    # Not bound into the container: parent missing -> disabled, not an error.
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "missing"))
    assert resolve_stage_dir(None) is None


class _FakeLoader:
    """Mimics ShardedFeatureLoader: reads shard files by path from shards_dir."""

    def __init__(self, *, shards_dir: Path, start_step: int, batches_per_shard: int):
        self.shard_files = sorted(Path(shards_dir).glob("*.pt"))
        self.batches_per_shard = batches_per_shard
        self.num_workers = 2
        self.step = start_step
        self.loader = None

    def next(self):
        self.loader = object()
        shard = self.shard_files[(self.step // self.batches_per_shard) % len(self.shard_files)]
        self.step += 1
        return shard.read_bytes()


def test_staged_loader_matches_unstaged_stream(tmp_path) -> None:
    # Problem: staging must not change what training sees. Expect the staged
    # wrapper to return the same byte stream as reading the originals, from
    # a mid-shard resume point.
    sources = _make_sources(tmp_path / "src", 5)
    plain = _FakeLoader(shards_dir=tmp_path / "src", start_step=7, batches_per_shard=3)
    staged = StagedShardedFeatureLoader(
        _FakeLoader,
        shards_dir=tmp_path / "src",
        stage_dir=tmp_path / "stage",
        start_step=7,
        ahead=2,
        min_free_bytes=0,
        batches_per_shard=3,
    )
    try:
        # Start shard (7 // 3 = 2) and the next one are staged before workers start.
        assert _is_local(tmp_path / "stage" / sources[2].name)
        assert _is_local(tmp_path / "stage" / sources[3].name)
        for _ in range(20):
            assert staged.next() == plain.next()
        assert staged.batches_per_shard == 3
        assert "staging/local_shard_frac" in staged.staging_metrics()
    finally:
        staged.close()


@pytest.mark.parametrize("ahead", [0, -1])
def test_stager_rejects_nonpositive_ahead(tmp_path, ahead) -> None:
    with pytest.raises(ValueError):
        _stager(tmp_path, ahead=ahead)
