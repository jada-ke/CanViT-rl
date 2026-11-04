"""Stage upcoming IN21k feature shards to node-local disk during training.

Problem: CanViT-pretrain's shard loader mmaps 6.1 GiB feature shards on shared
storage and each DataLoader worker copies strided samples out of them. On nibi
that scattered access reads at ~50-130 MB/s per job (whole-file sequential
reads reach ~515 MB/s), so training waits on data ~2/3 of the time.

Solution: the loader reads shards through a directory of per-shard symlinks
that initially point at the shared-storage originals. A background thread
copies the next few shards sequentially to local disk and atomically swaps
each symlink for the local copy; once training has moved past a shard, the
local copy is swapped back for a symlink so disk use stays bounded (~ahead+2
shards). Workers open shards by path, so they pick up whichever file is there
when they reach that shard; a shard that is not staged in time is read from
shared storage exactly as before. The local copy is byte-identical, so it does
not matter if torch.load's zip reader and its mmap open different inodes of
the same path across a swap.

The upstream loader, sample order and exact resume are unchanged.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Keep this much local disk free beyond the shard being copied, so staging
# never fills $SLURM_TMPDIR for other jobs on the node.
DEFAULT_MIN_FREE_BYTES = 20 * 1024**3


def copy_shard(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` sequentially, then drop ``src`` from page cache.

    Page cache counts against the Slurm job's memory limit; the source pages
    are never read again (workers read the local copy), so release them.
    """
    shutil.copyfile(src, dst)
    if hasattr(os, "posix_fadvise"):
        fd = os.open(src, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def resolve_stage_dir(stage_dir: Path | None) -> Path | None:
    """Return a usable staging directory, defaulting to ``$SLURM_TMPDIR``.

    Returns None (staging disabled) when no directory is given and
    ``$SLURM_TMPDIR`` is unset, or when the directory is not visible/writable,
    e.g. because ``$SLURM_TMPDIR`` was not bound into the Apptainer container.
    """
    if stage_dir is None:
        tmpdir = os.environ.get("SLURM_TMPDIR")
        if not tmpdir:
            return None
        stage_dir = Path(tmpdir) / "canvit_shards"
    parent = stage_dir.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        log.warning(
            "Shard staging disabled: %s is not a writable directory here. "
            "Inside Apptainer, add --bind \"$SLURM_TMPDIR\" to the exec command.",
            parent,
        )
        return None
    return stage_dir


def build_link_dir(source_files: list[Path], link_dir: Path) -> dict[str, Path]:
    """Create ``link_dir`` with one symlink per source shard, same file names.

    Returns {file name: resolved source path}. An entry that already exists as
    a regular file of the source's size is kept (already staged); anything
    else is reset to a symlink pointing at the source.
    """
    link_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    # Resolve each parent directory once instead of every file: resolve()
    # stats every path component, which is slow for ~3k files on /project.
    resolved_dirs: dict[Path, Path] = {}
    for src in source_files:
        parent = resolved_dirs.setdefault(src.parent, src.parent.resolve())
        src = parent / src.name
        sources[src.name] = src
        link = link_dir / src.name
        if link.is_symlink():
            if Path(os.readlink(link)) == src:
                continue
        elif link.is_file() and link.stat().st_size == src.stat().st_size:
            continue
        _point_at(link, src)
    return sources


def _point_at(link: Path, src: Path) -> None:
    """Atomically make ``link`` a symlink to ``src`` (replacing any file)."""
    tmp = link.with_name(f".{link.name}.link")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(src, tmp)
    os.replace(tmp, link)


class ShardStager:
    """Background copier keeping shards [current, current+ahead] on local disk.

    ``link_files`` is the loader's shard order (paths inside the link dir);
    ``sources`` maps each file name to its shared-storage original. Call
    ``stage_blocking`` before DataLoader workers fork, ``start`` after, and
    ``set_position`` with the shard index training is currently consuming.
    """

    def __init__(
        self,
        link_files: list[Path],
        sources: dict[str, Path],
        *,
        ahead: int,
        behind: int = 1,
        copy_threads: int = 2,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
        copy_fn: Callable[[Path, Path], Any] = copy_shard,
    ) -> None:
        if ahead < 1:
            raise ValueError("ahead must be >= 1")
        self.link_files = list(link_files)
        self.sources = sources
        self.ahead = ahead
        self.behind = behind
        self.copy_threads = copy_threads
        self.min_free_bytes = min_free_bytes
        self.copy_fn = copy_fn
        self.n_shards = len(self.link_files)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._position = 0
        self._opened = -1
        self._staged: set[int] = {
            i for i, link in enumerate(self.link_files)
            if link.is_file() and not link.is_symlink()
        }
        self._in_flight: dict[int, Future] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._thread: threading.Thread | None = None
        self._warned_disk = False
        # Counters for logging: shards workers opened while local vs. remote.
        self.entered_local = 0
        self.entered_remote = 0
        self.bytes_copied = 0
        self.copy_seconds = 0.0

    # ---- public API -------------------------------------------------------

    def stage_blocking(self, shard_counter: int, count: int = 1) -> None:
        """Copy ``count`` shards from ``shard_counter`` synchronously (no threads).

        Used before DataLoader workers fork, so no thread is running at fork.
        """
        self._position = shard_counter
        for i in range(count):
            self._copy((shard_counter + i) % self.n_shards)

    def start(self) -> None:
        """Start the background staging thread (call after workers fork)."""
        if self._thread is not None:
            return
        self._pool = ThreadPoolExecutor(self.copy_threads, thread_name_prefix="shard-copy")
        self._thread = threading.Thread(target=self._run, name="shard-stager", daemon=True)
        self._thread.start()

    def set_position(self, shard_counter: int) -> None:
        """Report the (unbounded) shard counter training is consuming."""
        with self._lock:
            if shard_counter == self._position:
                return
            self._position = shard_counter
        self._wake.set()

    def note_opened(self, shard_counter: int) -> None:
        """Record whether the shard workers are opening now is local.

        Workers prefetch ahead of the main loop, so they open shard N before
        training consumes its first batch; the caller passes that earlier
        counter so LOCAL/REMOTE reflects what workers actually read.
        """
        with self._lock:
            if shard_counter <= self._opened:
                return
            self._opened = shard_counter
            idx = shard_counter % self.n_shards
            local = idx in self._staged
        if local:
            self.entered_local += 1
        else:
            self.entered_remote += 1
        log.info(
            "Shard %s opened by workers: %s (local %d / remote %d so far)",
            self.link_files[idx].name,
            "LOCAL" if local else "REMOTE (staging behind)",
            self.entered_local,
            self.entered_remote,
        )

    def is_staged(self, shard_counter: int) -> bool:
        with self._lock:
            return shard_counter % self.n_shards in self._staged

    def metrics(self) -> dict[str, float]:
        entered = self.entered_local + self.entered_remote
        return {
            "staging/local_shard_frac": self.entered_local / entered if entered else 0.0,
            "staging/shards_staged": float(len(self._staged)),
            "staging/copy_mb_per_sec": (
                self.bytes_copied / 1e6 / self.copy_seconds if self.copy_seconds > 0 else 0.0
            ),
        }

    def close(self, wait: bool = True) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5 if wait else 0)
        if self._pool is not None:
            self._pool.shutdown(wait=wait, cancel_futures=True)

    # ---- internals --------------------------------------------------------

    def _windows(self) -> tuple[list[int], set[int]]:
        """Return (shards to stage, nearest first) and (shards to keep)."""
        with self._lock:
            pos = self._position
        n = self.n_shards
        wanted = []
        for i in range(self.ahead + 1):
            idx = (pos + i) % n
            if idx not in wanted:
                wanted.append(idx)
        keep = set(wanted) | {(pos - j) % n for j in range(1, self.behind + 1)}
        return wanted, keep

    def _run(self) -> None:
        while not self._stop.is_set():
            # An unexpected error must not silently end staging for the rest
            # of the job; log it and retry on the next tick.
            try:
                wanted, keep = self._windows()
                self._evict(keep)
                self._submit(wanted)
            except Exception:
                log.exception("Shard staging loop error; retrying")
            self._wake.wait(timeout=1.0)
            self._wake.clear()

    def _evict(self, keep: set[int]) -> None:
        with self._lock:
            stale = [i for i in self._staged if i not in keep and i not in self._in_flight]
        for idx in stale:
            link = self.link_files[idx]
            # Swapping back to a symlink unlinks the local copy; workers that
            # still have it mmapped keep their mapping until they move on.
            _point_at(link, self.sources[link.name])
            with self._lock:
                self._staged.discard(idx)
            log.debug("Evicted local copy of %s", link.name)

    def _submit(self, wanted: list[int]) -> None:
        assert self._pool is not None
        with self._lock:
            for idx, fut in list(self._in_flight.items()):
                if fut.done():
                    del self._in_flight[idx]
            todo = [i for i in wanted if i not in self._staged and i not in self._in_flight]
            slots = self.copy_threads - len(self._in_flight)
        for idx in todo[: max(slots, 0)]:
            fut = self._pool.submit(self._copy, idx)
            with self._lock:
                self._in_flight[idx] = fut
            fut.add_done_callback(lambda _f: self._wake.set())

    def _has_space(self, src: Path, link_dir: Path) -> bool:
        need = src.stat().st_size + self.min_free_bytes
        free = shutil.disk_usage(link_dir).free
        if free >= need:
            return True
        if not self._warned_disk:
            log.warning(
                "Shard staging paused: %.1f GiB free in %s, need %.1f GiB. "
                "Request more with #SBATCH --tmp.",
                free / 1024**3, link_dir, need / 1024**3,
            )
            self._warned_disk = True
        return False

    def _copy(self, idx: int) -> None:
        link = self.link_files[idx]
        src = self.sources[link.name]
        with self._lock:
            if idx in self._staged:
                return
        if not self._has_space(src, link.parent):
            return
        tmp = link.with_name(f".{link.name}.partial")
        t0 = time.perf_counter()
        try:
            self.copy_fn(src, tmp)
            os.replace(tmp, link)
        except Exception:
            log.exception("Staging %s failed; it will be read from shared storage", link.name)
            if tmp.exists():
                tmp.unlink()
            return
        dt = time.perf_counter() - t0
        size = src.stat().st_size
        with self._lock:
            self._staged.add(idx)
            self.bytes_copied += size
            self.copy_seconds += dt
        log.info(
            "Staged %s: %.2f GiB in %.1fs (%.0f MB/s)",
            link.name, size / 1024**3, dt, size / 1e6 / max(dt, 1e-9),
        )


class StagedShardedFeatureLoader:
    """``ShardedFeatureLoader`` that reads shards via a local staging directory.

    Same ``next()`` contract as the wrapped loader. ``loader_cls`` is the
    upstream class; all other keyword arguments are passed through to it.
    """

    def __init__(
        self,
        loader_cls: type,
        *,
        shards_dir: Path,
        stage_dir: Path,
        start_step: int,
        ahead: int = 3,
        copy_threads: int = 2,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
        **loader_kwargs: Any,
    ) -> None:
        source_files = sorted(Path(shards_dir).glob("*.pt"))
        if not source_files:
            raise FileNotFoundError(f"No shards found in {shards_dir}")
        sources = build_link_dir(source_files, stage_dir)
        self.inner = loader_cls(shards_dir=stage_dir, start_step=start_step, **loader_kwargs)
        # Use the loader's own shard order: with tar_dir it may filter shards.
        self.stager = ShardStager(
            list(self.inner.shard_files),
            sources,
            ahead=ahead,
            copy_threads=copy_threads,
            min_free_bytes=min_free_bytes,
        )
        self.batches_per_shard = int(self.inner.batches_per_shard)
        self.consumed = start_step
        # DataLoader workers run up to prefetch_factor (default 2) batches
        # each ahead of the main loop.
        self.worker_lead_batches = 2 * int(getattr(self.inner, "num_workers", 0) or 0)
        start_shard = start_step // self.batches_per_shard
        log.info(
            "Shard staging: %d shards linked in %s; staging shards %d-%d before workers start",
            len(source_files), stage_dir, start_shard, start_shard + 1,
        )
        # Copy the first two shards in the main thread: no thread may run when
        # DataLoader forks its workers, and copying both up front avoids the
        # background thread racing the workers for shard start+1 at startup
        # (they open it after ~batches_per_shard - worker_lead batches).
        self.stager.stage_blocking(start_shard, count=2)
        self.stager.note_opened(start_shard)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found normally; guard "inner" so a
        # partially constructed instance does not recurse forever.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def next(self):
        started = self.inner.loader is not None
        batch = self.inner.next()
        if not started:
            self.stager.start()
        self.consumed += 1
        self.stager.set_position(self.consumed // self.batches_per_shard)
        self.stager.note_opened(
            (self.consumed + self.worker_lead_batches) // self.batches_per_shard
        )
        return batch

    def staging_metrics(self) -> dict[str, float]:
        return self.stager.metrics()

    def close(self) -> None:
        self.stager.close()
