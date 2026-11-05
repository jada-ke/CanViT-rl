"""Measure how fast IN21k feature shards copy from shared storage to local disk.

Problem: shard staging (canvit_rl/in21k/shard_staging.py) needs ~260 MB/s per
job to keep sweep-17/37 training fed, but on nibi a single sequential copy
stream reached only ~80-100 MB/s (job 26201418). Whether parallel byte-range
reads go faster depends on the shared filesystem, so measure it before
spending a GPU run.

Solution: copy one *different* shard per configuration (so no run reads a
shard another run already pulled into page cache) with the same
``copy_shard`` function training uses, and report MB/s. The last config copies
two shards at once to show the aggregate rate with --stage-copy-threads 2.
Each local copy is removed right after it is timed.

Example (CPU-only job, inside the training container with the same binds plus
--bind "$SLURM_TMPDIR"):
    salloc --time=0:30:00 --cpus-per-task=4 --mem=8G --tmp=20G
    python scripts/analysis/in21k/bench_shard_copy.py \\
        --shards-dir /features/dinov3_vitb16/512/shards --first-shard 2600
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from canvit_rl.in21k.shard_staging import copy_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Local directory for copies. Default: $SLURM_TMPDIR/shard_copy_bench.",
    )
    parser.add_argument(
        "--first-shard",
        type=int,
        default=2600,
        help="Index of the first shard to use; pick one no training job read recently.",
    )
    parser.add_argument("--streams", type=str, default="1,4,8,16")
    parser.add_argument(
        "--concurrent-streams",
        type=int,
        default=8,
        help="Streams per shard for the final two-shards-at-once run (0 skips it).",
    )
    return parser.parse_args()


def timed_copy(src: Path, dest_dir: Path, streams: int) -> tuple[float, int]:
    dst = dest_dir / src.name
    size = src.stat().st_size
    t0 = time.perf_counter()
    copy_shard(src, dst, streams=streams)
    elapsed = time.perf_counter() - t0
    dst.unlink()
    return elapsed, size


def main() -> None:
    args = parse_args()
    dest = args.dest
    if dest is None:
        tmpdir = os.environ.get("SLURM_TMPDIR")
        if not tmpdir:
            raise SystemExit("Pass --dest or run inside a Slurm job with $SLURM_TMPDIR bound.")
        dest = Path(tmpdir) / "shard_copy_bench"
    dest.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.shards_dir.glob("*.pt"))
    streams_list = [int(s) for s in args.streams.split(",") if s]
    needed = len(streams_list) + (2 if args.concurrent_streams > 0 else 0)
    picked = shards[args.first_shard : args.first_shard + needed]
    if len(picked) < needed:
        raise SystemExit(f"Need {needed} shards from index {args.first_shard}; found {len(picked)}.")

    print(f"shards_dir={args.shards_dir} dest={dest}")
    print("Training needs ~260 MB/s per job (one 6 GiB shard every ~25 s).")
    for streams, src in zip(streams_list, picked):
        elapsed, size = timed_copy(src, dest, streams)
        print(
            f"streams={streams:<3d} {src.name}: {size / 1024**3:.2f} GiB in {elapsed:6.1f}s "
            f"= {size / 1e6 / elapsed:6.0f} MB/s",
            flush=True,
        )
    if args.concurrent_streams > 0:
        pair = picked[len(streams_list) :]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(lambda s: timed_copy(s, dest, args.concurrent_streams), pair)
            )
        elapsed = time.perf_counter() - t0
        total = sum(size for _, size in results)
        print(
            f"2 shards at once x streams={args.concurrent_streams}: "
            f"{total / 1024**3:.2f} GiB in {elapsed:6.1f}s = {total / 1e6 / elapsed:6.0f} MB/s aggregate"
        )


if __name__ == "__main__":
    main()
