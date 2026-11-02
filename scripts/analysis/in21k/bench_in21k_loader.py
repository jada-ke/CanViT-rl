"""Time the two IN21k dense-SAC data paths separately: images vs feature shards.

Problem: IN21k dense SAC runs dropped from ~770-1160 to ~280 glimpses/s with
the GPU mostly idle, and the training log does not show whether image reads or
feature-shard reads are the slow part (canvit-pretrain's per-shard timing logs
are INFO-level and never enabled in the RL script).

Solution: reproduce the training loader's access pattern without the GPU.
Each phase uses a fresh slice of samples from one shard so earlier phases do not
warm the page cache for later ones:

1. shard open        torch.load(mmap=True), as AllShardsDataset does
2. features, 1 proc  clone patches + cls for strided samples
3. images, 1 proc    raw file bytes from --feature-image-root
4. decode, 1 proc    PIL decode + canvit preprocess on in-memory bytes (CPU only)
5. features, N procs strided across --num-workers forked workers
6. images, N procs   read + decode + preprocess across --num-workers workers

Result: phases 5 and 6 give images/s for each path at training concurrency;
the lower one caps throughput. Glimpses/s = images/s * (t + 1).

Example (inside the training container, same binds as the training job):
    python scripts/analysis/in21k/bench_in21k_loader.py \\
        --feature-base-dir /features --feature-image-root /data/train
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import time
from pathlib import Path

import torch
from canvit_pytorch.preprocess import preprocess
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-base-dir", type=Path, required=True)
    parser.add_argument("--feature-image-root", type=Path, required=True)
    parser.add_argument("--teacher-name", type=str, default="dinov3_vitb16")
    parser.add_argument("--scene-resolution", type=int, default=512)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard to read (sorted order). Default: middle shard, to avoid "
        "shards recent jobs just read and the filesystem may still cache.",
    )
    parser.add_argument("--num-samples", type=int, default=256, help="Samples per phase.")
    parser.add_argument("--num-workers", type=int, default=7, help="Match training --num-workers.")
    parser.add_argument("--t", type=int, default=4, help="Policy glimpses per episode.")
    return parser.parse_args()


# Shard and transform are set in the parent before forking so workers share
# them copy-on-write, matching how the training DataLoader workers see them.
_SHARD: dict | None = None
_IMAGE_ROOT: Path | None = None
_TRANSFORM = None


def _read_features(indices: list[int]) -> tuple[int, int]:
    assert _SHARD is not None
    nbytes = 0
    for i in indices:
        patches = _SHARD["patches"][i].clone()
        cls = _SHARD["cls"][i].clone()
        nbytes += patches.numel() * patches.element_size() + cls.numel() * cls.element_size()
    return len(indices), nbytes


def _read_images(indices: list[int], decode: bool) -> tuple[int, int]:
    assert _SHARD is not None and _IMAGE_ROOT is not None
    nbytes = 0
    for i in indices:
        data = (_IMAGE_ROOT / _SHARD["paths"][i]).read_bytes()
        nbytes += len(data)
        if decode:
            with Image.open(io.BytesIO(data)) as f:
                _TRANSFORM(f.convert("RGB"))
    return len(indices), nbytes


def _worker_features(indices: list[int]) -> tuple[int, int]:
    return _read_features(indices)


def _worker_images(indices: list[int]) -> tuple[int, int]:
    return _read_images(indices, decode=True)


def report(name: str, n: int, nbytes: int, seconds: float, t: int) -> None:
    ips = n / seconds if seconds > 0 else float("inf")
    print(
        f"{name:28s} {n:5d} imgs {seconds:7.2f}s  "
        f"{seconds / max(n, 1) * 1e3:7.1f} ms/img  {ips:7.1f} img/s  "
        f"(= {ips * (t + 1):7.0f} glimpses/s)  {nbytes / max(seconds, 1e-9) / 1e6:7.1f} MB/s",
        flush=True,
    )


def main() -> None:
    global _SHARD, _IMAGE_ROOT, _TRANSFORM
    args = parse_args()
    shards_dir = args.feature_base_dir / args.teacher_name / str(args.scene_resolution) / "shards"
    shard_files = sorted(shards_dir.glob("*.pt"))
    if not shard_files:
        raise SystemExit(f"No shards in {shards_dir}")
    shard_index = args.shard_index if args.shard_index is not None else len(shard_files) // 2
    shard_path = shard_files[shard_index % len(shard_files)]
    print(f"shards_dir={shards_dir} ({len(shard_files)} shards)\nshard={shard_path.name}")

    t0 = time.perf_counter()
    _SHARD = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    print(f"{'1. shard open (mmap)':28s} {time.perf_counter() - t0:7.2f}s")
    _IMAGE_ROOT = args.feature_image_root
    _TRANSFORM = preprocess(args.scene_resolution)

    failed = set(_SHARD.get("failed_indices", []))
    valid = [i for i in range(len(_SHARD["paths"])) if i not in failed]
    n = args.num_samples
    if len(valid) < 6 * n:
        n = len(valid) // 6
        print(f"shard has {len(valid)} usable samples; using {n} per phase")
    # Disjoint sample slices per phase so no phase reads data a previous one cached.
    slices = [valid[k * n : (k + 1) * n] for k in range(6)]

    def strided(indices: list[int], workers: int) -> list[list[int]]:
        # Same split as AllShardsDataset: worker w takes samples w, w+N, w+2N, ...
        return [indices[w::workers] for w in range(workers)]

    t0 = time.perf_counter()
    k, b = _read_features(slices[1])
    report("2. features, 1 proc", k, b, time.perf_counter() - t0, args.t)

    t0 = time.perf_counter()
    k, b = _read_images(slices[2], decode=False)
    report("3. image read, 1 proc", k, b, time.perf_counter() - t0, args.t)

    # Decode-only: pre-read bytes untimed, so this measures CPU work alone.
    blobs = [(_IMAGE_ROOT / _SHARD["paths"][i]).read_bytes() for i in slices[3]]
    t0 = time.perf_counter()
    for data in blobs:
        with Image.open(io.BytesIO(data)) as f:
            _TRANSFORM(f.convert("RGB"))
    report("4. decode+resize, 1 proc", len(blobs), sum(map(len, blobs)), time.perf_counter() - t0, args.t)

    ctx = mp.get_context("fork")
    with ctx.Pool(args.num_workers) as pool:
        t0 = time.perf_counter()
        parts = pool.map(_worker_features, strided(slices[4], args.num_workers))
        report(
            f"5. features, {args.num_workers} procs",
            sum(p[0] for p in parts),
            sum(p[1] for p in parts),
            time.perf_counter() - t0,
            args.t,
        )
        t0 = time.perf_counter()
        parts = pool.map(_worker_images, strided(slices[5], args.num_workers))
        report(
            f"6. images, {args.num_workers} procs",
            sum(p[0] for p in parts),
            sum(p[1] for p in parts),
            time.perf_counter() - t0,
            args.t,
        )

    print(
        "\nReference: Sep runs trained at 770-1160 glimpses/s, Oct runs at ~280.\n"
        "Whichever of rows 5/6 is lower is the data-path ceiling; if both are well\n"
        "above training throughput, the bottleneck is elsewhere (GPU/CPU contention)."
    )


if __name__ == "__main__":
    main()
