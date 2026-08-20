from pathlib import Path

import torch
from PIL import Image

from canvit_rl.in21k import FixedDenseSubsetLoader


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_fixed_dense_subset_loader_streams_records(tmp_path: Path) -> None:
    shards_dir = tmp_path / "features" / "dinov3_vitb16" / "512" / "shards"
    shards_dir.mkdir(parents=True)
    image_root = tmp_path / "images"
    paths = [f"class_{idx % 2}/sample_{idx}.JPEG" for idx in range(6)]
    for idx, rel_path in enumerate(paths):
        _write_image(
            image_root / rel_path,
            color=(idx * 30, idx * 20, idx * 10),
        )

    torch.save(
        {
            "paths": paths,
            "patches": torch.arange(6 * 4 * 3, dtype=torch.float16).reshape(6, 4, 3),
            "cls": torch.arange(6 * 3, dtype=torch.float16).reshape(6, 3),
            "class_idxs": torch.arange(6, dtype=torch.long),
            "failed_indices": [],
        },
        shards_dir / "shard_000.pt",
    )

    loader = FixedDenseSubsetLoader(
        shards_dir=shards_dir,
        image_size=32,
        batch_size=2,
        subset_size=4,
        subset_seed=123,
        subset_shards=1,
        image_root=image_root,
        tar_dir=None,
    )

    # Problem: the fixed-subset loader previously kept full image/feature
    # tensors on the object. Solution: store only selected shard row references
    # and materialize mini-batches in next(). Result: large deterministic
    # subsets avoid host-memory OOM while preserving the same public loader API.
    assert not hasattr(loader, "images")
    assert not hasattr(loader, "patches")
    assert loader.dataset._shard_cache == {}

    first = loader.next()
    assert first[0].shape == (2, 3, 32, 32)
    assert first[1].shape == (2, 4, 3)
    assert first[2].shape == (2, 3)
    assert first[3].shape == (2,)
    assert len(loader.dataset._shard_cache) == 1

    loader.reset()
    replayed = loader.next()
    for actual, expected in zip(replayed, first, strict=True):
        torch.testing.assert_close(actual, expected)

    same_seed_loader = FixedDenseSubsetLoader(
        shards_dir=shards_dir,
        image_size=32,
        batch_size=2,
        subset_size=4,
        subset_seed=123,
        subset_shards=1,
        image_root=image_root,
        tar_dir=None,
    )
    same_seed_first = same_seed_loader.next()
    assert same_seed_loader.selected == loader.selected
    for actual, expected in zip(same_seed_first, first, strict=True):
        torch.testing.assert_close(actual, expected)
