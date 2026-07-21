"""Dense-feature batch adapters for IN21k policy pretraining.

CanViT-pretrain owns shard discovery, tar/image loading, resume positioning,
and preprocessing. This module keeps the RL side small: pull one dense-feature
batch, move it to the training device, cast stored fp16 teacher features to
fp32, and standardize them with the model-owned normalizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from canvit_pytorch.preprocess import preprocess
from PIL import Image
from torch import Tensor


class DenseTrainLoader(Protocol):
    """Minimal protocol implemented by canvit_pretrain ShardedFeatureLoader."""

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return images, raw dense patches, raw CLS tokens, and labels."""


class TensorStandardizer(Protocol):
    """Callable standardizer contract exposed by CanViT pretraining models."""

    def __call__(self, values: Tensor) -> Tensor:
        """Standardize a tensor with model-owned statistics."""


@dataclass(frozen=True)
class DenseTrainBatch:
    """Device-ready dense distillation targets for one RL rollout batch."""

    images: Tensor
    labels: Tensor
    scene_target: Tensor
    cls_target: Tensor
    raw_scene_target: Tensor
    raw_cls_target: Tensor
    glimpse_images: Tensor | None = None


def dense_glimpse_images(batch: DenseTrainBatch) -> Tensor:
    """Return the image tensor used for policy-selected non-t0 glimpses."""
    return batch.images if batch.glimpse_images is None else batch.glimpse_images


class FixedDenseSubsetLoader:
    """Small deterministic dense-feature subset loader for smoke experiments."""

    def __init__(
        self,
        *,
        shards_dir: Path,
        image_size: int,
        batch_size: int,
        subset_size: int,
        subset_seed: int,
        subset_shards: int,
        image_root: Path | None,
        tar_dir: Path | None,
    ) -> None:
        if subset_size <= 0:
            raise ValueError("subset_size must be positive.")
        if subset_shards <= 0:
            raise ValueError("subset_shards must be positive.")
        validate_dense_feature_source(feature_image_root=image_root, tar_dir=tar_dir)
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(subset_seed)
        shard_files = sorted(Path(shards_dir).glob("*.pt"))[:subset_shards]
        if not shard_files:
            raise FileNotFoundError(f"No dense-feature shards found in {shards_dir}")
        candidates = self._collect_candidates(
            shard_files,
            image_root=image_root,
        )
        if subset_size > len(candidates):
            source_hint = (
                f" with images under {image_root}"
                if image_root is not None
                else ""
            )
            raise ValueError(
                f"Requested subset_size={subset_size}, but only found "
                f"{len(candidates)} usable samples{source_hint} in "
                f"{len(shard_files)} shard(s). Increase --subset-shards, reduce "
                "--subset-size/--eval-images, or point --eval-feature-base-dir "
                "at split-specific feature shards."
            )
        chosen = torch.randperm(len(candidates), generator=self.generator)[:subset_size]
        selected = [candidates[int(i)] for i in chosen.tolist()]
        self.images, self.patches, self.cls, self.labels = self._load_selected(
            selected,
            image_size=image_size,
            image_root=image_root,
            tar_dir=tar_dir,
        )
        self.initial_order = torch.randperm(subset_size, generator=self.generator)
        self.order = self.initial_order.clone()
        self.pos = 0

    @staticmethod
    def _collect_candidates(
        shard_files: list[Path],
        *,
        image_root: Path | None,
    ) -> list[tuple[Path, int]]:
        """Collect usable shard row references without materializing features."""
        candidates: list[tuple[Path, int]] = []
        for shard_path in shard_files:
            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            failed = set(shard.get("failed_indices", []))
            for idx, rel_path in enumerate(shard["paths"]):
                if idx in failed:
                    continue
                if image_root is not None and not (image_root / str(rel_path)).is_file():
                    # Problem: train/validation/test feature rows can share a
                    # feature base while their pixels live under different
                    # roots. Solution: when an image root is provided, keep
                    # only rows whose stored relative path resolves under that
                    # root. Result: eval roots select matching split rows
                    # instead of failing later on train-only image paths.
                    continue
                candidates.append((shard_path, idx))
            del shard
        return candidates

    @staticmethod
    def _open_tar_reader(tar_dir: Path, shard_path: Path):
        """Open the tar reader matching one dense-feature shard."""
        from canvit_rl.pretrain_IN21k.pretrain_modules import install_pretrain_train_shim

        install_pretrain_train_shim()
        from canvit_pretrain.train.data.tar_images import TarImageReader, load_tar_index

        tar_path = tar_dir / f"{shard_path.stem}.tar"
        return TarImageReader(tar_path, index=load_tar_index(tar_path))

    @staticmethod
    def _load_selected(
        selected: list[tuple[Path, int]],
        *,
        image_size: int,
        image_root: Path | None,
        tar_dir: Path | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Materialize a small selected subset into memory."""
        transform = preprocess(image_size)
        images: list[Tensor] = []
        patches: list[Tensor] = []
        cls_tokens: list[Tensor] = []
        labels: list[int] = []
        current_shard_path: Path | None = None
        current_shard = None
        tar_reader = None
        try:
            for shard_path, sample_idx in selected:
                if current_shard_path != shard_path:
                    if tar_reader is not None:
                        tar_reader.close()
                    current_shard_path = shard_path
                    current_shard = torch.load(
                        shard_path,
                        map_location="cpu",
                        weights_only=False,
                        mmap=True,
                    )
                    tar_reader = (
                        FixedDenseSubsetLoader._open_tar_reader(tar_dir, shard_path)
                        if tar_dir is not None
                        else None
                    )
                assert current_shard is not None
                rel_path = current_shard["paths"][sample_idx]
                if tar_reader is not None:
                    image = tar_reader.read_image(rel_path)
                else:
                    assert image_root is not None
                    with Image.open(image_root / rel_path) as image_file:
                        image = image_file.convert("RGB")
                # Problem: small-subset experiments should exercise the same
                # student image preprocessing as shard training. Solution:
                # load images through CanViT-pretrain's preprocess transform
                # while materializing only the selected records. Result: tiny
                # runs use realistic pixels/features without scanning the full
                # dataset every batch.
                images.append(transform(image))
                patches.append(current_shard["patches"][sample_idx].clone())
                cls_tokens.append(current_shard["cls"][sample_idx].clone())
                labels.append(int(current_shard["class_idxs"][sample_idx]))
        finally:
            if tar_reader is not None:
                tar_reader.close()
        return (
            torch.stack(images),
            torch.stack(patches),
            torch.stack(cls_tokens),
            torch.tensor(labels, dtype=torch.long),
        )

    def _next_indices(self) -> Tensor:
        """Return a reshuffled mini-batch of subset indices."""
        pieces: list[Tensor] = []
        remaining = self.batch_size
        while remaining > 0:
            if self.pos >= len(self.order):
                self.order = torch.randperm(len(self.order), generator=self.generator)
                self.pos = 0
            take = min(remaining, len(self.order) - self.pos)
            pieces.append(self.order[self.pos : self.pos + take])
            self.pos += take
            remaining -= take
        return torch.cat(pieces)

    def reset(self) -> None:
        """Replay the same materialized subset order for deterministic eval."""
        self.order = self.initial_order.clone()
        self.pos = 0

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the next reshuffled batch from the fixed subset."""
        idx = self._next_indices()
        return (
            self.images.index_select(0, idx),
            self.patches.index_select(0, idx),
            self.cls.index_select(0, idx),
            self.labels.index_select(0, idx),
        )


class PairedDenseShardLoader:
    """Small paired-shard loader for hidden-image/oracle-target diagnostics."""

    def __init__(
        self,
        *,
        hidden_shards_dir: Path,
        target_shards_dir: Path,
        image_size: int,
        batch_size: int,
        hidden_image_root: Path | None,
        hidden_tar_dir: Path | None,
        target_image_root: Path,
        shuffle_seed: int,
    ) -> None:
        validate_dense_feature_source(
            feature_image_root=hidden_image_root,
            tar_dir=hidden_tar_dir,
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(shuffle_seed)
        self.images, self.glimpse_images, self.patches, self.cls, self.labels = (
            self._load_paired_records(
                hidden_shards_dir=hidden_shards_dir,
                target_shards_dir=target_shards_dir,
                image_size=image_size,
                hidden_image_root=hidden_image_root,
                hidden_tar_dir=hidden_tar_dir,
                target_image_root=target_image_root,
            )
        )
        self.order = torch.randperm(self.images.shape[0], generator=self.generator)
        self.initial_order = self.order.clone()
        self.pos = 0

    @staticmethod
    @staticmethod
    def _list_shards(shards_dir: Path) -> list[Path]:
        """Return sorted dense-feature shard files from one source."""
        shard_files = sorted(Path(shards_dir).glob("*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No dense-feature shards found in {shards_dir}")
        return shard_files

    @staticmethod
    def _sample_key(rel_path: str) -> str:
        """Return the stable key used to join independently shuffled exports."""
        return Path(rel_path).name

    @classmethod
    def _index_hidden_records(
        cls,
        *,
        hidden_shards_dir: Path,
    ) -> dict[str, tuple[Path, str]]:
        """Index usable hidden rows by sample basename across all hidden shards."""
        index: dict[str, tuple[Path, str]] = {}
        for hidden_shard_path in cls._list_shards(hidden_shards_dir):
            hidden_shard = torch.load(
                hidden_shard_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            failed = set(hidden_shard.get("failed_indices", []))
            for idx, rel_path in enumerate(hidden_shard["paths"]):
                if idx in failed:
                    continue
                key = cls._sample_key(str(rel_path))
                if key in index:
                    raise ValueError(
                        f"Duplicate hidden sample key {key!r}; use unique staged "
                        "filenames before feature export."
                    )
                index[key] = (hidden_shard_path, str(rel_path))
            del hidden_shard
        return index

    @staticmethod
    def _load_hidden_image(
        *,
        rel_path: str,
        hidden_image_root: Path | None,
        hidden_tar_reader,
    ) -> Image.Image:
        """Load the hidden t0 image for one paired shard row."""
        if hidden_tar_reader is not None:
            return hidden_tar_reader.read_image(rel_path)
        assert hidden_image_root is not None
        with Image.open(hidden_image_root / rel_path) as image_file:
            return image_file.convert("RGB")

    @staticmethod
    def _load_target_image(
        *,
        rel_path: str,
        target_image_root: Path,
    ) -> Image.Image:
        """Load the oracle image if available for non-t0 glimpses."""
        with Image.open(target_image_root / rel_path) as image_file:
            return image_file.convert("RGB")

    @classmethod
    def _load_paired_records(
        cls,
        *,
        hidden_shards_dir: Path,
        target_shards_dir: Path,
        image_size: int,
        hidden_image_root: Path | None,
        hidden_tar_dir: Path | None,
        target_image_root: Path,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Materialize paired hidden images and oracle targets into memory."""
        transform = preprocess(image_size)
        hidden_images: list[Tensor] = []
        target_images: list[Tensor] = []
        patches: list[Tensor] = []
        cls_tokens: list[Tensor] = []
        labels: list[int] = []
        hidden_index = cls._index_hidden_records(hidden_shards_dir=hidden_shards_dir)
        tar_readers: dict[Path, object] = {}
        target_keys_seen: set[str] = set()
        try:
            for target_shard_path in cls._list_shards(target_shards_dir):
                target_shard = torch.load(
                    target_shard_path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
                target_failed = set(target_shard.get("failed_indices", []))
                for idx, target_rel_path in enumerate(target_shard["paths"]):
                    if idx in target_failed:
                        continue
                    if not (target_image_root / str(target_rel_path)).is_file():
                        # Problem: paired eval target shards may include rows
                        # from another image split when the feature base is
                        # shared. Solution: skip target rows that do not exist
                        # under the requested target image root. Result: paired
                        # eval samples only rows with matching target pixels.
                        continue
                    key = cls._sample_key(str(target_rel_path))
                    if key in target_keys_seen:
                        raise ValueError(
                            f"Duplicate target sample key {key!r}; use unique "
                            "staged filenames before feature export."
                        )
                    target_keys_seen.add(key)
                    hidden_record = hidden_index.get(key)
                    if hidden_record is None:
                        continue
                    hidden_shard_path, hidden_rel_path = hidden_record
                    hidden_tar_reader = None
                    if hidden_tar_dir is not None:
                        hidden_tar_reader = tar_readers.get(hidden_shard_path)
                        if hidden_tar_reader is None:
                            hidden_tar_reader = FixedDenseSubsetLoader._open_tar_reader(
                                hidden_tar_dir,
                                hidden_shard_path,
                            )
                            tar_readers[hidden_shard_path] = hidden_tar_reader
                    # Problem: hidden and oracle parquet exports can be shuffled
                    # independently, so shard filenames and row indices are not
                    # stable pair identifiers. Solution: join rows by the
                    # staged sample basename and then load hidden pixels plus
                    # oracle features/images from their own sources. Result:
                    # paired SAC training is correct even after independent
                    # parquet/shard shuffles.
                    hidden_image = cls._load_hidden_image(
                        rel_path=hidden_rel_path,
                        hidden_image_root=hidden_image_root,
                        hidden_tar_reader=hidden_tar_reader,
                    )
                    target_image = cls._load_target_image(
                        rel_path=str(target_rel_path),
                        target_image_root=target_image_root,
                    )
                    hidden_images.append(transform(hidden_image))
                    target_images.append(transform(target_image))
                    patches.append(target_shard["patches"][idx].clone())
                    cls_tokens.append(target_shard["cls"][idx].clone())
                    labels.append(int(target_shard["class_idxs"][idx]))
                del target_shard
        finally:
            for reader in tar_readers.values():
                reader.close()
        if not hidden_images:
            raise ValueError("No usable paired dense-shard rows were found.")
        missing = len(target_keys_seen) - len(hidden_images)
        if missing:
            print(
                f"PairedDenseShardLoader skipped {missing} target rows without "
                "matching hidden basename."
            )
        return (
            torch.stack(hidden_images),
            torch.stack(target_images),
            torch.stack(patches),
            torch.stack(cls_tokens),
            torch.tensor(labels, dtype=torch.long),
        )

    def _next_indices(self) -> Tensor:
        """Return a reshuffled mini-batch of paired indices."""
        pieces: list[Tensor] = []
        remaining = self.batch_size
        while remaining > 0:
            if self.pos >= len(self.order):
                self.order = torch.randperm(len(self.order), generator=self.generator)
                self.pos = 0
            take = min(remaining, len(self.order) - self.pos)
            pieces.append(self.order[self.pos : self.pos + take])
            self.pos += take
            remaining -= take
        return torch.cat(pieces)

    def reset(self) -> None:
        """Replay the same materialized paired order for deterministic eval."""
        self.order = self.initial_order.clone()
        self.pos = 0

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return hidden images, oracle images, targets, and labels."""
        idx = self._next_indices()
        return (
            self.images.index_select(0, idx),
            self.glimpse_images.index_select(0, idx),
            self.patches.index_select(0, idx),
            self.cls.index_select(0, idx),
            self.labels.index_select(0, idx),
        )


def validate_dense_feature_source(
    *,
    feature_image_root: Path | str | None,
    tar_dir: Path | str | None,
) -> tuple[Path | None, Path | None]:
    """Validate the mutually exclusive image source expected by pretrain shards."""
    image_root_path = Path(feature_image_root) if feature_image_root is not None else None
    tar_dir_path = Path(tar_dir) if tar_dir is not None else None
    if (image_root_path is None) == (tar_dir_path is None):
        raise ValueError(
            "Exactly one of feature_image_root or tar_dir must be provided for "
            "IN21k dense-feature loading."
        )
    return image_root_path, tar_dir_path


def apply_dense_feature_config(
    cfg: object,
    *,
    feature_base_dir: Path | str,
    feature_image_root: Path | str | None = None,
    tar_dir: Path | str | None = None,
) -> object:
    """Set dense-feature paths on a canvit_pretrain Config-like object.

    This deliberately avoids importing ``canvit_pretrain.train`` here, because
    that package currently imports optional plotting/probe modules at package
    import time. Keeping this helper duck-typed lets scripts configure the
    actual pretrain ``Config`` object after they create it.
    """
    image_root_path, tar_dir_path = validate_dense_feature_source(
        feature_image_root=feature_image_root,
        tar_dir=tar_dir,
    )
    setattr(cfg, "feature_base_dir", Path(feature_base_dir))
    setattr(cfg, "feature_image_root", image_root_path)
    setattr(cfg, "tar_dir", tar_dir_path)
    return cfg


def load_dense_train_batch(
    *,
    train_loader: DenseTrainLoader,
    device: torch.device,
    scene_norm: TensorStandardizer,
    cls_norm: TensorStandardizer,
    non_blocking: bool = True,
) -> DenseTrainBatch:
    """Load and standardize one CanViT-pretrain dense-feature train batch."""
    batch_parts = train_loader.next()
    if len(batch_parts) == 5:
        images, glimpse_images, raw_patches, raw_cls, labels = batch_parts
    else:
        images, raw_patches, raw_cls, labels = batch_parts
        glimpse_images = None
    images = images.to(device=device, non_blocking=non_blocking)
    if glimpse_images is not None:
        glimpse_images = glimpse_images.to(device=device, non_blocking=non_blocking)
    labels = labels.to(device=device, non_blocking=non_blocking)

    # Problem: dense features are stored as fp16 in shards, but distillation
    # rewards compare feature geometry and normalizer statistics in fp32.
    # Solution: materialize raw patch/CLS targets as float32 before
    # standardization. Result: RL rewards match CanViT-pretrain's loss/metric
    # path while preserving the raw targets for denormalized reward metrics.
    raw_scene_target = raw_patches.to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    raw_cls_target = raw_cls.to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    scene_target = scene_norm(raw_scene_target)
    cls_target = cls_norm(raw_cls_target.unsqueeze(1)).squeeze(1)
    return DenseTrainBatch(
        images=images,
        labels=labels,
        scene_target=scene_target,
        cls_target=cls_target,
        raw_scene_target=raw_scene_target,
        raw_cls_target=raw_cls_target,
        glimpse_images=glimpse_images,
    )


def init_normalizer_stats_from_shard(
    *,
    shards_dir: Path,
    scene_norm: object,
    cls_norm: object,
    device: torch.device,
    max_samples: int,
) -> None:
    """Initialize pretraining normalizers from the first dense-feature shard."""
    shard_files = sorted(Path(shards_dir).glob("*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No dense-feature shards found in {shards_dir}")
    shard_path = shard_files[0]
    shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    n_total = shard["patches"].shape[0]
    n = min(n_total, max_samples) if max_samples > 0 else n_total

    # Problem: the upstream initializer lives in canvit_pretrain.train.loop,
    # but importing that loop also imports optional plotting/probe modules in
    # this environment. Solution: keep the same mmap/subset/set_stats behavior
    # here without touching the training loop package. Result: fresh dense SAC
    # runs can initialize model-owned standardizers from shards reliably.
    patches = shard["patches"][:n].clone().float().to(device)
    cls = shard["cls"][:n].clone().float().to(device)
    scene_norm.set_stats(patches)
    cls_norm.set_stats(cls.unsqueeze(1))
    del shard, patches, cls
    if device.type == "cuda":
        torch.cuda.empty_cache()
