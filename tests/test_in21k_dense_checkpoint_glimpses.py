"""Glimpse-count persistence for IN21k dense SAC resume checkpoints."""

from canvit_rl.in21k.checkpoints import _checkpoint_glimpses


def test_checkpoint_glimpses_uses_saved_count() -> None:
    checkpoint = {"glimpses": 12345, "args": {"batch_size": 8, "t": 4}}
    assert _checkpoint_glimpses(checkpoint, batch=100) == 12345


def test_checkpoint_glimpses_estimates_legacy_from_saved_args() -> None:
    # Legacy checkpoints lack "glimpses": estimate batch * batch_size * (t + 1).
    checkpoint = {"args": {"batch_size": 8, "t": 4}}
    assert _checkpoint_glimpses(checkpoint, batch=100) == 100 * 8 * 5


def test_checkpoint_glimpses_without_args_is_zero() -> None:
    assert _checkpoint_glimpses({}, batch=100) == 0


from pathlib import Path

from canvit_rl.in21k.checkpoints import format_glimpse_count, resolve_dense_checkpoint_dir


def test_format_glimpse_count() -> None:
    assert format_glimpse_count(2_000_000) == "2M"
    assert format_glimpse_count(2_500_000) == "2.5M"
    assert format_glimpse_count(500_000) == "500K"
    assert format_glimpse_count(1_234_567) == "1.23457M"
    assert format_glimpse_count(999) == "999"


def _resolve(**overrides) -> Path:
    kwargs = dict(
        checkpoint_dir=None,
        checkpoint_root=Path("checkpoints/in1k_dense_sac"),
        experiment_name="sweep-15-full-t4_s44",
        start_batch=25_001,
        end_batch=50_000,
        batch_size=8,
        t=4,
        resume_glimpses=1_000_000,
    )
    kwargs.update(overrides)
    return resolve_dense_checkpoint_dir(**kwargs)


def test_resume_targets_total_glimpses() -> None:
    # 1M already trained + 25k batches * 8 * 5 = 1M more -> 2M folder.
    assert _resolve() == Path("checkpoints/in1k_dense_sac/sweep-15-full-t4_s44/2M")


def test_fresh_run_targets_glimpses() -> None:
    assert _resolve(start_batch=1, end_batch=25_000, resume_glimpses=0) == Path(
        "checkpoints/in1k_dense_sac/sweep-15-full-t4_s44/1M"
    )


def test_explicit_dir_and_no_name_fallbacks() -> None:
    assert _resolve(checkpoint_dir=Path("custom")) == Path("custom")
    assert _resolve(experiment_name=None) == Path("checkpoints/in1k_dense_sac")


def test_atomic_torch_save_keeps_previous_file_on_failure(tmp_path, monkeypatch) -> None:
    # Problem: a job killed mid-save used to leave a truncated latest.pt.
    # Expect: a failed save leaves the previous checkpoint loadable and
    # unchanged; a successful save replaces it and leaves no temp file.
    import torch

    from canvit_rl.in21k import checkpoints

    path = tmp_path / "latest.pt"
    checkpoints.atomic_torch_save({"batch": 1}, path)
    assert torch.load(path)["batch"] == 1

    real_save = torch.save

    def dying_save(obj, f):
        real_save(obj, f)
        with open(f, "r+b") as handle:  # simulate a write cut off mid-file
            handle.truncate(10)
        raise KeyboardInterrupt("killed")

    monkeypatch.setattr(checkpoints.torch, "save", dying_save)
    try:
        checkpoints.atomic_torch_save({"batch": 2}, path)
    except KeyboardInterrupt:
        pass
    monkeypatch.setattr(checkpoints.torch, "save", real_save)
    assert torch.load(path)["batch"] == 1

    checkpoints.atomic_torch_save({"batch": 3}, path)
    assert torch.load(path)["batch"] == 3
    assert not (tmp_path / ".latest.pt.tmp").exists()
