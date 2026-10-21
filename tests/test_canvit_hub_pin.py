"""Tests for pinning canvit/* Hub repos to the canvit-pytorch 0.1 revision."""

from canvit_rl.environment import hub


def test_canvit_repo_resolves_to_pinned_revision(monkeypatch):
    calls = []
    monkeypatch.delenv("CANVIT_HUB_REVISION", raising=False)
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda repo_id, revision: calls.append((repo_id, revision)) or "/snap",
    )
    assert hub.pinned_repo("canvit/some-model") == "/snap"
    assert calls == [("canvit/some-model", "canvit-pytorch-0.1")]


def test_non_canvit_and_local_paths_pass_through(monkeypatch, tmp_path):
    monkeypatch.setattr(hub, "snapshot_download", lambda **_: "/unexpected")
    assert hub.pinned_repo("facebook/dinov3") == "facebook/dinov3"
    assert hub.pinned_repo(str(tmp_path)) == str(tmp_path)


def test_empty_env_disables_pin(monkeypatch):
    monkeypatch.setenv("CANVIT_HUB_REVISION", "")
    monkeypatch.setattr(hub, "snapshot_download", lambda **_: "/unexpected")
    assert hub.pinned_repo("canvit/some-model") == "canvit/some-model"
