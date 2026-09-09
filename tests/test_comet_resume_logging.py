from __future__ import annotations

import argparse
import sys
import types

import torch

from canvit_rl.policies.canvas_state.logging import make_comet_experiment


class _FakeExperiment:
    instances: list["_FakeExperiment"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = None
        self.tags: list[str] = []
        self.parameters = None
        _FakeExperiment.instances.append(self)

    def get_key(self) -> str:
        return "fresh-comet-key"

    def set_name(self, name: str) -> None:
        self.name = name

    def add_tags(self, tags: list[str]) -> None:
        self.tags.extend(tags)

    def log_parameters(self, parameters: dict) -> None:
        self.parameters = parameters


class _FakeExistingExperiment(_FakeExperiment):
    instances: list["_FakeExistingExperiment"] = []

    def __init__(self, previous_experiment: str, **kwargs):
        self.previous_experiment = previous_experiment
        super().__init__(**kwargs)
        _FakeExistingExperiment.instances.append(self)


def test_canvas_comet_resume_uses_checkpoint_experiment_key(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "latest.pt"
    torch.save({"comet_experiment_key": "saved-comet-key"}, checkpoint)
    fake_comet = types.SimpleNamespace(
        Experiment=_FakeExperiment,
        ExistingExperiment=_FakeExistingExperiment,
    )
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    _FakeExperiment.instances.clear()
    _FakeExistingExperiment.instances.clear()
    args = argparse.Namespace(
        no_comet=False,
        comet_project="canvas-sac",
        comet_workspace=None,
        experiment_name="resume-run",
        comet_tags="resume,canvas",
        comet_experiment_key=None,
        resume=checkpoint,
    )

    experiment = make_comet_experiment(args)

    # Problem: trainer resume needs Comet continuity even when the user only
    # passes --resume. Solution: load the saved key and construct
    # ExistingExperiment. Result: all future logs append to the original run.
    assert isinstance(experiment, _FakeExistingExperiment)
    assert experiment.previous_experiment == "saved-comet-key"
    assert args.comet_experiment_key == "saved-comet-key"
    assert experiment.name == "resume-run"
    assert experiment.tags == ["resume", "canvas"]


def test_canvas_comet_fresh_run_stores_new_experiment_key(monkeypatch) -> None:
    fake_comet = types.SimpleNamespace(
        Experiment=_FakeExperiment,
        ExistingExperiment=_FakeExistingExperiment,
    )
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    _FakeExperiment.instances.clear()
    _FakeExistingExperiment.instances.clear()
    args = argparse.Namespace(
        no_comet=False,
        comet_project="canvas-sac",
        comet_workspace=None,
        experiment_name=None,
        comet_tags="",
        comet_experiment_key=None,
        resume=None,
    )

    experiment = make_comet_experiment(args)

    assert isinstance(experiment, _FakeExperiment)
    assert not isinstance(experiment, _FakeExistingExperiment)
    assert args.comet_experiment_key == "fresh-comet-key"
