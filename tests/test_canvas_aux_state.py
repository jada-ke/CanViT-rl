"""ADE Canvas SAC aux-state selection (probe entropy vs reconstruction norm)."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from canvit_rl.policies.canvas_state.args import (
    add_canvas_sac_args,
    validate_canvas_sac_args,
)
from canvit_rl.policies.canvas_state.checkpoints import (
    canvas_state_representation,
    checkpoint_aux_state_kind,
    inherit_canvas_state_args,
)
from canvit_rl.policies.canvas_state.models import CanvasStateActor
from canvit_rl.policies.canvas_state.state import (
    canvas_aux_state_kind,
    canvas_aux_state_map,
)
from canvit_rl.policies.canvas_state.training import build_canvas_sac_networks


def _parse(*argv: str, **overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_canvas_sac_args(parser)
    args = parser.parse_args(list(argv))
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _in21k_checkpoint(tmp_path, *, aux_token="_reconstruction_norm", **arg_overrides):
    """Write an IN21k-dense-SAC-shaped latest.pt with metadata only + actor."""
    saved_args = {
        "d_model": 8,
        "rff_dim": 4,
        "rff_seed": 0,
        "canvas_entropy_state": True,  # IN21k meaning: reconstruction norm
        "reconstruction_norm_state": False,
        "disable_canvas_avg_pool": False,
        "disable_canvas_max_pool": True,
        "disable_viewpoint_history_state": False,
        **arg_overrides,
    }
    actor = CanvasStateActor(
        canvas_feature_dim=4,
        d_model=8,
        rff_dim=4,
        rff_seed=0,
        use_entropy_state=aux_token != "",
        aux_state_channels=1 if aux_token else 0,
        use_canvas_max_pool=False,
    )
    path = tmp_path / "latest.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "args": saved_args,
            "state_representation": (
                f"current_canvas_layernorm{aux_token}_with_viewpoint_history"
            ),
        },
        path,
    )
    return path, actor


class _FakeCanViT:
    """Minimal model exposing the two canvas readouts used by aux maps."""

    def __init__(self, grid: int, dim: int, teacher_dim: int):
        self.grid = grid
        self.teacher = torch.nn.Linear(dim, teacher_dim)

    def get_spatial(self, canvas: torch.Tensor) -> torch.Tensor:
        return canvas

    def predict_teacher_scene(self, canvas: torch.Tensor) -> torch.Tensor:
        return self.teacher(canvas)


def test_aux_state_kind_resolution():
    assert canvas_aux_state_kind(SimpleNamespace()) is None
    assert (
        canvas_aux_state_kind(SimpleNamespace(canvas_entropy_state=True))
        == "segmentation_entropy"
    )
    assert (
        canvas_aux_state_kind(SimpleNamespace(reconstruction_norm_state=True))
        == "reconstruction_norm"
    )
    with pytest.raises(ValueError):
        canvas_aux_state_kind(
            SimpleNamespace(canvas_entropy_state=True, reconstruction_norm_state=True)
        )


def test_reconstruction_norm_is_not_a_cli_flag():
    # The aux kind comes from the init/resume checkpoint, not the CLI.
    with pytest.raises(SystemExit):
        _parse("--reconstruction-norm-state")


def test_validate_rejects_both_aux_kinds():
    validate_canvas_sac_args(_parse(reconstruction_norm_state=True))
    with pytest.raises(ValueError, match="at most one"):
        validate_canvas_sac_args(
            _parse("--canvas-entropy-state", reconstruction_norm_state=True)
        )


def test_state_representation_labels_aux_signal():
    # The ADE eval script parses this suffix to rebuild the matching aux map.
    assert canvas_state_representation(
        _parse(reconstruction_norm_state=True)
    ) == (
        "current_canvas_layernorm_reconstruction_norm_with_viewpoint_history"
    )
    assert canvas_state_representation(_parse("--canvas-entropy-state")) == (
        "current_canvas_layernorm_entropy_with_viewpoint_history"
    )
    assert canvas_state_representation(_parse()) == (
        "current_canvas_layernorm_with_viewpoint_history"
    )


def test_aux_state_map_dispatch():
    torch.manual_seed(0)
    grid, dim, classes = 4, 6, 5
    model = _FakeCanViT(grid=grid, dim=dim, teacher_dim=8)
    probe = torch.nn.Linear(dim, classes)
    state = SimpleNamespace(canvas=torch.randn(2, grid * grid, dim))

    assert (
        canvas_aux_state_map(
            kind=None, model=model, probe=probe, state=state, canvas_grid_size=grid
        )
        is None
    )

    recon = canvas_aux_state_map(
        kind="reconstruction_norm",
        model=model,
        probe=probe,
        state=state,
        canvas_grid_size=grid,
    )
    assert recon.shape == (2, 1, grid, grid)
    # Per-image min-max normalization, matching IN21k dense SAC.
    flat = recon.flatten(1)
    assert torch.allclose(flat.min(dim=1).values, torch.zeros(2), atol=1e-6)
    assert torch.allclose(flat.max(dim=1).values, torch.ones(2), atol=1e-5)
    expected = model.predict_teacher_scene(state.canvas).norm(dim=-1)
    assert torch.equal(recon.flatten(1).argmax(dim=1), expected.argmax(dim=1))

    # Probe entropy path: the fake probe acts per token, so reshape to the
    # [B, C, G, G] layout canvas_segmentation_entropy expects.
    class _ChannelProbe(torch.nn.Module):
        def forward(self, spatial: torch.Tensor) -> torch.Tensor:
            return probe(spatial).permute(0, 3, 1, 2)

    entropy = canvas_aux_state_map(
        kind="segmentation_entropy",
        model=model,
        probe=_ChannelProbe(),
        state=state,
        canvas_grid_size=grid,
    )
    assert entropy.shape == (2, 1, grid, grid)
    assert float(entropy.detach().max()) <= 1.0 + 1e-6
    assert not torch.allclose(entropy, recon)

    with pytest.raises(ValueError):
        canvas_aux_state_map(
            kind="nope", model=model, probe=probe, state=state, canvas_grid_size=grid
        )


def test_checkpoint_aux_state_kind_parsing():
    rep = "current_canvas_layernorm{}_with_viewpoint_history"
    assert checkpoint_aux_state_kind({"state_representation": rep.format("")}) is None
    assert (
        checkpoint_aux_state_kind({"state_representation": rep.format("_entropy")})
        == "segmentation_entropy"
    )
    assert (
        checkpoint_aux_state_kind(
            {"state_representation": rep.format("_reconstruction_norm")}
        )
        == "reconstruction_norm"
    )
    assert checkpoint_aux_state_kind({}) is None
    for unsupported in ("_detail_debt", "_reconstruction_norm_cos_prev"):
        with pytest.raises(ValueError, match="not supported"):
            checkpoint_aux_state_kind(
                {"state_representation": rep.format(unsupported)}
            )


def test_init_actor_checkpoint_inherits_in21k_state_and_loads(tmp_path):
    path, in21k_actor = _in21k_checkpoint(tmp_path)
    # CLI deliberately disagrees: wrong sizes, ADE entropy flag, max pool on.
    args = _parse(
        "--canvas-entropy-state",
        "--init-actor-checkpoint",
        str(path),
    )

    inherit_canvas_state_args(args)
    validate_canvas_sac_args(args)

    assert canvas_aux_state_kind(args) == "reconstruction_norm"
    assert (args.d_model, args.rff_dim, args.rff_seed) == (8, 4, 0)
    assert args.disable_canvas_max_pool is True
    assert canvas_state_representation(args) == (
        "current_canvas_layernorm_reconstruction_norm_with_viewpoint_history"
    )
    ade_actor, *_ = build_canvas_sac_networks(
        args=args,
        canvas_feature_dim=4,
        device=torch.device("cpu"),
    )
    # Strict load: same parameter names and shapes as the IN21k actor.
    ade_actor.load_state_dict(in21k_actor.state_dict())


def test_inherit_without_aux_clears_cli_entropy(tmp_path):
    path, _ = _in21k_checkpoint(tmp_path, aux_token="", canvas_entropy_state=False)
    args = _parse("--canvas-entropy-state", "--init-actor-checkpoint", str(path))
    inherit_canvas_state_args(args)
    assert canvas_aux_state_kind(args) is None


def test_inherit_is_noop_without_checkpoint_or_metadata(tmp_path):
    args = _parse("--canvas-entropy-state")
    inherit_canvas_state_args(args)
    assert canvas_aux_state_kind(args) == "segmentation_entropy"

    bare = tmp_path / "actor_final.pt"
    torch.save({"weight": torch.zeros(1)}, bare)
    args = _parse("--canvas-entropy-state", "--init-actor-checkpoint", str(bare))
    inherit_canvas_state_args(args)
    assert canvas_aux_state_kind(args) == "segmentation_entropy"
