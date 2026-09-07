from scripts.sweeps.scan_optuna import _format_cli_flags, _param_to_cli_flag


def test_param_to_cli_flag_converts_snake_case_to_kebab_case():
    assert _param_to_cli_flag("actor_lr") == "--actor-lr"
    assert _param_to_cli_flag("critic_rff_dim") == "--critic-rff-dim"


def test_format_cli_flags_adds_continuations_except_last_line():
    # Problem: best-trial params used to require manual snake_case to CLI flag
    # conversion. Solution: format them as trainer-compatible kebab-case flags
    # with shell continuations. Result: users can paste the block directly into
    # non-Optuna training commands.
    lines = _format_cli_flags(
        {
            "actor_lr": 1e-4,
            "learning_starts": 512,
            "gamma": 0.25,
        }
    )

    assert lines == [
        "  --actor-lr 0.0001 \\",
        "  --learning-starts 512 \\",
        "  --gamma 0.25",
    ]
