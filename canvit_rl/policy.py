"""Compatibility wrapper for baseline policy modules.

New code should import from `canvit_rl.policies.mlp`. This wrapper keeps older
checkpoints, scripts, and tests that import `canvit_rl.policy` working.
"""

from canvit_rl.policies.mlp import MLPPolicy

__all__ = ["MLPPolicy"]
