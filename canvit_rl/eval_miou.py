"""Compatibility wrapper for ADE20K mIoU evaluation.

New code should run `python -m canvit_rl.ade20k.eval_miou`.
"""

from canvit_rl.ade20k.eval_miou import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
