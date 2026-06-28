"""Precision helpers for frozen CanViT inference inside RL loops."""

from __future__ import annotations

import torch


def resolve_canvit_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve the requested CanViT inference dtype for the current device."""
    if requested == "float32":
        return torch.float32
    if requested != "bfloat16":
        raise ValueError(f"Unsupported CanViT dtype: {requested}")
    if device.type != "cuda":
        print(
            "Requested bfloat16 CanViT inference, but device is not CUDA; "
            "using float32 for compatibility."
        )
        return torch.float32
    return torch.bfloat16


def configure_frozen_canvit_precision(
    *,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    requested: str,
    device: torch.device,
) -> torch.dtype:
    """Put frozen CanViT in bf16/fp32 while keeping the probe in fp32."""
    dtype = resolve_canvit_dtype(requested, device)
    model.to(device=device, dtype=dtype)
    # Fixed by Codex on 2026-06-28
    # Problem: model.to(dtype=torch.bfloat16) also casts the upstream
    # VPEEncoder's frozen RFF matrix, but VPEEncoder.forward disables autocast
    # and builds fp32 coordinate features before multiplying by that matrix.
    # Solution: keep VPEEncoder modules in fp32 after the bulk CanViT cast.
    # Result: frozen CanViT can still run bf16 patch/canvas inference without
    # dtype mismatches in viewpoint encoding.
    for module in model.modules():
        if module.__class__.__name__ == "VPEEncoder":
            module.to(device=device, dtype=torch.float32)
    probe.to(device=device, dtype=torch.float32)
    return dtype
