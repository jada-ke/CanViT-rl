"""
canvit_rl/env.py

Gymnasium environment wrapping a CanViT episode.

Observation: recurrent_cls token — shape (1, 768) squeezed to (768,)
Action:      [cx, cy, scale] in [-1, 1] x [-1, 1] x (0, 1]
Reward:      cosine similarity between canvas CLS and frozen DINOv3 teacher CLS
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch import (
    CanViTForPretrainingHFHub,
    Viewpoint,
    sample_at_viewpoint,
)
from canvit_pytorch.preprocess import preprocess
from canvit_pytorch.teacher import load_teacher


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CanViTEnvConfig:
    checkpoint: str = (
        "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
    )
    teacher_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    scene_size_px: int = 512
    glimpse_size_px: int = 128
    canvas_grid_size: int = 64
    max_steps: int = 10
    cls_dim: int = 768


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CanViTEnv(gym.Env):
    """
    A single-image CanViT episode as a Gymnasium environment.

    At each step the agent outputs a Viewpoint (center + scale) and receives:
    - obs:    recurrent_cls token, shape (cls_dim,)
    - reward: cosine similarity gain vs previous step
    - done:   True after max_steps

    The environment is intentionally image-agnostic at init time.
    Call reset(image=...) with a pre-processed [1, 3, H, W] tensor,
    or let reset() generate a random synthetic image for unit tests.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: CanViTEnvConfig | None = None, device: torch.device | None = None):
        super().__init__()
        self.cfg = config or CanViTEnvConfig()
        self.device = device or get_device()

        # --- Spaces ---
        # Action: [cx, cy, scale] — all in [-1, 1]; scale will be re-mapped to (0, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Observation: recurrent_cls flattened
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.cls_dim,),
            dtype=np.float32,
        )

        # --- Model (loaded once, frozen) ---
        checkpoint = os.environ.get("CANVIT_CHECKPOINT", self.cfg.checkpoint)
        self._model = (
            CanViTForPretrainingHFHub.from_pretrained(checkpoint)
            .eval()
            .to(self.device)
        )
        for p in self._model.parameters():
            p.requires_grad_(False)

        # --- Teacher (loaded once, frozen) ---
        self._teacher = load_teacher(self.cfg.teacher_repo, self.device)

        self._preprocess = preprocess(self.cfg.scene_size_px)

        # --- Episode state ---
        self._image: torch.Tensor | None = None
        self._state = None
        self._teacher_cls: torch.Tensor | None = None
        self._prev_sim: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        image: torch.Tensor | None = (options or {}).get("image", None)
        if image is None:
            # Synthetic random image for testing / debugging (no dataset needed)
            image = torch.rand(
                1, 3, self.cfg.scene_size_px, self.cfg.scene_size_px,
                device=self.device,
            )
        else:
            image = image.to(self.device)
            if image.ndim == 3:
                image = image.unsqueeze(0)

        self._image = image
        self._state = self._model.init_state(
            batch_size=1, canvas_grid_size=self.cfg.canvas_grid_size
        )

        # Compute teacher CLS once per episode — target for the reward signal
        with torch.inference_mode():
            self._teacher_cls = self._teacher.forward_norm_features(image).cls  # [1, 768]

        self._prev_sim = 0.0
        self._step_count = 0

        # First observation: full-scene glimpse (no agent action yet)
        vp = Viewpoint.full_scene(batch_size=1, device=self.device)
        obs, _ = self._take_glimpse(vp)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._image is not None, "Call reset() before step()."

        vp = self._action_to_viewpoint(action)
        obs, sim = self._take_glimpse(vp)

        reward = float(sim - self._prev_sim)
        self._prev_sim = sim
        self._step_count += 1

        terminated = self._step_count >= self.cfg.max_steps
        return obs, reward, terminated, False, {"cosine_sim": sim}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _take_glimpse(self, vp: Viewpoint) -> tuple[np.ndarray, float]:
        """Step the model with a viewpoint, return (obs, cosine_sim)."""
        # Fixed by Codex on 2026-05-29
        # Problem: Type checkers saw episode tensors as Optional inside _take_glimpse.
        # Solution: Guard the reset-initialized tensors before passing them to tensor APIs.
        # Result: sample_at_viewpoint and cosine_similarity receive concrete Tensor values.
        assert self._image is not None, "Call reset() before taking a glimpse."
        assert self._teacher_cls is not None, "Call reset() before computing reward."
        image = self._image
        teacher_cls = self._teacher_cls

        glimpse = sample_at_viewpoint(
            spatial=image,
            viewpoint=vp,
            glimpse_size_px=self.cfg.glimpse_size_px,
        )
        with torch.inference_mode():
            out = self._model(glimpse=glimpse, state=self._state, viewpoint=vp)

        self._state = out.state

        # recurrent_cls: [1, 1, 768] → [1, 768]
        cls = out.state.recurrent_cls.squeeze(1).float()

        # Cosine similarity between canvas CLS and frozen teacher CLS
        sim = float(F.cosine_similarity(cls, teacher_cls, dim=-1).item())

        obs = cls.squeeze(0).cpu().numpy().astype(np.float32)
        return obs, sim

    def _action_to_viewpoint(self, action: np.ndarray) -> Viewpoint:
        """
        Map raw action [-1, 1]^3 to a Viewpoint.
        action[0:2] → centers (cx, cy) in [-1, 1]
        action[2]   → scale in (0, 1], mapped via (action[2] + 1) / 2
        """
        cx, cy, s_raw = float(action[0]), float(action[1]), float(action[2])
        scale = (s_raw + 1.0) / 2.0  # remap [-1, 1] → [0, 1]
        scale = max(scale, 0.05)       # avoid degenerate zero-scale glimpses

        centers = torch.tensor([[cx, cy]], device=self.device)
        scales = torch.tensor([scale], device=self.device)
        return Viewpoint(centers=centers, scales=scales)
