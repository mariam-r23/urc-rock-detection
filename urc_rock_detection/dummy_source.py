"""Dummy camera source: synthetic rock/crater scenes with exact ground-truth boxes (no ROS).

Each scene is held for ``hold`` consecutive frames (like a rover that has not moved yet), so the node's
temporal confirmation (min_hits) behaves as it would on a real stream.
IMPORTANT: these scenes come from the same generator the detector was tuned on, so the detection rate
measured on them is optimistic. They stand in for the team's rock-hazard bag, they do not replace it.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .synthetic import generate_scene

Box = List[int]  # [x, y, w, h]


class DummySource:
    def __init__(self, n_scenes: int = 40, hold: int = 5,
                 levels: Sequence[str] = ("easy", "medium"), seed0: int = 1000):
        if n_scenes < 1 or hold < 1 or not levels:
            raise ValueError("n_scenes, hold and levels must be positive / non-empty")
        self.n_scenes, self.hold = int(n_scenes), int(hold)
        self.levels, self.seed0 = list(levels), int(seed0)
        self._cache: Tuple[int, np.ndarray, List[Box]] = (-1, None, [])

    def __len__(self) -> int:
        return self.n_scenes * self.hold

    def scene(self, k: int) -> Tuple[np.ndarray, List[Box]]:
        """BGR image and ground-truth [x, y, w, h] boxes of scene k."""
        if k == self._cache[0]:
            return self._cache[1], self._cache[2]
        level = self.levels[k % len(self.levels)]
        img, gt = generate_scene(self.seed0 + k, level)
        boxes = [[int(x), int(y), int(w), int(h)] for (_label, x, y, w, h) in gt]
        self._cache = (k, img, boxes)
        return img, boxes

    def frame(self, i: int) -> Tuple[np.ndarray, List[Box]]:
        """Frame i of the stream (i in [0, len))."""
        return self.scene(i // self.hold)

    def labels_by_frame(self) -> dict:
        """{"0": [[x,y,w,h],...], ...} keyed by frame index (format used by `evaluate bag --labels`)."""
        return {str(i): self.frame(i)[1] for i in range(len(self))}
