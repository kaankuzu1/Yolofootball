"""Shared detection vocabulary.

Every backend maps its own class names onto these labels so that downstream
code (tracking, event detection) only ever sees one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

BALL = "ball"
PLAYER = "player"
GOALKEEPER = "goalkeeper"
REFEREE = "referee"
GOAL = "goal"

CLASSES = (BALL, PLAYER, GOALKEEPER, REFEREE, GOAL)


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame.

    ``xyxy`` is in pixel coordinates of the frame as passed to ``detect``.
    """

    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)
