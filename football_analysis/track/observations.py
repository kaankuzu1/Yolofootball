"""One internal shape for a detection, whichever detector produced it.

Two detection types exist in the package today: the pipeline's
:class:`football_analysis.types.Detection` (``class_name``, ``bbox``) and the
detection module's own (``label``, ``xyxy``).  This layer accepts either, or a
plain dict, and converts at the door so nothing downstream has to care.

Goal keypoints travel in ``attributes["keypoints"]`` (pipeline type) or a
``keypoints`` attribute (anything else): four ``(x, y)`` or ``(x, y, conf)``
rows in the order *left post base, right post base, right crossbar end, left
crossbar end*, left and right as they appear in the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

__all__ = ["Obs", "to_obs", "to_obs_list"]


@dataclass
class Obs:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    keypoints: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def diameter(self) -> float:
        return 0.5 * (self.w + self.h)

    @property
    def foot(self) -> tuple[float, float]:
        return (self.cx, self.y2)

    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _keypoints(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 4 or arr.shape[1] < 2:
        return None
    return arr


def to_obs(det: Any) -> Obs:
    """Convert any supported detection object into an :class:`Obs`."""
    if isinstance(det, Obs):
        return det
    if isinstance(det, dict):
        cls = det.get("cls") or det.get("class_name") or det.get("label")
        box = det.get("xyxy") or det.get("bbox")
        if isinstance(box, dict):
            box = (box["x1"], box["y1"], box["x2"], box["y2"])
        conf = det.get("conf", det.get("confidence", 1.0))
        kp = _keypoints(det.get("keypoints"))
        return Obs(str(cls).lower(), *map(float, box), float(conf), kp)
    if hasattr(det, "class_name") and hasattr(det, "bbox"):
        attrs = getattr(det, "attributes", None) or {}
        b = det.bbox
        return Obs(str(det.class_name).lower(), float(b.x1), float(b.y1), float(b.x2),
                   float(b.y2), float(det.confidence), _keypoints(attrs.get("keypoints")))
    if hasattr(det, "label") and hasattr(det, "xyxy"):
        x1, y1, x2, y2 = det.xyxy
        return Obs(str(det.label).lower(), float(x1), float(y1), float(x2), float(y2),
                   float(det.confidence), _keypoints(getattr(det, "keypoints", None)))
    raise TypeError(f"unsupported detection type: {type(det).__name__}")


def to_obs_list(dets: Iterable[Any]) -> list[Obs]:
    return [to_obs(d) for d in dets]
