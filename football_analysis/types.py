"""Core value types shared by every stage of the pipeline.

Coordinate convention
---------------------
Every bounding box in this package is expressed in the coordinate space of
``VideoFrame.image`` -- that is, *after* any resizing the reader applied.
``VideoFrame.scale`` records how that space relates to the source clip, so a
consumer that needs source pixels can divide by it.  Keeping one space for
detection, tracking, event logic and drawing removes a whole class of
off-by-a-resize bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "BBox",
    "Point",
    "VideoFrame",
    "Detection",
    "Track",
    "ObjectClass",
    "BALL",
    "PLAYER",
    "GOAL",
]

# The three object classes the system is required to find on the pitch.  They
# are plain strings so that a detector can pass through whatever its model
# emits without importing an enum.
BALL = "ball"
PLAYER = "player"
GOAL = "goal"
ObjectClass = str


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Point") -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass(frozen=True)
class BBox:
    """An axis-aligned box, ``x1 <= x2`` and ``y1 <= y2``, in frame pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"degenerate bbox: {self!r}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Where the object meets the ground -- the useful anchor for players."""
        return Point((self.x1 + self.x2) / 2.0, self.y2)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))

    def scaled(self, factor: float) -> "BBox":
        return BBox(self.x1 * factor, self.y1 * factor,
                    self.x2 * factor, self.y2 * factor)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - inter
        return float(inter / union) if union > 0 else 0.0

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(x, y, x + w, y + h)

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class VideoFrame:
    """One decoded frame plus everything needed to place it in time."""

    index: int
    """Zero-based position of this frame in the *source* clip."""

    timestamp_s: float
    """Presentation time in seconds from the start of the source clip."""

    image: np.ndarray
    """BGR pixels, possibly resized from the source."""

    source_size: tuple[int, int]
    """``(width, height)`` of the clip on disk."""

    scale: float = 1.0
    """``processed_size / source_size``.  1.0 when no resizing happened."""

    timestamp_is_exact: bool = True
    """False when the timestamp was derived from a nominal fps, not the container."""

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)

    def to_source_bbox(self, box: BBox) -> BBox:
        """Map a box from this frame's space back to source-clip pixels."""
        return box.scaled(1.0 / self.scale) if self.scale != 1.0 else box


@dataclass
class Detection:
    """A single-frame observation of one object, produced by a detector."""

    class_name: ObjectClass
    bbox: BBox
    confidence: float
    class_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "confidence": round(float(self.confidence), 4),
        }
        if self.class_id is not None:
            payload["class_id"] = self.class_id
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload


@dataclass
class Track:
    """A detection that has been given a stable identity across frames."""

    track_id: int
    class_name: ObjectClass
    bbox: BBox
    confidence: float
    frame_index: int
    timestamp_s: float
    velocity: Point | None = None
    """Motion in pixels per second, when the tracker can estimate it."""

    age: int = 0
    """How many frames this track has existed for."""

    time_since_update: int = 0
    """Frames since this track was last matched to a detection (0 = fresh)."""

    label: str | None = None
    """Human-facing name, e.g. ``"Player 1"``.  Falls back to the id."""

    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        return self.time_since_update == 0

    @property
    def display_name(self) -> str:
        return self.label or f"{self.class_name}#{self.track_id}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "confidence": round(float(self.confidence), 4),
            "frame_index": self.frame_index,
            "timestamp_s": round(float(self.timestamp_s), 3),
        }
        if self.velocity is not None:
            payload["velocity"] = {"vx": self.velocity.x, "vy": self.velocity.y}
        if self.label:
            payload["label"] = self.label
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload
