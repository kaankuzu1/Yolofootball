"""Where recordings and their results live, and the goal corners that go with them.

Every clip gets its results beside it, named after it::

    2026-09-24 18.05.12.mp4             the recording
    2026-09-24 18.05.12.goal.json       the goal's four corners and its size
    2026-09-24 18.05.12.events.json     the timeline (same JSON as the CLI's)
    2026-09-24 18.05.12.annotated.mp4   the clip with boxes and captions drawn on
    2026-09-24 18.05.12.states.jsonl    the per-frame record, for --replay

so a folder of sessions can be browsed in Finder, and any clip reopened in the
app picks its goal and results back up. The goal file is the same shape
``scripts/pick_goal_corners.py --save`` writes, plus the goal's size and the
frame size the corners were clicked on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from football_analysis.detection.goal import CORNER_ORDER

Corners = list[list[float]]


def default_output_dir() -> Path:
    """``~/Movies/Football Analysis`` on a Mac, the platform's videos folder elsewhere."""
    try:
        from PySide6.QtCore import QStandardPaths

        movies = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MoviesLocation)
    except Exception:  # pragma: no cover - Qt missing or not initialised
        movies = ""
    base = Path(movies) if movies else Path.home() / "Movies"
    return base / "Football Analysis"


def new_recording_path(folder: str | Path, now: datetime | None = None) -> Path:
    """A fresh, readable, sortable file name that does not overwrite anything."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H.%M.%S")
    folder = Path(folder)
    path = folder / f"{stamp}.mp4"
    n = 2
    while path.exists():
        path = folder / f"{stamp} ({n}).mp4"
        n += 1
    return path


def _sibling(clip: str | Path, suffix: str) -> Path:
    clip = Path(clip)
    return clip.with_name(f"{clip.stem}{suffix}")


def goal_path(clip: str | Path) -> Path:
    return _sibling(clip, ".goal.json")


def events_path(clip: str | Path) -> Path:
    return _sibling(clip, ".events.json")


def annotated_path(clip: str | Path) -> Path:
    return _sibling(clip, ".annotated.mp4")


def states_path(clip: str | Path) -> Path:
    return _sibling(clip, ".states.jsonl")


def csv_path(clip: str | Path) -> Path:
    return _sibling(clip, ".events.csv")


@dataclass
class GoalSetup:
    """The goal as clicked on one frame: corners in that frame's pixels."""

    corners: Corners
    """Left post base, right post base, right crossbar end, left crossbar end."""

    frame_size: tuple[int, int]
    """``(width, height)`` of the frame the corners were clicked on."""

    size_m: tuple[float, float] = (3.0, 2.0)
    """Goal mouth width and height in metres."""

    def scaled_to(self, frame_size: tuple[int, int]) -> Corners:
        """The corners on a frame of another size, e.g. a 1080p recording of a 720p click."""
        sx = frame_size[0] / float(self.frame_size[0])
        sy = frame_size[1] / float(self.frame_size[1])
        return [[x * sx, y * sy] for x, y in self.corners]

    def to_dict(self) -> dict:
        return {
            "corner_order": list(CORNER_ORDER),
            "corners": [[round(x, 1), round(y, 1)] for x, y in self.corners],
            "frame_size": list(self.frame_size),
            "goal_size_m": list(self.size_m),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalSetup":
        corners = [[float(x), float(y)] for x, y in data["corners"]]
        if len(corners) != 4:
            raise ValueError("a goal needs four corners")
        size = data.get("frame_size")
        if not size:
            # A file from pick_goal_corners.py carries no frame size; its
            # corners are at the clip's own resolution, which the caller knows.
            raise ValueError("goal file has no frame_size")
        return cls(
            corners=corners,
            frame_size=(int(size[0]), int(size[1])),
            size_m=tuple(float(v) for v in data.get("goal_size_m", (3.0, 2.0))),
        )


def save_goal(clip: str | Path, goal: GoalSetup) -> Path:
    path = goal_path(clip)
    path.write_text(json.dumps(goal.to_dict(), indent=1))
    return path


def load_goal(clip: str | Path, clip_size: tuple[int, int] | None = None) -> GoalSetup | None:
    """The goal saved beside ``clip``, or ``None``.

    ``clip_size`` fills in the frame size for a file written by
    ``pick_goal_corners.py``, whose corners are at the clip's own resolution.
    """
    path = goal_path(clip)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if "frame_size" not in data and clip_size is not None:
            data["frame_size"] = list(clip_size)
        return GoalSetup.from_dict(data)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def processed_corners(corners: Corners, source_size: tuple[int, int],
                      resize_width: int | None, resize_height: int | None = None) -> Corners:
    """Corners at the clip's resolution, moved into the pipeline's processed frame.

    The same conversion the CLI does for ``--goal-corners``: the pipeline
    resizes every frame to ``video.resize_width`` before anything reads it,
    and ``geometry.goal_corners_px`` is in that resized space.
    """
    if resize_width is None or not source_size[0]:
        return [list(p) for p in corners]
    sx = resize_width / float(source_size[0])
    sy = resize_height / float(source_size[1]) if resize_height else sx
    return [[x * sx, y * sy] for x, y in corners]
