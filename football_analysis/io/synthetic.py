"""Generate a synthetic 1v1 clip.

Tests need a real video file -- one that OpenCV actually decodes, with a real
container and real timestamps -- and CI cannot depend on someone's phone
footage.  So this renders one: a green pitch shot from the side, a goal at the
right, two coloured player rectangles, and a ball that is dribbled, passed and
shot.

It is a test fixture and a demo aid, not training data.  Nothing here should
ever be used to evaluate detection quality.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

__all__ = ["make_sample_clip", "SampleClipTruth"]

_PITCH = (58, 120, 52)     # BGR, a dull grass green
_LINE = (230, 230, 230)
_P1 = (40, 60, 200)        # red kit
_P2 = (200, 120, 40)       # blue kit
_BALL = (245, 245, 245)
_NET = (240, 240, 240)


class SampleClipTruth:
    """What the generated clip contains, for tests that want to assert on it."""

    def __init__(self, duration_s: float, fps: float, size: tuple[int, int]) -> None:
        self.duration_s = duration_s
        self.fps = fps
        self.size = size
        self.pass_at_s = duration_s * 0.45
        self.shot_at_s = duration_s * 0.80

    @property
    def frame_count(self) -> int:
        return int(round(self.duration_s * self.fps))


def _ball_position(t: float, duration: float, width: int, height: int) -> tuple[float, float]:
    """Dribble left, pass across, then shoot at the goal."""
    ground = height * 0.78
    pass_at = duration * 0.45
    shot_at = duration * 0.80

    if t < pass_at:
        # Dribbled along with player 1, bobbling slightly.
        progress = t / pass_at
        x = width * (0.18 + 0.12 * progress)
        y = ground - 6 - 4 * abs(math.sin(t * 6.0))
    elif t < shot_at:
        # In flight to player 2, with an arc.
        progress = (t - pass_at) / (shot_at - pass_at)
        x = width * (0.30 + 0.32 * progress)
        y = ground - 6 - 40 * math.sin(math.pi * progress)
    else:
        # Struck at the goal, rising.
        progress = min(1.0, (t - shot_at) / (duration - shot_at))
        x = width * (0.62 + 0.33 * progress)
        y = ground - 6 - 70 * progress
    return x, y


def _player_positions(t: float, duration: float, width: int, height: int):
    ground = height * 0.78
    pass_at = duration * 0.45
    p1_x = width * (0.16 + 0.10 * min(1.0, t / pass_at)) - 4 * math.sin(t * 3.0)
    p2_progress = max(0.0, (t - pass_at * 0.5)) / max(1e-6, duration - pass_at * 0.5)
    p2_x = width * (0.70 - 0.08 * min(1.0, p2_progress))
    return (p1_x, ground), (p2_x, ground)


def make_sample_clip(
    path: str | Path,
    *,
    duration_s: float = 6.0,
    fps: float = 25.0,
    width: int = 640,
    height: int = 360,
) -> SampleClipTruth:
    """Render a clip to ``path`` and describe what is in it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    truth = SampleClipTruth(duration_s, fps, (width, height))
    writer = None
    for codec in ("mp4v", "MJPG"):
        candidate = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"no codec available to write {out}")

    ground = int(height * 0.78)
    try:
        for i in range(truth.frame_count):
            t = i / fps
            frame = np.full((height, width, 3), _PITCH, dtype=np.uint8)

            # Pitch markings and a little texture so motion detection has
            # something to work against.
            cv2.line(frame, (0, ground), (width, ground), _LINE, 2)
            cv2.line(frame, (int(width * 0.5), ground), (int(width * 0.5), height), _LINE, 1)
            frame[ground:, :] = cv2.addWeighted(
                frame[ground:, :], 0.92,
                np.full_like(frame[ground:, :], (40, 100, 40)), 0.08, 0,
            )

            # The goal, side-on at the right edge.
            goal_x1, goal_y1 = int(width * 0.88), ground - int(height * 0.30)
            goal_x2, goal_y2 = int(width * 0.99), ground
            cv2.rectangle(frame, (goal_x1, goal_y1), (goal_x2, goal_y2), _NET, 2)
            for gx in range(goal_x1, goal_x2, 8):
                cv2.line(frame, (gx, goal_y1), (gx, goal_y2), _NET, 1)
            for gy in range(goal_y1, goal_y2, 8):
                cv2.line(frame, (goal_x1, gy), (goal_x2, gy), _NET, 1)

            (p1_x, p1_y), (p2_x, p2_y) = _player_positions(t, duration_s, width, height)
            for (px, py), color in (((p1_x, p1_y), _P1), ((p2_x, p2_y), _P2)):
                half_w, body_h = int(width * 0.018), int(height * 0.20)
                cv2.rectangle(
                    frame,
                    (int(px - half_w), int(py - body_h)),
                    (int(px + half_w), int(py)),
                    color, -1,
                )
                cv2.circle(frame, (int(px), int(py - body_h - half_w)), half_w, color, -1)

            bx, by = _ball_position(t, duration_s, width, height)
            cv2.circle(frame, (int(bx), int(by)), max(3, int(width * 0.012)), _BALL, -1)

            writer.write(frame)
    finally:
        writer.release()

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"clip was not written to {out}")
    return truth
