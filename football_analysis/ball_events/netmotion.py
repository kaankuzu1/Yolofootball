"""Net ripple: frame-difference energy over the goal net (plan §4.4, condition 4).

With a fixed camera, the net is the one part of the goal that moves, and it
moves when it takes a ball.  So the mean absolute grey-level change between
consecutive frames, over the net and nowhere else, spikes at a goal.  Almost
nobody uses this signal because it only exists when the camera does not move.

Pixels under the ball and under the players are masked out before averaging,
so neither the ball flying in nor a player walking past the posts reads as
the net moving.

This is the one part of the goal rule that needs pixels, and the event rules
never see pixels -- they read the per-frame record.  So the reading is taken
where the frames are, once per frame, and stored on the record as
``FrameState.attributes["net_motion"]``::

    meter = NetMotionMeter()
    for frame in reader:
        ...
        state = state_from_tracks(frame, tracks)
        state.attributes["net_motion"] = meter.measure(frame.image, state)

The meter keeps the last goal box it saw, since the goal is static and a
detector can miss it for a frame.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

__all__ = ["NetMotionMeter", "net_region"]


def net_region(bbox: tuple[float, float, float, float] | None,
               quad: np.ndarray | None, margin: float) -> np.ndarray | None:
    """The polygon the net occupies in the image: the mouth grown by ``margin``."""
    if quad is not None and len(quad) == 4:
        poly = np.asarray(quad, float)
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        poly = np.array([[x1, y2], [x2, y2], [x2, y1], [x1, y1]], float)
    else:
        return None
    c = poly.mean(axis=0)
    return c + (poly - c) * (1.0 + margin)


class NetMotionMeter:
    """Measures net motion one frame at a time.

    Parameters
    ----------
    margin:
        Grow the goal mouth by this fraction to cover the net behind it.
    scale:
        Work at this fraction of the frame's resolution.  A net ripple is a
        large, low-frequency change; quarter resolution sees it fine.
    min_pixels:
        Fewer unmasked pixels than this (at working resolution) = no reading.
    """

    def __init__(self, margin: float = 0.25, scale: float = 0.25, min_pixels: int = 32,
                 ball_mask_diameters: float = 1.5) -> None:
        self.margin = margin
        self.scale = scale
        self.min_pixels = min_pixels
        self.ball_mask_diameters = ball_mask_diameters
        self._prev: np.ndarray | None = None
        self._goal: tuple[tuple[float, float, float, float] | None, np.ndarray | None] = (None, None)
        self._last_ball: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self._prev = None
        self._goal = (None, None)
        self._last_ball = None

    def measure(self, image: np.ndarray | None, state: Any) -> float | None:
        """The net-motion reading for this frame, or ``None`` if there is none.

        ``state`` is read by attribute (a :class:`football_analysis.state.FrameState`
        or anything shaped like one): its goal, ball and players.
        """
        if image is None or image.size == 0:
            self._prev = None
            return None
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        sw, sh = max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale)))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.int16)
        prev, self._prev = self._prev, small
        self._remember_goal(state)
        current = self._ball_disc(getattr(state, "ball", None), h)
        last, self._last_ball = self._last_ball, current
        if prev is None or prev.shape != small.shape:
            return None
        bbox, quad = self._goal
        region = net_region(bbox, quad, self.margin)
        if region is None:
            return None

        mask = np.zeros((sh, sw), np.uint8)
        cv2.fillPoly(mask, [np.round(region * self.scale).astype(np.int32)], 1)
        for p in getattr(state, "players", None) or []:
            x1, y1, x2, y2 = _box(p.bbox)
            cv2.rectangle(mask, _pt(x1, y1, self.scale), _pt(x2, y2, self.scale), 0, -1)
        # Mask the ball where it is now and where it was a frame ago: both
        # places changed because the ball moved, not because the net did.
        for disc in (current, last):
            if disc is not None:
                cx, cy, r = disc
                cv2.circle(mask, _pt(cx, cy, self.scale), int(round(r * self.scale)) + 1, 0, -1)

        used = mask.astype(bool)
        if int(used.sum()) < self.min_pixels:
            return None
        diff = np.abs(small - prev)
        return float(diff[used].mean())

    def _ball_disc(self, ball: Any, frame_height: int) -> tuple[float, float, float] | None:
        if ball is None or getattr(ball, "position", None) is None:
            return None
        if getattr(ball, "bbox", None) is not None:
            x1, y1, x2, y2 = _box(ball.bbox)
            r = self.ball_mask_diameters * max(x2 - x1, y2 - y1)
        else:
            r = 0.05 * frame_height
        return (float(ball.position.x), float(ball.position.y), r)

    def _remember_goal(self, state: Any) -> None:
        goal = getattr(state, "goal", None)
        if goal is None:
            return
        quad = getattr(goal, "quad", None) or []
        if len(quad) == 4:
            self._goal = (None, np.array([[p.x, p.y] for p in quad], float))
        elif getattr(goal, "bbox", None) is not None and self._goal[1] is None:
            self._goal = (_box(goal.bbox), None)


def _box(b: Any) -> tuple[float, float, float, float]:
    if hasattr(b, "x1"):
        return (float(b.x1), float(b.y1), float(b.x2), float(b.y2))
    x1, y1, x2, y2 = b
    return (float(x1), float(y1), float(x2), float(y2))


def _pt(x: float, y: float, s: float) -> tuple[int, int]:
    return (int(round(x * s)), int(round(y * s)))
