"""Draw boxes, trails and event captions onto a frame.

:class:`FrameAnnotator` keeps the small amount of state an overlay needs --
the ball's recent positions for its trail, and which events are recent enough
to still be captioned -- so the pipeline can hand it one frame at a time.

It always draws on a copy.  The frame the detector saw is never modified,
because a stage running after the annotator would otherwise see painted pixels.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import cv2
import numpy as np

from football_analysis.config import OutputConfig
from football_analysis.events import Event
from football_analysis.types import BALL, Detection, Track, VideoFrame
from football_analysis.viz.palette import (
    BANNER_COLOR,
    TEXT_COLOR,
    color_for_class,
    color_for_track,
)

__all__ = ["FrameAnnotator"]

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _format_clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(minutes):02d}:{rest:06.3f}"


def _scaled_font(frame_width: int) -> tuple[float, int]:
    """Font scale and thickness that stay readable at any output size."""
    scale = max(0.4, min(1.1, frame_width / 1280.0 * 0.6))
    thickness = max(1, int(round(scale * 2)))
    return scale, thickness


class FrameAnnotator:
    """Stateful overlay renderer for one run."""

    # A trail segment is drawn only when consecutive ball sightings are close
    # enough in time and space to plausibly be the same ball travelling. On
    # crowded footage the ball track jumps between objects, and joining those
    # jumps draws confident lines across the pitch that are simply wrong.
    _TRAIL_MAX_GAP_S = 0.25
    _TRAIL_MAX_JUMP_FRACTION = 0.25

    def __init__(self, config: OutputConfig | None = None) -> None:
        self.config = config or OutputConfig()
        self._trail: deque[tuple[int, int, float]] = deque(
            maxlen=max(1, self.config.trail_length)
        )
        self._recent_events: list[Event] = []

    def reset(self) -> None:
        self._trail.clear()
        self._recent_events.clear()

    def note_events(self, events: Iterable[Event]) -> None:
        """Register events so they appear in the caption banner for a while."""
        self._recent_events.extend(events)

    # -- primitives -------------------------------------------------------

    def _draw_box(
        self,
        canvas: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
        color: tuple[int, int, int],
        caption: str | None,
        font_scale: float,
        thickness: int,
    ) -> None:
        x1, y1, x2, y2 = box_xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if not caption or not self.config.draw_labels:
            return

        (text_w, text_h), baseline = cv2.getTextSize(
            caption, _FONT, font_scale, thickness
        )
        # Put the caption above the box, or inside it when there is no room.
        top = y1 - text_h - baseline - 2
        if top < 0:
            top = y1 + 2
        cv2.rectangle(
            canvas,
            (x1, top),
            (x1 + text_w + 6, top + text_h + baseline + 2),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            caption,
            (x1 + 3, top + text_h + 1),
            _FONT,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_trail(self, canvas: np.ndarray, thickness: int) -> None:
        points = list(self._trail)
        if len(points) < 2:
            return
        max_jump = canvas.shape[1] * self._TRAIL_MAX_JUMP_FRACTION

        for i in range(1, len(points)):
            x1, y1, t1 = points[i - 1]
            x2, y2, t2 = points[i]
            if t2 - t1 > self._TRAIL_MAX_GAP_S:
                continue  # the ball was lost in between; do not invent a path
            if float(np.hypot(x2 - x1, y2 - y1)) > max_jump:
                continue  # a jump this big is a tracking error, not a kick

            # Older segments are thinner, so the trail reads as a direction.
            weight = i / len(points)
            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                color_for_class(BALL),
                max(1, int(round(thickness * weight))),
                cv2.LINE_AA,
            )

    def _draw_banner(
        self,
        canvas: np.ndarray,
        timestamp_s: float,
        font_scale: float,
        thickness: int,
    ) -> None:
        window = self.config.event_banner_seconds
        self._recent_events = [
            e for e in self._recent_events
            if timestamp_s - float(e.timestamp_s) <= window
        ]
        lines = [
            f"{e.clock}  {e.label()}  ({e.confidence:.2f})"
            for e in self._recent_events[-4:]
        ]
        if not lines:
            return

        height, width = canvas.shape[:2]
        (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, font_scale, thickness)
        line_h = text_h + baseline + 6
        box_h = line_h * len(lines) + 8
        top = height - box_h - 8

        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, top), (width - 8, height - 8), BANNER_COLOR, -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (16, top + line_h * (i + 1) - baseline),
                _FONT,
                font_scale,
                TEXT_COLOR,
                thickness,
                cv2.LINE_AA,
            )

    def _draw_clock(
        self, canvas: np.ndarray, frame: VideoFrame, font_scale: float, thickness: int
    ) -> None:
        text = _format_clock(frame.timestamp_s)
        if not frame.timestamp_is_exact:
            text += " ~"  # marks a timestamp derived from the nominal fps
        (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
        cv2.rectangle(
            canvas, (8, 8), (8 + text_w + 10, 8 + text_h + baseline + 8), BANNER_COLOR, -1
        )
        cv2.putText(
            canvas, text, (13, 8 + text_h + 4), _FONT, font_scale,
            TEXT_COLOR, thickness, cv2.LINE_AA,
        )

    # -- entry point ------------------------------------------------------

    def annotate(
        self,
        frame: VideoFrame,
        *,
        tracks: Sequence[Track] = (),
        detections: Sequence[Detection] = (),
        events: Iterable[Event] = (),
    ) -> np.ndarray:
        """Return a drawn-on copy of ``frame.image``.

        Tracks are preferred over detections when both are given, because a
        track carries an identity worth showing.  Untracked detections are only
        drawn when there are no tracks at all -- otherwise the frame doubles up.
        """
        canvas = frame.image.copy()
        font_scale, thickness = _scaled_font(canvas.shape[1])
        self.note_events(events)

        if self.config.draw_boxes:
            if tracks:
                for track in tracks:
                    color = color_for_track(track.track_id, track.class_name)
                    caption = f"{track.display_name} {track.confidence:.2f}"
                    self._draw_box(
                        canvas, track.bbox.as_int_tuple(), color, caption,
                        font_scale, thickness,
                    )
            elif detections:
                for detection in detections:
                    color = color_for_class(detection.class_name)
                    caption = f"{detection.class_name} {detection.confidence:.2f}"
                    self._draw_box(
                        canvas, detection.bbox.as_int_tuple(), color, caption,
                        font_scale, thickness,
                    )

        if self.config.draw_ball_trail:
            ball = next((t for t in tracks if t.class_name == BALL), None)
            if ball is None:
                ball_detection = next(
                    (d for d in detections if d.class_name == BALL), None
                )
                center = ball_detection.bbox.center if ball_detection else None
            else:
                center = ball.bbox.center
            if center is not None:
                self._trail.append(
                    (int(round(center.x)), int(round(center.y)), frame.timestamp_s)
                )
            self._draw_trail(canvas, thickness + 1)

        if self.config.draw_clock:
            self._draw_clock(canvas, frame, font_scale, thickness)
        if self.config.draw_events:
            self._draw_banner(canvas, frame.timestamp_s, font_scale, thickness)
        return canvas
