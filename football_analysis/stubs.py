"""Placeholder stages, so the pipeline runs before the real ones exist.

None of this is the system.  The YOLO detector, the real tracker and the real
event logic are being built separately; these three classes exist so that the
plumbing around them -- reading, resizing, timing, drawing, writing, the JSON
schema, the CLI -- can be exercised and tested today, and so that a missing
stage degrades to "no events" rather than to a crash.

They are deliberately shallow:

``StubDetector``       motion blobs, not object recognition.  It has no idea
                       what a football is; it reports big movers as players and
                       small fast movers as a ball, which is right often enough
                       to prove the wiring and wrong often enough that nobody
                       will mistake it for the real thing.
``GreedyIouTracker``   a genuine, simple tracker: greedy IoU association with
                       age-out.  Good enough to be a fallback and a baseline to
                       measure a real tracker against.
``StubEventDetector``  proximity only.  It emits ``ball_touch`` and
                       ``possession_change`` and nothing else -- it does not
                       attempt tackles, tricks, shots or goals, because those
                       need the real logic.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from football_analysis.config import Config
from football_analysis.events import Event, EventType
from football_analysis.interfaces import BaseDetector, BaseEventDetector, BaseTracker
from football_analysis.state import FrameState
from football_analysis.types import BALL, BBox, Detection, PLAYER, Point, Track, VideoFrame

__all__ = [
    "StubDetector",
    "GreedyIouTracker",
    "StubEventDetector",
    "NullDetector",
    "NullEventDetector",
]


class NullDetector(BaseDetector):
    """Finds nothing.  The honest default when no weights are available."""

    def detect(self, frame: VideoFrame) -> list[Detection]:
        return []

    def describe(self) -> dict[str, Any]:
        return {"name": "NullDetector", "placeholder": True}


class StubDetector(BaseDetector):
    """Motion-blob placeholder standing in for the YOLO detector."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        min_area: int = 120,
        max_area_fraction: float = 0.35,
    ) -> None:
        self.config = config or Config()
        self.min_area = min_area
        self.max_area_fraction = max_area_fraction
        """A blob covering more than this much of the frame is the background
        model warming up or the exposure shifting, not a person."""
        self._background = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=32, detectShadows=False
        )

    def reset(self) -> None:
        self._background = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=32, detectShadows=False
        )

    def detect(self, frame: VideoFrame) -> list[Detection]:
        mask = self._background.apply(frame.image)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = float(frame.image.shape[0] * frame.image.shape[1])
        max_area = frame_area * self.max_area_fraction

        blobs: list[tuple[float, BBox]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            blobs.append((float(area), BBox.from_xywh(x, y, w, h)))

        if not blobs:
            return []

        blobs.sort(key=lambda item: item[0], reverse=True)
        detections: list[Detection] = []

        # The two biggest movers are called players; the biggest remaining
        # roughly-square blob is called the ball.
        for _, box in blobs[: self.config.tracking.max_players]:
            detections.append(
                Detection(class_name=PLAYER, bbox=box, confidence=0.50,
                          attributes={"placeholder": True})
            )
        for _, box in blobs[self.config.tracking.max_players:]:
            aspect = box.width / box.height if box.height else 99.0
            if 0.6 <= aspect <= 1.6:
                detections.append(
                    Detection(class_name=BALL, bbox=box, confidence=0.30,
                              attributes={"placeholder": True})
                )
                break
        return detections

    def describe(self) -> dict[str, Any]:
        return {"name": "StubDetector", "placeholder": True, "method": "MOG2 motion blobs"}


class GreedyIouTracker(BaseTracker):
    """Greedy IoU association with age-out, per class.

    Tracks are matched to detections of the same class, best IoU first.  A
    track that goes unmatched is kept alive for ``max_age_frames`` (the ball
    gets its own, shorter budget) and then dropped.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._tracks: list[Track] = []
        self._next_id = 1
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
        self._last_timestamp = None

    def _max_age(self, class_name: str) -> int:
        cfg = self.config.tracking
        return cfg.ball_max_age_frames if class_name == BALL else cfg.max_age_frames

    def _claim_label(self, class_name: str) -> str | None:
        """Take a free name from the pool, so a 1v1 never grows a third player.

        A player who was briefly lost keeps their name, because their track is
        still alive (just unmatched) and still holds the label.
        """
        if class_name != PLAYER:
            return None
        in_use = {
            t.label for t in self._tracks if t.class_name == PLAYER and t.label
        }
        for label in self.config.tracking.player_labels:
            if label not in in_use:
                return label
        return f"Player {len(in_use) + 1}"

    def update(self, frame: VideoFrame, detections: list[Detection]) -> list[Track]:
        dt = None
        if self._last_timestamp is not None:
            dt = frame.timestamp_s - self._last_timestamp
        self._last_timestamp = frame.timestamp_s

        unmatched = list(range(len(detections)))
        matched: set[int] = set()

        # Score every (track, detection) pair of the same class, then take them
        # greedily in descending IoU.
        pairs: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx in unmatched:
                detection = detections[d_idx]
                if detection.class_name != track.class_name:
                    continue
                iou = track.bbox.iou(detection.bbox)
                if iou >= self.config.tracking.iou_threshold:
                    pairs.append((iou, t_idx, d_idx))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, t_idx, d_idx in pairs:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
            matched.add(d_idx)

            track = self._tracks[t_idx]
            detection = detections[d_idx]
            if dt and dt > 0:
                previous = track.bbox.center
                current = detection.bbox.center
                track.velocity = Point(
                    (current.x - previous.x) / dt, (current.y - previous.y) / dt
                )
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.frame_index = frame.index
            track.timestamp_s = frame.timestamp_s
            track.age += 1
            track.time_since_update = 0

        for t_idx, track in enumerate(self._tracks):
            if t_idx not in used_tracks:
                track.time_since_update += 1
                track.age += 1

        for d_idx, detection in enumerate(detections):
            if d_idx in matched:
                continue
            if detection.class_name == PLAYER:
                # Every player track that has not aged out counts, not just the
                # ones matched this frame: a player who is briefly occluded is
                # still one of the two people on the pitch.
                live_players = sum(1 for t in self._tracks if t.class_name == PLAYER)
                if live_players >= self.config.tracking.max_players:
                    continue
            track = Track(
                track_id=self._next_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                confidence=detection.confidence,
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                attributes=dict(detection.attributes),
            )
            track.label = self._claim_label(detection.class_name)
            self._next_id += 1
            self._tracks.append(track)

        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self._max_age(t.class_name)
        ]
        # Snapshots, not the live objects: an event detector that keeps a track
        # to compare against a later frame must not find it mutated underneath.
        return [
            replace(t, attributes=dict(t.attributes))
            for t in self._tracks
            if t.time_since_update == 0
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "name": "GreedyIouTracker",
            "iou_threshold": self.config.tracking.iou_threshold,
            "max_age_frames": self.config.tracking.max_age_frames,
        }


class NullEventDetector(BaseEventDetector):
    """Calls nothing.  The default when the real event logic is absent."""

    def update(self, state: FrameState) -> list[Event]:
        return []

    def describe(self) -> dict[str, Any]:
        return {"name": "NullEventDetector", "placeholder": True}


class StubEventDetector(BaseEventDetector):
    """Proximity-only placeholder: ball touches and possession changes.

    It deliberately does not attempt tackles, tricks, shots or goals.  Those
    require the real logic, and a plausible-looking wrong answer is worse than
    no answer.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._holder: int | None = None
        self._holder_since: float | None = None
        self._close_frames = 0
        self._candidate: int | None = None
        self._last_touch_s: float | None = None

    def reset(self) -> None:
        self._holder = None
        self._holder_since = None
        self._close_frames = 0
        self._candidate = None
        self._last_touch_s = None

    def update(self, state: FrameState) -> list[Event]:
        if state.ball is None or not state.players:
            self._close_frames = 0
            self._candidate = None
            return []

        # A coasted position is not a sighting: calling a touch off one invents
        # an event at a place nobody ever saw the ball.
        if state.ball.interpolated:
            return []

        nearest = state.nearest_player_to_ball()
        if nearest is None:
            return []
        player, distance = nearest

        if distance > self.config.events.possession_distance_px:
            self._close_frames = 0
            self._candidate = None
            return []

        if self._candidate == player.track_id:
            self._close_frames += 1
        else:
            self._candidate = player.track_id
            self._close_frames = 1

        if self._close_frames < self.config.events.possession_min_frames:
            return []

        events: list[Event] = []
        confidence = float(min(0.6, 0.3 + 0.1 * self._close_frames))
        ball_track_ids = [state.ball.track_id] if state.ball.track_id else []

        gap_ok = (
            self._last_touch_s is None
            or state.timestamp_s - self._last_touch_s >= self.config.events.min_gap_s
        )
        if gap_ok:
            events.append(
                Event(
                    type=EventType.BALL_TOUCH,
                    timestamp_s=state.timestamp_s,
                    frame_index=state.frame_index,
                    confidence=confidence,
                    player_id=player.player_id,
                    track_ids=[player.track_id, *ball_track_ids],
                    detail={"distance_px": round(distance, 1), "placeholder": True},
                    source="stubs.StubEventDetector",
                )
            )
            self._last_touch_s = state.timestamp_s

        if self._holder is not None and self._holder != player.track_id:
            events.append(
                Event(
                    type=EventType.POSSESSION_CHANGE,
                    timestamp_s=state.timestamp_s,
                    frame_index=state.frame_index,
                    confidence=confidence,
                    player_id=player.player_id,
                    track_ids=[player.track_id, self._holder],
                    detail={"from_track_id": self._holder, "placeholder": True},
                    source="stubs.StubEventDetector",
                )
            )
        if self._holder != player.track_id:
            self._holder = player.track_id
            self._holder_since = state.timestamp_s
        return events

    def describe(self) -> dict[str, Any]:
        return {
            "name": "StubEventDetector",
            "placeholder": True,
            "emits": [EventType.BALL_TOUCH.value, EventType.POSSESSION_CHANGE.value],
        }
