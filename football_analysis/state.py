"""The per-frame world-state record -- the interface between the pipeline and
the event logic.

Everything upstream of events (detection, tracking, pose, geometry) writes one
:class:`FrameState` per processed frame.  Every event module reads only that
record and never touches the video.  That is what lets the event rules be
developed against a cached run in seconds instead of re-running inference each
time, and it is why the record, not the frame, is what
:class:`~football_analysis.interfaces.EventDetector` is handed.

The one field that is easy to get wrong is :attr:`BallState.interpolated`.  A
ball tracker coasts through occlusions -- a player's legs hide the ball for
five frames and the tracker keeps predicting where it must be.  Those predicted
positions are not observations, and a rule that treats them as observations
will call passes and shots off a position nobody ever saw.  So the flag travels
with the position, and a rule that cares must check it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from football_analysis.types import BALL, BBox, GOAL, PLAYER, Point, Track, VideoFrame

__all__ = [
    "STATE_SCHEMA_VERSION",
    "Keypoint",
    "PlayerState",
    "BallState",
    "GoalState",
    "FrameState",
    "StateCacheWriter",
    "read_state_cache",
    "read_state_cache_header",
    "state_from_tracks",
    "tracks_from_state",
]

STATE_SCHEMA_VERSION = "1.0"


def _point_to_dict(point: Point | None) -> dict[str, float] | None:
    return None if point is None else {"x": point.x, "y": point.y}


def _point_from_dict(payload: dict[str, float] | None) -> Point | None:
    return None if payload is None else Point(payload["x"], payload["y"])


def _bbox_from_dict(payload: dict[str, float] | None) -> BBox | None:
    if payload is None:
        return None
    return BBox(payload["x1"], payload["y1"], payload["x2"], payload["y2"])


@dataclass
class Keypoint:
    """One pose joint, in the coordinate space of the processed frame."""

    name: str
    x: float
    y: float
    confidence: float = 0.0

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Keypoint":
        return cls(
            name=payload["name"],
            x=payload["x"],
            y=payload["y"],
            confidence=payload.get("confidence", 0.0),
        )


@dataclass
class PlayerState:
    """One player as of this frame."""

    player_id: str
    """Stable, human-facing -- ``"Player 1"``.  What events refer to."""

    track_id: int
    bbox: BBox
    confidence: float
    velocity: Point | None = None
    """Pixels per second, when the tracker estimates it."""

    keypoints: list[Keypoint] = field(default_factory=list)
    """Pose joints, when a pose stage has run.  Empty otherwise."""

    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> Point:
        """Where the player meets the ground -- the anchor for distances."""
        return self.bbox.bottom_center

    def keypoint(self, name: str) -> Keypoint | None:
        return next((k for k in self.keypoints if k.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "player_id": self.player_id,
            "track_id": self.track_id,
            "bbox": self.bbox.to_dict(),
            "confidence": round(float(self.confidence), 4),
        }
        if self.velocity is not None:
            payload["velocity"] = _point_to_dict(self.velocity)
        if self.keypoints:
            payload["keypoints"] = [k.to_dict() for k in self.keypoints]
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlayerState":
        return cls(
            player_id=payload["player_id"],
            track_id=payload["track_id"],
            bbox=_bbox_from_dict(payload["bbox"]),
            confidence=payload.get("confidence", 0.0),
            velocity=_point_from_dict(payload.get("velocity")),
            keypoints=[Keypoint.from_dict(k) for k in payload.get("keypoints", [])],
            attributes=payload.get("attributes", {}),
        )


@dataclass
class BallState:
    """The ball as of this frame, observed or coasted."""

    position: Point
    confidence: float
    bbox: BBox | None = None
    velocity: Point | None = None

    interpolated: bool = False
    """True when this position was predicted through an occlusion rather than
    observed.  A rule that treats a coasted position as a sighting will invent
    events; check this before calling a touch, a pass or a shot."""

    track_id: int | None = None
    frames_since_seen: int = 0
    """How long ago the ball was last actually detected.  0 = this frame."""

    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_observed(self) -> bool:
        return not self.interpolated

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "position": _point_to_dict(self.position),
            "confidence": round(float(self.confidence), 4),
            "interpolated": bool(self.interpolated),
            "frames_since_seen": int(self.frames_since_seen),
        }
        if self.bbox is not None:
            payload["bbox"] = self.bbox.to_dict()
        if self.velocity is not None:
            payload["velocity"] = _point_to_dict(self.velocity)
        if self.track_id is not None:
            payload["track_id"] = self.track_id
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BallState":
        return cls(
            position=_point_from_dict(payload["position"]),
            confidence=payload.get("confidence", 0.0),
            bbox=_bbox_from_dict(payload.get("bbox")),
            velocity=_point_from_dict(payload.get("velocity")),
            interpolated=payload.get("interpolated", False),
            track_id=payload.get("track_id"),
            frames_since_seen=payload.get("frames_since_seen", 0),
            attributes=payload.get("attributes", {}),
        )


@dataclass
class GoalState:
    """The goal as of this frame."""

    bbox: BBox | None = None
    quad: list[Point] = field(default_factory=list)
    """The four mouth corners, once geometry has solved them.  Empty until then."""

    confidence: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"confidence": round(float(self.confidence), 4)}
        if self.bbox is not None:
            payload["bbox"] = self.bbox.to_dict()
        if self.quad:
            payload["quad"] = [_point_to_dict(p) for p in self.quad]
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoalState":
        return cls(
            bbox=_bbox_from_dict(payload.get("bbox")),
            quad=[_point_from_dict(p) for p in payload.get("quad", [])],
            confidence=payload.get("confidence", 0.0),
            attributes=payload.get("attributes", {}),
        )


@dataclass
class FrameState:
    """Everything the event logic is allowed to know about one frame."""

    frame_index: int
    timestamp_s: float
    players: list[PlayerState] = field(default_factory=list)
    ball: BallState | None = None
    goal: GoalState | None = None

    possessor_id: str | None = None
    """``player_id`` of whoever has the ball, when a possession stage has run."""

    frame_size: tuple[int, int] = (0, 0)
    """``(width, height)`` of the processed frame -- the space every box is in."""

    scale: float = 1.0
    """Processed size over source size, for mapping back to source pixels."""

    timestamp_is_exact: bool = True
    """False when the time came from a nominal fps rather than the container."""

    attributes: dict[str, Any] = field(default_factory=dict)

    def player(self, player_id: str) -> PlayerState | None:
        return next((p for p in self.players if p.player_id == player_id), None)

    def player_by_track(self, track_id: int) -> PlayerState | None:
        return next((p for p in self.players if p.track_id == track_id), None)

    @property
    def possessor(self) -> PlayerState | None:
        return self.player(self.possessor_id) if self.possessor_id else None

    def nearest_player_to_ball(self) -> tuple[PlayerState, float] | None:
        """The closest player and their distance, or ``None`` without a ball.

        Distance is measured from the player's feet, which is where the ball
        actually is when they have it.
        """
        if self.ball is None or not self.players:
            return None
        nearest = min(
            self.players, key=lambda p: p.position.distance_to(self.ball.position)
        )
        return nearest, nearest.position.distance_to(self.ball.position)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_index": self.frame_index,
            "timestamp_s": round(float(self.timestamp_s), 4),
            "players": [p.to_dict() for p in self.players],
            "frame_size": list(self.frame_size),
            "scale": self.scale,
        }
        if self.ball is not None:
            payload["ball"] = self.ball.to_dict()
        if self.goal is not None:
            payload["goal"] = self.goal.to_dict()
        if self.possessor_id is not None:
            payload["possessor_id"] = self.possessor_id
        if not self.timestamp_is_exact:
            payload["timestamp_is_exact"] = False
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameState":
        size = payload.get("frame_size", [0, 0])
        return cls(
            frame_index=payload["frame_index"],
            timestamp_s=payload["timestamp_s"],
            players=[PlayerState.from_dict(p) for p in payload.get("players", [])],
            ball=BallState.from_dict(payload["ball"]) if payload.get("ball") else None,
            goal=GoalState.from_dict(payload["goal"]) if payload.get("goal") else None,
            possessor_id=payload.get("possessor_id"),
            frame_size=(int(size[0]), int(size[1])),
            scale=payload.get("scale", 1.0),
            timestamp_is_exact=payload.get("timestamp_is_exact", True),
            attributes=payload.get("attributes", {}),
        )


def state_from_tracks(frame: VideoFrame, tracks: Sequence[Track]) -> FrameState:
    """Build the record for one frame from its tracks.

    Pose keypoints, the goal quad and the possessor are left empty here: the
    stages that produce them fill them in.  A tracker that carries extras on a
    track's ``attributes`` -- ``keypoints`` from a pose stage, ``interpolated``
    from a ball tracker that coasts -- has them picked up automatically.
    """
    players: list[PlayerState] = []
    ball: BallState | None = None
    goal: GoalState | None = None

    for track in tracks:
        if track.class_name == PLAYER:
            raw_keypoints = track.attributes.get("keypoints") or []
            players.append(
                PlayerState(
                    player_id=track.display_name,
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    velocity=track.velocity,
                    keypoints=[
                        k if isinstance(k, Keypoint) else Keypoint.from_dict(k)
                        for k in raw_keypoints
                    ],
                    attributes={
                        k: v for k, v in track.attributes.items() if k != "keypoints"
                    },
                )
            )
        elif track.class_name == BALL and ball is None:
            # A tracker that coasts should say so; if it does not, an unmatched
            # track is the honest fallback signal that this is not a sighting.
            interpolated = bool(
                track.attributes.get("interpolated", track.time_since_update > 0)
            )
            ball = BallState(
                position=track.bbox.center,
                confidence=track.confidence,
                bbox=track.bbox,
                velocity=track.velocity,
                interpolated=interpolated,
                track_id=track.track_id,
                frames_since_seen=track.time_since_update,
                attributes={
                    k: v for k, v in track.attributes.items() if k != "interpolated"
                },
            )
        elif track.class_name == GOAL and goal is None:
            raw_quad = track.attributes.get("quad") or []
            goal = GoalState(
                bbox=track.bbox,
                quad=[
                    p if isinstance(p, Point) else Point(p["x"], p["y"])
                    for p in raw_quad
                ],
                confidence=track.confidence,
                attributes={k: v for k, v in track.attributes.items() if k != "quad"},
            )

    return FrameState(
        frame_index=frame.index,
        timestamp_s=frame.timestamp_s,
        players=players,
        ball=ball,
        goal=goal,
        frame_size=frame.size,
        scale=frame.scale,
        timestamp_is_exact=frame.timestamp_is_exact,
    )


def tracks_from_state(state: FrameState) -> list[Track]:
    """Turn a record back into drawable tracks.

    The overlay draws tracks, but after refinement the records are the truth --
    so the drawing has to come from them, or the annotated clip will caption a
    player the timeline calls someone else.
    """
    tracks: list[Track] = []
    for player in state.players:
        tracks.append(
            Track(
                track_id=player.track_id,
                class_name=PLAYER,
                bbox=player.bbox,
                confidence=player.confidence,
                frame_index=state.frame_index,
                timestamp_s=state.timestamp_s,
                velocity=player.velocity,
                label=player.player_id,
                attributes=dict(player.attributes),
            )
        )
    if state.ball is not None and state.ball.bbox is not None:
        tracks.append(
            Track(
                track_id=state.ball.track_id or 0,
                class_name=BALL,
                bbox=state.ball.bbox,
                confidence=state.ball.confidence,
                frame_index=state.frame_index,
                timestamp_s=state.timestamp_s,
                velocity=state.ball.velocity,
                label="ball",
                time_since_update=state.ball.frames_since_seen,
                attributes={**state.ball.attributes,
                            "interpolated": state.ball.interpolated},
            )
        )
    if state.goal is not None and state.goal.bbox is not None:
        tracks.append(
            Track(
                track_id=0,
                class_name=GOAL,
                bbox=state.goal.bbox,
                confidence=state.goal.confidence,
                frame_index=state.frame_index,
                timestamp_s=state.timestamp_s,
                label="goal",
                attributes=dict(state.goal.attributes),
            )
        )
    return tracks


class StateCacheWriter:
    """Write one record per line (JSON Lines) as a run proceeds.

    Line-delimited rather than one big document so a run that is interrupted
    still leaves a readable cache, and so a long clip never has to be held in
    memory to be written.
    """

    def __init__(
        self, path: str | Path, header_extra: dict[str, Any] | None = None
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.records_written = 0
        self._closed = False
        header: dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "record": "header",
        }
        if header_extra:
            # Clip-level findings that belong to no single frame -- the camera
            # calibration, the identity report, where the ball was lost. A
            # replay needs them as much as the live run did.
            #
            # Nested under its own key rather than merged: a stage's summary is
            # its own vocabulary, and one that happened to contain "record" or
            # "schema_version" would corrupt the header and make the whole
            # cache unreadable.
            header["tracking"] = header_extra
        self._handle.write(json.dumps(header, default=str) + "\n")

    def write(self, state: FrameState) -> None:
        if self._closed:
            raise ValueError("state cache is closed")
        self._handle.write(json.dumps(state.to_dict(), ensure_ascii=False) + "\n")
        self.records_written += 1

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "StateCacheWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_state_cache_header(path: str | Path) -> dict[str, Any]:
    """Read just the header line: the clip-level findings and the schema."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("record") == "header":
                return payload
            break
    return {}


def read_state_cache(path: str | Path) -> Iterator[FrameState]:
    """Stream records back from a cache written by :class:`StateCacheWriter`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("record") == "header":
                version = payload.get("schema_version")
                if version != STATE_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path} was written by state schema {version}, "
                        f"this build reads {STATE_SCHEMA_VERSION}"
                    )
                continue
            try:
                yield FrameState.from_dict(payload)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number} is not a state record: {exc}") from exc
