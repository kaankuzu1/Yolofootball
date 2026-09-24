"""The event schema -- the contract every downstream stage writes into.

An analysis run produces exactly one :class:`EventTimeline`, which serialises
to the JSON document the CLI writes out.  The shape is deliberately flat and
self-describing: one list of events, each stamped with a time, a confidence, a
type, the player it belongs to, and a free-form ``detail`` object for whatever
is specific to that event type.

Adding a new event type means adding a member to :class:`EventType` and
documenting the keys it puts in ``detail``.  It should never mean changing the
envelope, because consumers (the annotated video, any UI, any export) read the
envelope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "SCHEMA_VERSION",
    "EventType",
    "Event",
    "SourceInfo",
    "PlayerInfo",
    "EventTimeline",
]

SCHEMA_VERSION = "1.0"


class EventType(str, Enum):
    """What happened.  The first five are what the system is asked to report.

    ``detail`` keys each type is expected to populate (all optional -- a
    consumer must tolerate their absence):

    ``TACKLE``    ``winner_player_id``, ``loser_player_id``, ``contact_point``
    ``TRICK``     ``trick_name`` (e.g. "stepover", "nutmeg"), ``touch_count``
    ``PASS``      ``receiver_player_id``, ``distance_px``, ``completed`` (bool)
    ``SHOT``      ``on_target`` (bool), ``distance_to_goal_px``, ``speed_px_s``
    ``GOAL``      ``scoring_player_id``, ``shot_event_id``

    ``PUSH_PAST`` is supporting rather than headline, and exists because a
    strict 1v1 has no teammate: a deliberate forward knock beyond the defender
    to re-collect is a real, different event, and calling it a pass would be
    wrong.  Its ``detail`` keys are ``beaten_player_id``, ``distance_px`` and
    ``recollected``.

    The remaining types are supporting signal: they are cheap to emit, they
    make the timeline debuggable, and the five headline types are usually
    derived from them.  A consumer that only wants the headline events can
    filter with :meth:`EventType.is_primary`.
    """

    TACKLE = "tackle"
    TRICK = "trick"
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"

    # Supporting events.
    PUSH_PAST = "push_past"
    BALL_TOUCH = "ball_touch"
    POSSESSION_CHANGE = "possession_change"
    OUT_OF_PLAY = "out_of_play"

    @property
    def is_primary(self) -> bool:
        return self in _PRIMARY_TYPES

    @classmethod
    def primary(cls) -> tuple["EventType", ...]:
        return _PRIMARY_TYPES


_PRIMARY_TYPES: tuple[EventType, ...] = (
    EventType.TACKLE,
    EventType.TRICK,
    EventType.PASS,
    EventType.SHOT,
    EventType.GOAL,
)


def _format_clock(seconds: float) -> str:
    """``73.4`` -> ``"01:13.400"`` -- what a human reads off a scrub bar."""
    if seconds < 0:
        seconds = 0.0
    minutes, rest = divmod(float(seconds), 60.0)
    return f"{int(minutes):02d}:{rest:06.3f}"


@dataclass
class Event:
    """One thing that happened, at one moment (or over one span) in the clip."""

    type: EventType
    timestamp_s: float
    """When it happened, in seconds from the start of the source clip."""

    confidence: float
    """0..1.  How sure the producing stage is that this event is real."""

    frame_index: int | None = None
    """Source-clip frame the event was called on, when one applies."""

    end_timestamp_s: float | None = None
    """For events with duration (a tackle, a trick).  ``None`` = instantaneous."""

    end_frame_index: int | None = None

    player_id: str | None = None
    """Who the event belongs to -- the tackler, the passer, the shooter."""

    secondary_player_id: str | None = None
    """The other party, where there is one -- the tackled, the receiver."""

    track_ids: list[int] = field(default_factory=list)
    """Tracks that evidenced this event, for drawing and for debugging."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Type-specific payload.  See :class:`EventType` for the expected keys."""

    source: str | None = None
    """Which stage emitted this, e.g. ``"rule_engine.tackle"``.  For triage."""

    id: str | None = None
    """Assigned by :class:`EventTimeline` on insert; stable within one run."""

    def __post_init__(self) -> None:
        self.type = EventType(self.type)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if float(self.timestamp_s) < 0:
            raise ValueError(f"timestamp_s must be >= 0, got {self.timestamp_s!r}")
        if self.end_timestamp_s is not None:
            if float(self.end_timestamp_s) < float(self.timestamp_s):
                raise ValueError(
                    "end_timestamp_s must not precede timestamp_s "
                    f"({self.end_timestamp_s} < {self.timestamp_s})"
                )

    @property
    def duration_s(self) -> float:
        if self.end_timestamp_s is None:
            return 0.0
        return float(self.end_timestamp_s) - float(self.timestamp_s)

    @property
    def clock(self) -> str:
        return _format_clock(self.timestamp_s)

    def label(self) -> str:
        """Short text for drawing on a frame or listing in a terminal."""
        who = self.player_id or ""
        name = self.detail.get("trick_name") if self.type is EventType.TRICK else None
        head = name or self.type.value
        return f"{head} {who}".strip()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "timestamp_s": round(float(self.timestamp_s), 3),
            "clock": self.clock,
            "confidence": round(float(self.confidence), 4),
        }
        if self.frame_index is not None:
            payload["frame_index"] = int(self.frame_index)
        if self.end_timestamp_s is not None:
            payload["end_timestamp_s"] = round(float(self.end_timestamp_s), 3)
            payload["duration_s"] = round(self.duration_s, 3)
        if self.end_frame_index is not None:
            payload["end_frame_index"] = int(self.end_frame_index)
        if self.player_id is not None:
            payload["player_id"] = self.player_id
        if self.secondary_player_id is not None:
            payload["secondary_player_id"] = self.secondary_player_id
        if self.track_ids:
            payload["track_ids"] = list(self.track_ids)
        if self.detail:
            payload["detail"] = self.detail
        if self.source is not None:
            payload["source"] = self.source
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Event":
        known = {
            "type", "timestamp_s", "confidence", "frame_index",
            "end_timestamp_s", "end_frame_index", "player_id",
            "secondary_player_id", "track_ids", "detail", "source", "id",
        }
        kwargs = {k: v for k, v in payload.items() if k in known}
        kwargs["type"] = EventType(payload["type"])
        return cls(**kwargs)


@dataclass
class SourceInfo:
    """What the clip on disk actually is.  Filled in by the reader."""

    path: str
    width: int
    height: int
    fps: float
    """Nominal frame rate reported by the container."""

    frame_count: int | None = None
    duration_s: float | None = None
    variable_frame_rate: bool = False
    """True when observed inter-frame gaps did not match the nominal fps."""

    processed_width: int | None = None
    processed_height: int | None = None
    segment_start_s: float | None = None
    segment_end_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PlayerInfo:
    """A participant the timeline refers to by ``player_id``."""

    player_id: str
    label: str
    track_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"player_id": self.player_id, "label": self.label}
        if self.track_id is not None:
            payload["track_id"] = self.track_id
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload


@dataclass
class EventTimeline:
    """The output document: source metadata, participants, and the events."""

    source: SourceInfo
    events: list[Event] = field(default_factory=list)
    players: list[PlayerInfo] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    config: dict[str, Any] = field(default_factory=dict)
    """The resolved configuration the run used, so a result is reproducible."""

    stats: dict[str, Any] = field(default_factory=dict)
    """Run counters -- frames read, frames processed, wall time."""

    schema_version: str = SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def add(self, event: Event) -> Event:
        """Append one event, assigning it a stable id if it lacks one."""
        if event.id is None:
            event.id = f"evt_{len(self.events) + 1:04d}"
        self.events.append(event)
        return event

    def extend(self, events: Iterable[Event]) -> None:
        for event in events:
            self.add(event)

    def sort(self) -> None:
        """Order by time.  Ids are left alone so links between events survive."""
        self.events.sort(key=lambda e: (float(e.timestamp_s), e.type.value))

    def of_type(self, *types: EventType) -> list[Event]:
        wanted = set(types)
        return [e for e in self.events if e.type in wanted]

    def primary(self) -> list[Event]:
        return [e for e in self.events if e.type.is_primary]

    def needs_review(self) -> list[Event]:
        """Events a stage flagged as uncertain, for a human to look at.

        A rule that is unsure says so in ``detail["needs_review"]`` with a
        ``review_reason``, rather than quietly picking a side.  Surfacing them
        is the difference between a timeline that can be trusted and one that
        merely looks confident.
        """
        return [e for e in self.events if e.detail.get("needs_review")]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for event in self.events:
            out[event.type.value] = out.get(event.type.value, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source": self.source.to_dict(),
            "players": [p.to_dict() for p in self.players],
            "counts": self.counts(),
            "needs_review": [
                {
                    "id": e.id,
                    "type": e.type.value,
                    "clock": e.clock,
                    "reason": e.detail.get("review_reason"),
                }
                for e in self.needs_review()
            ],
            "stats": self.stats,
            "config": self.config,
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path, indent: int | None = 2) -> Path:
        out = Path(path)
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(indent=indent), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventTimeline":
        source_payload = dict(payload.get("source", {}))
        source = SourceInfo(
            path=source_payload.pop("path", ""),
            width=source_payload.pop("width", 0),
            height=source_payload.pop("height", 0),
            fps=source_payload.pop("fps", 0.0),
            **source_payload,
        )
        timeline = cls(
            source=source,
            created_at=payload.get("created_at", ""),
            config=payload.get("config", {}),
            stats=payload.get("stats", {}),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )
        timeline.players = [
            PlayerInfo(
                player_id=p["player_id"],
                label=p.get("label", p["player_id"]),
                track_id=p.get("track_id"),
                attributes=p.get("attributes", {}),
            )
            for p in payload.get("players", [])
        ]
        timeline.events = [Event.from_dict(e) for e in payload.get("events", [])]
        return timeline

    @classmethod
    def read_json(cls, path: str | Path) -> "EventTimeline":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
