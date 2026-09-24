"""Hand-labelled events: the file format, loading, and a draft to correct.

One JSON file per clip, under ``assets/annotations/events/``::

    {
      "format": "football-event-labels",
      "version": 1,
      "clip": "assets/clips/my_1v1.mp4",
      "fps": 25.0,
      "span_s": [0.0, 180.0],
      "labelled_types": ["tackle", "trick", "pass", "shot", "goal"],
      "notes": "who labelled it, and how",
      "events": [
        {"type": "shot", "timestamp_s": 12.40, "player_id": "Player 1"},
        {"type": "goal", "clock": "00:12.960"},
        {"type": "tackle", "frame_index": 431},
        {"type": "pass", "window_s": [16.8, 20.4], "note": "ball too small to see the release"},
        {"type": "trick", "timestamp_s": 40.1, "ambiguous": true}
      ]
    }

``labelled_types`` is the promise that matters: for every type listed, *every*
occurrence inside ``span_s`` is labelled, so a system event of that type that
matches nothing is a false positive.  A type not listed is not scored at all,
which is what lets a clip labelled only for goals be used without the scorer
calling every tackle invented.

Each label gives its time one of four ways: ``timestamp_s``, ``clock``
(``mm:ss.fff``), ``frame_index`` (needs ``fps``), or ``window_s`` when the
labeller knows the event happened but not exactly when.  ``ambiguous: true``
marks a moment the labeller could not call: a system event on it is neither
right nor wrong, and not finding it is not a miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_analysis.events import EventType

FORMAT = "football-event-labels"
VERSION = 1

__all__ = ["FORMAT", "VERSION", "Label", "LabelSet", "load_labels", "draft_labels"]


def _parse_clock(text: str) -> float:
    """``"01:13.400"`` or ``"1:02:03.5"`` or ``"13.4"`` -> seconds."""
    total = 0.0
    for part in str(text).strip().split(":"):
        total = total * 60.0 + float(part)
    return total


@dataclass(frozen=True)
class Label:
    """One labelled event.  ``start_s == end_s`` unless the label is a window."""

    type: str
    start_s: float
    end_s: float
    ambiguous: bool = False
    player_id: str | None = None
    note: str | None = None

    @property
    def timestamp_s(self) -> float:
        return 0.5 * (self.start_s + self.end_s)

    @property
    def is_window(self) -> bool:
        return self.end_s > self.start_s

    def distance_s(self, t: float) -> float:
        """How far ``t`` falls outside the label: 0 anywhere inside a window."""
        if t < self.start_s:
            return self.start_s - t
        if t > self.end_s:
            return t - self.end_s
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.is_window:
            out["window_s"] = [round(self.start_s, 3), round(self.end_s, 3)]
        else:
            out["timestamp_s"] = round(self.start_s, 3)
        if self.player_id is not None:
            out["player_id"] = self.player_id
        if self.ambiguous:
            out["ambiguous"] = True
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class LabelSet:
    """Every label for one clip, plus what the labeller promised to cover."""

    labels: list[Label]
    labelled_types: tuple[str, ...]
    span_s: tuple[float, float] | None = None
    clip: str | None = None
    fps: float | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def in_span(self, t: float) -> bool:
        return self.span_s is None or self.span_s[0] <= t <= self.span_s[1]

    def of_type(self, event_type: str) -> list[Label]:
        return [lab for lab in self.labels if lab.type == event_type]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"format": FORMAT, "version": VERSION}
        if self.clip is not None:
            out["clip"] = self.clip
        if self.fps is not None:
            out["fps"] = self.fps
        if self.span_s is not None:
            out["span_s"] = list(self.span_s)
        out["labelled_types"] = list(self.labelled_types)
        if self.notes:
            out["notes"] = self.notes
        out.update(self.extra)
        out["events"] = [lab.to_dict() for lab in sorted(self.labels, key=lambda lab: lab.start_s)]
        return out

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, where: str = "labels") -> "LabelSet":
        if payload.get("format") != FORMAT:
            raise ValueError(f"{where}: 'format' must be {FORMAT!r}")
        if int(payload.get("version", 0)) != VERSION:
            raise ValueError(f"{where}: unsupported version {payload.get('version')!r}")
        known = {t.value for t in EventType}
        fps = payload.get("fps")
        labels = []
        for i, raw in enumerate(payload.get("events", [])):
            labels.append(_parse_label(raw, fps=fps, known=known, where=f"{where}: events[{i}]"))
        types = payload.get("labelled_types")
        if types is None:
            types = sorted({lab.type for lab in labels})
        for t in types:
            if t not in known:
                raise ValueError(f"{where}: labelled_types has unknown type {t!r}")
        stray = sorted({lab.type for lab in labels} - set(types))
        if stray:
            raise ValueError(
                f"{where}: labels of type {stray} but labelled_types does not list them"
            )
        span = payload.get("span_s")
        reserved = {"format", "version", "clip", "fps", "span_s", "labelled_types", "notes", "events"}
        return cls(
            labels=labels,
            labelled_types=tuple(types),
            span_s=(float(span[0]), float(span[1])) if span else None,
            clip=payload.get("clip"),
            fps=float(fps) if fps else None,
            notes=payload.get("notes"),
            extra={k: v for k, v in payload.items() if k not in reserved},
        )


def _parse_label(raw: dict[str, Any], *, fps: float | None, known: set[str], where: str) -> Label:
    event_type = raw.get("type")
    if event_type not in known:
        raise ValueError(f"{where}: unknown type {event_type!r} (known: {sorted(known)})")
    given = [k for k in ("timestamp_s", "clock", "frame_index", "window_s") if k in raw]
    if len(given) != 1:
        raise ValueError(f"{where}: give exactly one of timestamp_s, clock, frame_index, window_s")
    key = given[0]
    if key == "window_s":
        start, end = (float(v) for v in raw["window_s"])
        if end < start:
            raise ValueError(f"{where}: window_s ends before it starts")
    else:
        if key == "timestamp_s":
            start = float(raw["timestamp_s"])
        elif key == "clock":
            start = _parse_clock(raw["clock"])
        else:
            if not fps:
                raise ValueError(f"{where}: frame_index needs the file's fps")
            start = int(raw["frame_index"]) / float(fps)
        end = start
    if start < 0:
        raise ValueError(f"{where}: negative time")
    return Label(
        type=event_type,
        start_s=start,
        end_s=end,
        ambiguous=bool(raw.get("ambiguous", False)),
        player_id=raw.get("player_id"),
        note=raw.get("note"),
    )


def load_labels(path: str | Path) -> LabelSet:
    path = Path(path)
    return LabelSet.from_dict(json.loads(path.read_text(encoding="utf-8")), where=str(path))


def draft_labels(
    events: list[dict[str, Any]],
    *,
    clip: str | None = None,
    fps: float | None = None,
    types: list[str] | None = None,
) -> LabelSet:
    """A label file seeded from a system run, for a person to correct.

    Correcting is much faster than labelling from scratch, but it anchors the
    labeller on what the system saw, so every drafted label carries a
    ``note`` saying it has not been checked.  Delete the note once it has.
    """
    wanted = set(types) if types else {t.value for t in EventType.primary()}
    labels = [
        Label(
            type=e["type"],
            start_s=float(e["timestamp_s"]),
            end_s=float(e["timestamp_s"]),
            player_id=e.get("player_id"),
            note="unchecked: drafted from the system's output",
        )
        for e in events
        if e.get("type") in wanted
    ]
    return LabelSet(
        labels=labels,
        labelled_types=tuple(t for t in (x.value for x in EventType) if t in wanted),
        clip=clip,
        fps=fps,
        notes="Draft from a system run. Check every label against the video, add what it missed, "
        "and remove the 'unchecked' notes.",
    )
