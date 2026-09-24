"""The contracts between pipeline stages.

Three stages sit between a decoded frame and an event timeline, and each one
is owned by a different piece of work:

``Detector``       finds the ball, the players and the goal in one frame
``Tracker``        gives those detections identities that persist across frames
``EventDetector``  reads the world-state record over time and calls what happened

A fourth contract, :class:`StateRefiner`, is not a stage: it is a second look
at the finished records, for readings that are only worth taking once the
boxes have settled.

The first two are handed frames.  The third is not: it is handed one
:class:`~football_analysis.state.FrameState` per frame and never sees the
video, which is what lets event rules be re-run in seconds against a cached
run instead of re-running inference each time.

They are :class:`typing.Protocol` classes rather than base classes on purpose:
an implementation does not import anything from here, it just has to have the
right methods.  That keeps the detection work and the event work independent of
each other and of this module.

Each protocol has an abstract-ish companion base (``BaseDetector`` and friends)
for implementations that would rather inherit shared plumbing -- ``reset`` and
``describe`` come free.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from football_analysis.events import Event
from football_analysis.state import FrameState
from football_analysis.types import Detection, Track, VideoFrame

__all__ = [
    "Detector",
    "Tracker",
    "EventDetector",
    "FrameObserver",
    "StateRefiner",
    "BaseDetector",
    "BaseTracker",
    "BaseEventDetector",
]


@runtime_checkable
class Detector(Protocol):
    """Finds objects in a single frame.

    Implementations must return boxes in the coordinate space of
    ``frame.image`` (see :mod:`football_analysis.types`), and must return an
    empty list rather than raising when a frame contains nothing.
    """

    def detect(self, frame: VideoFrame) -> list[Detection]:
        """Detect every object of interest in one frame."""
        ...

    def reset(self) -> None:
        """Forget any per-clip state.  Called once before a run starts."""
        ...

    def describe(self) -> dict[str, Any]:
        """Report what this detector is, for the run's provenance record."""
        ...


@runtime_checkable
class Tracker(Protocol):
    """Turns per-frame detections into identities that persist.

    ``update`` is called once per processed frame, in order, and returns the
    full set of live tracks for that frame -- not just the ones that changed.

    The returned tracks are snapshots of that frame: an implementation must not
    hand back objects it will keep mutating, because event logic routinely
    holds on to a track to compare it against a later frame.
    """

    def update(self, frame: VideoFrame, detections: list[Detection]) -> list[Track]:
        """Associate this frame's detections with existing tracks."""
        ...

    def reset(self) -> None:
        ...

    def describe(self) -> dict[str, Any]:
        ...


@runtime_checkable
class EventDetector(Protocol):
    """Reads the world-state record and calls tackles, tricks, passes, goals.

    ``update`` is handed one :class:`~football_analysis.state.FrameState` per
    processed frame, in order, and may return events as soon as it is sure of
    them.  ``finalize`` is called once after the last frame, for events that
    could only be confirmed in hindsight (a shot that turned out to be a goal,
    a sequence of touches that turned out to be a trick).

    Two things to respect in the record: ``state.ball.interpolated`` marks a
    position that was coasted through an occlusion rather than observed, and
    every coordinate is in ``state.frame_size`` space, not source pixels.
    """

    def update(self, state: FrameState) -> list[Event]:
        """Advance the state machine by one frame; return anything decided."""
        ...

    def finalize(self) -> list[Event]:
        """Flush events still pending when the clip ended."""
        ...

    def reset(self) -> None:
        ...

    def describe(self) -> dict[str, Any]:
        ...


@runtime_checkable
class StateRefiner(Protocol):
    """A second pass over the finished records, with the clip still available.

    An observer runs inside the analysis loop, one frame at a time, before the
    tracker has settled anything.  Some readings cannot be taken then.  Pose
    keypoints are the case this exists for: reading a skeleton is expensive and
    only worth doing for the players who turn out to be real, on the boxes the
    tracker finally settled on -- which on a crowded clip is two players rather
    than twenty provisional ones.

    So a refiner runs after ``finalize_states`` and before the rules, and it is
    handed the records plus the path to the clip it may re-read.  It returns the
    records, refined; it may mutate them in place and return the same list.

    It runs before the state cache is written, so what it adds survives into a
    ``--replay`` and the rules see it either way.
    """

    def refine(self, states: list[FrameState], video_path: str) -> list[FrameState]:
        """Return the records with whatever this refiner adds filled in."""
        ...

    def describe(self) -> dict[str, Any]:
        ...


@runtime_checkable
class FrameObserver(Protocol):
    """Measures something in the pixels and writes it onto the record.

    Event logic is denied the frame on purpose, but a few decisions genuinely
    need pixels -- whether the net rippled, for instance, which no amount of
    box geometry can tell you.  An observer is the sanctioned way across that
    line: it runs in the analysis loop, where the frame exists, and leaves its
    finding in ``state.attributes`` for the rules to read.

    Because the finding lands on the record, it is written to the state cache
    too, so a replay sees exactly what the live run saw.

    An observer writes to ``state.attributes`` and returns nothing.  It must
    not modify anything else on the record: the stage that owns a field is the
    one that filled it.
    """

    def observe(self, frame: VideoFrame, state: FrameState) -> None:
        """Measure this frame and annotate ``state.attributes``."""
        ...

    def reset(self) -> None:
        ...

    def describe(self) -> dict[str, Any]:
        ...


class _Described:
    """Shared ``reset``/``describe`` so implementations need not repeat them."""

    def reset(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": type(self).__name__}


class BaseDetector(_Described):
    """Convenience base for detectors.  Subclasses implement :meth:`detect`."""

    def detect(self, frame: VideoFrame) -> list[Detection]:
        raise NotImplementedError


class BaseTracker(_Described):
    """Convenience base for trackers.  Subclasses implement :meth:`update`."""

    def update(self, frame: VideoFrame, detections: list[Detection]) -> list[Track]:
        raise NotImplementedError


class BaseEventDetector(_Described):
    """Convenience base for event logic.  Subclasses implement :meth:`update`."""

    def update(self, state: FrameState) -> list[Event]:
        raise NotImplementedError

    def finalize(self) -> list[Event]:
        return []
