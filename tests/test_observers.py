"""The pixel-observer seam, the composite stage, and producer-named events.

Event logic is denied the frame on purpose, but a few rules genuinely need a
pixel reading. These tests pin the sanctioned route across that line.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.adapters import CompositeEventDetector, MeasurementObserver
from football_analysis.config import Config
from football_analysis.events import Event, EventType
from football_analysis.interfaces import BaseEventDetector, EventDetector, FrameObserver
from football_analysis.pipeline import AnalysisPipeline, default_event_stage
from football_analysis.state import FrameState, read_state_cache
from football_analysis.stubs import GreedyIouTracker, StubDetector
from football_analysis.types import BBox, Detection, VideoFrame


class FakeMeter:
    """A meter in the usual shape: pixels in, a number (or None) out."""

    def __init__(self, values=None):
        self.values = list(values) if values is not None else None
        self.calls = 0
        self.resets = 0

    def measure(self, image, state):
        self.calls += 1
        if self.values is None:
            return float(image.mean())
        return self.values.pop(0) if self.values else None

    def reset(self):
        self.resets += 1


def _frame(index=0, timestamp_s=0.0, fill=40):
    return VideoFrame(
        index=index,
        timestamp_s=timestamp_s,
        image=np.full((180, 320, 3), fill, np.uint8),
        source_size=(320, 180),
    )


def test_observer_satisfies_the_contract():
    assert isinstance(MeasurementObserver(FakeMeter(), "net_motion"), FrameObserver)


def test_a_reading_lands_on_the_record():
    observer = MeasurementObserver(FakeMeter([0.4217]), "net_motion")
    state = FrameState(frame_index=0, timestamp_s=0.0)
    observer.observe(_frame(), state)

    assert state.attributes["net_motion"] == pytest.approx(0.422)


def test_no_reading_leaves_the_key_absent():
    # "the net did not move" and "nobody looked" are different facts, and a
    # rule that cannot tell them apart will conclude the wrong thing.
    observer = MeasurementObserver(FakeMeter([None]), "net_motion")
    state = FrameState(frame_index=0, timestamp_s=0.0)
    observer.observe(_frame(), state)

    assert "net_motion" not in state.attributes


def test_a_zero_reading_is_recorded_not_dropped():
    observer = MeasurementObserver(FakeMeter([0.0]), "net_motion")
    state = FrameState(frame_index=0, timestamp_s=0.0)
    observer.observe(_frame(), state)

    assert state.attributes["net_motion"] == 0.0


def test_observer_resets_its_meter():
    meter = FakeMeter()
    MeasurementObserver(meter, "k").reset()
    assert meter.resets == 1


def test_something_that_cannot_measure_is_rejected():
    with pytest.raises(TypeError, match="measure"):
        MeasurementObserver(object(), "k")


class RecordingEvents(BaseEventDetector):
    """Captures the records it was handed."""

    def __init__(self):
        self.seen = []

    def update(self, state):
        self.seen.append(state)
        return []


def test_the_pipeline_runs_observers_before_the_rules_read_the_record(
    sample_clip, config
):
    path, _ = sample_clip
    meter = FakeMeter()
    events = RecordingEvents()
    AnalysisPipeline(
        config,
        detector=StubDetector(config),
        tracker=GreedyIouTracker(config),
        event_detector=events,
        observers=[MeasurementObserver(meter, "net_motion")],
    ).run(path)

    assert meter.calls == len(events.seen)
    # The rules must see the reading, not a record that gets it afterwards.
    assert all("net_motion" in s.attributes for s in events.seen)


def test_readings_are_written_to_the_cache_so_a_replay_sees_them(
    sample_clip, config, tmp_path
):
    path, _ = sample_clip
    cache = tmp_path / "state.jsonl"
    config.output.state_cache_path = str(cache)
    AnalysisPipeline(
        config,
        detector=StubDetector(config),
        tracker=GreedyIouTracker(config),
        event_detector=RecordingEvents(),
        observers=[MeasurementObserver(FakeMeter(), "net_motion")],
    ).run(path)

    records = list(read_state_cache(cache))
    assert records and all("net_motion" in r.attributes for r in records)


def test_observers_are_reported_in_the_provenance_record(sample_clip, config):
    path, _ = sample_clip
    result = AnalysisPipeline(
        config,
        detector=StubDetector(config),
        tracker=GreedyIouTracker(config),
        event_detector=RecordingEvents(),
        observers=[MeasurementObserver(FakeMeter(), "net_motion")],
    ).run(path)

    observers = result.timeline.stats["stages"]["observers"]
    assert observers[0]["key"] == "net_motion"
    assert observers[0]["name"] == "FakeMeter"


# -- the composite ---------------------------------------------------------


class Emitter(BaseEventDetector):
    def __init__(self, event_type, at=1.0, late=None):
        self.event_type = event_type
        self.at = at
        self.late = late
        self.resets = 0

    def update(self, state):
        if abs(state.timestamp_s - self.at) < 1e-6:
            return [Event(type=self.event_type, timestamp_s=state.timestamp_s,
                          confidence=0.9)]
        return []

    def finalize(self):
        if self.late is None:
            return []
        return [Event(type=self.late, timestamp_s=2.0, confidence=0.9)]

    def reset(self):
        self.resets += 1


def test_composite_satisfies_the_contract():
    assert isinstance(CompositeEventDetector(Emitter(EventType.PASS)), EventDetector)


def test_composite_merges_both_update_and_finalize():
    composite = CompositeEventDetector(
        Emitter(EventType.PASS, at=1.0, late=EventType.GOAL),
        Emitter(EventType.TACKLE, at=1.0),
    )
    state = FrameState(frame_index=25, timestamp_s=1.0)

    assert {e.type for e in composite.update(state)} == {
        EventType.PASS, EventType.TACKLE
    }
    assert [e.type for e in composite.finalize()] == [EventType.GOAL]


def test_composite_resets_every_stage():
    stages = [Emitter(EventType.PASS), Emitter(EventType.SHOT)]
    CompositeEventDetector(*stages).reset()
    assert [s.resets for s in stages] == [1, 1]


def test_composite_ignores_absent_stages():
    # A stage that is not built yet is None, not a crash.
    composite = CompositeEventDetector(Emitter(EventType.PASS), None, [None])
    assert len(composite.detectors) == 1


def test_composite_describes_its_stages():
    payload = CompositeEventDetector(Emitter(EventType.PASS)).describe()
    assert payload["stages"][0]["name"] == "Emitter"


def test_a_failing_stage_is_not_swallowed():
    # A half-analysed timeline presented as a whole one is worse than a failure.
    class Broken(BaseEventDetector):
        def update(self, state):
            raise RuntimeError("rule blew up")

    with pytest.raises(RuntimeError, match="rule blew up"):
        CompositeEventDetector(Emitter(EventType.PASS), Broken()).update(
            FrameState(frame_index=0, timestamp_s=0.0)
        )


# -- de-duplication --------------------------------------------------------


class NamedRebound(BaseEventDetector):
    """A shot and its rebound by the same player, inside min_gap_s, both named."""

    def update(self, state):
        return []

    def finalize(self):
        return [
            Event(type=EventType.SHOT, timestamp_s=1.00, confidence=0.9,
                  player_id="Player 1", id="ball_0001"),
            Event(type=EventType.SHOT, timestamp_s=1.20, confidence=0.9,
                  player_id="Player 1", id="ball_0002"),
        ]


class UnnamedSpam(BaseEventDetector):
    """A stage that fires every frame and does not name its events."""

    def update(self, state):
        return [Event(type=EventType.SHOT, timestamp_s=state.timestamp_s,
                      confidence=0.9, player_id="Player 1")]


def _run(sample_clip, config, event_detector):
    path, _ = sample_clip
    config.events.enabled_types = ["shot"]
    config.events.min_gap_s = 0.5
    return AnalysisPipeline(
        config,
        detector=StubDetector(config),
        tracker=GreedyIouTracker(config),
        event_detector=event_detector,
    ).run(path)


def test_producer_named_events_survive_the_gap_rule(sample_clip, config):
    # A shot and the rebound the same player puts away 0.2s later are two
    # shots. The stage named them, so it has already de-duplicated.
    result = _run(sample_clip, config, NamedRebound())
    assert [e.id for e in result.events] == ["ball_0001", "ball_0002"]


def test_unnamed_repeats_are_still_collapsed(sample_clip, config):
    # Fired every frame at 25fps; the gap rule must thin them to one per 0.5s.
    result = _run(sample_clip, config, UnnamedSpam())
    times = [e.timestamp_s for e in result.events]

    assert len(times) < result.timeline.stats["frames_processed"] / 5
    assert all(b - a >= 0.5 for a, b in zip(times, times[1:]))


# -- defaults --------------------------------------------------------------


def test_the_default_event_stage_is_the_real_rule_engine():
    detector, observers = default_event_stage(Config())
    assert type(detector).__name__ != "StubEventDetector"
    assert isinstance(detector, EventDetector)
    # The goal rule needs a pixel reading, so its observer comes with it.
    assert [o.key for o in observers] == ["net_motion"]
