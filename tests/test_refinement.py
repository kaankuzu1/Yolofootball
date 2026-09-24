"""The two-phase run: a tracker that can only answer from the whole clip.

The identity lock clusters every crop at once and the ball smoother runs
backward, so the tracker's best records exist only after the last frame. The
rules must read those, never the provisional ones -- a provisional identity is
the wrong player, and an event attributed to the wrong player is worse than no
event.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.adapters import MeasurementObserver
from football_analysis.interfaces import BaseEventDetector, BaseTracker
from football_analysis.pipeline import AnalysisPipeline, default_tracker
from football_analysis.state import (
    BallState,
    FrameState,
    PlayerState,
    read_state_cache,
    read_state_cache_header,
    state_from_tracks,
)
from football_analysis.config import Config
from football_analysis.types import BBox, Detection, Point, Track


class ProvisionalTracker(BaseTracker):
    """Returns provisional identities, then corrects them for the whole clip."""

    def __init__(self, *, summary=None, refined_count=None):
        self._frames = []
        self._summary = summary
        self.refined_count = refined_count
        self.finalize_calls = 0

    def update(self, frame, detections):
        self._frames.append((frame.index, frame.timestamp_s))
        return [
            Track(
                track_id=99,
                class_name="player",
                bbox=BBox(10, 10, 50, 150),
                confidence=0.5,
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                label="Provisional",
                attributes={"provisional": True},
            )
        ]

    def finalize_states(self):
        self.finalize_calls += 1
        frames = self._frames
        if self.refined_count is not None:
            frames = frames[: self.refined_count]
        return [
            FrameState(
                frame_index=index,
                timestamp_s=timestamp,
                players=[
                    PlayerState(
                        player_id="Player 1", track_id=1,
                        bbox=BBox(10, 10, 50, 150), confidence=0.95,
                        attributes={"identity_confidence": 0.9},
                    )
                ],
                ball=BallState(position=Point(60, 140), confidence=0.6),
                frame_size=(320, 180),
            )
            for index, timestamp in frames
        ]

    def summary(self):
        return self._summary

    def reset(self):
        self._frames = []


class RecordingEvents(BaseEventDetector):
    def __init__(self):
        self.seen = []

    def update(self, state):
        self.seen.append(state)
        return []


class SteadyMeter:
    def measure(self, image, state):
        return 7.5

    def reset(self):
        return None


def _run(sample_clip, config, tracker, events=None, observers=()):
    path, _ = sample_clip
    events = events or RecordingEvents()
    result = AnalysisPipeline(
        config,
        detector=_OnePlayerDetector(),
        tracker=tracker,
        event_detector=events,
        observers=list(observers),
    ).run(path)
    return result, events


class _OnePlayerDetector:
    def detect(self, frame):
        return [Detection(class_name="player", bbox=BBox(10, 10, 50, 150), confidence=0.9)]

    def reset(self):
        return None

    def describe(self):
        return {"name": "_OnePlayerDetector"}


def test_the_rules_read_the_refined_records_not_the_provisional_ones(
    sample_clip, config
):
    tracker = ProvisionalTracker()
    result, events = _run(sample_clip, config, tracker)

    assert tracker.finalize_calls == 1
    assert events.seen, "the rules should have run"
    # Provisional said "Provisional"/99; the lock says "Player 1"/1.
    assert all(s.players[0].player_id == "Player 1" for s in events.seen)
    assert all(s.players[0].track_id == 1 for s in events.seen)
    assert result.timeline.stats["states_refined"] is True


def test_observer_readings_survive_refinement(sample_clip, config):
    # The trap: refined records are built by the tracker, which knows nothing
    # about observers. Taking them wholesale would drop every pixel reading,
    # and the only symptom would be a goal rule quietly getting less sure.
    tracker = ProvisionalTracker()
    _, events = _run(
        sample_clip, config, tracker,
        observers=[MeasurementObserver(SteadyMeter(), "net_motion")],
    )

    assert events.seen
    assert all(s.attributes.get("net_motion") == 7.5 for s in events.seen)


def test_the_tracker_wins_a_collision_on_its_own_attributes(sample_clip, config):
    class Colliding(ProvisionalTracker):
        def finalize_states(self):
            states = super().finalize_states()
            for state in states:
                state.attributes["net_motion"] = 1.0
            return states

    _, events = _run(
        sample_clip, config, Colliding(),
        observers=[MeasurementObserver(SteadyMeter(), "net_motion")],
    )
    # It just recomputed the value, so its answer is the one to keep.
    assert all(s.attributes["net_motion"] == 1.0 for s in events.seen)


def test_the_cache_holds_the_refined_records(sample_clip, config, tmp_path):
    cache = tmp_path / "state.jsonl"
    config.output.state_cache_path = str(cache)
    _run(sample_clip, config, ProvisionalTracker())

    records = list(read_state_cache(cache))
    assert records
    assert all(r.players[0].player_id == "Player 1" for r in records)


def test_clip_level_findings_ride_in_the_cache_header(sample_clip, config, tmp_path):
    cache = tmp_path / "state.jsonl"
    config.output.state_cache_path = str(cache)
    summary = {"calibration": "goal-pnp", "ball_gaps": [[10, 14]]}
    _run(sample_clip, config, ProvisionalTracker(summary=summary))

    header = read_state_cache_header(cache)
    assert header["tracking"]["calibration"] == "goal-pnp"
    assert header["tracking"]["ball_gaps"] == [[10, 14]]
    assert header["schema_version"] == "1.0"
    # The header must not break reading the records.
    assert list(read_state_cache(cache))


def test_a_summary_cannot_corrupt_the_cache_header(sample_clip, config, tmp_path):
    # A stage's summary is its own vocabulary. Merged flat, one carrying
    # "record" or "schema_version" would make the whole cache unreadable.
    cache = tmp_path / "state.jsonl"
    config.output.state_cache_path = str(cache)
    hostile = {"record": "not-a-header", "schema_version": "99.0"}
    _run(sample_clip, config, ProvisionalTracker(summary=hostile))

    header = read_state_cache_header(cache)
    assert header["record"] == "header"
    assert header["schema_version"] == "1.0"
    assert header["tracking"]["record"] == "not-a-header"
    assert list(read_state_cache(cache)), "the records must still be readable"


def test_a_frame_by_frame_tracker_passes_its_records_through(sample_clip, config):
    from football_analysis.stubs import GreedyIouTracker

    result, events = _run(sample_clip, config, GreedyIouTracker(config))
    assert events.seen
    assert result.timeline.stats["states_refined"] is False


def test_a_tracker_returning_nothing_keeps_the_provisional_records(
    sample_clip, config, caplog
):
    class Empty(ProvisionalTracker):
        def finalize_states(self):
            return []

    with caplog.at_level("WARNING"):
        _, events = _run(sample_clip, config, Empty())

    assert events.seen, "a tracker that refines nothing must not lose the run"
    assert "keeping the provisional records" in caplog.text


def test_a_short_refinement_is_flagged(sample_clip, config, caplog):
    with caplog.at_level("WARNING"):
        _run(sample_clip, config, ProvisionalTracker(refined_count=5))
    assert "for" in caplog.text and "processed frames" in caplog.text


def test_the_default_tracker_is_the_real_one():
    tracker = default_tracker(Config())
    assert type(tracker).__name__ == "WorldStateTracker"
    assert callable(getattr(tracker, "finalize_states", None))


def test_the_default_tracker_takes_the_config_sections():
    config = Config()
    config.tracking.max_players = 2
    config.geometry = {"goal_width_m": 7.32, "goal_height_m": 2.44}
    tracker = default_tracker(config)

    assert tracker.geometry_config.goal_width_m == pytest.approx(7.32)
    assert tracker.layer.players.num_identities == 2


def test_an_unknown_geometry_key_is_rejected_rather_than_ignored():
    from football_analysis.config import ConfigError

    config = Config()
    config.geometry = {"goal_widht_m": 7.32}  # typo
    with pytest.raises(ConfigError, match="geometry"):
        default_tracker(config)


def test_an_unknown_track_key_is_rejected_rather_than_ignored():
    from football_analysis.config import ConfigError

    config = Config()
    config.track = {"ball.no_such_setting": 1}
    with pytest.raises(ConfigError, match="track"):
        default_tracker(config)


def test_nested_track_settings_are_applied():
    config = Config()
    config.track = {"players.num_identities": 3}
    assert default_tracker(config).layer.players.num_identities == 3


def test_the_clip_is_drawn_from_the_refined_records(sample_clip, config, tmp_path):
    # The video captioning "player#21" while the JSON says "Player 1" is worse
    # than either being wrong on its own: the two outputs contradict each other.
    from football_analysis.viz import FrameAnnotator

    drawn: list[list[str]] = []

    class SpyAnnotator(FrameAnnotator):
        def annotate(self, frame, *, tracks=(), detections=(), events=()):
            drawn.append([t.display_name for t in tracks])
            return super().annotate(
                frame, tracks=tracks, detections=detections, events=events
            )

    path, _ = sample_clip
    config.output.video_path = str(tmp_path / "annotated.mp4")
    AnalysisPipeline(
        config,
        detector=_OnePlayerDetector(),
        tracker=ProvisionalTracker(),
        event_detector=RecordingEvents(),
        annotator=SpyAnnotator(config.output),
    ).run(path)

    assert drawn
    # Provisional said "Provisional"; the locked identity is "Player 1".
    assert all("Player 1" in names for names in drawn)
    assert not any("Provisional" in names for names in drawn)


def test_tracks_from_state_round_trips_the_record():
    from football_analysis.state import tracks_from_state

    state = FrameState(
        frame_index=7,
        timestamp_s=0.28,
        players=[PlayerState(player_id="Player 2", track_id=2,
                             bbox=BBox(10, 10, 50, 150), confidence=0.8)],
        ball=BallState(position=Point(60, 140), confidence=0.4,
                       bbox=BBox(54, 134, 66, 146), interpolated=True,
                       track_id=5, frames_since_seen=2),
        frame_size=(320, 180),
    )
    tracks = tracks_from_state(state)

    player = next(t for t in tracks if t.class_name == "player")
    ball = next(t for t in tracks if t.class_name == "ball")
    assert player.display_name == "Player 2"
    assert player.track_id == 2
    assert ball.attributes["interpolated"] is True
    assert ball.time_since_update == 2


def test_a_record_with_no_boxes_draws_nothing_rather_than_raising():
    from football_analysis.state import tracks_from_state

    # A ball known only by position, with no box, has nothing to draw.
    state = FrameState(
        frame_index=1, timestamp_s=0.04,
        ball=BallState(position=Point(10, 10), confidence=0.3, bbox=None),
    )
    assert tracks_from_state(state) == []
