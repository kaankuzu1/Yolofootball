"""The contracts the detection, tracking and event threads must satisfy.

These tests are the executable version of the interface documentation: if a
real implementation passes them, it will drop into the pipeline unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.config import Config
from football_analysis.interfaces import Detector, EventDetector, Tracker
from football_analysis.stubs import (
    GreedyIouTracker,
    NullDetector,
    NullEventDetector,
    StubDetector,
    StubEventDetector,
)
from football_analysis.state import BallState, FrameState, PlayerState, state_from_tracks
from football_analysis.types import BBox, Detection, Point, VideoFrame


def _frame(index=0, timestamp_s=0.0, width=320, height=180):
    return VideoFrame(
        index=index,
        timestamp_s=timestamp_s,
        image=np.zeros((height, width, 3), np.uint8),
        source_size=(width, height),
    )


@pytest.mark.parametrize(
    "stage, protocol",
    [
        (NullDetector(), Detector),
        (StubDetector(), Detector),
        (GreedyIouTracker(), Tracker),
        (NullEventDetector(), EventDetector),
        (StubEventDetector(), EventDetector),
    ],
)
def test_stages_satisfy_their_protocol(stage, protocol):
    assert isinstance(stage, protocol)
    assert isinstance(stage.describe(), dict)
    stage.reset()


def test_an_empty_frame_yields_no_detections_rather_than_raising():
    assert NullDetector().detect(_frame()) == []
    # A featureless frame must not be reported as one frame-sized player.
    assert StubDetector().detect(_frame()) == []


def test_returned_tracks_are_snapshots_not_live_objects():
    tracker = GreedyIouTracker()
    first = tracker.update(
        _frame(0, 0.0),
        [Detection(class_name="player", bbox=BBox(50, 40, 90, 150), confidence=0.9)],
    )
    held = first[0]
    tracker.update(
        _frame(1, 0.04),
        [Detection(class_name="player", bbox=BBox(80, 40, 120, 150), confidence=0.9)],
    )
    # Event logic compares a held track against a later frame; it must not have
    # been rewritten underneath.
    assert held.bbox.x1 == pytest.approx(50)


def test_tracker_keeps_one_identity_across_frames():
    tracker = GreedyIouTracker()
    box = BBox(50, 40, 90, 150)
    first = tracker.update(
        _frame(0, 0.0), [Detection(class_name="player", bbox=box, confidence=0.9)]
    )
    second = tracker.update(
        _frame(1, 0.04),
        [Detection(class_name="player", bbox=BBox(52, 40, 92, 150), confidence=0.9)],
    )

    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert second[0].age > first[0].age
    assert second[0].velocity is not None, "a matched track should estimate motion"


def test_tracker_gives_the_two_players_distinct_ids_and_labels():
    tracker = GreedyIouTracker()
    tracks = tracker.update(
        _frame(0, 0.0),
        [
            Detection(class_name="player", bbox=BBox(20, 40, 60, 150), confidence=0.9),
            Detection(class_name="player", bbox=BBox(200, 40, 240, 150), confidence=0.9),
        ],
    )

    assert len({t.track_id for t in tracks}) == 2
    assert sorted(t.label for t in tracks) == ["Player 1", "Player 2"]


def test_tracker_respects_the_two_player_limit_of_a_1v1():
    tracker = GreedyIouTracker()
    crowd = [
        Detection(class_name="player", bbox=BBox(x, 40, x + 30, 150), confidence=0.9)
        for x in (10, 60, 110, 160, 210)
    ]
    tracks = tracker.update(_frame(0, 0.0), crowd)
    assert len([t for t in tracks if t.class_name == "player"]) == 2


def test_a_1v1_never_grows_a_third_player_through_track_churn():
    # Tracks are lost and re-acquired constantly; labels must not accumulate.
    tracker = GreedyIouTracker()
    seen_labels = set()
    for i in range(30):
        # Jump the boxes far enough that IoU association fails every few frames.
        offset = (i % 3) * 90
        frame = _frame(i, i * 0.04)
        tracks = tracker.update(
            frame,
            [
                Detection(class_name="player",
                          bbox=BBox(10 + offset, 40, 50 + offset, 150), confidence=0.9),
                Detection(class_name="player",
                          bbox=BBox(150 + offset, 40, 190 + offset, 150), confidence=0.9),
            ],
        )
        seen_labels.update(t.label for t in tracks if t.label)
        assert len([t for t in tracks if t.class_name == "player"]) <= 2

    assert seen_labels <= {"Player 1", "Player 2"}, seen_labels


def test_tracker_ages_out_a_track_that_stops_being_seen():
    config = Config()
    config.tracking.max_age_frames = 3
    tracker = GreedyIouTracker(config)
    tracker.update(
        _frame(0, 0.0),
        [Detection(class_name="player", bbox=BBox(20, 40, 60, 150), confidence=0.9)],
    )
    for i in range(1, 8):
        live = tracker.update(_frame(i, i * 0.04), [])
    assert live == []


def test_tracker_does_not_match_across_classes():
    tracker = GreedyIouTracker()
    box = BBox(50, 50, 70, 70)
    tracker.update(_frame(0, 0.0), [Detection(class_name="ball", bbox=box, confidence=0.5)])
    tracks = tracker.update(
        _frame(1, 0.04), [Detection(class_name="player", bbox=box, confidence=0.9)]
    )
    # Same place, different class: a new identity, not a reused one.
    assert [t.class_name for t in tracks] == ["player"]


def test_stub_event_detector_needs_sustained_proximity():
    config = Config()
    config.events.possession_min_frames = 3
    detector = StubEventDetector(config)
    tracker = GreedyIouTracker(config)

    emitted = []
    for i in range(6):
        frame = _frame(i, i * 0.04)
        detections = [
            Detection(class_name="player", bbox=BBox(50, 40, 90, 150), confidence=0.9),
            Detection(class_name="ball", bbox=BBox(66, 140, 78, 152), confidence=0.5),
        ]
        state = state_from_tracks(frame, tracker.update(frame, detections))
        emitted.extend(detector.update(state))

    assert emitted, "sustained proximity should eventually be called a touch"
    assert all(0.0 <= e.confidence <= 1.0 for e in emitted)
    assert all(e.timestamp_s >= 0 for e in emitted)


def test_stub_event_detector_ignores_a_frame_with_no_ball():
    detector = StubEventDetector()
    tracker = GreedyIouTracker()
    frame = _frame(0, 0.0)
    detections = [Detection(class_name="player", bbox=BBox(50, 40, 90, 150), confidence=0.9)]
    assert detector.update(state_from_tracks(frame, tracker.update(frame, detections))) == []


def test_a_coasted_ball_position_is_never_called_a_touch():
    # The tracker coasts the ball through an occlusion. Those positions are
    # predictions, not sightings, and must not produce events.
    config = Config()
    config.events.possession_min_frames = 1
    detector = StubEventDetector(config)
    state = FrameState(
        frame_index=4,
        timestamp_s=0.16,
        players=[PlayerState(player_id="Player 1", track_id=1,
                             bbox=BBox(50, 40, 90, 150), confidence=0.9)],
        ball=BallState(position=Point(70, 152), confidence=0.4, interpolated=True),
        frame_size=(320, 180),
    )
    assert detector.update(state) == []

    observed = FrameState(
        frame_index=5, timestamp_s=0.20, players=state.players,
        ball=BallState(position=Point(70, 152), confidence=0.4, interpolated=False),
        frame_size=(320, 180),
    )
    assert detector.update(observed), "an observed ball at the same place should count"


def test_stub_event_detector_does_not_guess_at_the_hard_events():
    # A placeholder that invented tackles and goals would be worse than useless.
    emitted = StubEventDetector().describe()["emits"]
    assert set(emitted) == {"ball_touch", "possession_change"}
