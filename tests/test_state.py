"""The world-state record: the interface event logic reads, and its cache."""

from __future__ import annotations

import json

import numpy as np
import pytest

from football_analysis.state import (
    STATE_SCHEMA_VERSION,
    BallState,
    FrameState,
    GoalState,
    Keypoint,
    PlayerState,
    StateCacheWriter,
    read_state_cache,
    state_from_tracks,
)
from football_analysis.types import BBox, Point, Track, VideoFrame


def _frame(index=4, timestamp_s=0.16, width=320, height=180, scale=0.5):
    return VideoFrame(
        index=index,
        timestamp_s=timestamp_s,
        image=np.zeros((height, width, 3), np.uint8),
        source_size=(int(width / scale), int(height / scale)),
        scale=scale,
    )


def _track(track_id, class_name, box, **kwargs):
    return Track(
        track_id=track_id,
        class_name=class_name,
        bbox=BBox(*box),
        confidence=kwargs.pop("confidence", 0.8),
        frame_index=kwargs.pop("frame_index", 4),
        timestamp_s=kwargs.pop("timestamp_s", 0.16),
        **kwargs,
    )


def test_record_is_built_from_tracks():
    state = state_from_tracks(
        _frame(),
        [
            _track(1, "player", (20, 40, 60, 150), label="Player 1"),
            _track(2, "ball", (70, 140, 82, 152), confidence=0.4),
            _track(3, "goal", (280, 60, 318, 150), confidence=0.7),
        ],
    )

    assert [p.player_id for p in state.players] == ["Player 1"]
    assert state.ball is not None and state.goal is not None
    assert state.ball.position.as_tuple() == (76.0, 146.0)
    assert state.frame_size == (320, 180)
    assert state.scale == pytest.approx(0.5)
    assert state.frame_index == 4


def test_a_coasted_ball_is_marked_interpolated():
    # An unmatched ball track is a prediction, not a sighting.
    state = state_from_tracks(
        _frame(), [_track(2, "ball", (70, 140, 82, 152), time_since_update=4)]
    )
    assert state.ball.interpolated is True
    assert state.ball.is_observed is False
    assert state.ball.frames_since_seen == 4


def test_an_observed_ball_is_not_marked_interpolated():
    state = state_from_tracks(
        _frame(), [_track(2, "ball", (70, 140, 82, 152), time_since_update=0)]
    )
    assert state.ball.interpolated is False
    assert state.ball.is_observed is True


def test_a_tracker_that_reports_coasting_itself_is_believed():
    # A Kalman ball tracker knows it is coasting even on a frame it "matched".
    state = state_from_tracks(
        _frame(),
        [_track(2, "ball", (70, 140, 82, 152), attributes={"interpolated": True})],
    )
    assert state.ball.interpolated is True


def test_pose_keypoints_ride_through_on_a_track():
    state = state_from_tracks(
        _frame(),
        [
            _track(1, "player", (20, 40, 60, 150), label="Player 1",
                   attributes={"keypoints": [
                       {"name": "left_ankle", "x": 25.0, "y": 148.0, "confidence": 0.7}
                   ]}),
        ],
    )
    ankle = state.players[0].keypoint("left_ankle")
    assert ankle is not None
    assert ankle.point.as_tuple() == (25.0, 148.0)
    assert state.players[0].keypoint("nose") is None


def test_goal_quad_rides_through_on_a_track():
    quad = [{"x": 280.0, "y": 60.0}, {"x": 318.0, "y": 60.0},
            {"x": 318.0, "y": 150.0}, {"x": 280.0, "y": 150.0}]
    state = state_from_tracks(
        _frame(), [_track(3, "goal", (280, 60, 318, 150), attributes={"quad": quad})]
    )
    assert len(state.goal.quad) == 4
    assert state.goal.quad[0].as_tuple() == (280.0, 60.0)


def test_a_frame_with_nothing_in_it_is_still_a_valid_record():
    state = state_from_tracks(_frame(), [])
    assert state.players == []
    assert state.ball is None
    assert state.nearest_player_to_ball() is None


def test_nearest_player_measures_from_the_feet():
    state = state_from_tracks(
        _frame(),
        [
            _track(1, "player", (20, 40, 60, 150), label="Player 1"),
            _track(2, "player", (200, 40, 240, 150), label="Player 2"),
            _track(3, "ball", (44, 144, 56, 156)),
        ],
    )
    player, distance = state.nearest_player_to_ball()
    assert player.player_id == "Player 1"
    # Feet are at (40, 150); the ball is at (50, 150).
    assert distance == pytest.approx(10.0, abs=0.5)


def test_lookup_helpers():
    state = state_from_tracks(
        _frame(), [_track(1, "player", (20, 40, 60, 150), label="Player 1")]
    )
    state.possessor_id = "Player 1"

    assert state.player("Player 1").track_id == 1
    assert state.player("Player 9") is None
    assert state.player_by_track(1).player_id == "Player 1"
    assert state.possessor.player_id == "Player 1"


def test_record_round_trips_through_json():
    original = FrameState(
        frame_index=12,
        timestamp_s=0.48,
        players=[
            PlayerState(
                player_id="Player 1", track_id=1, bbox=BBox(20, 40, 60, 150),
                confidence=0.91, velocity=Point(12.0, -3.0),
                keypoints=[Keypoint("left_ankle", 25.0, 148.0, 0.7)],
            )
        ],
        ball=BallState(
            position=Point(70.0, 146.0), confidence=0.35, bbox=BBox(64, 140, 76, 152),
            interpolated=True, track_id=2, frames_since_seen=3,
        ),
        goal=GoalState(bbox=BBox(280, 60, 318, 150), confidence=0.7,
                       quad=[Point(280, 60), Point(318, 60)]),
        possessor_id="Player 1",
        frame_size=(320, 180),
        scale=0.5,
        timestamp_is_exact=False,
    )

    restored = FrameState.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.frame_index == 12
    assert restored.possessor_id == "Player 1"
    assert restored.timestamp_is_exact is False
    assert restored.players[0].keypoints[0].name == "left_ankle"
    assert restored.players[0].velocity.as_tuple() == (12.0, -3.0)
    assert restored.ball.interpolated is True
    assert restored.ball.frames_since_seen == 3
    assert len(restored.goal.quad) == 2


def test_cache_writes_json_lines_with_a_header(tmp_path):
    path = tmp_path / "state.jsonl"
    with StateCacheWriter(path) as writer:
        for i in range(3):
            writer.write(state_from_tracks(_frame(index=i, timestamp_s=i * 0.04), []))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {
        "schema_version": STATE_SCHEMA_VERSION, "record": "header"
    }
    assert len(lines) == 4  # header plus three records
    assert writer.records_written == 3


def test_cache_streams_back_in_order(tmp_path):
    path = tmp_path / "state.jsonl"
    with StateCacheWriter(path) as writer:
        for i in range(5):
            writer.write(state_from_tracks(_frame(index=i, timestamp_s=i * 0.04), []))

    records = list(read_state_cache(path))
    assert [r.frame_index for r in records] == [0, 1, 2, 3, 4]


def test_cache_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "state.jsonl"
    with StateCacheWriter(path) as writer:
        writer.write(state_from_tracks(_frame(), []))
    assert path.exists()


def test_writing_to_a_closed_cache_is_refused(tmp_path):
    writer = StateCacheWriter(tmp_path / "state.jsonl")
    writer.close()
    with pytest.raises(ValueError, match="closed"):
        writer.write(state_from_tracks(_frame(), []))


def test_a_cache_from_another_schema_is_refused(tmp_path):
    path = tmp_path / "state.jsonl"
    path.write_text('{"schema_version": "99.0", "record": "header"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="state schema"):
        list(read_state_cache(path))


def test_a_corrupt_record_names_its_line(tmp_path):
    path = tmp_path / "state.jsonl"
    path.write_text(
        '{"schema_version": "%s", "record": "header"}\n{"nonsense": true}\n'
        % STATE_SCHEMA_VERSION,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":2"):
        list(read_state_cache(path))
