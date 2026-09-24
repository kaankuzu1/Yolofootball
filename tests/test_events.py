"""The output schema: validation, ids, ordering, and JSON round-tripping."""

from __future__ import annotations

import json

import pytest

from football_analysis.events import (
    SCHEMA_VERSION,
    Event,
    EventTimeline,
    EventType,
    PlayerInfo,
    SourceInfo,
)


def _source() -> SourceInfo:
    return SourceInfo(path="clip.mp4", width=1920, height=1080, fps=30.0)


def test_primary_types_are_the_five_asked_for():
    assert {t.value for t in EventType.primary()} == {
        "tackle", "trick", "pass", "shot", "goal"
    }
    assert EventType.GOAL.is_primary
    assert not EventType.BALL_TOUCH.is_primary


def test_event_rejects_impossible_values():
    with pytest.raises(ValueError):
        Event(type=EventType.PASS, timestamp_s=1.0, confidence=1.5)
    with pytest.raises(ValueError):
        Event(type=EventType.PASS, timestamp_s=-1.0, confidence=0.5)
    with pytest.raises(ValueError):
        Event(type=EventType.TACKLE, timestamp_s=5.0, confidence=0.5, end_timestamp_s=4.0)


def test_event_clock_is_human_readable():
    assert Event(type=EventType.GOAL, timestamp_s=73.4, confidence=0.9).clock == "01:13.400"
    assert Event(type=EventType.GOAL, timestamp_s=0.0, confidence=0.9).clock == "00:00.000"


def test_event_duration_and_label():
    event = Event(
        type=EventType.TRICK,
        timestamp_s=10.0,
        end_timestamp_s=11.5,
        confidence=0.8,
        player_id="Player 1",
        detail={"trick_name": "nutmeg"},
    )
    assert event.duration_s == pytest.approx(1.5)
    assert event.label() == "nutmeg Player 1"


def test_timeline_assigns_stable_ids_and_counts():
    timeline = EventTimeline(source=_source())
    timeline.add(Event(type=EventType.PASS, timestamp_s=1.0, confidence=0.7))
    timeline.add(Event(type=EventType.PASS, timestamp_s=2.0, confidence=0.7))
    timeline.add(Event(type=EventType.GOAL, timestamp_s=3.0, confidence=0.9))

    assert [e.id for e in timeline] == ["evt_0001", "evt_0002", "evt_0003"]
    assert timeline.counts() == {"pass": 2, "goal": 1}
    assert len(timeline) == 3


def test_timeline_sorts_by_time_without_renaming_ids():
    timeline = EventTimeline(source=_source())
    late = timeline.add(Event(type=EventType.GOAL, timestamp_s=9.0, confidence=0.9))
    early = timeline.add(Event(type=EventType.PASS, timestamp_s=1.0, confidence=0.7))
    timeline.sort()

    assert [e.id for e in timeline] == [early.id, late.id]
    assert late.id == "evt_0001"  # ids survive sorting, so cross-references hold


def test_timeline_filters_by_type():
    timeline = EventTimeline(source=_source())
    timeline.add(Event(type=EventType.BALL_TOUCH, timestamp_s=1.0, confidence=0.5))
    timeline.add(Event(type=EventType.SHOT, timestamp_s=2.0, confidence=0.8))

    assert [e.type for e in timeline.primary()] == [EventType.SHOT]
    assert len(timeline.of_type(EventType.BALL_TOUCH)) == 1


def test_timeline_json_round_trip():
    timeline = EventTimeline(source=_source())
    timeline.players.append(PlayerInfo(player_id="Player 1", label="Player 1", track_id=1))
    timeline.add(
        Event(
            type=EventType.TACKLE,
            timestamp_s=12.25,
            end_timestamp_s=12.9,
            frame_index=367,
            confidence=0.71,
            player_id="Player 1",
            secondary_player_id="Player 2",
            track_ids=[1, 2],
            detail={"winner_player_id": "Player 1"},
            source="rules.tackle",
        )
    )

    payload = json.loads(timeline.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["events"][0]["type"] == "tackle"
    assert payload["events"][0]["duration_s"] == pytest.approx(0.65)

    restored = EventTimeline.from_dict(payload)
    original, copy = timeline.events[0], restored.events[0]
    assert copy.type is original.type
    assert copy.timestamp_s == pytest.approx(original.timestamp_s)
    assert copy.secondary_player_id == original.secondary_player_id
    assert copy.track_ids == original.track_ids
    assert copy.detail == original.detail
    assert restored.source.fps == pytest.approx(30.0)
    assert restored.players[0].track_id == 1


def test_timeline_writes_a_file(tmp_path):
    timeline = EventTimeline(source=_source())
    timeline.add(Event(type=EventType.SHOT, timestamp_s=4.0, confidence=0.6))
    path = timeline.write_json(tmp_path / "nested" / "events.json")

    assert path.exists()
    assert EventTimeline.read_json(path).events[0].type is EventType.SHOT


def test_uncertain_events_are_surfaced_for_review():
    # A rule that is unsure says so rather than quietly picking a side; the
    # timeline has to carry that forward or the flag is pointless.
    timeline = EventTimeline(source=_source())
    timeline.add(Event(type=EventType.GOAL, timestamp_s=10.0, confidence=0.55,
                       detail={"needs_review": True,
                               "review_reason": "ball lost behind the keeper"}))
    timeline.add(Event(type=EventType.PASS, timestamp_s=11.0, confidence=0.9))

    flagged = timeline.needs_review()
    assert [e.type for e in flagged] == [EventType.GOAL]

    payload = json.loads(timeline.to_json())
    assert payload["needs_review"] == [
        {
            "id": "evt_0001",
            "type": "goal",
            "clock": "00:10.000",
            "reason": "ball lost behind the keeper",
        }
    ]


def test_a_clean_timeline_surfaces_an_empty_review_list():
    timeline = EventTimeline(source=_source())
    timeline.add(Event(type=EventType.PASS, timestamp_s=1.0, confidence=0.9))
    assert timeline.needs_review() == []
    assert json.loads(timeline.to_json())["needs_review"] == []
