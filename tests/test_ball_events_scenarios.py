"""The ball-event rules against scripted 1v1 moments with a known answer.

Each scenario in :mod:`football_analysis.ball_events.synthetic` is a short
moment -- a goal, a save, a post, a knock past the defender, a feeder's pass --
and the event it must produce.  Clean trajectories must all be right; with
detector-like noise and missed frames the rules must still be right nearly
always, and must never turn a non-goal into a goal.
"""

from __future__ import annotations

import pytest

from football_analysis.ball_events import BallEventConfig, detect_ball_events
from football_analysis.ball_events.synthetic import SCENARIOS
from football_analysis.events import EventType


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_clean_scenario(name: str) -> None:
    scenario = SCENARIOS[name]()
    result = detect_ball_events(scenario.scene.build(), BallEventConfig.from_dict(scenario.rules))
    assert scenario.failures(result) == [], [
        (e.clock, e.type.value, e.detail.get("outcome") or e.detail.get("cause")) for e in result.events
    ]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_noisy_scenario(name: str) -> None:
    """1.5 px of jitter and 15% of frames missed, over ten seeds."""
    passed = 0
    for seed in range(10):
        scenario = SCENARIOS[name]()
        scenario.scene.noise_px, scenario.scene.drop_rate, scenario.scene.seed = 1.5, 0.15, seed
        result = detect_ball_events(scenario.scene.build(), BallEventConfig.from_dict(scenario.rules))
        passed += not scenario.failures(result)
        if scenario.expect.get("goal") == 0:
            # Goal precision is the plan's hardest target: noise may cost a
            # label elsewhere, never a phantom goal.
            assert not any(e.type is EventType.GOAL for e in result.events), seed
    assert passed >= 9, f"{name}: {passed}/10"


def test_goal_links_to_its_shot() -> None:
    result = detect_ball_events(SCENARIOS["goal"]().scene.build())
    shot = next(e for e in result.events if e.type is EventType.SHOT)
    goal = next(e for e in result.events if e.type is EventType.GOAL)
    assert goal.detail["shot_event_id"] == shot.id
    assert goal.detail["scoring_player_id"] == shot.player_id == "Player 1"
    assert goal.timestamp_s > shot.timestamp_s
    assert goal.confidence == pytest.approx(0.95)
    assert set(goal.detail["conditions"].values()) == {"pass"}


def test_shot_timestamp_is_the_kick() -> None:
    result = detect_ball_events(SCENARIOS["goal"]().scene.build())
    shot = next(e for e in result.events if e.type is EventType.SHOT)
    assert shot.timestamp_s == pytest.approx(2.0, abs=0.08)


def test_every_event_says_which_rule_fired() -> None:
    for name in SCENARIOS:
        result = detect_ball_events(SCENARIOS[name]().scene.build())
        for event in result.events:
            assert event.source and event.source.startswith("ball_events.")
            assert 0.0 <= event.confidence <= 1.0


def test_goal_in_a_track_gap_is_flagged_not_called() -> None:
    result = detect_ball_events(SCENARIOS["goal_in_gap"]().scene.build())
    assert not [e for e in result.events if e.type is EventType.GOAL]
    shot = next(e for e in result.events if e.type is EventType.SHOT)
    assert shot.detail["needs_review"] is True
    assert result.review and result.review[0]["kind"] == "shot"


def test_dribble_is_one_possession() -> None:
    result = detect_ball_events(SCENARIOS["dribble"]().scene.build())
    assert [s.player_id for s in result.spells] == ["Player 1"]
    assert result.possession.possessor[50] == "Player 1"
