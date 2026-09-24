"""The ball-event rules piece by piece, and wired into the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.ball_events import (
    BallEventConfig,
    BallEventDetector,
    ClipState,
    GoalGeometry,
    NetMotionMeter,
    PlayerObs,
    detect_ball_events,
)
from football_analysis.ball_events.goal import GoalCheck, judge
from football_analysis.ball_events.possession import foot_distance
from football_analysis.ball_events.state import BRIDGED, DETECTED, SMOOTHED
from football_analysis.ball_events.synthetic import SCENARIOS, to_frame_states
from football_analysis.events import EventType
from football_analysis.interfaces import EventDetector
from football_analysis.pipeline import replay
from football_analysis.state import BallState, FrameState, GoalState, StateCacheWriter
from football_analysis.types import BBox, Point


# -- possession geometry ---------------------------------------------------------


def test_foot_distance_is_zero_at_the_feet_and_grows_in_player_heights() -> None:
    cfg = BallEventConfig()
    p = PlayerObs("P", (100, 0, 140, 100))
    assert foot_distance(np.array([120, 95]), p, cfg) == 0.0
    # 50 px right of the box edge on a 100 px player is half a height.
    assert foot_distance(np.array([190, 95]), p, cfg) == pytest.approx(0.5)
    # At head height, far from the foot band.
    assert foot_distance(np.array([120, 10]), p, cfg) > 0.5


# -- goal verdicts -----------------------------------------------------------------


def _check(**conditions: str) -> GoalCheck:
    base = {"entry": "pass", "absorption": "pass", "no_reemergence": "pass", "net_motion": "pass"}
    base.update(conditions)
    return judge(GoalCheck(entry_frame=10, conditions=base))


def test_four_conditions_is_a_confident_goal() -> None:
    c = _check()
    assert (c.verdict, c.confidence, c.review_reason) == ("goal", 0.95, None)


def test_three_conditions_and_no_reading_is_a_goal_with_less_confidence() -> None:
    c = _check(net_motion="unknown")
    assert (c.verdict, c.confidence) == ("goal", 0.85)


def test_a_ball_that_comes_back_out_is_not_a_goal() -> None:
    assert _check(no_reemergence="fail").verdict == "no_goal"


def test_a_ball_at_the_wrong_depth_is_not_a_goal() -> None:
    assert _check(entry="fail").verdict == "no_goal"


def test_split_evidence_is_flagged_for_review() -> None:
    c = _check(net_motion="fail")
    assert c.verdict == "possible_goal"
    assert "net_motion" in c.review_reason
    c = _check(absorption="unknown", net_motion="unknown")
    assert c.verdict == "possible_goal"
    assert c.confidence < 0.85


def test_goal_geometry_from_a_box() -> None:
    g = GoalGeometry.from_bbox((100, 50, 200, 110))
    assert g.height_px == pytest.approx(60)
    assert g.signed_distance((150, 80)) > 0
    assert g.signed_distance((250, 80)) == pytest.approx(-50)
    # 2 m goal, 0.22 m ball: 60 px posts predict a 6.6 px ball at that depth.
    assert g.expected_ball_diameter(2.0, 0.22) == pytest.approx(6.6)


# -- configuration ------------------------------------------------------------------


def test_config_rejects_unknown_keys_and_modes() -> None:
    with pytest.raises(ValueError):
        BallEventConfig.from_dict({"not_a_key": 1})
    with pytest.raises(ValueError):
        BallEventConfig.from_dict({"pass_mode": "sometimes"})


def test_config_reads_the_shared_event_settings() -> None:
    from football_analysis.config import Config

    cfg = Config()
    cfg.events.possession_min_frames = 5
    cfg.events.shot_goal_cone_deg = 20.0
    rules = BallEventConfig.from_config(cfg)
    assert (rules.possession_min_frames, rules.shot_cone_deg) == (5, 20.0)


@pytest.mark.parametrize(
    "scenario, rules, passes",
    [
        ("feeder_pass", {"feeders": ["Feeder"]}, 1),
        ("feeder_pass", {"feeders": ["Feeder"], "pass_mode": "never"}, 0),
        # Three identities and no feeder or teams: a transfer counts, at lower confidence.
        ("feeder_pass", {}, 1),
        # A strict 1v1: the ball reaching the other player is a turnover.
        ("turnover", {}, 0),
        ("turnover", {"pass_mode": "any"}, 1),
        ("turnover", {"teams": {"Player 1": "A", "Player 2": "A"}}, 1),
        ("turnover", {"teams": {"Player 1": "A", "Player 2": "B"}, "pass_mode": "teams"}, 0),
    ],
)
def test_pass_mode_decides_what_a_transfer_is(scenario: str, rules: dict, passes: int) -> None:
    result = detect_ball_events(SCENARIOS[scenario]().scene.build(), BallEventConfig.from_dict(rules))
    assert sum(e.type is EventType.PASS for e in result.events) == passes


def test_pass_confidence_is_lower_without_team_information() -> None:
    known = detect_ball_events(SCENARIOS["feeder_pass"]().scene.build(),
                               BallEventConfig.from_dict({"feeders": ["Feeder"]}))
    guessed = detect_ball_events(SCENARIOS["feeder_pass"]().scene.build())
    c_known = next(e for e in known.events if e.type is EventType.PASS).confidence
    c_guess = next(e for e in guessed.events if e.type is EventType.PASS).confidence
    assert c_known > c_guess


# -- the per-frame record ------------------------------------------------------------


def test_clip_state_from_frame_states_keeps_provenance_and_the_goal() -> None:
    clip = SCENARIOS["goal"]().scene.build()
    states = to_frame_states(clip)
    states[10].ball.interpolated = True
    rebuilt = ClipState.from_frame_states(states)
    assert len(rebuilt) == len(clip)
    assert rebuilt.ball_source[10] == SMOOTHED
    assert rebuilt.ball_source[11] == DETECTED
    assert rebuilt.goal is not None
    np.testing.assert_allclose(rebuilt.goal.quad, clip.goal.quad)
    np.testing.assert_allclose(rebuilt.net_motion, clip.net_motion)
    assert rebuilt.frame_size == clip.frame_size


def test_short_gaps_in_a_raw_track_are_bridged_and_marked() -> None:
    states = [
        FrameState(frame_index=k, timestamp_s=k / 25, frame_size=(640, 360),
                   ball=None if k in (3, 4) else BallState(Point(10.0 * k, 100.0), 0.9))
        for k in range(8)
    ]
    clip = ClipState.from_frame_states(states)
    assert list(clip.ball_source[2:6]) == [DETECTED, BRIDGED, BRIDGED, DETECTED]
    assert clip.ball_xy[3, 0] == pytest.approx(30.0)


def test_detector_satisfies_the_stage_contract() -> None:
    assert isinstance(BallEventDetector(), EventDetector)


def test_detector_decides_in_finalize() -> None:
    detector = BallEventDetector()
    clip = SCENARIOS["goal"]().scene.build()
    early = [e for s in to_frame_states(clip) for e in detector.update(s)]
    assert early == []
    events = detector.finalize()
    assert {e.type for e in events} >= {EventType.SHOT, EventType.GOAL}
    assert detector.possessor_at(1.0) == "Player 1"
    assert detector.describe()["summary"]["counts"]["goal"] == 1
    detector.reset()
    assert detector.finalize() == []


def test_replay_from_a_state_cache(tmp_path) -> None:
    from football_analysis.config import Config

    path = tmp_path / "state.jsonl"
    with StateCacheWriter(path) as writer:
        for state in to_frame_states(SCENARIOS["goal"]().scene.build()):
            writer.write(state)
    cfg = Config()
    cfg.events.enabled_types = ["shot", "goal"]
    timeline = replay(path, BallEventDetector(cfg), cfg)
    assert [e.type.value for e in timeline.events] == ["shot", "goal"]
    goal = timeline.events[1]
    assert goal.detail["shot_event_id"] == timeline.events[0].id


# -- net motion ------------------------------------------------------------------------


def _frame(net_shift: int = 0, ball_at: tuple[int, int] | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((360, 640, 3), 90, np.uint8)
    # A textured net, so moving it changes pixels.
    net = (rng.random((80, 80)) * 255).astype(np.uint8)
    img[140:220, 500:580] = np.roll(net, net_shift, axis=1)[:, :, None]
    if ball_at is not None:
        x, y = ball_at
        img[y - 5:y + 5, x - 5:x + 5] = 250
    return img


def _state(ball_at: tuple[int, int] | None = None) -> FrameState:
    ball = None
    if ball_at is not None:
        x, y = ball_at
        ball = BallState(Point(float(x), float(y)), 0.9, bbox=BBox(x - 5, y - 5, x + 5, y + 5))
    return FrameState(frame_index=0, timestamp_s=0.0, ball=ball, frame_size=(640, 360),
                      goal=GoalState(bbox=BBox(505, 145, 575, 215), confidence=0.9))


def test_net_motion_spikes_when_the_net_moves() -> None:
    meter = NetMotionMeter()
    assert meter.measure(_frame(), _state()) is None  # no previous frame yet
    still = meter.measure(_frame(), _state())
    moved = meter.measure(_frame(net_shift=3), _state())
    assert still == pytest.approx(0.0, abs=0.01)
    assert moved > 10 * max(still, 0.1)


def test_net_motion_ignores_the_ball_itself() -> None:
    meter = NetMotionMeter()
    meter.measure(_frame(ball_at=(520, 170)), _state((520, 170)))
    reading = meter.measure(_frame(ball_at=(540, 180)), _state((540, 180)))
    assert reading == pytest.approx(0.0, abs=0.05)


def test_net_motion_without_a_goal_is_no_reading() -> None:
    meter = NetMotionMeter()
    state = FrameState(frame_index=0, timestamp_s=0.0)
    meter.measure(_frame(), state)
    assert meter.measure(_frame(net_shift=3), state) is None
