"""Tier 2: pose features, candidate windows, the nutmeg rule, the stage itself.

The model tests train a tiny classifier on a few simulated clips, so they check
the plumbing and the contracts, not accuracy.  Accuracy is measured by
``python -m football_analysis.body_events evaluate``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from football_analysis.body_events.dataset import WindowDataset, synthetic_samples
from football_analysis.body_events.features import (
    CONTACT_FEATURES,
    SKILL_FEATURES,
    contact_features,
    contact_windows,
    skill_features,
    skill_windows,
)
from football_analysis.body_events.rules import find_nutmegs
from football_analysis.body_events.series import ClipSeries
from football_analysis.body_events.synthetic import SCENARIOS, generate
from football_analysis.events import Event, EventType
from football_analysis.interfaces import EventDetector
from football_analysis.state import BallState, FrameState, Keypoint, PlayerState
from football_analysis.types import BBox, Point


def _scaled(states: list[FrameState], k: float) -> list[FrameState]:
    """The same clip filmed k times bigger: every coordinate times k."""
    out = []
    for s in states:
        players = [
            replace(p, bbox=BBox(*(v * k for v in p.bbox.as_tuple())),
                    keypoints=[Keypoint(q.name, q.x * k, q.y * k, q.confidence) for q in p.keypoints])
            for p in s.players
        ]
        ball = None if s.ball is None else replace(
            s.ball, position=Point(s.ball.position.x * k, s.ball.position.y * k), bbox=None)
        out.append(replace(s, players=players, ball=ball))
    return out


# -- the simulator ---------------------------------------------------------------


def test_simulator_is_reproducible():
    a, b = generate("tackle_won", 5), generate("tackle_won", 5)
    assert len(a.states) == len(b.states)
    assert a.states[10].players[0].bbox == b.states[10].players[0].bbox
    assert generate("tackle_won", 6).states[10].players[0].bbox != a.states[10].players[0].bbox


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_scenario_renders_two_posed_players(scenario):
    seq = generate(scenario, 3)
    assert {p.player_id for p in seq.states[0].players} == {seq.attacker_id, seq.defender_id}
    posed = sum(1 for s in seq.states for p in s.players if len(p.keypoints) == 17)
    assert posed > 0.6 * 2 * len(seq.states)
    assert 0 <= seq.event_start_s <= seq.event_end_s <= seq.states[-1].timestamp_s


# -- windows and features --------------------------------------------------------


def test_contact_window_lands_on_the_tackle():
    seq = generate("tackle_won", 11)
    clip = ClipSeries.from_states(seq.states)
    windows = contact_windows(clip)
    assert windows
    tc = seq.params["contact_s"]
    best = min(windows, key=lambda w: abs(clip.t[w.anchor] - tc))
    assert abs(clip.t[best.anchor] - tc) < 0.35
    assert best.defender_id == seq.defender_id


def test_no_contact_window_in_a_lone_dribble():
    clip = ClipSeries.from_states(generate("dribble", 4).states)
    assert contact_windows(clip) == []
    assert skill_windows(clip)


def test_features_are_complete_and_scale_free():
    seq = generate("tackle_failed", 2)
    small = ClipSeries.from_states(seq.states)
    big = ClipSeries.from_states(_scaled(seq.states, 2.0))
    ws, wb = contact_windows(small), contact_windows(big)
    assert len(ws) == len(wb) >= 1
    fs, fb = contact_features(small, ws[0]), contact_features(big, wb[0])
    assert set(fs) == set(CONTACT_FEATURES)
    for name in ("min_gap_h", "min_def_ankle_ball_h", "def_leg_ext_max", "att_has_before"):
        assert fs[name] == pytest.approx(fb[name], rel=1e-6, abs=1e-9), name

    sk_s, sk_b = skill_windows(small), skill_windows(big)
    f1, f2 = skill_features(small, sk_s[0]), skill_features(big, sk_b[0])
    assert set(f1) == set(SKILL_FEATURES)
    for name in ("foot_rel_per_ball", "ball_disp_h", "hip_sway_h"):
        assert f1[name] == pytest.approx(f2[name], rel=1e-6, abs=1e-9), name


def test_features_tolerate_missing_pose_and_ball():
    seq = generate("tackle_won", 8)
    stripped = [replace(s, ball=None, players=[replace(p, keypoints=[]) for p in s.players])
                for s in seq.states]
    clip = ClipSeries.from_states(stripped)
    for w in contact_windows(clip):
        f = contact_features(clip, w)
        assert np.isnan(f["min_def_ankle_ball_h"])
        assert not np.isnan(f["min_gap_h"])


def test_synthetic_labels_follow_the_script():
    samples = synthetic_samples(generate("stepover", 21))
    tricks = [s for s in samples if s.kind == "skill" and s.label == "trick"]
    assert tricks and all(s.sub_label == "stepover" for s in tricks)
    tackle = synthetic_samples(generate("tackle_won", 21))
    assert any(s.kind == "contact" and s.label == "tackle_won" for s in tackle)


# -- the nutmeg rule --------------------------------------------------------------


def test_nutmeg_rule_fires_on_a_nutmeg_and_names_the_players():
    fired = 0
    for seed in range(6):
        seq = generate("nutmeg", seed)
        calls = find_nutmegs(ClipSeries.from_states(seq.states))
        if calls:
            fired += 1
            assert calls[0].attacker_id == seq.attacker_id
            assert calls[0].defender_id == seq.defender_id
    assert fired >= 4


def test_nutmeg_rule_ignores_a_ball_at_another_depth():
    """Push the ball a little further from the camera: in the image it still
    crosses the gap between the feet, but it no longer passes *through* it."""
    seq = generate("nutmeg", 1)
    assert find_nutmegs(ClipSeries.from_states(seq.states))
    h = np.median([p.bbox.height for s in seq.states for p in s.players])
    shifted = [s if s.ball is None else replace(
        s, ball=replace(s.ball, position=Point(s.ball.position.x, s.ball.position.y - 0.15 * h)))
        for s in seq.states]
    assert find_nutmegs(ClipSeries.from_states(shifted)) == []


def test_nutmeg_rule_quiet_on_plain_dribbles():
    for seed in range(4):
        assert find_nutmegs(ClipSeries.from_states(generate("dribble", seed).states)) == []


# -- the stage ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    pytest.importorskip("sklearn")
    from football_analysis.body_events.model import BodyEventModel

    samples = [s for name in SCENARIOS for seed in range(6)
               for s in synthetic_samples(generate(name, 1000 + seed))]
    return BodyEventModel.train(samples)


def test_detector_satisfies_the_protocol(tiny_model):
    from football_analysis.body_events import BodyEventDetector

    assert isinstance(BodyEventDetector(model=tiny_model), EventDetector)


def test_detector_emits_schema_valid_events(tiny_model):
    from football_analysis.body_events import BodyEventDetector

    det = BodyEventDetector(model=tiny_model)
    seq = generate("tackle_won", 77)
    for s in seq.states:
        assert det.update(s) == []
    events = det.finalize()
    for e in events:
        assert isinstance(e, Event)
        assert e.type in (EventType.TACKLE, EventType.TRICK)
        assert 0 <= e.confidence <= 1
        assert "needs_review" in e.detail and isinstance(e.detail["review_reasons"], list)
        Event.from_dict(e.to_dict())  # round-trips through the JSON schema
    tackles = [e for e in events if e.type is EventType.TACKLE]
    if tackles:
        t = tackles[0]
        assert t.detail["outcome"] in ("won", "failed")
        assert {t.player_id, t.secondary_player_id} == {seq.attacker_id, seq.defender_id}
    # finalize leaves it ready for the next clip
    assert det.finalize() == []



def test_describe_reports_what_the_stage_looked_at(tiny_model):
    """So "no tackles" and "nothing usable to read" are told apart in stats."""
    from football_analysis.body_events import BodyEventDetector

    det = BodyEventDetector(model=tiny_model)
    assert "summary" not in det.describe()
    for s in generate("tackle_won", 77).states:
        det.update(s)
    events = det.finalize()
    summary = det.describe()["summary"]
    assert summary["players"] == 2 and summary["abstained_reason"] is None
    assert summary["contact"]["windows"] >= 1
    assert summary["contact"]["reported"] == sum(e.type is EventType.TACKLE for e in events)
    assert 0 < summary["pose_coverage"] <= 1

def test_model_round_trips_through_disk(tiny_model, tmp_path):
    from football_analysis.body_events.model import BodyEventModel

    path = tiny_model.save(tmp_path / "m.joblib")
    loaded = BodyEventModel.load(path)
    clip = ClipSeries.from_states(generate("dribble", 3).states)
    rows = [skill_features(clip, w) for w in skill_windows(clip)]
    assert np.allclose(loaded.skill.predict_proba(rows), tiny_model.skill.predict_proba(rows))


def test_single_player_clip_gives_no_events(tiny_model):
    from football_analysis.body_events import detect_body_events

    seq = generate("dribble", 2)
    solo = [replace(s, players=s.players[:1]) for s in seq.states]
    assert detect_body_events(solo, tiny_model) == []



def test_no_tackle_without_the_ball_seen_near_the_pair(tiny_model):
    """With the ball never detected there is nothing to judge a leg against,
    so the stage abstains instead of guessing."""
    from football_analysis.body_events import detect_body_events

    seq = generate("tackle_won", 77)
    blind = [replace(s, ball=None) for s in seq.states]
    assert [e for e in detect_body_events(blind, tiny_model) if e.type is EventType.TACKLE] == []


def test_nutmeg_rule_does_not_fire_on_the_player_who_had_the_ball():
    """Whoever was on the ball on the way in is the attacker, never the victim."""
    for seed in range(20):
        seq = generate("nutmeg", 500 + seed)
        for call in find_nutmegs(ClipSeries.from_states(seq.states)):
            assert call.defender_id == seq.defender_id

# -- the dataset store ---------------------------------------------------------------


def test_dataset_round_trip(tmp_path):
    seq = generate("tackle_failed", 9)
    clip = ClipSeries.from_states(seq.states)
    w = contact_windows(clip)[0]
    ds = WindowDataset(tmp_path)
    ds.add_window("clip_c_0001", seq.states, {
        "kind": "contact", "label": "tackle_failed", "source_clip": "clip.mp4",
        "start_s": clip.t[w.start], "end_s": clip.t[w.end], "anchor_s": clip.t[w.anchor],
        "player_id": w.attacker_id, "other_player_id": w.defender_id, "origin": "real",
    })
    ds.add_window("clip_s_0002", seq.states, {
        "kind": "skill", "label": "", "source_clip": "clip.mp4", "start_s": 0.5, "end_s": 1.7,
        "player_id": seq.attacker_id, "origin": "real",
    })
    samples = ds.samples()
    assert len(samples) == 1  # the unlabelled window is not a sample
    assert samples[0].label == "tackle_failed" and samples[0].origin == "real"
    # Stored records round keypoints to 0.01 px, so features agree to that.
    assert samples[0].features["min_gap_h"] == pytest.approx(
        contact_features(clip, w)["min_gap_h"], rel=1e-3)
    assert ds.counts() == {"contact:tackle_failed": 1, "skill:unlabelled": 1}


def test_dataset_rejects_an_unknown_label(tmp_path):
    seq = generate("dribble", 1)
    ds = WindowDataset(tmp_path)
    ds.add_window("w", seq.states, {"kind": "skill", "label": "rabona", "start_s": 0.5,
                                    "end_s": 1.5, "player_id": seq.attacker_id})
    with pytest.raises(ValueError, match="rabona"):
        ds.samples()


# -- pose ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = _FakeTensor(xyxy)

    def __len__(self):
        return len(self.xyxy.a)


class _FakeResult:
    def __init__(self, boxes, kps):
        kps = np.asarray(kps, dtype=float)
        self.boxes = _FakeBoxes(boxes)
        self.keypoints = type("K", (), {"xy": _FakeTensor(kps[..., :2]),
                                        "conf": _FakeTensor(kps[..., 2])})()


class _FakePose:
    """Returns, for every crop, two skeletons: one filling the crop's middle
    and a neighbour at the left edge -- the overlapping-players case."""

    def predict(self, crops, **kwargs):
        out = []
        for c in crops:
            h, w = c.shape[:2]
            main = [w * 0.25, h * 0.15, w * 0.75, h * 0.85]
            edge = [0, h * 0.2, w * 0.2, h * 0.9]
            kp_main = np.tile([w * 0.5, h * 0.5, 0.9], (17, 1))
            kp_edge = np.tile([w * 0.1, h * 0.5, 0.9], (17, 1))
            out.append(_FakeResult([edge, main], [kp_edge, kp_main]))
        return out


def test_pose_keeps_the_tracked_players_skeleton_in_frame_pixels():
    from football_analysis.pose import PoseConfig, PoseEstimator

    est = PoseEstimator(PoseConfig(crop_height=200, pad=0.2), model=_FakePose())
    image = np.zeros((720, 1280, 3), np.uint8)
    box = BBox(600, 300, 660, 450)
    (kps,) = est.estimate(image, [box])
    assert len(kps) == 17 and kps[15].name == "left_ankle"
    # The picked skeleton is the one centred on the box, mapped back to the frame.
    assert kps[0].x == pytest.approx(box.center.x, abs=2)
    assert kps[0].y == pytest.approx(box.center.y, abs=2)


def test_pose_skips_players_too_small_to_read():
    from football_analysis.pose import PoseConfig, PoseEstimator

    est = PoseEstimator(PoseConfig(min_box_height=30), model=_FakePose())
    assert est.estimate(np.zeros((100, 100, 3), np.uint8), [BBox(10, 10, 20, 30)]) == [[]]


def test_pose_tracker_returns_new_snapshots_with_keypoints():
    from football_analysis.pose import PoseConfig, PoseEstimator, PoseTracker
    from football_analysis.types import Track, VideoFrame

    kept = Track(track_id=1, class_name="player", bbox=BBox(600, 300, 660, 450),
                 confidence=0.9, frame_index=0, timestamp_s=0.0)

    class Inner:
        def update(self, frame, detections):
            return [kept]

    tracker = PoseTracker(Inner(), PoseEstimator(PoseConfig(), model=_FakePose()))
    frame = VideoFrame(index=0, timestamp_s=0.0, image=np.zeros((720, 1280, 3), np.uint8),
                       source_size=(1280, 720))
    (out,) = tracker.update(frame, [])
    assert out is not kept and "keypoints" not in kept.attributes
    assert len(out.attributes["keypoints"]) == 17

    from football_analysis.state import state_from_tracks

    state = state_from_tracks(frame, [out])
    assert state.players[0].keypoint("left_ankle") is not None


def test_stage_reads_the_live_pipelines_state_cache(tiny_model):
    """Contract check on records shaped exactly as the pipeline writes them:
    it triggers nothing, but a renamed field or lost keypoints would show."""
    from pathlib import Path

    from football_analysis.body_events import detect_body_events
    from football_analysis.state import read_state_cache

    path = Path(__file__).resolve().parents[1] / "assets/body_events/real/broadcast_08fd33_4.2p.states.jsonl"
    if not path.exists():
        pytest.skip("live-pipeline state cache not present")
    summary: dict = {}
    detect_body_events(list(read_state_cache(path)), tiny_model, summary=summary)
    assert summary["players"] == 2 and summary["abstained_reason"] is None
    assert summary["pose_coverage"] > 0.9 and summary["ball_observed_frac"] > 0.5
