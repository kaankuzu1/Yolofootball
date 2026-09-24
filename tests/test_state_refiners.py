"""The second-pass seam: readings taken once the tracker has settled the boxes.

Pose is what this exists for. Reading a skeleton costs real time and is only
worth spending on the players who turned out to be real, on the boxes the
tracker finally settled on. So it does not belong in the analysis loop next to
the observers; it belongs after ``finalize_states`` and before the rules.

The ordering is the whole point, and all of it is testable without a pose
model: a refiner must see the settled records, and what it adds must reach both
the state cache and the rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from football_analysis.adapters import PoseRefiner
from football_analysis.config import Config, ConfigError
from football_analysis.interfaces import BaseEventDetector, BaseTracker, StateRefiner
from football_analysis.pipeline import (
    AnalysisPipeline,
    default_event_stage,
    default_state_refiners,
    find_weights,
)
from football_analysis.state import BallState, FrameState, Keypoint, PlayerState, read_state_cache
from football_analysis.types import BBox, Detection, Point, Track


class _OnePlayerDetector:
    def detect(self, frame):
        return [Detection(class_name="player", bbox=BBox(10, 10, 50, 150), confidence=0.9)]

    def reset(self):
        return None


class _SettlingTracker(BaseTracker):
    """Provisional in the loop, settled only at the end -- the two-phase case."""

    def __init__(self):
        self._frames = []

    def update(self, frame, detections):
        self._frames.append((frame.index, frame.timestamp_s))
        return [
            Track(
                track_id=99, class_name="player", bbox=BBox(10, 10, 50, 150),
                confidence=0.5, frame_index=frame.index,
                timestamp_s=frame.timestamp_s, label="Provisional",
            )
        ]

    def finalize_states(self):
        return [
            FrameState(
                frame_index=index,
                timestamp_s=timestamp,
                players=[
                    PlayerState(
                        player_id="Player 1", track_id=1,
                        bbox=BBox(10, 10, 50, 150), confidence=0.95,
                    )
                ],
                ball=BallState(position=Point(60, 140), confidence=0.6),
                frame_size=(320, 180),
            )
            for index, timestamp in self._frames
        ]

    def reset(self):
        self._frames = []


class _RecordingEvents(BaseEventDetector):
    def __init__(self):
        self.seen = []

    def update(self, state):
        self.seen.append(state)
        return []


class _SpyRefiner:
    """Stands in for pose: notes what it was handed, marks every record."""

    def __init__(self, *, raises=False, returns=None):
        self.calls = []
        self.resets = 0
        self.raises = raises
        self.returns = returns

    def refine(self, states, video_path):
        self.calls.append((list(states), video_path))
        if self.raises:
            raise RuntimeError("no weights")
        for state in states:
            for player in state.players:
                player.keypoints = [Keypoint(name="left_ankle", x=20.0, y=145.0, confidence=0.8)]
        return states if self.returns is None else self.returns

    def reset(self):
        self.resets += 1

    def describe(self):
        return {"name": "_SpyRefiner"}


def _run(sample_clip, config, refiners, events=None):
    path, _ = sample_clip
    events = events or _RecordingEvents()
    result = AnalysisPipeline(
        config,
        detector=_OnePlayerDetector(),
        tracker=_SettlingTracker(),
        event_detector=events,
        refiners=list(refiners),
    ).run(path)
    return result, events


# -- the seam itself ---------------------------------------------------------


def test_a_refiner_sees_the_settled_records_not_the_provisional_ones(sample_clip, config):
    refiner = _SpyRefiner()
    _run(sample_clip, config, [refiner])

    assert len(refiner.calls) == 1
    states, video_path = refiner.calls[0]
    assert states, "the refiner should have been handed the run's records"
    # Provisional said "Provisional"/99. If a refiner ran before the tracker
    # settled, it would attach a skeleton to an identity about to be replaced.
    assert all(s.players[0].player_id == "Player 1" for s in states)
    assert Path(video_path) == Path(sample_clip[0])


def test_what_a_refiner_adds_reaches_the_rules(sample_clip, config):
    events = _RecordingEvents()
    _run(sample_clip, config, [_SpyRefiner()], events=events)

    assert events.seen
    assert all(s.players[0].keypoints for s in events.seen), (
        "the rules must read the refined records, or tackle detection has no pose"
    )


def test_what_a_refiner_adds_reaches_the_state_cache(sample_clip, config, tmp_path):
    # The cache is written after the refiners on purpose: a --replay that could
    # not see the keypoints would silently score differently from the live run.
    config.output.state_cache_path = tmp_path / "states.jsonl"
    _run(sample_clip, config, [_SpyRefiner()])

    cached = list(read_state_cache(config.output.state_cache_path))
    assert cached
    assert all(s.players[0].keypoints for s in cached)
    assert cached[0].players[0].keypoints[0].name == "left_ankle"


def test_refiners_run_in_order_each_seeing_the_last_one_s_work(sample_clip, config):
    first, second = _SpyRefiner(), _SpyRefiner()
    _run(sample_clip, config, [first, second])

    assert second.calls, "the second refiner should have run"
    states, _ = second.calls[0]
    assert all(s.players[0].keypoints for s in states)


def test_refiners_are_reset_before_each_run(sample_clip, config):
    refiner = _SpyRefiner()
    _run(sample_clip, config, [refiner])
    assert refiner.resets == 1


# -- failure is not fatal ----------------------------------------------------


def test_a_refiner_that_raises_costs_its_readings_not_the_run(sample_clip, config):
    # Pose weights that will not load should cost the tackles, not the goals.
    events = _RecordingEvents()
    result, _ = _run(sample_clip, config, [_SpyRefiner(raises=True)], events=events)

    assert result.timeline is not None
    assert events.seen, "the rules should still have run"
    assert not any(s.players[0].keypoints for s in events.seen)


def test_a_refiner_that_returns_nothing_does_not_throw_the_clip_away(sample_clip, config):
    events = _RecordingEvents()
    _run(sample_clip, config, [_SpyRefiner(returns=[])], events=events)
    assert events.seen, "an empty return is a bug in the refiner, not an empty clip"


def test_a_later_refiner_still_runs_after_an_earlier_one_failed(sample_clip, config):
    second = _SpyRefiner()
    _run(sample_clip, config, [_SpyRefiner(raises=True), second])
    assert second.calls


# -- the pose refiner --------------------------------------------------------


class _FakeEstimator:
    """A pose model that puts one keypoint on every box it is given."""

    def __init__(self):
        self.boxes_seen = 0

    def estimate(self, image, boxes):
        self.boxes_seen += len(boxes)
        return [
            [Keypoint(name="left_ankle", x=float(b.x1), y=float(b.y2), confidence=0.9)]
            for b in boxes
        ]

    def describe(self):
        return {"name": "_FakeEstimator"}


def test_pose_refiner_fills_keypoints_from_the_clip(sample_clip, config):
    estimator = _FakeEstimator()
    refiner = PoseRefiner(estimator, target_width=config.video.resize_width)
    events = _RecordingEvents()
    path, _ = sample_clip
    AnalysisPipeline(
        config,
        detector=_OnePlayerDetector(),
        tracker=_SettlingTracker(),
        event_detector=events,
        refiners=[refiner],
    ).run(path)

    assert estimator.boxes_seen > 0
    assert all(s.players[0].keypoints for s in events.seen)


def test_pose_refiner_is_a_state_refiner():
    assert isinstance(PoseRefiner(_FakeEstimator()), StateRefiner)


def test_pose_refiner_describes_its_estimator():
    described = PoseRefiner(_FakeEstimator()).describe()
    assert described["name"] == "PoseRefiner"
    assert described["estimator"]["name"] == "_FakeEstimator"


# -- the defaults ------------------------------------------------------------


def test_pose_is_skipped_rather_than_downloaded_when_the_weights_are_missing(caplog):
    cfg = Config()
    cfg.pose = {"weights": "definitely-not-here.pt"}
    cfg.detection.weights_search_paths = ["no/such/directory"]

    with caplog.at_level("WARNING"):
        assert default_state_refiners(cfg.validate()) == []
    assert any("definitely-not-here.pt" in r.message for r in caplog.records)


def test_pose_settings_come_from_the_config(tmp_path, monkeypatch):
    import football_analysis.pose as pose

    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"")
    # Build the estimator without the 20 MB of real weights behind it.
    real = pose.PoseEstimator
    monkeypatch.setattr(pose, "PoseEstimator", lambda config: real(config, model=object()))

    cfg = Config()
    cfg.pose = {"weights": str(weights), "crop_height": 224, "device": "cpu"}

    refiners = default_state_refiners(cfg.validate())
    assert len(refiners) == 1
    assert refiners[0].estimator.config.crop_height == 224
    assert refiners[0].estimator.config.device == "cpu"
    # A bare name is resolved to the file found on disk, never left to be
    # downloaded behind the run's back.
    assert refiners[0].estimator.config.weights == str(weights)
    # The resize the records were produced under, or the boxes miss the pixels.
    assert refiners[0].target_width == cfg.video.resize_width


def test_a_pose_checkpoint_that_will_not_load_costs_pose_not_the_run(tmp_path, caplog):
    # Found on disk, so the missing-weights branch does not fire; it is the
    # load that fails. A corrupt checkpoint should cost the tackles, not the run.
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"not a checkpoint")
    cfg = Config()
    cfg.pose = {"weights": str(weights)}

    with caplog.at_level("WARNING"):
        assert default_state_refiners(cfg.validate()) == []
    assert any("pose" in r.message for r in caplog.records)


def test_an_unknown_pose_key_is_rejected_rather_than_ignored(tmp_path):
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"")
    cfg = Config()
    cfg.pose = {"weights": str(weights), "crp_height": 224}
    with pytest.raises(ConfigError):
        default_state_refiners(cfg.validate())


def test_the_default_event_stage_includes_tackles_and_tricks():
    stage, _ = default_event_stage(Config().validate())
    described = stage.describe()
    names = [s["name"] for s in described.get("stages", [described])]
    assert "BodyEventDetector" in names


def test_analyze_does_not_wire_defaults_around_a_caller_s_own_event_stage(monkeypatch):
    # A caller who brings their own rules brings their inputs too: we cannot
    # know what readings those rules expect.
    import football_analysis.pipeline as pipeline

    called = []
    monkeypatch.setattr(pipeline, "default_state_refiners", lambda c: called.append(c) or [])
    captured = {}

    class _Recorder(AnalysisPipeline):
        def __init__(self, *a, **kw):
            captured.update(kw)
            super().__init__(*a, **kw)

        def run(self, *a, **kw):
            return None

    monkeypatch.setattr(pipeline, "AnalysisPipeline", _Recorder)
    pipeline.analyze("unused.mp4", Config(), event_detector=_RecordingEvents(),
                     detector=_OnePlayerDetector(), tracker=_SettlingTracker())

    assert captured["refiners"] == []
    assert not called


# -- the weights lookup ------------------------------------------------------


def test_find_weights_looks_under_the_search_paths(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pose.pt").write_bytes(b"")
    assert find_weights("pose.pt", [str(tmp_path / "models")]) is not None
    assert find_weights("pose.pt", [str(tmp_path)]) is None


def test_find_weights_does_not_search_for_a_path_the_caller_spelled_out(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pose.pt").write_bytes(b"")
    # A caller who wrote a directory meant that directory.
    assert find_weights("elsewhere/pose.pt", [str(tmp_path / "models")]) is None


def test_the_run_record_says_which_refiners_ran(sample_clip, config):
    # Reading events.json should tell you whether pose ran, because "no
    # tackles" and "no pose" look identical from the timeline alone.
    result, _ = _run(sample_clip, config, [_SpyRefiner()])
    assert [r["name"] for r in result.timeline.stats["stages"]["refiners"]] == ["_SpyRefiner"]


def test_pose_is_skipped_when_nothing_would_read_it(tmp_path, monkeypatch):
    # A goals-only pass over a long clip should not pay for pose.
    import football_analysis.pose as pose

    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"")
    real = pose.PoseEstimator
    monkeypatch.setattr(pose, "PoseEstimator", lambda config: real(config, model=object()))

    cfg = Config()
    cfg.pose = {"weights": str(weights)}
    cfg.events.enabled_types = ["goal", "pass", "shot"]
    assert default_state_refiners(cfg.validate()) == []

    cfg.events.enabled_types = ["goal", "tackle"]
    assert len(default_state_refiners(cfg.validate())) == 1


def test_weights_are_found_from_any_working_directory(tmp_path, monkeypatch):
    # The bug this exists for: `assets/models` is relative, so running
    # `football-analyse ~/clips/match.mp4` from the clips folder found no
    # weights and silently downgraded to the placeholder detector.
    import football_analysis.pipeline as pipeline

    checkout = tmp_path / "checkout"
    (checkout / "assets" / "models").mkdir(parents=True)
    (checkout / "assets" / "models" / "pose.pt").write_bytes(b"")
    elsewhere = tmp_path / "clips"
    elsewhere.mkdir()

    monkeypatch.setattr(pipeline, "_working_dir", lambda: elsewhere)
    monkeypatch.setattr(pipeline, "_checkout_root", lambda: checkout)

    found = pipeline.find_weights("pose.pt", ["assets/models"])
    assert found is not None and found.name == "pose.pt"


def test_the_working_directory_still_wins_over_the_checkout(tmp_path, monkeypatch):
    # Someone who put their own weights where the config points meant those.
    import football_analysis.pipeline as pipeline

    checkout = tmp_path / "checkout"
    (checkout / "assets" / "models").mkdir(parents=True)
    (checkout / "assets" / "models" / "pose.pt").write_bytes(b"")
    here = tmp_path / "here"
    (here / "assets" / "models").mkdir(parents=True)
    (here / "assets" / "models" / "pose.pt").write_bytes(b"")

    monkeypatch.setattr(pipeline, "_working_dir", lambda: here)
    monkeypatch.setattr(pipeline, "_checkout_root", lambda: checkout)

    found = pipeline.find_weights("pose.pt", ["assets/models"])
    assert found.resolve() == (here / "assets" / "models" / "pose.pt").resolve()


def test_an_absolute_search_path_is_not_rewritten_against_the_checkout(tmp_path, monkeypatch):
    import football_analysis.pipeline as pipeline

    models = tmp_path / "models"
    models.mkdir()
    (models / "pose.pt").write_bytes(b"")
    monkeypatch.setattr(pipeline, "_checkout_root", lambda: tmp_path / "nowhere")

    found = pipeline.find_weights("pose.pt", [str(models)])
    assert found.resolve() == (models / "pose.pt").resolve()
