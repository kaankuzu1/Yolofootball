"""The whole loop, with fake stages so the assertions are about plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_analysis.config import Config
from football_analysis.events import Event, EventType
from football_analysis.interfaces import BaseDetector, BaseEventDetector, BaseTracker
from football_analysis.pipeline import AnalysisPipeline, analyze, replay
from football_analysis.io.video_reader import VideoReader
from football_analysis.state import FrameState, read_state_cache
from football_analysis.viz import FrameAnnotator
from football_analysis.types import BBox, Detection, Track


class ScriptedDetector(BaseDetector):
    """Emits one player and one ball every frame, at predictable places."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [
            Detection(class_name="player", bbox=BBox(50, 40, 90, 150), confidence=0.9),
            Detection(class_name="ball", bbox=BBox(95, 130, 107, 142), confidence=0.6),
        ]


class PassThroughTracker(BaseTracker):
    def update(self, frame, detections):
        return [
            Track(
                track_id=i + 1,
                class_name=d.class_name,
                bbox=d.bbox,
                confidence=d.confidence,
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                label="Player 1" if d.class_name == "player" else None,
            )
            for i, d in enumerate(detections)
        ]


class ScriptedEvents(BaseEventDetector):
    """Fires a pass on frame 3 and flushes a goal at the end."""

    def __init__(self, confidence=0.9):
        self.confidence = confidence
        self.finalized = False
        self.seen_states = []

    def update(self, state):
        self.seen_states.append(state)
        if state.frame_index == 3:
            return [
                Event(
                    type=EventType.PASS,
                    timestamp_s=state.timestamp_s,
                    frame_index=state.frame_index,
                    confidence=self.confidence,
                    player_id="Player 1",
                )
            ]
        return []

    def finalize(self):
        self.finalized = True
        return [Event(type=EventType.GOAL, timestamp_s=99.0, confidence=0.95)]


def _pipeline(config, **stages):
    return AnalysisPipeline(
        config,
        detector=stages.get("detector") or ScriptedDetector(),
        tracker=stages.get("tracker") or PassThroughTracker(),
        event_detector=stages.get("event_detector") or ScriptedEvents(),
    )


def test_runs_end_to_end_and_collects_events(sample_clip, config):
    path, _ = sample_clip
    result = _pipeline(config).run(path)

    types = [e.type for e in result.events]
    assert EventType.PASS in types
    assert EventType.GOAL in types, "finalize() events must reach the timeline"
    assert all(e.id for e in result.events), "every event gets an id"


def test_every_frame_reaches_the_detector(sample_clip, config):
    path, _ = sample_clip
    detector = ScriptedDetector()
    result = _pipeline(config, detector=detector).run(path)

    assert detector.calls == result.timeline.stats["frames_processed"]
    assert detector.calls > 10


def test_timeline_records_source_config_and_stats(sample_clip, config):
    path, _ = sample_clip
    result = _pipeline(config).run(path)
    timeline = result.timeline

    assert timeline.source.path.endswith("sample.mp4")
    assert timeline.source.processed_width == 320
    # The resolved config travels with the result, so a run is reproducible.
    assert timeline.config["video"]["resize_width"] == 320
    assert timeline.stats["frames_processed"] > 0
    assert timeline.stats["stages"]["detector"]["name"] == "ScriptedDetector"


def test_events_are_sorted_by_time(sample_clip, config):
    path, _ = sample_clip
    result = _pipeline(config).run(path)
    times = [e.timestamp_s for e in result.events]
    assert times == sorted(times)


def test_low_confidence_events_are_dropped(sample_clip, config):
    path, _ = sample_clip
    config.events.min_confidence = 0.9
    result = _pipeline(config, event_detector=ScriptedEvents(confidence=0.2)).run(path)

    assert EventType.PASS not in [e.type for e in result.events]
    assert EventType.GOAL in [e.type for e in result.events]  # 0.95 clears the bar


def test_disabled_event_types_never_reach_the_timeline(sample_clip, config):
    path, _ = sample_clip
    config.events.enabled_types = ["goal"]
    result = _pipeline(config).run(path)

    assert {e.type for e in result.events} == {EventType.GOAL}


def test_duplicate_events_within_the_gap_are_collapsed(sample_clip, config):
    path, _ = sample_clip

    class Spammer(BaseEventDetector):
        def update(self, state):
            return [Event(type=EventType.PASS, timestamp_s=state.timestamp_s,
                          confidence=0.9, player_id="Player 1")]

    config.events.min_gap_s = 1.0
    result = _pipeline(config, event_detector=Spammer()).run(path)
    times = [e.timestamp_s for e in result.events]

    assert len(times) < 10, "a pass every frame should collapse to a handful"
    assert all(b - a >= 1.0 for a, b in zip(times, times[1:]))


def test_segment_config_limits_what_is_processed(sample_clip, config):
    path, _ = sample_clip
    config.video.start_s = 2.0
    config.video.end_s = 3.0
    result = _pipeline(config).run(path)

    assert 0 < result.timeline.stats["frames_processed"] < 40
    assert result.timeline.source.segment_start_s == pytest.approx(2.0)


def test_writes_json_and_annotated_video_when_asked(sample_clip, config, tmp_path):
    path, _ = sample_clip
    config.output.json_path = str(tmp_path / "events.json")
    config.output.video_path = str(tmp_path / "annotated.mp4")
    result = _pipeline(config).run(path)

    assert result.json_path is not None and result.json_path.exists()
    assert result.video_path is not None and result.video_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["events"], "the written document should carry the events"


def test_no_output_paths_means_no_files_written(sample_clip, config):
    path, _ = sample_clip
    result = _pipeline(config).run(path)
    assert result.json_path is None
    assert result.video_path is None


def test_progress_hook_reports_each_phase_separately(sample_clip, config):
    # A run makes more than one pass over the clip. A progress line that does
    # not say which pass it is looks like the run has restarted.
    path, _ = sample_clip
    updates = []
    result = _pipeline(config).run(path, progress=updates.append)
    frames = result.timeline.stats["frames_processed"]

    by_phase = {}
    for update in updates:
        by_phase.setdefault(update.phase, []).append(update)

    assert set(by_phase) == {"analyse", "events"}
    for phase, phase_updates in by_phase.items():
        assert len(phase_updates) == frames, phase
        assert phase_updates[-1].frames_processed == frames
        assert phase_updates[-1].timestamp_s > phase_updates[0].timestamp_s


def test_events_are_counted_as_the_event_phase_progresses(sample_clip, config):
    path, _ = sample_clip
    updates = []
    _pipeline(config).run(path, progress=updates.append)

    event_phase = [u for u in updates if u.phase == "events"]
    assert event_phase[-1].events_so_far >= event_phase[0].events_so_far
    assert event_phase[-1].events_so_far > 0


def test_stages_are_reset_between_runs(sample_clip, config):
    path, _ = sample_clip
    events = ScriptedEvents()
    pipeline = _pipeline(config, event_detector=events)

    first = pipeline.run(path)
    second = pipeline.run(path)
    assert len(first.events) == len(second.events)
    assert events.finalized


def test_analyze_falls_back_to_the_placeholder_when_there_are_no_weights(
    sample_clip, config, tmp_path
):
    # The point of the stubs: a clip in, a valid document out, no models needed.
    path, _ = sample_clip
    config.detection.model_path = str(tmp_path / "absent.pt")
    config.video.max_frames = 5
    result = analyze(path, config)

    payload = json.loads(result.timeline.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["stats"]["frames_processed"] > 0
    assert payload["stats"]["stages"]["detector"]["placeholder"] is True


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "assets/models/forzasys_soccer.pt").is_file(),
    reason="forzasys_soccer.pt is not on disk (see 'Model weights' in the README)",
)
def test_analyze_uses_the_real_detector_when_the_weights_are_there(
    sample_clip, config
):
    # With weights available, a default run must analyse players, not motion.
    path, _ = sample_clip
    config.detection.model_path = "assets/models/forzasys_soccer.pt"
    config.video.max_frames = 2
    result = analyze(path, config)

    detector = result.timeline.stats["stages"]["detector"]
    assert detector["name"] == "Detector"
    assert detector.get("placeholder") is None


def test_a_failing_stage_still_releases_the_clip(sample_clip, config):
    path, _ = sample_clip

    class Exploding(BaseDetector):
        def detect(self, frame):
            raise RuntimeError("model blew up")

    with pytest.raises(RuntimeError, match="model blew up"):
        _pipeline(config, detector=Exploding()).run(path)
    # The reader is closed in a finally block, so the file is not left locked.
    assert _pipeline(config).run(path).timeline.stats["frames_processed"] > 0


def test_event_logic_is_handed_the_record_not_the_frame(sample_clip, config):
    # The whole point of the world-state record: event rules never see pixels.
    path, _ = sample_clip
    events = ScriptedEvents()
    _pipeline(config, event_detector=events).run(path)

    state = events.seen_states[5]
    assert isinstance(state, FrameState)
    assert not hasattr(state, "image")
    assert state.frame_size == (320, 180)
    assert [p.player_id for p in state.players] == ["Player 1"]
    assert state.ball is not None


def test_the_record_carries_ball_and_player_geometry(sample_clip, config):
    path, _ = sample_clip
    events = ScriptedEvents()
    _pipeline(config, event_detector=events).run(path)
    state = events.seen_states[5]

    nearest = state.nearest_player_to_ball()
    assert nearest is not None
    player, distance = nearest
    assert player.player_id == "Player 1"
    assert distance > 0
    # Distance is from the feet, which is where the ball is when they have it.
    assert player.position.y == pytest.approx(player.bbox.y2)


def test_state_cache_is_written_and_replays(sample_clip, config, tmp_path):
    path, _ = sample_clip
    cache = tmp_path / "state.jsonl"
    config.output.state_cache_path = str(cache)
    result = _pipeline(config).run(path)

    assert result.state_cache_path == cache
    assert cache.exists()

    records = list(read_state_cache(cache))
    assert len(records) == result.timeline.stats["frames_processed"]
    assert all(isinstance(r, FrameState) for r in records)


def test_replay_reproduces_the_events_without_the_video(sample_clip, config, tmp_path):
    path, _ = sample_clip
    cache = tmp_path / "state.jsonl"
    config.output.state_cache_path = str(cache)
    live = _pipeline(config).run(path)

    replayed = replay(cache, ScriptedEvents(), config)

    assert [e.type for e in replayed] == [e.type for e in live.events]
    assert [round(e.timestamp_s, 3) for e in replayed] == [
        round(e.timestamp_s, 3) for e in live.events
    ]
    assert replayed.stats["replayed_from_cache"] == str(cache)


def test_replay_applies_the_config_so_thresholds_can_be_tuned(sample_clip, config, tmp_path):
    # Tuning a threshold should mean re-reading a file, not re-running inference.
    path, _ = sample_clip
    cache = tmp_path / "state.jsonl"
    config.output.state_cache_path = str(cache)
    _pipeline(config).run(path)

    strict = Config()
    strict.events.enabled_types = ["goal"]
    assert {e.type for e in replay(cache, ScriptedEvents(), strict)} == {EventType.GOAL}


def test_replay_rejects_a_cache_from_another_schema(tmp_path):
    cache = tmp_path / "state.jsonl"
    cache.write_text('{"schema_version": "99.0", "record": "header"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="state schema"):
        replay(cache, ScriptedEvents(), Config())


class LateOnlyEvents(BaseEventDetector):
    """Emits nothing per frame and one event from finalize(), like real logic.

    Real pass/shot/goal rules need lookahead -- whether a ball ended at a
    player's feet or in the net is only known later -- so the headline events
    arrive from finalize(). They still have to be drawn on the frame where they
    happened.
    """

    def __init__(self, timestamp_s=2.0):
        self.timestamp_s = timestamp_s

    def update(self, state):
        return []

    def finalize(self):
        return [
            Event(
                type=EventType.GOAL,
                timestamp_s=self.timestamp_s,
                confidence=0.95,
                player_id="Player 1",
            )
        ]


class SpyAnnotator(FrameAnnotator):
    """Records which events were handed to each frame."""

    def __init__(self, config=None):
        super().__init__(config)
        self.handed: list[tuple[float, list]] = []

    def annotate(self, frame, *, tracks=(), detections=(), events=()):
        events = list(events)
        self.handed.append((frame.timestamp_s, events))
        return super().annotate(
            frame, tracks=tracks, detections=detections, events=events
        )


def test_finalize_events_are_drawn_on_the_frame_where_they_happened(
    sample_clip, config, tmp_path
):
    path, _ = sample_clip
    config.output.video_path = str(tmp_path / "annotated.mp4")
    annotator = SpyAnnotator(config.output)
    pipeline = AnalysisPipeline(
        config,
        detector=ScriptedDetector(),
        tracker=PassThroughTracker(),
        event_detector=LateOnlyEvents(timestamp_s=2.0),
        annotator=annotator,
    )
    result = pipeline.run(path)

    drawn = [(t, e) for t, events in annotator.handed for e in events]
    assert drawn, "a finalize-time event must still reach the overlay"
    timestamp, event = drawn[0]
    assert event.type is EventType.GOAL
    # Drawn at the moment it happened, not at the end of the clip.
    assert timestamp == pytest.approx(2.0, abs=0.1)
    assert result.video_path is not None and result.video_path.exists()


def test_every_event_reaches_the_overlay_exactly_once(sample_clip, config, tmp_path):
    path, _ = sample_clip
    config.output.video_path = str(tmp_path / "annotated.mp4")
    annotator = SpyAnnotator(config.output)
    pipeline = AnalysisPipeline(
        config,
        detector=ScriptedDetector(),
        tracker=PassThroughTracker(),
        event_detector=LateOnlyEvents(timestamp_s=2.0),
        annotator=annotator,
    )
    result = pipeline.run(path)

    drawn_ids = [e.id for _, events in annotator.handed for e in events]
    assert sorted(drawn_ids) == sorted(e.id for e in result.events)
    assert len(drawn_ids) == len(set(drawn_ids)), "no event should be drawn twice"


def test_an_event_on_the_last_frame_is_still_drawn(sample_clip, config, tmp_path):
    # Rounding can put a finalize-time event a hair past the final frame.
    path, _ = sample_clip
    config.output.video_path = str(tmp_path / "annotated.mp4")
    config.video.max_frames = 25
    annotator = SpyAnnotator(config.output)

    with VideoReader(path, target_width=320, max_frames=25) as reader:
        last_timestamp = list(reader)[-1].timestamp_s

    pipeline = AnalysisPipeline(
        config,
        detector=ScriptedDetector(),
        tracker=PassThroughTracker(),
        event_detector=LateOnlyEvents(timestamp_s=last_timestamp + 0.001),
        annotator=annotator,
    )
    pipeline.run(path)

    assert [e for _, events in annotator.handed for e in events], (
        "an event a millisecond past the last frame should land on it"
    )


def test_an_event_timed_beyond_the_clip_warns_rather_than_vanishing(
    sample_clip, config, tmp_path, caplog
):
    path, _ = sample_clip
    config.output.video_path = str(tmp_path / "annotated.mp4")
    pipeline = AnalysisPipeline(
        config,
        detector=ScriptedDetector(),
        tracker=PassThroughTracker(),
        event_detector=LateOnlyEvents(timestamp_s=999.0),
        annotator=SpyAnnotator(config.output),
    )
    with caplog.at_level("WARNING"):
        result = pipeline.run(path)

    # It stays in the timeline; it just could not be drawn, and that is said.
    assert EventType.GOAL in [e.type for e in result.events]
    assert "after the last rendered frame" in caplog.text


def test_rendering_draws_every_analysed_frame(sample_clip, config, tmp_path):
    path, _ = sample_clip
    config.output.video_path = str(tmp_path / "annotated.mp4")
    result = _pipeline(config).run(path)

    assert result.timeline.stats["frames_rendered"] == (
        result.timeline.stats["frames_processed"]
    )


def test_rendering_respects_the_segment(sample_clip, config, tmp_path):
    # The render pass re-reads the clip, so it must re-apply the same segment.
    path, _ = sample_clip
    config.video.start_s = 2.0
    config.video.end_s = 3.0
    config.output.video_path = str(tmp_path / "annotated.mp4")
    result = _pipeline(config).run(path)

    with VideoReader(result.video_path) as reader:
        rendered = list(reader)
    assert len(rendered) == result.timeline.stats["frames_processed"]


def test_no_video_is_written_when_none_was_asked_for(sample_clip, config):
    path, _ = sample_clip
    result = _pipeline(config, event_detector=LateOnlyEvents()).run(path)
    assert result.video_path is None
    assert result.timeline.stats["frames_rendered"] == 0
