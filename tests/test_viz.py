"""The overlay: drawing without destroying the frame underneath."""

from __future__ import annotations

import numpy as np

from football_analysis.config import OutputConfig
from football_analysis.events import Event, EventType
from football_analysis.types import BBox, Detection, Track, VideoFrame
from football_analysis.viz import FrameAnnotator, color_for_track


def _frame(width=320, height=180):
    return VideoFrame(
        index=10,
        timestamp_s=0.4,
        image=np.zeros((height, width, 3), np.uint8),
        source_size=(width, height),
    )


def _track(track_id=1, class_name="player", box=(20, 20, 60, 140), label="Player 1"):
    return Track(
        track_id=track_id,
        class_name=class_name,
        bbox=BBox(*box),
        confidence=0.8,
        frame_index=10,
        timestamp_s=0.4,
        label=label,
    )


def test_annotate_draws_without_modifying_the_original():
    frame = _frame()
    original = frame.image.copy()
    canvas = FrameAnnotator().annotate(frame, tracks=[_track()])

    assert canvas.shape == frame.image.shape
    assert canvas.sum() > 0, "something should have been drawn"
    # A later stage must never see painted-on pixels.
    assert np.array_equal(frame.image, original)


def test_detections_are_drawn_only_when_there_are_no_tracks():
    detection = Detection(class_name="ball", bbox=BBox(10, 10, 22, 22), confidence=0.4)

    only_detections = FrameAnnotator().annotate(_frame(), detections=[detection])
    assert only_detections.sum() > 0

    config = OutputConfig(draw_ball_trail=False, draw_clock=False, draw_events=False)
    with_tracks = FrameAnnotator(config).annotate(
        _frame(), tracks=[_track()], detections=[detection]
    )
    without = FrameAnnotator(config).annotate(_frame(), tracks=[_track()])
    # The detection must not be drawn a second time next to its own track.
    assert np.array_equal(with_tracks, without)


def test_drawing_can_be_switched_off():
    config = OutputConfig(
        draw_boxes=False, draw_labels=False, draw_tracks=False,
        draw_ball_trail=False, draw_events=False, draw_clock=False,
    )
    canvas = FrameAnnotator(config).annotate(_frame(), tracks=[_track()])
    assert canvas.sum() == 0


def test_event_banner_expires_after_its_window():
    config = OutputConfig(
        draw_boxes=False, draw_ball_trail=False, draw_clock=False,
        event_banner_seconds=1.0,
    )
    annotator = FrameAnnotator(config)
    event = Event(type=EventType.PASS, timestamp_s=0.4, confidence=0.9, player_id="Player 1")

    fresh = annotator.annotate(_frame(), events=[event])
    assert fresh.sum() > 0

    later = _frame()
    later.timestamp_s = 5.0
    assert annotator.annotate(later).sum() == 0


def test_boxes_near_the_top_edge_still_get_a_caption():
    # A caption drawn above y=0 would be clipped away silently.
    canvas = FrameAnnotator().annotate(_frame(), tracks=[_track(box=(5, 0, 40, 60))])
    assert canvas[:30, :60].sum() > 0


def test_ball_trail_accumulates_across_frames():
    config = OutputConfig(draw_boxes=False, draw_clock=False, draw_events=False)
    annotator = FrameAnnotator(config)
    canvas = None
    for i in range(6):
        frame = _frame()
        frame.timestamp_s = i * 0.04
        ball = _track(track_id=9, class_name="ball",
                      box=(20 + i * 12, 100, 32 + i * 12, 112), label=None)
        canvas = annotator.annotate(frame, tracks=[ball])
    assert canvas.sum() > 0, "the trail should be visible after several frames"


def test_each_player_gets_its_own_colour_and_the_ball_is_constant():
    assert color_for_track(1, "player") != color_for_track(2, "player")
    assert color_for_track(1, "ball") == color_for_track(7, "ball")


def _ball_frames(annotator, positions, *, dt=0.04):
    """Feed a sequence of (x, timestamp) ball sightings; return the last canvas."""
    canvas = None
    for x, t in positions:
        frame = _frame()
        frame.timestamp_s = t
        ball = _track(track_id=9, class_name="ball",
                      box=(x, 100, x + 12, 112), label=None)
        canvas = annotator.annotate(frame, tracks=[ball])
    return canvas


def _trail_only_config():
    return OutputConfig(draw_boxes=False, draw_clock=False, draw_events=False)


def test_trail_is_not_drawn_across_a_gap_in_time():
    # The ball vanished for a second; joining the two sightings would invent a path.
    canvas = _ball_frames(
        FrameAnnotator(_trail_only_config()),
        [(20, 0.0), (26, 0.04), (40, 1.5), (46, 1.54)],
    )
    # Only the two short in-time segments should appear, not the long jump.
    assert canvas[100:115, 60:200].sum() == 0


def test_trail_is_not_drawn_across_an_implausible_jump():
    # A ball track that leaps most of the frame width is a tracking error.
    canvas = _ball_frames(
        FrameAnnotator(_trail_only_config()), [(10, 0.0), (280, 0.04)]
    )
    assert canvas.sum() == 0


def test_a_normally_moving_ball_still_gets_a_trail():
    canvas = _ball_frames(
        FrameAnnotator(_trail_only_config()),
        [(20 + i * 10, i * 0.04) for i in range(6)],
    )
    assert canvas.sum() > 0
