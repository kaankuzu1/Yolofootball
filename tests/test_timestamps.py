"""Timestamp behaviour under the conditions phone footage actually creates.

A clip written by OpenCV has a constant frame rate, so the awkward cases --
a container that reports no presentation time, one that stalls, one whose real
gaps disagree with its nominal fps -- are exercised against the reader's timing
logic directly rather than through a file that cannot be made to misbehave.
"""

from __future__ import annotations

import cv2
import pytest

from football_analysis.io.video_reader import VideoReader


class FakeCapture:
    """A stand-in for ``cv2.VideoCapture``.

    OpenCV's capture object is a C extension whose methods cannot be patched,
    so the awkward containers are simulated with an object of our own that
    answers ``get`` however the test needs.
    """

    def __init__(self, *, pos_msec=0.0, fps=25.0, width=640, height=360, frames=150):
        self.pos_msec = pos_msec
        self.props = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FRAME_COUNT: frames,
        }
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pos_msec
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        return True

    def read(self):
        return False, None

    def grab(self):
        return False

    def release(self):
        self.released = True


def _swap_capture(reader, fake):
    """Replace a reader's real capture, releasing the one it opened."""
    reader._cap.release()
    reader._cap = fake


def test_a_constant_rate_clip_is_not_flagged_as_variable(sample_clip):
    path, _ = sample_clip
    with VideoReader(path) as reader:
        list(reader)
    assert reader.is_variable_frame_rate is False
    assert reader.inexact_timestamp_count == 0


def test_uneven_gaps_are_detected_as_a_variable_frame_rate(sample_clip):
    path, _ = sample_clip
    reader = VideoReader(path)
    try:
        # 25fps nominal (0.040s); half these frames arrive far off that rate.
        reader._gap_samples = [0.040, 0.075, 0.040, 0.012, 0.040, 0.080,
                               0.040, 0.011, 0.040, 0.070, 0.040, 0.090]
        assert reader.is_variable_frame_rate is True
        assert reader.source_info().variable_frame_rate is True
    finally:
        reader.close()


def test_a_few_stray_gaps_do_not_flag_a_steady_clip(sample_clip):
    path, _ = sample_clip
    reader = VideoReader(path)
    try:
        reader._gap_samples = [0.040] * 40 + [0.075]
        assert reader.is_variable_frame_rate is False
    finally:
        reader.close()


def test_too_few_samples_is_reported_as_steady_not_guessed(sample_clip):
    path, _ = sample_clip
    reader = VideoReader(path)
    try:
        reader._gap_samples = [0.040, 0.090]
        assert reader.is_variable_frame_rate is False
    finally:
        reader.close()


def test_a_missing_presentation_time_falls_back_to_the_nominal_rate(sample_clip):
    path, _ = sample_clip
    with VideoReader(path) as reader:
        _swap_capture(reader, FakeCapture(pos_msec=0.0))
        timestamp, exact = reader._timestamp_for(10)

    assert timestamp == pytest.approx(10 / reader.fps)
    assert exact is False, "a derived timestamp must be marked inexact"


def test_a_stalled_presentation_time_still_moves_forward(sample_clip):
    path, _ = sample_clip
    with VideoReader(path) as reader:
        _swap_capture(reader, FakeCapture(pos_msec=1000.0))  # stuck at 1.0s
        reader._last_timestamp_s = 1.0
        timestamp, exact = reader._timestamp_for(30)

    # Two events must never share a timestamp just because the container stalled.
    assert timestamp > 1.0
    assert exact is False


def test_a_backwards_presentation_time_is_rejected(sample_clip):
    path, _ = sample_clip
    with VideoReader(path) as reader:
        _swap_capture(reader, FakeCapture(pos_msec=2000.0))  # 2.0s, backwards
        reader._last_timestamp_s = 5.0
        timestamp, exact = reader._timestamp_for(125)

    assert timestamp > 5.0
    assert exact is False


def test_an_absurd_container_frame_rate_falls_back_to_a_usable_one(sample_clip, monkeypatch):
    path, _ = sample_clip
    # A container claiming 0fps is common on clips that have been re-encoded.
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *a, **k: FakeCapture(fps=0.0, width=640, height=360)
    )
    with VideoReader(path) as reader:
        assert reader.fps == pytest.approx(30.0)  # the documented fallback
        assert reader._fps_is_nominal is True
        assert reader.source_info().fps == pytest.approx(30.0)
