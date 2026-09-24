"""Encoding annotated frames back out."""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.io.video_reader import VideoReader
from football_analysis.io.video_writer import AnnotatedVideoWriter, VideoWriteError


def _frame(width=320, height=180, value=90):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_writes_a_clip_that_reads_back(tmp_path):
    out = tmp_path / "annotated.mp4"
    with AnnotatedVideoWriter(out, fps=25.0) as writer:
        for i in range(10):
            writer.write(_frame(value=20 * i))

    assert writer.frames_written == 10
    assert out.exists() and out.stat().st_size > 0

    with VideoReader(out) as reader:
        frames = list(reader)
    assert len(frames) == pytest.approx(10, abs=1)
    assert frames[0].size == (320, 180)


def test_size_is_taken_from_the_first_frame(tmp_path):
    # The reader may have resized, so the writer must not need to be told.
    writer = AnnotatedVideoWriter(tmp_path / "out.mp4", fps=25.0)
    writer.write(_frame(width=256, height=144))
    assert writer.size == (256, 144)
    writer.close()


def test_mismatched_later_frames_are_resized_not_corrupted(tmp_path):
    with AnnotatedVideoWriter(tmp_path / "out.mp4", fps=25.0) as writer:
        writer.write(_frame(320, 180))
        writer.write(_frame(640, 360))
    assert writer.frames_written == 2


def test_creates_missing_parent_directories(tmp_path):
    out = tmp_path / "deep" / "nested" / "out.mp4"
    with AnnotatedVideoWriter(out, fps=25.0) as writer:
        writer.write(_frame())
    assert out.exists()


def test_falls_back_to_an_available_codec(tmp_path):
    # avc1 is often missing; the writer must still produce a file.
    with AnnotatedVideoWriter(tmp_path / "out.mp4", fps=25.0) as writer:
        writer.write(_frame())
    assert writer.codec in ("avc1", "mp4v")


def test_rejects_empty_frames_and_use_after_close(tmp_path):
    writer = AnnotatedVideoWriter(tmp_path / "out.mp4", fps=25.0)
    with pytest.raises(ValueError):
        writer.write(np.zeros((0, 0, 3), np.uint8))
    writer.write(_frame())
    writer.close()
    with pytest.raises(VideoWriteError, match="closed"):
        writer.write(_frame())


def test_rejects_a_nonsense_frame_rate(tmp_path):
    with pytest.raises(ValueError):
        AnnotatedVideoWriter(tmp_path / "out.mp4", fps=0)
