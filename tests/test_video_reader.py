"""Reading clips: timestamps, resizing, segments, stride and failure modes."""

from __future__ import annotations

import pytest

from football_analysis.io.video_reader import VideoReader, VideoSourceError, probe


def test_reads_every_frame_with_increasing_timestamps(sample_clip):
    path, truth = sample_clip
    with VideoReader(path) as reader:
        frames = list(reader)

    assert len(frames) == pytest.approx(truth.frame_count, abs=2)
    times = [f.timestamp_s for f in frames]
    assert times[0] == pytest.approx(0.0, abs=1e-6)
    assert times == sorted(times)
    # Strictly increasing: a repeated timestamp would put two events at one moment.
    assert all(b > a for a, b in zip(times, times[1:]))
    assert times[-1] == pytest.approx(truth.duration_s, abs=0.2)


def test_timestamps_track_the_container_not_just_the_index(sample_clip):
    path, truth = sample_clip
    with VideoReader(path) as reader:
        frames = list(reader)

    nominal_gap = 1.0 / truth.fps
    for frame in frames[:50]:
        assert frame.timestamp_s == pytest.approx(frame.index * nominal_gap, abs=0.02)
    # A constant-rate clip must not be mislabelled as variable.
    assert reader.is_variable_frame_rate is False


def test_frame_carries_source_size_and_index(sample_clip):
    path, truth = sample_clip
    with VideoReader(path) as reader:
        frame = next(iter(reader))

    assert frame.index == 0
    assert frame.source_size == truth.size
    assert frame.scale == pytest.approx(1.0)
    assert frame.timestamp_is_exact


def test_resize_width_preserves_aspect_ratio(sample_clip):
    path, truth = sample_clip
    source_w, source_h = truth.size
    with VideoReader(path, target_width=320) as reader:
        frame = next(iter(reader))

    assert frame.size == (320, round(320 * source_h / source_w))
    assert frame.scale == pytest.approx(320 / source_w)
    # The frame shrank but its place in the source clip did not move.
    assert frame.source_size == truth.size


def test_resize_both_dimensions_forces_the_size(sample_clip):
    path, _ = sample_clip
    with VideoReader(path, target_width=200, target_height=200) as reader:
        assert next(iter(reader)).size == (200, 200)


def test_segment_limits_the_range_and_keeps_source_timestamps(sample_clip):
    path, truth = sample_clip
    start, end = 2.0, 4.0
    with VideoReader(path, start_s=start, end_s=end) as reader:
        frames = list(reader)

    assert frames, "the segment should not be empty"
    # Timestamps stay anchored to the source clip, not rebased to the segment.
    assert frames[0].timestamp_s == pytest.approx(start, abs=0.15)
    assert all(start - 0.15 <= f.timestamp_s < end for f in frames)
    assert frames[0].index > 0
    assert reader.source_info().segment_start_s == pytest.approx(start)


def test_stride_skips_frames_without_distorting_time(sample_clip):
    path, _ = sample_clip
    with VideoReader(path) as full:
        all_frames = list(full)
    with VideoReader(path, frame_stride=3) as strided:
        strided_frames = list(strided)

    assert len(strided_frames) == pytest.approx(len(all_frames) / 3, abs=2)
    assert [f.index for f in strided_frames[:4]] == [
        f.index for f in all_frames[0:12:3]
    ]
    # Each kept frame keeps the timestamp it had in the full read.
    by_index = {f.index: f.timestamp_s for f in all_frames}
    for frame in strided_frames:
        assert frame.timestamp_s == pytest.approx(by_index[frame.index])


def test_output_fps_accounts_for_stride(sample_clip):
    path, truth = sample_clip
    with VideoReader(path, frame_stride=2) as reader:
        assert reader.output_fps == pytest.approx(truth.fps / 2, abs=0.01)


def test_max_frames_stops_early(sample_clip):
    path, _ = sample_clip
    with VideoReader(path, max_frames=5) as reader:
        assert len(list(reader)) == 5


def test_source_info_describes_the_clip(sample_clip):
    path, truth = sample_clip
    info = probe(path)

    assert info.width, info.height == truth.size
    assert info.fps == pytest.approx(truth.fps, abs=0.5)
    assert info.frame_count == pytest.approx(truth.frame_count, abs=2)
    assert info.path.endswith("sample.mp4")


def test_rejects_a_missing_file(tmp_path):
    with pytest.raises(VideoSourceError, match="no such file"):
        VideoReader(tmp_path / "nope.mp4")


def test_rejects_a_file_that_is_not_a_video(tmp_path):
    not_video = tmp_path / "notes.txt"
    not_video.write_text("this is not a clip", encoding="utf-8")
    with pytest.raises(VideoSourceError):
        VideoReader(not_video)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_stride": 0},
        {"start_s": -1.0},
        {"start_s": 5.0, "end_s": 5.0},
        {"start_s": 5.0, "end_s": 1.0},
    ],
)
def test_rejects_impossible_arguments(sample_clip, kwargs):
    path, _ = sample_clip
    with pytest.raises(ValueError):
        VideoReader(path, **kwargs)


def test_reader_is_reusable_as_a_context_manager(sample_clip):
    path, _ = sample_clip
    reader = VideoReader(path, max_frames=2)
    with reader:
        assert len(list(reader)) == 2
    with pytest.raises(VideoSourceError, match="closed"):
        list(reader)
