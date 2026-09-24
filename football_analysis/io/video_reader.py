"""Frame-by-frame video input with timestamps you can put in a report.

Why this is not just ``cap.read()`` in a loop
---------------------------------------------
Phone footage is the normal case here, and phone footage lies.  The container
reports a nominal fps that is an average, not a rate: a clip recorded at
"30fps" routinely delivers frames 28ms apart in good light and 45ms apart in
bad.  Deriving an event timestamp as ``index / fps`` on such a clip drifts, and
a drifting timestamp is a wrong answer to the only question this system is
being asked.

So the reader prefers the container's own presentation timestamp
(``CAP_PROP_POS_MSEC``) for every frame, and falls back to the nominal rate
only when that value is missing or stuck -- recording, per frame, which of the
two it used.  It also watches the gaps between presentation times and flags the
clip as variable frame rate when they disagree with the nominal rate, so a
consumer knows not to trust ``fps`` for arithmetic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from football_analysis.events import SourceInfo
from football_analysis.types import VideoFrame

logger = logging.getLogger(__name__)

__all__ = ["VideoReader", "VideoSourceError"]

# A clip is called variable frame rate when this fraction of its inter-frame
# gaps stray from the nominal gap by more than _VFR_TOLERANCE.
_VFR_SAMPLE_FRAMES = 120
_VFR_TOLERANCE = 0.25
_VFR_MIN_OUTLIER_RATIO = 0.10

# Fallback when a container reports no usable frame rate at all.
_DEFAULT_FPS = 30.0


class VideoSourceError(RuntimeError):
    """The clip could not be opened, or could not be read as video."""


class VideoReader:
    """Iterate a clip as :class:`VideoFrame` objects.

    Parameters
    ----------
    path:
        The clip to read.
    target_width, target_height:
        Resize each frame before yielding it.  Give one and the other is
        derived from the source aspect ratio; give both and the aspect ratio is
        forced.  Give neither and frames come through at source size.
    start_s, end_s:
        Process only this segment of the clip.  ``end_s`` is exclusive.
    frame_stride:
        Yield every Nth frame.  ``2`` halves the work at the cost of halving
        temporal resolution; timestamps stay exact either way.
    max_frames:
        Stop after this many yielded frames.  Useful for smoke runs.

    Use it as a context manager, or call :meth:`close` yourself::

        with VideoReader("match.mp4", target_width=960) as reader:
            for frame in reader:
                ...
    """

    def __init__(
        self,
        path: str | Path,
        *,
        target_width: int | None = None,
        target_height: int | None = None,
        start_s: float = 0.0,
        end_s: float | None = None,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise VideoSourceError(f"no such file: {self.path}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if start_s < 0:
            raise ValueError(f"start_s must be >= 0, got {start_s}")
        if end_s is not None and end_s <= start_s:
            raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s})")

        self.start_s = float(start_s)
        self.end_s = float(end_s) if end_s is not None else None
        self.frame_stride = int(frame_stride)
        self.max_frames = max_frames

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise VideoSourceError(f"OpenCV could not open: {self.path}")

        self.source_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.source_width <= 0 or self.source_height <= 0:
            self._cap.release()
            raise VideoSourceError(
                f"{self.path} reports a zero-sized frame; it is probably not a video"
            )

        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Some containers report absurd rates (0, inf, 1000) for phone footage.
        self.fps = raw_fps if 0.0 < raw_fps < 1000.0 else _DEFAULT_FPS
        self._fps_is_nominal = not (0.0 < raw_fps < 1000.0)
        if self._fps_is_nominal:
            logger.warning(
                "%s reports fps=%r; assuming %.1f for fallback timestamps",
                self.path.name, raw_fps, self.fps,
            )

        raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.source_frame_count = raw_count if raw_count > 0 else None
        self.source_duration_s = (
            self.source_frame_count / self.fps if self.source_frame_count else None
        )

        self.target_size = self._resolve_target_size(target_width, target_height)
        self.scale = (
            self.target_size[0] / self.source_width if self.target_size else 1.0
        )

        self._closed = False
        self._exhausted = False
        self._sought = False
        self._last_timestamp_s: float | None = None
        self._gap_samples: list[float] = []
        self._inexact_timestamps = 0

        self.frames_read = 0
        """Frames pulled off the decoder, including ones skipped by stride."""

        self.frames_yielded = 0
        """Frames actually handed to the caller."""

    # -- setup helpers ----------------------------------------------------

    def _resolve_target_size(
        self, width: int | None, height: int | None
    ) -> tuple[int, int] | None:
        if width is None and height is None:
            return None
        if width is not None and height is not None:
            return (int(width), int(height))
        aspect = self.source_height / self.source_width
        if width is not None:
            return (int(width), max(1, int(round(int(width) * aspect))))
        return (max(1, int(round(int(height) / aspect))), int(height))

    # -- timing -----------------------------------------------------------

    def _timestamp_for(self, index: int) -> tuple[float, bool]:
        """Presentation time of the frame just read, and whether it is exact.

        ``CAP_PROP_POS_MSEC`` is read *after* the frame is grabbed, so it is the
        time of that frame.  It is rejected when it is missing, when it goes
        backwards (a seek artefact), or when it stalls at the previous value --
        all of which happen on real phone clips.
        """
        raw_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        nominal = index / self.fps

        if raw_ms <= 0.0 and index > 0:
            return nominal, False
        if not np.isfinite(raw_ms):
            return nominal, False

        timestamp = raw_ms / 1000.0
        last = self._last_timestamp_s
        if last is not None and timestamp <= last:
            # Stalled or went backwards: step forward by one nominal gap so the
            # timeline stays strictly increasing.
            return last + (1.0 / self.fps), False
        return timestamp, True

    def _note_gap(self, timestamp_s: float) -> None:
        if self._last_timestamp_s is not None and len(self._gap_samples) < _VFR_SAMPLE_FRAMES:
            gap = timestamp_s - self._last_timestamp_s
            if gap > 0:
                self._gap_samples.append(gap)

    @property
    def is_variable_frame_rate(self) -> bool:
        """Whether observed frame gaps disagreed with the nominal rate."""
        if len(self._gap_samples) < 10:
            return False
        nominal_gap = 1.0 / self.fps
        outliers = sum(
            1 for gap in self._gap_samples
            if abs(gap - nominal_gap) > _VFR_TOLERANCE * nominal_gap
        )
        return (outliers / len(self._gap_samples)) > _VFR_MIN_OUTLIER_RATIO

    # -- seeking ----------------------------------------------------------

    def _seek_to_start(self) -> None:
        """Position the decoder at ``start_s``.

        A keyframe seek is tried first because skipping a minute of frames one
        at a time is slow.  Seeking is unreliable on some containers, so the
        result is verified and the reader falls back to reading forward.
        """
        self._sought = True
        if self.start_s <= 0.0:
            return

        target_ms = self.start_s * 1000.0
        if self._cap.set(cv2.CAP_PROP_POS_MSEC, target_ms):
            landed_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if 0.0 < landed_ms <= target_ms + 1.0:
                return
            # Overshot, or the container ignored us: rewind and walk forward.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        logger.debug("seek unreliable on %s; skipping forward to %.3fs",
                     self.path.name, self.start_s)
        while True:
            ok = self._cap.grab()
            if not ok:
                self._exhausted = True
                return
            pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pos_ms / 1000.0 >= self.start_s:
                self._cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    max(0.0, self._cap.get(cv2.CAP_PROP_POS_FRAMES) - 1),
                )
                return

    # -- iteration --------------------------------------------------------

    def __iter__(self) -> Iterator[VideoFrame]:
        if self._closed:
            raise VideoSourceError("reader is closed")
        if not self._sought:
            self._seek_to_start()

        while not self._exhausted:
            if self.max_frames is not None and self.frames_yielded >= self.max_frames:
                break

            ok, image = self._cap.read()
            if not ok or image is None:
                self._exhausted = True
                break

            index = max(0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            timestamp_s, exact = self._timestamp_for(index)
            self.frames_read += 1

            if self.end_s is not None and timestamp_s >= self.end_s:
                self._exhausted = True
                break

            self._note_gap(timestamp_s)
            self._last_timestamp_s = timestamp_s
            if not exact:
                self._inexact_timestamps += 1

            # Stride is applied after timing so skipped frames still inform the
            # frame-rate estimate and the timeline stays anchored to the source.
            if (self.frames_read - 1) % self.frame_stride != 0:
                continue

            if self.target_size is not None:
                image = cv2.resize(
                    image, self.target_size, interpolation=cv2.INTER_AREA
                )

            self.frames_yielded += 1
            yield VideoFrame(
                index=index,
                timestamp_s=timestamp_s,
                image=image,
                source_size=(self.source_width, self.source_height),
                scale=self.scale,
                timestamp_is_exact=exact,
            )

    # -- metadata ---------------------------------------------------------

    def source_info(self) -> SourceInfo:
        """Describe the clip.  Most accurate once iteration has finished."""
        processed = self.target_size or (self.source_width, self.source_height)
        return SourceInfo(
            path=str(self.path),
            width=self.source_width,
            height=self.source_height,
            fps=round(self.fps, 4),
            frame_count=self.source_frame_count,
            duration_s=round(self.source_duration_s, 3) if self.source_duration_s else None,
            variable_frame_rate=self.is_variable_frame_rate,
            processed_width=processed[0],
            processed_height=processed[1],
            segment_start_s=self.start_s if self.start_s > 0 else None,
            segment_end_s=self.end_s,
        )

    @property
    def output_fps(self) -> float:
        """The rate an annotated clip should be written at, given the stride."""
        return self.fps / self.frame_stride

    @property
    def inexact_timestamp_count(self) -> int:
        """Frames whose time came from the nominal fps rather than the container."""
        return self._inexact_timestamps

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if not self._closed:
            self._cap.release()
            self._closed = True

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"VideoReader({self.path.name!r}, {self.source_width}x{self.source_height}"
            f"@{self.fps:.2f}fps, yielded={self.frames_yielded})"
        )


def probe(path: str | Path) -> SourceInfo:
    """Open a clip just long enough to describe it."""
    with VideoReader(path) as reader:
        return reader.source_info()
