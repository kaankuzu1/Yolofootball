"""Writing the annotated clip back out.

The writer is deliberately dumb: it takes frames that :mod:`football_analysis.viz`
has already drawn on and encodes them.  It exists as its own module because
codec selection is the one part of this that is environment-dependent, and it
is better to have that in one place than sprinkled through the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["AnnotatedVideoWriter", "VideoWriteError"]

# Tried in order.  mp4v is the one that is present in essentially every
# OpenCV build; avc1 produces smaller files but needs a licensed encoder.
_CODECS_BY_SUFFIX: dict[str, tuple[str, ...]] = {
    ".mp4": ("avc1", "mp4v"),
    ".m4v": ("mp4v",),
    ".avi": ("MJPG", "XVID"),
    ".mkv": ("mp4v", "MJPG"),
    ".mov": ("mp4v", "avc1"),
    ".webm": ("VP80",),
}
_FALLBACK_CODECS = ("mp4v", "MJPG")


class VideoWriteError(RuntimeError):
    """No codec available could open the output file for writing."""


class AnnotatedVideoWriter:
    """Encode drawn-on frames to a clip.

    The frame size is taken from the first frame written, so callers do not
    have to know it up front -- which matters because the reader may have
    resized.  Later frames of a different size are resized to match rather
    than silently corrupting the output.

    Use it as a context manager::

        with AnnotatedVideoWriter("out.mp4", fps=30.0) as writer:
            writer.write(annotated_frame)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        size: tuple[int, int] | None = None,
        codec: str | None = None,
    ) -> None:
        self.path = Path(path)
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = float(fps)
        self.size = size
        self.requested_codec = codec
        self.codec: str | None = None
        self.frames_written = 0
        self._writer: cv2.VideoWriter | None = None
        self._closed = False

        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if size is not None:
            self._open(size)

    def _candidate_codecs(self) -> tuple[str, ...]:
        if self.requested_codec:
            return (self.requested_codec,)
        return _CODECS_BY_SUFFIX.get(self.path.suffix.lower(), _FALLBACK_CODECS)

    def _open(self, size: tuple[int, int]) -> None:
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid output size: {size}")

        attempts: list[str] = []
        for codec in self._candidate_codecs():
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if writer.isOpened():
                self._writer = writer
                self.codec = codec
                self.size = (width, height)
                if attempts:
                    logger.info("codec %s unavailable; wrote with %s", attempts[0], codec)
                return
            writer.release()
            attempts.append(codec)

        raise VideoWriteError(
            f"could not open {self.path} for writing; tried codecs: {', '.join(attempts)}"
        )

    def write(self, image: np.ndarray) -> None:
        """Encode one frame.  Opens the file on the first call if needed."""
        if self._closed:
            raise VideoWriteError("writer is closed")
        if image is None or image.size == 0:
            raise ValueError("refusing to write an empty frame")

        height, width = image.shape[:2]
        if self._writer is None:
            self._open((width, height))
        assert self._writer is not None and self.size is not None

        if (width, height) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.release()
        self._closed = True
        if self.frames_written == 0:
            logger.warning("no frames were written to %s", self.path)

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"AnnotatedVideoWriter({self.path.name!r}, codec={self.codec}, "
            f"fps={self.fps:.2f}, frames={self.frames_written})"
        )
