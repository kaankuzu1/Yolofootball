"""Record camera frames to a clip whose timestamps mean wall-clock time.

A webcam does not deliver a steady frame rate. A MacBook camera in a dim hall
drops from 30 to 15 fps to expose longer, and a busy machine skips frames. A
writer that simply appends every frame it gets, at a nominal 30 fps, produces
a clip that runs fast wherever frames went missing -- and every timestamp the
analysis reports after that point is early.

So frames are placed by the time they arrived: frame *n* of the file shows
what the camera saw at ``n / fps`` seconds after recording started. A late
frame is repeated to fill the gap, an early one is dropped. Timestamps are then
right to within one frame (33 ms at 30 fps), whatever the camera did.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from football_analysis.io import AnnotatedVideoWriter


class PacedRecorder:
    """Writes a constant-frame-rate clip from frames that arrive irregularly."""

    def __init__(self, path: str | Path, *, fps: float = 30.0,
                 max_fill_s: float = 2.0) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.path = Path(path)
        self.fps = float(fps)
        self.max_fill_s = max_fill_s
        """A gap longer than this is filled only up to this much, so a stalled
        camera cannot make the writer spin out minutes of one frozen image."""
        self._writer = AnnotatedVideoWriter(self.path, fps=self.fps)
        self._t0: float | None = None
        self._last_t: float | None = None
        self.frames_written = 0
        self.frames_received = 0
        self.frames_repeated = 0
        self.frames_dropped = 0

    @property
    def duration_s(self) -> float:
        return self.frames_written / self.fps

    def write(self, image: np.ndarray, t: float) -> None:
        """Add the frame that arrived at monotonic time ``t`` seconds."""
        self.frames_received += 1
        if self._t0 is None:
            self._t0 = t
        elif self._last_t is not None and t - self._last_t > self.max_fill_s:
            # The camera stalled. Shift the origin so the gap costs at most
            # ``max_fill_s`` of frozen picture rather than all of it.
            self._t0 += (t - self._last_t) - self.max_fill_s
        self._last_t = t

        # How many frames the file should hold once this one is in.
        due = int((t - self._t0) * self.fps) + 1
        if due <= self.frames_written:
            self.frames_dropped += 1
            return
        repeats = due - self.frames_written
        for _ in range(repeats):
            self._writer.write(image)
        self.frames_written += repeats
        self.frames_repeated += repeats - 1

    def close(self) -> Path | None:
        self._writer.close()
        return self.path if self.frames_written else None

    def __enter__(self) -> "PacedRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
