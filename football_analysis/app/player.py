"""Plays a clip frame by frame so the window can seek to any event."""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QTimer, Signal


class ClipPlayer(QObject):
    """A clip on disk, decoded with OpenCV, paced by a timer.

    OpenCV rather than Qt's media player because the analysis reads clips
    with OpenCV too: a frame shown at 12.40 s here is the frame the timeline
    means by 12.40 s, with no second decoder's idea of time in between.
    """

    frame = Signal(object, float)
    """``(image_bgr, timestamp_s)``"""

    state_changed = Signal(bool)
    """True while playing."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._capture: cv2.VideoCapture | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.path: Path | None = None
        self.fps = 30.0
        self.duration_s = 0.0
        self.size = (0, 0)
        self.position_s = 0.0

    def open(self, path: str | Path) -> bool:
        self.close()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return False
        self._capture = capture
        self.path = Path(path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duration_s = float(count) / self.fps if count > 0 else 0.0
        self.size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._timer.setInterval(max(5, int(1000.0 / self.fps)))
        self.seek(0.0)
        return True

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def play(self) -> None:
        if self._capture is None:
            return
        if self.duration_s and self.position_s >= self.duration_s - 1.0 / self.fps:
            self.seek(0.0)
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        self._timer.stop()
        self.state_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def seek(self, t: float) -> None:
        if self._capture is None:
            return
        t = max(0.0, min(t, self.duration_s or t))
        # Seeking by frame number is exact for the constant-rate clips the app
        # records; by milliseconds some backends land on the nearest keyframe.
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * self.fps)))
        self._read()

    def step(self, frames: int) -> None:
        self.pause()
        self.seek(self.position_s + frames / self.fps)

    def _read(self) -> bool:
        assert self._capture is not None
        index = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        ok, image = self._capture.read()
        if not ok:
            return False
        self.position_s = index / self.fps
        self.frame.emit(image, self.position_s)
        return True

    def _tick(self) -> None:
        if self._capture is None or not self._read():
            self.pause()

    def close(self) -> None:
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.path = None
        self.position_s = 0.0
