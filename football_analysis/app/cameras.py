"""Which cameras there are, and a stream of BGR frames from the chosen one.

The list comes from Qt Multimedia, which on macOS asks AVFoundation. That is
what gives each camera its real name ("FaceTime HD Camera", "Logitech BRIO",
"Kaan's iPhone Camera") and a stable id, so a remembered choice survives a
USB camera being unplugged and plugged back into another port. OpenCV can
only number cameras, and the numbering moves when devices come and go.

OpenCV is kept as the fallback source: if Qt Multimedia cannot list anything
(a broken plugin, a Linux box without the right libraries) the app still
offers ``Kamera 0``, ``Kamera 1`` and so on rather than an empty menu.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

QT = "qt"
OPENCV = "opencv"

# Named sizes the sidebar offers. 720p is the default: analysis resizes every
# frame to 1280 wide anyway, so 1080p costs conversion time for pixels the
# detector never sees.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


@dataclass(frozen=True)
class CameraInfo:
    """One camera the user can pick."""

    id: str
    """Stable key for remembering the choice. OpenCV cameras use ``opencv:<n>``."""

    name: str
    backend: str = QT
    is_default: bool = False
    position: str = "unspecified"
    """``front``, ``back`` or ``unspecified``, as the OS reports it."""

    index: int | None = None
    """OpenCV device index, for the fallback backend only."""

    @property
    def label(self) -> str:
        tags = []
        if self.is_default:
            tags.append("varsayılan")
        if self.position == "front":
            tags.append("ön kamera")
        return f"{self.name} ({', '.join(tags)})" if tags else self.name


def order_cameras(cameras: Iterable[CameraInfo]) -> list[CameraInfo]:
    """Built-in first: the system default, then front-facing, then by name.

    On a MacBook the system default is the FaceTime camera until the user
    changes it in System Settings, which is exactly the one to start on.
    """
    return sorted(
        cameras,
        key=lambda c: (not c.is_default, c.position != "front", c.name.lower(), c.id),
    )


def pick_camera(cameras: list[CameraInfo], remembered_id: str | None) -> CameraInfo | None:
    """The remembered camera if it is still connected, else the first in order."""
    if not cameras:
        return None
    if remembered_id:
        for camera in cameras:
            if camera.id == remembered_id:
                return camera
    return cameras[0]


def _qt_cameras() -> list[CameraInfo]:
    try:
        from PySide6.QtMultimedia import QCameraDevice, QMediaDevices
    except ImportError as exc:  # pragma: no cover - depends on the Qt build
        logger.warning("Qt Multimedia unavailable (%s); using OpenCV camera numbers", exc)
        return []

    positions = {
        QCameraDevice.Position.FrontFace: "front",
        QCameraDevice.Position.BackFace: "back",
    }
    found = []
    for device in QMediaDevices.videoInputs():
        raw_id = bytes(device.id().data())
        found.append(
            CameraInfo(
                id=raw_id.decode("utf-8", "replace") or device.description(),
                name=device.description() or "Kamera",
                backend=QT,
                is_default=bool(device.isDefault()),
                position=positions.get(device.position(), "unspecified"),
            )
        )
    return found


def _opencv_cameras(max_index: int = 5) -> list[CameraInfo]:
    import cv2

    found = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                found.append(
                    CameraInfo(
                        id=f"opencv:{index}", name=f"Kamera {index}", backend=OPENCV,
                        is_default=index == 0, index=index,
                    )
                )
        finally:
            capture.release()
    return found


def list_cameras(*, allow_opencv_probe: bool = True) -> list[CameraInfo]:
    """Every camera the OS will hand us, built-in first."""
    cameras = _qt_cameras()
    if not cameras and allow_opencv_probe:
        cameras = _opencv_cameras()
    return order_cameras(cameras)


def choose_format(formats: Iterable[tuple[int, int, float, float]], target: tuple[int, int]):
    """Pick the camera mode closest to ``target`` that still runs at 25 fps or more.

    ``formats`` are ``(width, height, min_fps, max_fps)``. Returns the index of
    the chosen one, or ``None`` when there are none. Frame rate matters more
    than size: a tackle is over in a few frames, and a camera that drops to
    15 fps in a dim sports hall halves what the rules get to see.
    """
    best = None
    best_key = None
    tw, th = target
    for i, (w, h, _lo, hi) in enumerate(formats):
        fast_enough = hi >= 25.0
        size_error = abs(w * h - tw * th) / float(tw * th)
        key = (not fast_enough, round(size_error, 3), -hi)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


class FrameSource(QObject):
    """Emits ``frame(image_bgr, t_monotonic_s)`` for every frame the camera delivers."""

    frame = Signal(object, float)
    error = Signal(str)
    started = Signal(str)
    """Carries a short description of the mode in use, e.g. ``1280x720 @ 30 fps``."""

    def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def qimage_to_bgr(image) -> np.ndarray:
    """Copy a QImage into a contiguous BGR array."""
    from PySide6.QtGui import QImage

    if image.format() != QImage.Format.Format_BGR888:
        image = image.convertToFormat(QImage.Format.Format_BGR888)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * height)
    return buffer.reshape(height, stride)[:, : width * 3].reshape(height, width, 3).copy()


class QtCameraSource(FrameSource):
    """A camera opened through Qt Multimedia (AVFoundation on a Mac)."""

    def __init__(self, camera: CameraInfo, resolution: tuple[int, int] = RESOLUTIONS["720p"],
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.info = camera
        self.resolution = resolution
        self._camera = None
        self._session = None
        self._sink = None

    def _device(self):
        from PySide6.QtMultimedia import QMediaDevices

        for device in QMediaDevices.videoInputs():
            raw = bytes(device.id().data()).decode("utf-8", "replace") or device.description()
            if raw == self.info.id:
                return device
        return None

    def start(self) -> None:
        from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QVideoSink

        device = self._device()
        if device is None:
            self.error.emit(f"“{self.info.name}” artık bağlı değil.")
            return

        formats = list(device.videoFormats())
        chosen = choose_format(
            [(f.resolution().width(), f.resolution().height(),
              f.minFrameRate(), f.maxFrameRate()) for f in formats],
            self.resolution,
        )
        self._camera = QCamera(device)
        if chosen is not None:
            self._camera.setCameraFormat(formats[chosen])
        self._camera.errorOccurred.connect(self._on_error)

        self._sink = QVideoSink()
        self._sink.videoFrameChanged.connect(self._on_frame)
        self._session = QMediaCaptureSession()
        self._session.setCamera(self._camera)
        self._session.setVideoSink(self._sink)
        self._camera.start()

        if chosen is not None:
            f = formats[chosen]
            self.started.emit(
                f"{f.resolution().width()}x{f.resolution().height()} @ {f.maxFrameRate():.0f} fps"
            )
        else:
            self.started.emit("")

    def _on_frame(self, video_frame) -> None:
        if not video_frame.isValid():
            return
        image = video_frame.toImage()
        if image.isNull():
            return
        self.frame.emit(qimage_to_bgr(image), time.monotonic())

    def _on_error(self, _code, message: str) -> None:
        self.error.emit(message or "Kamera açılamadı.")

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
        if self._sink is not None:
            try:
                self._sink.videoFrameChanged.disconnect(self._on_frame)
            except (RuntimeError, TypeError):
                pass
        self._camera = self._session = self._sink = None


class OpenCvCameraSource(FrameSource):
    """The fallback: a numbered camera read by OpenCV on its own thread."""

    def __init__(self, camera: CameraInfo, resolution: tuple[int, int] = RESOLUTIONS["720p"],
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.info = camera
        self.resolution = resolution
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import cv2

        capture = cv2.VideoCapture(int(self.info.index or 0))
        if not capture.isOpened():
            self.error.emit(f"{self.info.name} açılamadı.")
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        self.started.emit(f"{width}x{height}" + (f" @ {fps:.0f} fps" if fps else ""))
        try:
            while not self._stop.is_set():
                ok, image = capture.read()
                if not ok:
                    self.error.emit(f"{self.info.name} görüntü vermiyor.")
                    return
                self.frame.emit(image, time.monotonic())
        finally:
            capture.release()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def open_source(camera: CameraInfo, resolution: tuple[int, int],
                parent: QObject | None = None) -> FrameSource:
    if camera.backend == OPENCV:
        return OpenCvCameraSource(camera, resolution, parent)
    return QtCameraSource(camera, resolution, parent)
