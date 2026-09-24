"""The two pieces of work that must not run on the window's thread.

``LiveDetector`` runs the detector on the newest camera frame, over and over.
It never queues: while it is busy with one frame, newer ones replace each
other, so the boxes lag the picture by one inference and never drift further
behind. On a MacBook with Apple silicon it runs on the GPU through MPS.

``AnalysisWorker`` runs the full analysis over a finished clip -- the same
:func:`football_analysis.analyze` call the command line makes -- and reports
which pass it is on.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from football_analysis.config import Config, load_config
from football_analysis.types import BALL, GOAL, PLAYER, VideoFrame

logger = logging.getLogger(__name__)

LIVE_WIDTH = 1280
"""Frames are resized to this before live detection, as the analysis does."""


def live_config(base: Config | None = None) -> Config:
    config = base or load_config(None)
    return config.validate()


def describe_device(requested: str = "auto") -> str:
    """What ``auto`` will run on, in words: ``Apple GPU (MPS)``, ``NVIDIA GPU``, ``CPU``."""
    from football_analysis.pipeline import _resolve_device

    try:
        device = _resolve_device(requested)
    except Exception:  # pragma: no cover - torch missing
        return "CPU"
    return {"mps": "Apple GPU (MPS)", "cuda": "NVIDIA GPU (CUDA)"}.get(
        device.split(":")[0], "CPU"
    )


@dataclass
class LiveBox:
    """A box to draw, in the camera frame's own pixels."""

    kind: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None


@dataclass
class LiveResult:
    boxes: list[LiveBox] = field(default_factory=list)
    latency_ms: float = 0.0
    fps: float = 0.0
    frame_t: float = 0.0


class LiveDetector(QThread):
    """Detects players, ball and goal on the newest frame, continuously."""

    result = Signal(object)
    status = Signal(str)
    failed = Signal(str)

    # The tracker keeps a small record per frame for its whole life. Live, that
    # life is unbounded, so it starts over now and then; identities may
    # renumber at that moment, which only changes a box's colour.
    TRACKER_RESET_FRAMES = 3000

    def __init__(self, config: Config | None = None, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._paused = threading.Event()

    def submit(self, image: np.ndarray, t: float) -> None:
        """Offer a frame. Replaces any frame not yet picked up."""
        with self._lock:
            self._latest = (image, t)
        self._wake.set()

    def pause(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()
            self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self.wait(5000)

    def _take(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            item, self._latest = self._latest, None
        return item

    def run(self) -> None:  # noqa: C901 - one loop, kept together on purpose
        from football_analysis.pipeline import (
            default_detector,
            default_tracker,
            resolve_weights,
        )

        config = live_config(self._config)
        if resolve_weights(config) is None:
            self.failed.emit(
                f"Model dosyası bulunamadı ({config.detection.model_path}). "
                "assets/models klasörüne koyun; kurulum için README'ye bakın."
            )
            return
        self.status.emit("Model yükleniyor…")
        try:
            detector = default_detector(config)
            tracker = default_tracker(config)
        except Exception as exc:  # pragma: no cover - depends on the machine
            logger.exception("live detector failed to load")
            self.failed.emit(f"Model yüklenemedi: {exc}")
            return
        self.status.emit(f"Canlı tespit hazır · {describe_device(config.detection.device)}")

        index = 0
        recent: list[float] = []
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            if self._stop.is_set():
                break
            if self._paused.is_set():
                continue
            item = self._take()
            if item is None:
                continue
            image, t = item
            height, width = image.shape[:2]
            scale = min(1.0, LIVE_WIDTH / float(width))
            small = image if scale == 1.0 else cv2.resize(
                image, (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            frame = VideoFrame(
                index=index, timestamp_s=float(index) / 30.0, image=small,
                source_size=(width, height), scale=scale,
            )
            started = time.monotonic()
            try:
                detections = detector.detect(frame)
                tracks = tracker.update(frame, detections)
            except Exception as exc:  # pragma: no cover - model/runtime failure
                logger.exception("live detection failed")
                self.failed.emit(f"Canlı tespit durdu: {exc}")
                return
            elapsed = time.monotonic() - started
            index += 1
            if index % self.TRACKER_RESET_FRAMES == 0:
                tracker.reset()

            now = time.monotonic()
            recent = [x for x in recent if now - x < 2.0] + [now]
            inv = 1.0 / scale
            boxes = []
            for track in tracks:
                if track.class_name not in (PLAYER, BALL, GOAL):
                    continue
                b = track.bbox
                boxes.append(LiveBox(
                    kind=track.class_name,
                    xyxy=(b.x1 * inv, b.y1 * inv, b.x2 * inv, b.y2 * inv),
                    confidence=float(track.confidence),
                    track_id=track.track_id if track.class_name == PLAYER else None,
                ))
            self.result.emit(LiveResult(
                boxes=boxes, latency_ms=elapsed * 1000.0,
                fps=len(recent) / 2.0, frame_t=t,
            ))


class Cancelled(Exception):
    pass


@dataclass
class AnalysisRequest:
    clip: Path
    events_json: Path
    annotated: Path | None
    states: Path | None
    goal_corners_px: list[list[float]] | None
    """Already in the processed frame's pixels."""
    goal_size_m: tuple[float, float] | None
    frame_stride: int = 1
    config: Config | None = None


PHASES = {
    "load": "Model yükleniyor",
    "analyse": "Oyuncular, top ve kale bulunuyor",
    "refine": "Duruşlar okunuyor",
    "events": "Olaylar çıkarılıyor",
    "render": "Çizimli video yazılıyor",
}


def build_config(request: AnalysisRequest) -> Config:
    config = request.config or load_config(None)
    config.output.json_path = str(request.events_json)
    config.output.video_path = str(request.annotated) if request.annotated else None
    config.output.state_cache_path = str(request.states) if request.states else None
    config.video.frame_stride = max(1, int(request.frame_stride))
    if request.goal_corners_px:
        config.geometry["goal_corners_px"] = [list(map(float, p)) for p in request.goal_corners_px]
    if request.goal_size_m:
        config.geometry["goal_width_m"] = float(request.goal_size_m[0])
        config.geometry["goal_height_m"] = float(request.goal_size_m[1])
    return config.validate()


class _WarningCollector(logging.Handler):
    """Keeps the warnings a run logs, so the app can show them afterwards."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover
            return
        if message not in self.messages:
            self.messages.append(message)


class AnalysisWorker(QThread):
    """Runs one clip through the whole system."""

    progress = Signal(str, float)
    """Phase text and overall fraction done, 0..1 (negative when unknown)."""

    finished_ok = Signal(object, list)
    """The :class:`PipelineResult` and the warnings logged along the way."""

    failed = Signal(str)

    # How much of the bar each pass gets. Detection dominates a run.
    _SPANS = {"analyse": (0.02, 0.75), "refine": (0.75, 0.9), "events": (0.9, 0.97)}

    def __init__(self, request: AnalysisRequest, parent=None) -> None:
        super().__init__(parent)
        self.request = request
        self._cancel = threading.Event()
        self._total_frames = 0

    def cancel(self) -> None:
        self._cancel.set()

    def _hook(self, update) -> None:
        if self._cancel.is_set():
            raise Cancelled()
        phase = update.phase
        lo, hi = self._SPANS.get(phase, (0.97, 1.0))
        if phase == "analyse" and self._total_frames:
            share = min(1.0, update.frames_processed / self._total_frames)
        elif phase == "events" and self._total_frames:
            share = min(1.0, update.frames_processed / self._total_frames)
        else:
            share = 0.0
        text = PHASES.get(phase, phase)
        if phase == "analyse" and update.elapsed_s > 3 and share > 0.01:
            remaining = update.elapsed_s * (1.0 - share) / share
            text += f" · yaklaşık {_duration(remaining)} kaldı"
        self.progress.emit(text, lo + (hi - lo) * share)

    def run(self) -> None:
        from football_analysis.io.video_reader import probe
        from football_analysis.pipeline import analyze, resolve_weights

        collector = _WarningCollector()
        root = logging.getLogger("football_analysis")
        root.addHandler(collector)
        try:
            config = build_config(self.request)
            if resolve_weights(config) is None:
                self.failed.emit(
                    f"Model dosyası bulunamadı ({config.detection.model_path}). "
                    "Olmadan yapılan analiz oyuncuları değil hareketi bulur, bu yüzden başlatılmadı."
                )
                return
            info = probe(self.request.clip)
            self._total_frames = max(1, int((info.frame_count or 0) / config.video.frame_stride))
            self.progress.emit(PHASES["load"], 0.01)
            result = analyze(self.request.clip, config, progress=self._hook)
        except Cancelled:
            self.failed.emit("")
            return
        except Exception as exc:
            logger.exception("analysis failed")
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        finally:
            root.removeHandler(collector)
        self.progress.emit("Bitti", 1.0)
        self.finished_ok.emit(result, collector.messages)


def _duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} sn"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} dk {seconds:02d} sn"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} sa {minutes:02d} dk"
