"""Seams between this package's stage contracts and concrete implementations.

A stage built in its own module is free to have its own vocabulary -- its own
``Detection`` record, its own idea of what it is handed per frame.  Forcing
every stage to import :mod:`football_analysis.types` would couple them all
together for no benefit, and rewriting a working stage to match a contract is
the wrong way round.

So the conversion lives here, in the layer that owns the contract.  An adapter
wraps a foreign stage and presents it as a
:class:`~football_analysis.interfaces.Detector` (or tracker, or event
detector), doing the coordinate and vocabulary translation in one place where
it can be tested.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from football_analysis.types import BBox, Detection, VideoFrame

logger = logging.getLogger(__name__)

__all__ = [
    "FrameDetectorAdapter",
    "adapt_detector",
    "MeasurementObserver",
    "PoseRefiner",
    "CompositeEventDetector",
]


class _FrameArrayDetector(Protocol):
    """A detector that takes raw pixels and returns records of its own."""

    def detect(self, frame: np.ndarray) -> Sequence[Any]: ...


def _to_bbox(xyxy: Sequence[float], frame_size: tuple[int, int]) -> BBox | None:
    """Build a :class:`BBox`, clamped to the frame, or ``None`` if degenerate.

    A model can emit a box a pixel or two outside the frame, or one whose
    corners have been rounded into each other.  Neither should take a run down,
    and neither should become a zero-area box that the event logic then divides
    by, so they are clamped and the empty ones are dropped.
    """
    if xyxy is None or len(xyxy) != 4:
        return None
    width, height = frame_size
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return BBox(x1, y1, x2, y2)


class FrameDetectorAdapter:
    """Present a pixels-in, own-records-out detector as a :class:`Detector`.

    Parameters
    ----------
    detector:
        Anything with ``detect(ndarray)``.
    label_attr, confidence_attr, box_attr:
        Where to read the class name, the score and the box from each returned
        record.  The defaults match a record with ``label``, ``confidence`` and
        ``xyxy`` attributes.
    label_map:
        Rename classes on the way through, e.g. ``{"goalkeeper": "player"}``.
        Unlisted labels pass through unchanged.
    keep:
        Only emit these class names (after mapping).  ``None`` keeps everything.
    """

    def __init__(
        self,
        detector: _FrameArrayDetector,
        *,
        label_attr: str = "label",
        confidence_attr: str = "confidence",
        box_attr: str = "xyxy",
        label_map: dict[str, str] | None = None,
        keep: Sequence[str] | None = None,
    ) -> None:
        self.detector = detector
        self.label_attr = label_attr
        self.confidence_attr = confidence_attr
        self.box_attr = box_attr
        self.label_map = label_map or {}
        self.keep = set(keep) if keep is not None else None
        self.dropped = 0
        """Records discarded for having an unusable box.  Worth logging."""

    def _read(self, record: Any, name: str) -> Any:
        if isinstance(record, dict):
            return record.get(name)
        return getattr(record, name, None)

    def detect(self, frame: VideoFrame) -> list[Detection]:
        raw = self.detector.detect(frame.image)
        out: list[Detection] = []
        for record in raw or ():
            label = self._read(record, self.label_attr)
            if label is None:
                continue
            label = self.label_map.get(str(label), str(label))
            if self.keep is not None and label not in self.keep:
                continue

            box = _to_bbox(self._read(record, self.box_attr), frame.size)
            if box is None:
                self.dropped += 1
                continue

            confidence = self._read(record, self.confidence_attr)
            out.append(
                Detection(
                    class_name=label,
                    bbox=box,
                    confidence=float(confidence) if confidence is not None else 0.0,
                )
            )
        return out

    def reset(self) -> None:
        self.dropped = 0
        inner = getattr(self.detector, "reset", None)
        if callable(inner):
            inner()

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": type(self.detector).__name__,
            "adapter": type(self).__name__,
        }
        inner = getattr(self.detector, "describe", None)
        if callable(inner):
            try:
                payload["inner"] = inner()
            except Exception:  # pragma: no cover - a stage's own reporting bug
                logger.debug("wrapped detector failed to describe itself", exc_info=True)
        config = getattr(self.detector, "config", None)
        if config is not None and "inner" not in payload:
            for attr in ("weights", "model_path", "imgsz", "device", "conf"):
                value = getattr(config, attr, None)
                if value is not None:
                    payload[attr] = value
        return payload


def adapt_detector(
    detector: Any,
    *,
    label_map: dict[str, str] | None = None,
    keep: Sequence[str] | None = None,
) -> Any:
    """Wrap ``detector`` only if it needs it.

    A detector already speaking this package's contract -- one whose ``detect``
    takes a :class:`VideoFrame` -- is returned untouched, so this is safe to
    call on anything.
    """
    import inspect

    detect = getattr(detector, "detect", None)
    if detect is None:
        raise TypeError(f"{detector!r} has no detect() method")

    try:
        params = list(inspect.signature(detect).parameters.values())
        annotation = params[0].annotation if params else None
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callables
        annotation = None

    if annotation in (VideoFrame, "VideoFrame") and label_map is None and keep is None:
        return detector
    return FrameDetectorAdapter(detector, label_map=label_map, keep=keep)


class MeasurementObserver:
    """Turn a ``measure(image, state) -> value`` meter into a frame observer.

    Meters tend to be written as plain measuring instruments -- give them
    pixels and the current record, get a number back -- which is the right
    shape for the thing itself and the wrong shape for the pipeline.  This puts
    the number on the record under an agreed key, so the pipeline needs to know
    nothing about any particular meter.

    ``None`` from the meter means "no reading on this frame" and leaves the key
    absent, which is different from a reading of zero and must stay that way:
    a rule that cannot tell "the net did not move" from "nobody looked" will
    draw the wrong conclusion.
    """

    def __init__(self, meter: Any, key: str, *, digits: int | None = 3) -> None:
        if not hasattr(meter, "measure"):
            raise TypeError(f"{meter!r} has no measure() method")
        self.meter = meter
        self.key = key
        self.digits = digits

    def observe(self, frame: VideoFrame, state: Any) -> None:
        value = self.meter.measure(frame.image, state)
        if value is None:
            return
        value = float(value)
        state.attributes[self.key] = (
            round(value, self.digits) if self.digits is not None else value
        )

    def reset(self) -> None:
        inner = getattr(self.meter, "reset", None)
        if callable(inner):
            inner()

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": type(self.meter).__name__,
            "observer": type(self).__name__,
            "key": self.key,
        }
        inner = getattr(self.meter, "describe", None)
        if callable(inner):
            try:
                payload["inner"] = inner()
            except Exception:  # pragma: no cover - a meter's own reporting bug
                logger.debug("meter failed to describe itself", exc_info=True)
        return payload


class CompositeEventDetector:
    """Fan one record out to several event detectors and merge what they call.

    The event logic is split by what it reasons about -- ball flight in one
    place, bodies in another -- and a run wants all of it.  Each detector sees
    every frame and every ``finalize``, and their events are concatenated; the
    pipeline sorts and de-duplicates afterwards, so order here does not matter.

    One detector raising does not silently lose the others' work: the exception
    propagates, because a half-analysed timeline presented as a whole one is
    worse than a failed run.
    """

    def __init__(self, *detectors: Any) -> None:
        flat: list[Any] = []
        for detector in detectors:
            if detector is None:
                continue
            if isinstance(detector, (list, tuple)):
                flat.extend(d for d in detector if d is not None)
            else:
                flat.append(detector)
        self.detectors = flat

    def update(self, state: Any) -> list[Any]:
        events: list[Any] = []
        for detector in self.detectors:
            events.extend(detector.update(state) or ())
        return events

    def finalize(self) -> list[Any]:
        events: list[Any] = []
        for detector in self.detectors:
            finalize = getattr(detector, "finalize", None)
            if callable(finalize):
                events.extend(finalize() or ())
        return events

    def reset(self) -> None:
        for detector in self.detectors:
            reset = getattr(detector, "reset", None)
            if callable(reset):
                reset()

    def describe(self) -> dict[str, Any]:
        stages = []
        for detector in self.detectors:
            describe = getattr(detector, "describe", None)
            if callable(describe):
                try:
                    stages.append(describe())
                    continue
                except Exception:  # pragma: no cover - a stage's reporting bug
                    logger.debug("stage failed to describe itself", exc_info=True)
            stages.append({"name": type(detector).__name__})
        return {"name": type(self).__name__, "stages": stages}


class PoseRefiner:
    """Fills pose keypoints into the finished records, by re-reading the clip.

    A :class:`~football_analysis.interfaces.StateRefiner`. It exists as a seam
    rather than a call inside the pipeline for the usual reason: the pipeline
    should not know what a skeleton is. It knows it runs refiners; this one
    knows it runs :func:`football_analysis.pose.attach_pose`.

    ``target_width`` / ``target_height`` must be the resize the records were
    produced under, or the boxes and the pixels will not line up. The pipeline
    passes its own video settings, so they always match.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.target_width = target_width
        self.target_height = target_height

    def refine(self, states: list[Any], video_path: str) -> list[Any]:
        from football_analysis.pose import attach_pose

        refined = attach_pose(
            states,
            video_path,
            self.estimator,
            target_width=self.target_width,
            target_height=self.target_height,
        )
        posed = sum(1 for s in refined for p in s.players if p.keypoints)
        players = sum(len(s.players) for s in refined)
        if players and not posed:
            # Not fatal -- the rules that need pose will simply find none and
            # abstain -- but silent is the wrong way for it to fail, because
            # the only other symptom is no tackles on a clip full of them.
            logger.warning(
                "pose found no keypoints on any of %d player records; "
                "tackle and trick detection will have nothing to read",
                players,
            )
        else:
            logger.info("pose filled %d of %d player records", posed, players)
        return refined

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"name": type(self).__name__}
        describe = getattr(self.estimator, "describe", None)
        if callable(describe):
            info["estimator"] = describe()
        return info
