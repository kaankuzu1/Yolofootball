"""YOLO-backed detector with a single ``detect(frame)`` entry point.

The football-specific weights and the stock COCO weights disagree about class
names, so both are normalised through :data:`NAME_MAP` onto the vocabulary in
``types.py``.

The ball is the hard case: on a wide side-angle view it covers roughly 15-25
pixels, which is close to the smallest object a 640-pixel-input YOLO can
resolve.  ``tile_ball`` runs a second pass over overlapping crops so the ball
is seen at something nearer its native scale.  It costs extra time per frame,
so it is opt-in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .types import BALL, GOALKEEPER, PLAYER, REFEREE, Detection

# Maps a lowercased class name from any backing model onto our vocabulary.
# Anything absent from this map is dropped.
NAME_MAP = {
    "ball": BALL,
    "sports ball": BALL,
    "soccer ball": BALL,
    "football": BALL,
    "player": PLAYER,
    "person": PLAYER,
    "goalkeeper": GOALKEEPER,
    "goal keeper": GOALKEEPER,
    "referee": REFEREE,
}


@dataclass
class DetectorConfig:
    weights: str
    conf: float = 0.25
    # The ball is small and low-contrast, so it is scored on its own threshold.
    ball_conf: float = 0.10
    iou: float = 0.5
    imgsz: int = 1280
    device: str = "cpu"
    # Half precision. Sent to Ultralytics under its current keyword; see
    # _precision_kwargs. Pointless on CPU, which is where this runs today.
    half: bool = False
    # Second pass over overlapping crops, ball class only.
    tile_ball: bool = False
    tile_grid: Tuple[int, int] = (3, 2)
    tile_overlap: float = 0.2
    tile_imgsz: int = 640
    # Reject ball detections whose surroundings are not pitch. Every false
    # positive measured on real footage was off the pitch -- a smear past the
    # touchline, an object in the stand -- so this is cheap precision.
    ball_on_grass: bool = False
    grass_fraction: float = 0.35
    # OpenCV HSV bounds for pitch green (hue 0-179).
    grass_hsv_low: Tuple[int, int, int] = (30, 40, 25)
    grass_hsv_high: Tuple[int, int, int] = (90, 255, 255)
    # Classes to keep; None keeps everything the map knows about.
    keep: Sequence[str] | None = None
    verbose: bool = False


class Detector:
    """Wraps an Ultralytics YOLO model behind ``detect(frame)``."""

    def __init__(self, config: DetectorConfig):
        from ultralytics import YOLO  # imported lazily: heavy

        self.config = config
        self.model = YOLO(config.weights)
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self._last_ms: float = 0.0

    # -- public API ----------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in one BGR frame.

        Returns detections in the frame's own pixel coordinates, sorted by
        descending confidence.
        """
        started = time.perf_counter()
        dets = self._predict(frame, imgsz=self.config.imgsz)
        if self.config.tile_ball:
            dets.extend(self._tiled_ball_pass(frame))
            dets = _nms_by_label(dets, iou_threshold=self.config.iou)
        dets = [d for d in dets if self._passes_threshold(d)]
        if self.config.ball_on_grass:
            dets = [d for d in dets if d.label != BALL or _on_grass(frame, d, self.config)]
        if self.config.keep is not None:
            keep = set(self.config.keep)
            dets = [d for d in dets if d.label in keep]
        self._last_ms = (time.perf_counter() - started) * 1000.0
        return sorted(dets, key=lambda d: -d.confidence)

    @property
    def last_ms(self) -> float:
        """Wall-clock milliseconds taken by the most recent ``detect`` call."""
        return self._last_ms

    def supports(self, label: str) -> bool:
        """Whether the loaded weights can ever emit ``label``."""
        return label in {NAME_MAP.get(n.lower()) for n in self.names.values()}

    # -- internals -----------------------------------------------------

    def _passes_threshold(self, det: Detection) -> bool:
        floor = self.config.ball_conf if det.label == BALL else self.config.conf
        return det.confidence >= floor

    def _predict(
        self,
        image: np.ndarray,
        imgsz: int,
        offset: Tuple[float, float] = (0.0, 0.0),
        classes: Iterable[int] | None = None,
    ) -> List[Detection]:
        # Predict at the lowest threshold we might keep, then filter per class
        # afterwards, so one pass serves both the ball and everything else.
        floor = min(self.config.conf, self.config.ball_conf)
        results = self.model.predict(
            image,
            imgsz=imgsz,
            conf=floor,
            iou=self.config.iou,
            device=self.config.device,
            classes=list(classes) if classes is not None else None,
            verbose=self.config.verbose,
            **_precision_kwargs(self.config.half),
        )
        dx, dy = offset
        out: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, conf, cls):
                label = NAME_MAP.get(self.names.get(class_id, "").lower())
                if label is None:
                    continue
                out.append(
                    Detection(
                        label=label,
                        confidence=float(score),
                        xyxy=(
                            float(box[0]) + dx,
                            float(box[1]) + dy,
                            float(box[2]) + dx,
                            float(box[3]) + dy,
                        ),
                    )
                )
        return out

    def _ball_class_ids(self) -> List[int]:
        return [i for i, n in self.names.items() if NAME_MAP.get(n.lower()) == BALL]

    def _tiled_ball_pass(self, frame: np.ndarray) -> List[Detection]:
        ball_ids = self._ball_class_ids()
        if not ball_ids:
            return []
        out: List[Detection] = []
        for crop, (x0, y0) in _tiles(frame, self.config.tile_grid, self.config.tile_overlap):
            out.extend(
                self._predict(
                    crop,
                    imgsz=self.config.tile_imgsz,
                    offset=(x0, y0),
                    classes=ball_ids,
                )
            )
        return out


# -- helpers -----------------------------------------------------------


def _precision_kwargs(half: bool) -> dict:
    """Half precision, under whichever keyword the installed Ultralytics wants.

    ``half=True`` became ``quantize="fp16"`` in Ultralytics 8.4, and passing the
    old name prints a deprecation warning on every single ``predict`` call.  The
    config field keeps the name ``half`` because that is what callers and the
    pipeline config use; only the wire format changes.  Full precision passes
    nothing at all, which is both the default and silent on every version.
    """
    if not half:
        return {}
    key = _precision_key()
    return {key: "fp16"} if key == "quantize" else {key: True}


@lru_cache(maxsize=1)
def _precision_key() -> str:
    # Cached, and looked up lazily, so importing this module does not pull in
    # ultralytics.
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception:  # pragma: no cover - very old or unusual install
        return "half"
    return "quantize" if "quantize" in DEFAULT_CFG_DICT else "half"



def _tiles(
    frame: np.ndarray, grid: Tuple[int, int], overlap: float
) -> Iterable[Tuple[np.ndarray, Tuple[float, float]]]:
    """Yield overlapping crops of ``frame`` with their top-left offsets."""
    height, width = frame.shape[:2]
    cols, rows = grid
    tile_w = width / cols
    tile_h = height / rows
    pad_x = tile_w * overlap
    pad_y = tile_h * overlap
    for row in range(rows):
        for col in range(cols):
            x0 = int(max(0, col * tile_w - pad_x))
            y0 = int(max(0, row * tile_h - pad_y))
            x1 = int(min(width, (col + 1) * tile_w + pad_x))
            y1 = int(min(height, (row + 1) * tile_h + pad_y))
            yield frame[y0:y1, x0:x1], (float(x0), float(y0))


def _nms_by_label(dets: List[Detection], iou_threshold: float) -> List[Detection]:
    """Greedy NMS applied independently within each label."""
    kept: List[Detection] = []
    for label in {d.label for d in dets}:
        group = sorted((d for d in dets if d.label == label), key=lambda d: -d.confidence)
        chosen: List[Detection] = []
        for det in group:
            if all(_iou(det.xyxy, other.xyxy) < iou_threshold for other in chosen):
                chosen.append(det)
        kept.extend(chosen)
    return kept


def _on_grass(frame: np.ndarray, det: Detection, config: "DetectorConfig") -> bool:
    """Whether a detection sits on pitch, judged by the green around it.

    The ball itself is white, so the ring around the box is what is sampled,
    not the box: a ball on grass has green surroundings, a ball-shaped thing in
    the crowd or on an advertising hoarding does not.
    """
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = det.xyxy
    pad = max(8.0, 1.5 * max(det.width, det.height))
    x0 = int(max(0, x1 - pad)); y0 = int(max(0, y1 - pad))
    x3 = int(min(width, x2 + pad)); y3 = int(min(height, y2 + pad))
    patch = frame[y0:y3, x0:x3]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(config.grass_hsv_low), np.array(config.grass_hsv_high))
    return float(mask.mean()) / 255.0 >= config.grass_fraction


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
