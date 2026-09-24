"""YOLO pose on player crops.

Why crops rather than the whole frame: on a side-angle clip a player can be
80 pixels tall, and a whole-frame pose model at 640 input sees that as a
25-pixel figure whose ankles are two pixels apart.  Cropping each tracked box,
padding it, and upscaling it to a fixed height gives the pose model the player
at something close to the scale it was trained on.

The one subtle part is the tackle itself: when two players overlap, the crop
for one of them contains both, and the pose model happily returns two
skeletons.  The skeleton kept is the one whose own box best matches the box
the tracker asked about, so a lunging defender's legs are never attributed to
the attacker they are lunging at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from football_analysis.state import FrameState, Keypoint
from football_analysis.types import PLAYER, BBox, Track, VideoFrame

logger = logging.getLogger(__name__)

__all__ = [
    "COCO_KEYPOINTS",
    "PoseConfig",
    "PoseEstimator",
    "PoseTracker",
    "attach_pose",
]

COCO_KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass
class PoseConfig:
    weights: str = "yolo11s-pose.pt"
    """Any Ultralytics pose checkpoint.  Downloaded on first use if it is a
    stock name; ``assets/models/`` is the place for a local copy."""

    crop_height: int = 320
    """Every crop is resized to this height before inference."""

    pad: float = 0.2
    """Crop padding, as a fraction of the box size on each side.  Generous,
    because a lunge puts a foot well outside the box the detector drew."""

    min_box_height: float = 24.0
    """Below this many pixels there is nothing to read; skip the player."""

    conf: float = 0.15
    device: str = "cpu"
    batch: int = 8


class PoseEstimator:
    """One COCO-17 skeleton per player box."""

    def __init__(self, config: PoseConfig | None = None, model: Any = None) -> None:
        self.config = config or PoseConfig()
        if model is None:
            from ultralytics import YOLO  # heavy; imported only when used

            model = YOLO(self.config.weights)
        self.model = model

    def describe(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "weights": str(self.config.weights),
            "crop_height": self.config.crop_height,
        }

    def estimate(self, image: np.ndarray, boxes: Sequence[BBox]) -> list[list[Keypoint]]:
        """Keypoints for each box, in ``image`` pixels.  ``[]`` where none found."""
        cfg = self.config
        h_img, w_img = image.shape[:2]
        crops: list[np.ndarray] = []
        metas: list[tuple[int, float, float, float, BBox]] = []
        out: list[list[Keypoint]] = [[] for _ in boxes]

        import cv2

        for i, box in enumerate(boxes):
            if box.height < cfg.min_box_height:
                continue
            px, py = box.width * cfg.pad, box.height * cfg.pad
            x0 = int(max(0, np.floor(box.x1 - px)))
            y0 = int(max(0, np.floor(box.y1 - py)))
            x1 = int(min(w_img, np.ceil(box.x2 + px)))
            y1 = int(min(h_img, np.ceil(box.y2 + py)))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            scale = cfg.crop_height / float(y1 - y0)
            crop = cv2.resize(
                image[y0:y1, x0:x1],
                (max(8, int(round((x1 - x0) * scale))), cfg.crop_height),
                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
            )
            # The target box, in crop pixels: what the kept skeleton must match.
            target = BBox(
                (box.x1 - x0) * scale, (box.y1 - y0) * scale,
                (box.x2 - x0) * scale, (box.y2 - y0) * scale,
            )
            crops.append(crop)
            metas.append((i, float(x0), float(y0), scale, target))

        for start in range(0, len(crops), cfg.batch):
            chunk = crops[start:start + cfg.batch]
            results = self.model.predict(
                chunk,
                imgsz=_imgsz_for(chunk),
                conf=cfg.conf,
                device=cfg.device,
                verbose=False,
            )
            for result, (i, x0, y0, scale, target) in zip(results, metas[start:start + cfg.batch]):
                picked = _pick_instance(result, target)
                if picked is None:
                    continue
                xy, conf = picked
                out[i] = [
                    Keypoint(
                        name=name,
                        x=float(xy[k, 0] / scale + x0),
                        y=float(xy[k, 1] / scale + y0),
                        confidence=float(conf[k]),
                    )
                    for k, name in enumerate(COCO_KEYPOINTS)
                ]
        return out


def _imgsz_for(crops: Sequence[np.ndarray]) -> int:
    longest = max(max(c.shape[:2]) for c in crops)
    return int(min(640, max(160, 32 * int(np.ceil(longest / 32)))))


def _pick_instance(result: Any, target: BBox) -> tuple[np.ndarray, np.ndarray] | None:
    """The skeleton in ``result`` that belongs to ``target``, if any."""
    kps = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if kps is None or boxes is None or len(boxes) == 0:
        return None
    xy = kps.xy.cpu().numpy()
    conf = kps.conf.cpu().numpy() if kps.conf is not None else np.ones(xy.shape[:2])
    xyxy = boxes.xyxy.cpu().numpy()
    best, best_score = None, 0.0
    for j in range(len(xyxy)):
        cand = BBox(*[float(v) for v in xyxy[j]])
        # IoU with the tracked box, plus a nudge toward the crop centre, so a
        # half-visible neighbour at the crop edge never wins a close call.
        score = cand.iou(target)
        score -= 0.1 * cand.center.distance_to(target.center) / max(1.0, target.height)
        if score > best_score:
            best, best_score = j, score
    if best is None or best_score < 0.2:
        return None
    return xy[best], conf[best]


class PoseTracker:
    """Wrap a tracker so every player track it returns carries keypoints.

    Satisfies the :class:`~football_analysis.interfaces.Tracker` protocol.  The
    keypoints go on ``track.attributes["keypoints"]``, which is where
    :func:`~football_analysis.state.state_from_tracks` looks for them, so the
    pipeline needs no change to carry pose into the record.
    """

    def __init__(self, tracker: Any, estimator: PoseEstimator, *, every_n: int = 1) -> None:
        self.tracker = tracker
        self.estimator = estimator
        self.every_n = max(1, int(every_n))
        self._count = 0

    def reset(self) -> None:
        self._count = 0
        reset = getattr(self.tracker, "reset", None)
        if callable(reset):
            reset()

    def describe(self) -> dict[str, Any]:
        inner = getattr(self.tracker, "describe", None)
        return {
            "name": type(self).__name__,
            "tracker": inner() if callable(inner) else type(self.tracker).__name__,
            "pose": self.estimator.describe(),
        }

    def update(self, frame: VideoFrame, detections: list[Any]) -> list[Track]:
        tracks = self.tracker.update(frame, detections)
        self._count += 1
        if (self._count - 1) % self.every_n:
            return tracks
        players = [t for t in tracks if t.class_name == PLAYER]
        if not players:
            return tracks
        skeletons = self.estimator.estimate(frame.image, [t.bbox for t in players])
        by_id = {id(t): kp for t, kp in zip(players, skeletons)}
        out: list[Track] = []
        for track in tracks:
            kp = by_id.get(id(track))
            if kp:
                # A new snapshot, never the inner tracker's object (stage contract).
                attributes = dict(track.attributes)
                attributes["keypoints"] = [k.to_dict() for k in kp]
                track = replace(track, attributes=attributes)
            out.append(track)
        return out

    def __getattr__(self, name: str) -> Any:
        # Pass through anything else (``finalize_states``, ``summary`` ...).
        return getattr(self.tracker, name)


def attach_pose(
    states: Iterable[FrameState],
    video_path: str | Path,
    estimator: PoseEstimator,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    frame_indices: set[int] | None = None,
) -> list[FrameState]:
    """Fill ``keypoints`` into finished records by re-reading the clip.

    ``target_width`` / ``target_height`` must match the resize the records were
    produced under, so the boxes and the pixels line up.  ``frame_indices``
    limits the work to the frames that matter (candidate windows, say).
    """
    from football_analysis.io import VideoReader

    states = list(states)
    by_index = {s.frame_index: s for s in states}
    wanted = set(by_index) if frame_indices is None else set(frame_indices) & set(by_index)
    if not wanted:
        return states
    last = max(wanted)
    with VideoReader(video_path, target_width=target_width, target_height=target_height) as reader:
        for frame in reader:
            if frame.index > last:
                break
            if frame.index not in wanted:
                continue
            state = by_index[frame.index]
            if not state.players:
                continue
            skeletons = estimator.estimate(frame.image, [p.bbox for p in state.players])
            for player, kp in zip(state.players, skeletons):
                player.keypoints = kp
    return states
