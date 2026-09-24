"""Score detector configurations against the hand-labelled ball ground truth.

Run from the project root::

    python -m football_analysis.detection.benchmark

Ball ground truth is one point per sampled frame (or ``null`` when the ball is
not visible).  A detection counts as a hit when its centre lands within
``TOLERANCE_PX`` of that point; every other ball detection in the frame is a
false positive, including all ball detections on a frame where the ball is
absent.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

import cv2

from .detector import Detector, DetectorConfig
from .types import BALL

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "assets" / "clips" / "broadcast_08fd33_4.mp4"
GT = ROOT / "assets" / "annotations" / "ball_gt_points.json"
TOLERANCE_PX = 30.0


@dataclass
class Score:
    name: str
    frames: int
    ball_visible: int
    ball_hits: int
    false_positives: int
    ms_median: float
    ms_p90: float
    players_median: float

    @property
    def recall(self) -> float:
        return self.ball_hits / self.ball_visible if self.ball_visible else 0.0

    @property
    def fp_per_frame(self) -> float:
        return self.false_positives / self.frames if self.frames else 0.0

    @property
    def precision(self) -> float:
        total = self.ball_hits + self.false_positives
        return self.ball_hits / total if total else 0.0


def evaluate(name: str, config: DetectorConfig, gt: Dict[str, List[float] | None]) -> Score:
    detector = Detector(config)
    capture = cv2.VideoCapture(str(CLIP))
    hits = fps_ = 0
    visible = 0
    times: List[float] = []
    players: List[int] = []

    for key in sorted(gt, key=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(key))
        ok, frame = capture.read()
        if not ok:
            continue
        dets = detector.detect(frame)
        times.append(detector.last_ms)
        players.append(sum(1 for d in dets if d.label in ("player", "goalkeeper", "referee")))

        balls = [d for d in dets if d.label == BALL]
        point = gt[key]
        if point is None:
            fps_ += len(balls)
            continue
        visible += 1
        matched = None
        for det in balls:  # dets are confidence-sorted, so this takes the best match
            cx, cy = det.center
            if (cx - point[0]) ** 2 + (cy - point[1]) ** 2 <= TOLERANCE_PX**2:
                matched = det
                break
        if matched is not None:
            hits += 1
        fps_ += len(balls) - (1 if matched is not None else 0)

    capture.release()
    return Score(
        name=name,
        frames=len(times),
        ball_visible=visible,
        ball_hits=hits,
        false_positives=fps_,
        ms_median=statistics.median(times) if times else 0.0,
        ms_p90=sorted(times)[int(len(times) * 0.9)] if times else 0.0,
        players_median=statistics.median(players) if players else 0.0,
    )


def configurations(models: Path) -> Dict[str, DetectorConfig]:
    coco = DetectorConfig(weights=str(models / "yolo11x.pt"), imgsz=1280, conf=0.4, ball_conf=0.25)
    ft = replace(coco, weights=str(models / "hayati_football.pt"))
    return {
        "yolo11n COCO @1280": replace(coco, weights=str(models / "yolo11n.pt")),
        "yolo11x COCO @1280": coco,
        "yolo11x COCO + tiled ball": replace(coco, tile_ball=True),
        "football-ft @1280": ft,
        "football-ft + tiled ball": replace(ft, tile_ball=True),
        "football-ft + tiled, ball_conf 0.10": replace(ft, tile_ball=True, ball_conf=0.10),
    }


def main() -> int:
    gt = json.loads(GT.read_text())
    scores = []
    for name, config in configurations(ROOT / "assets" / "models").items():
        started = time.perf_counter()
        score = evaluate(name, config, gt)
        scores.append(score)
        print(
            f"{score.name:38s} recall {score.recall:5.1%}  "
            f"precision {score.precision:5.1%}  "
            f"FP/frame {score.fp_per_frame:4.2f}  "
            f"{score.ms_median:6.0f} ms/frame (p90 {score.ms_p90:5.0f})  "
            f"players {score.players_median:4.0f}  "
            f"[{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
    out = ROOT / "assets" / "annotations" / "benchmark.json"
    out.write_text(json.dumps([s.__dict__ | {"recall": s.recall, "precision": s.precision} for s in scores], indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
