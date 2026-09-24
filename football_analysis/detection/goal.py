"""Goal detection.

No pretrained checkpoint we tested carries a goal class: COCO has none, the
football-specific weights are trained on ball/goalkeeper/player/referee, and
open-vocabulary prompting ("soccer goal net", "goalpost") does not find it
reliably.  That leaves two options, and which one applies is a property of the
camera, not of the model:

``StaticGoal``
    For a camera that does not move -- a tripod or a clamped phone, which is
    the 1v1 setup this system targets -- the goal is in the same pixels for the
    whole recording, so it is calibrated once and then costs nothing per frame.

``Detector`` with goal-trained weights
    For a camera that pans, the goal moves in the image and has to be detected.
    That needs fine-tuned weights; see ``docs/detection-report.md`` for what it
    took to train one and what it scored.

Either way the goal is stored as **four corners of the goal mouth**, not as a
bounding box.  A box is enough to answer "is the ball in the goal", but four
image points with known 3D positions are what camera calibration needs, which
is how the system calibrates without pitch markings.  The box is derived from
the corners, so nothing is lost by carrying both.

This module deliberately does **not** solve for camera pose.  That belongs to
:func:`football_analysis.geometry.calibrate_from_goal`, which also estimates
focal length when it is unknown (it usually is, on a phone), rejects a mirrored
corner order, picks the physically valid solution and reports its own
uncertainty.  Two PnP implementations would drift apart, so there is one, and
it lives in ``geometry``.  Pass ``StaticGoal.corners`` straight to it::

    from football_analysis.geometry import calibrate_from_goal, GoalModel

    goal = StaticGoal.load("camera_a_goal.json")
    calibration = calibrate_from_goal(goal.corners, (width, height), GoalModel(3.0, 2.0))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from .types import GOAL, Detection

Point = Tuple[float, float]

# Corner order, fixed so image points and 3D object points line up. This is the
# order the geometry/tracking layer consumes, so it is the one to match.
CORNER_ORDER = ("left_post_base", "right_post_base", "right_crossbar_end", "left_crossbar_end")

@dataclass(frozen=True)
class StaticGoal:
    """A goal fixed in image space, for a camera that does not move.

    ``corners`` are the four corners of the goal mouth in image pixels, in
    :data:`CORNER_ORDER`, matching ``GoalModel.mouth_corners()`` in
    :mod:`football_analysis.geometry.calibrate`: base of the left post, base of
    the right post, right end of the crossbar, left end of the crossbar. "Left"
    and "right" are as seen in the image, which makes the order unambiguous to
    annotate whichever side the camera is on.
    """

    corners: Tuple[Point, Point, Point, Point]

    # -- shape shared with Detector ------------------------------------

    def detect(self, frame=None) -> list[Detection]:
        """Return the goal, ignoring the frame. Same shape as ``Detector.detect``."""
        return [Detection(label=GOAL, confidence=1.0, xyxy=self.xyxy)]

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        """Axis-aligned box around the four corners."""
        xs = [p[0] for p in self.corners]
        ys = [p[1] for p in self.corners]
        return (min(xs), min(ys), max(xs), max(ys))

    def contains(self, point: Point) -> bool:
        """Whether a point falls inside the goal-mouth quadrilateral."""
        x, y = point
        inside = False
        n = len(self.corners)
        for i in range(n):
            x1, y1 = self.corners[i]
            x2, y2 = self.corners[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < cross:
                    inside = not inside
        return inside

    # -- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"corner_order": list(CORNER_ORDER), "corners": [list(p) for p in self.corners]},
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "StaticGoal":
        data = json.loads(Path(path).read_text())
        return cls(corners=tuple(tuple(float(v) for v in p) for p in data["corners"]))

    @classmethod
    def from_box(cls, xyxy: Tuple[float, float, float, float]) -> "StaticGoal":
        """Build from an axis-aligned box, for a goal viewed close to square on.

        This is a fallback: the corners it invents are the box's own, so the
        pose it yields is only as good as that approximation.
        """
        x1, y1, x2, y2 = xyxy
        return cls(corners=((x1, y2), (x2, y2), (x2, y1), (x1, y1)))

    @classmethod
    def from_click(cls, frame) -> "StaticGoal":
        """Click the four goal-mouth corners on one frame, in CORNER_ORDER."""
        import cv2

        picked: list[Point] = []

        def on_mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
                picked.append((float(x), float(y)))

        window = "click: left post base, right post base, right crossbar end, left crossbar end"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)
        while len(picked) < 4:
            shown = frame.copy()
            for i, (px, py) in enumerate(picked):
                cv2.circle(shown, (int(px), int(py)), 6, (255, 120, 255), -1)
                cv2.putText(shown, CORNER_ORDER[i], (int(px) + 8, int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 255), 2)
            cv2.imshow(window, shown)
            if cv2.waitKey(20) == 27:
                break
        cv2.destroyWindow(window)
        if len(picked) != 4:
            raise ValueError("four goal corners are required")
        return cls(corners=tuple(picked))
