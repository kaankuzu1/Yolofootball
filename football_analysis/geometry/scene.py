"""One object that answers every geometric question the event logic asks.

:class:`SceneGeometry` hides which calibration is behind it.  In order of
preference (plan §3):

1. ``goal_pnp``      the goal's four mouth corners and its known size
2. ``ground_points`` four or more measured ground points (cones)
3. ``player_height`` a scale fitted to the players' own heights
4. ``none``          pixels only

With 1 or 2 there is a full camera, so points on the ground have real
coordinates.  With 3 there is only a scale.  Each method answers what it can
and returns NaN for what it cannot, so a caller never needs to branch on
which one is in use -- but ``method`` and ``quality`` are there when it should.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from football_analysis.geometry.calibrate import (
    CalibrationError,
    GoalModel,
    calibrate_from_goal,
    calibrate_from_ground_points,
)
from football_analysis.geometry.camera import CameraCalibration
from football_analysis.geometry.config import GeometryConfig
from football_analysis.geometry.height_scale import HeightScaleModel

__all__ = ["SceneGeometry", "build_scene_geometry", "aggregate_goal_corners"]


@dataclass
class SceneGeometry:
    goal: GoalModel
    calibration: CameraCalibration | None = None
    height_scale: HeightScaleModel | None = None
    goal_quad_px: np.ndarray | None = None
    """Goal mouth corners in pixels, keypoint order, or ``None``."""
    goal_quad_source: str = "none"
    """``configured``, ``keypoints``, ``bbox`` or ``none``."""
    ball_diameter_m: float = 0.22
    notes: list[str] = field(default_factory=list)

    @property
    def method(self) -> str:
        if self.calibration is not None:
            return self.calibration.method
        if self.height_scale is not None:
            return "player_height"
        return "none"

    @property
    def has_world(self) -> bool:
        return self.calibration is not None

    # -- points ------------------------------------------------------------

    def ground_xy(self, pixels: np.ndarray) -> np.ndarray:
        """Ground ``(X, Y)`` in metres for foot points; NaN without a camera."""
        px = np.atleast_2d(np.asarray(pixels, dtype=float))
        if self.calibration is None:
            return np.full((len(px), 2), np.nan)
        return self.calibration.image_to_ground(px)

    def px_per_metre(self, foot_px: np.ndarray) -> np.ndarray:
        """Pixels per metre of height at a ground point (the local scale)."""
        px = np.atleast_2d(np.asarray(foot_px, dtype=float))
        if self.calibration is not None:
            return self.calibration.vertical_px_per_metre(px)
        if self.height_scale is not None:
            return self.height_scale.px_per_metre(px[:, 1])
        return np.full(len(px), np.nan)

    def speed_mps(self, foot_px: np.ndarray, velocity_px_s: np.ndarray) -> np.ndarray:
        """Ground speed of things moving along the ground, metres per second.

        With a camera this pushes the pixel velocity through the local
        image-to-ground mapping, so motion into the picture counts in full.
        With only the height scale it is pixel speed over the local scale,
        which is right across the image and an underestimate into it.
        """
        px = np.atleast_2d(np.asarray(foot_px, dtype=float))
        v = np.atleast_2d(np.asarray(velocity_px_s, dtype=float))
        if self.calibration is not None:
            dt = 0.02
            a = self.calibration.image_to_ground(px)
            b = self.calibration.image_to_ground(px + v * dt)
            return np.linalg.norm(b - a, axis=1) / dt
        return np.linalg.norm(v, axis=1) / self.px_per_metre(px)

    def distance_m(self, a_px: Sequence[float], b_px: Sequence[float]) -> float:
        """Ground distance between two foot-level points.

        Exact with a camera.  With only the height scale it is the pixel
        distance over the local scale, which is right across the image and an
        underestimate into it.
        """
        a = np.asarray(a_px, dtype=float).reshape(1, 2)
        b = np.asarray(b_px, dtype=float).reshape(1, 2)
        if self.calibration is not None:
            ga, gb = self.ground_xy(a)[0], self.ground_xy(b)[0]
            return float(np.linalg.norm(ga - gb))
        if self.height_scale is not None:
            s = float(self.height_scale.px_per_metre(0.5 * (a[0, 1] + b[0, 1])))
            return float(np.linalg.norm(a - b) / s)
        return float("nan")

    def ball_world(self, centre_px: np.ndarray, diameter_px: np.ndarray | None = None):
        """``(on_ground_xyz, from_size_xyz)``; see :meth:`CameraCalibration.ball_position`."""
        c = np.atleast_2d(np.asarray(centre_px, dtype=float))
        if self.calibration is None:
            nan = np.full((len(c), 3), np.nan)
            return nan, (nan.copy() if diameter_px is not None else None)
        return self.calibration.ball_position(c, diameter_px, self.ball_diameter_m)

    # -- the goal ------------------------------------------------------------

    def goal_polygon_world(self) -> np.ndarray:
        return self.goal.mouth_corners()

    def goal_net_world(self) -> np.ndarray:
        return self.goal.net_box()

    def goal_net_polygon_px(self) -> np.ndarray | None:
        """Outline of the whole goal (mouth plus net) in pixels, when calibrated."""
        if self.calibration is None:
            return self.goal_quad_px
        import cv2

        pts = self.calibration.project(self.goal.net_box()).astype(np.float32)
        if not np.all(np.isfinite(pts)):
            return self.goal_quad_px
        return cv2.convexHull(pts).reshape(-1, 2)

    def in_goal_mouth_px(self, point_px: Sequence[float]) -> bool:
        if self.goal_quad_px is None:
            return False
        import cv2

        quad = self.goal_quad_px.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(quad, (float(point_px[0]), float(point_px[1])), False) >= 0

    # -- serialisation ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "goal": {"width_m": self.goal.width_m, "height_m": self.goal.height_m,
                     "depth_m": self.goal.depth_m},
            "goal_quad_px": None if self.goal_quad_px is None
            else np.round(self.goal_quad_px, 2).tolist(),
            "goal_quad_source": self.goal_quad_source,
            "ball_diameter_m": self.ball_diameter_m,
            "camera": None if self.calibration is None else self.calibration.to_dict(),
            "height_scale": None if self.height_scale is None else self.height_scale.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneGeometry":
        g = d.get("goal", {})
        hs = d.get("height_scale")
        return cls(
            goal=GoalModel(g.get("width_m", 3.0), g.get("height_m", 2.0), g.get("depth_m", 1.0)),
            calibration=CameraCalibration.from_dict(d["camera"]) if d.get("camera") else None,
            height_scale=HeightScaleModel(hs["a"], hs["b"], hs["person_height_m"], hs["samples"],
                                          hs["residual_px"]) if hs else None,
            goal_quad_px=None if d.get("goal_quad_px") is None else np.array(d["goal_quad_px"]),
            goal_quad_source=d.get("goal_quad_source", "none"),
            ball_diameter_m=d.get("ball_diameter_m", 0.22),
            notes=list(d.get("notes", [])),
        )


def aggregate_goal_corners(
    keypoint_sets: Sequence[np.ndarray], min_conf: float = 0.3
) -> np.ndarray | None:
    """Per-corner median of the goal keypoints seen across a clip.

    The camera does not move and neither does the goal, so every frame is
    another measurement of the same four points; the median shrugs off frames
    where a player stands in front of a post.
    """
    rows: list[list[np.ndarray]] = [[], [], [], []]
    for kp in keypoint_sets:
        kp = np.asarray(kp, dtype=float)
        for c in range(4):
            conf = kp[c, 2] if kp.shape[1] > 2 else 1.0
            if conf >= min_conf and np.all(np.isfinite(kp[c, :2])):
                rows[c].append(kp[c, :2])
    if any(len(r) == 0 for r in rows):
        return None
    return np.array([np.median(np.array(r), axis=0) for r in rows])


def build_scene_geometry(
    config: GeometryConfig,
    image_size: tuple[int, int],
    *,
    goal_keypoints: Sequence[np.ndarray] = (),
    goal_boxes: Sequence[tuple[float, float, float, float]] = (),
    player_samples: tuple[Sequence[float], Sequence[float]] | None = None,
) -> SceneGeometry:
    """Pick the best calibration the evidence allows (see module docstring)."""
    goal = GoalModel(config.goal_width_m, config.goal_height_m, config.goal_depth_m)
    scene = SceneGeometry(goal=goal, ball_diameter_m=config.ball_diameter_m)
    kwargs = dict(focal_px=config.focal_px, estimate_focal_length=config.estimate_focal,
                  assumed_hfov_deg=config.assumed_hfov_deg)

    if config.goal_corners_px is not None:
        scene.goal_quad_px, scene.goal_quad_source = np.asarray(config.goal_corners_px, float), "configured"
    elif goal_keypoints:
        quad = aggregate_goal_corners(goal_keypoints, config.min_goal_keypoint_confidence)
        if quad is not None:
            scene.goal_quad_px, scene.goal_quad_source = quad, "keypoints"
    if scene.goal_quad_px is None and goal_boxes:
        x1, y1, x2, y2 = np.median(np.asarray(goal_boxes, dtype=float), axis=0)
        scene.goal_quad_px = np.array([[x1, y2], [x2, y2], [x2, y1], [x1, y1]])
        scene.goal_quad_source = "bbox"
        scene.notes.append(
            "The goal was found only as a box, not as four corners, so it marks where the "
            "goal is but cannot calibrate the camera; a box's corners are not the goal's."
        )

    if scene.goal_quad_source in ("configured", "keypoints"):
        try:
            scene.calibration = calibrate_from_goal(scene.goal_quad_px, image_size, goal, **kwargs)
        except CalibrationError as exc:
            scene.notes.append(f"Goal calibration failed: {exc}")
    if scene.calibration is None and config.ground_points_px and config.ground_points_m:
        try:
            scene.calibration = calibrate_from_ground_points(
                config.ground_points_m, config.ground_points_px, image_size, **kwargs)
        except CalibrationError as exc:
            scene.notes.append(f"Ground-point calibration failed: {exc}")
    if scene.calibration is not None and scene.calibration.quality.get("verdict") == "weak":
        scene.notes.append(
            "The calibration is weak: the goal is seen nearly face-on or edge-on, or too "
            "small, so metric distances carry large errors (see camera.quality)."
        )

    if player_samples is not None:
        scene.height_scale = HeightScaleModel.fit(*player_samples, person_height_m=config.player_height_m)
    if scene.calibration is None and scene.height_scale is None:
        scene.notes.append("No calibration: distances are in pixels only.")
    return scene
