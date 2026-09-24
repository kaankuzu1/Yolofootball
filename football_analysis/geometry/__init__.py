"""Camera geometry from the goal itself (plan §3).

No pitch homography -- a 1v1 area has none of the markings one needs.  Instead
the goal's four mouth corners and its known size give the camera's pose (and,
usually, its focal length), which yields a ground plane, a scale anywhere in
the frame, and the goal as a 3-D object.  Measured cones and the players' own
heights are the fallbacks.
"""

from football_analysis.geometry.calibrate import (
    CalibrationError,
    GoalModel,
    calibrate_from_goal,
    calibrate_from_ground_points,
    calibrate_planar,
    estimate_focal,
)
from football_analysis.geometry.camera import CameraCalibration
from football_analysis.geometry.config import GeometryConfig
from football_analysis.geometry.height_scale import HeightScaleModel
from football_analysis.geometry.scene import SceneGeometry, aggregate_goal_corners, build_scene_geometry

__all__ = [
    "CalibrationError",
    "CameraCalibration",
    "GeometryConfig",
    "GoalModel",
    "HeightScaleModel",
    "SceneGeometry",
    "aggregate_goal_corners",
    "build_scene_geometry",
    "calibrate_from_goal",
    "calibrate_from_ground_points",
    "calibrate_planar",
    "estimate_focal",
]
