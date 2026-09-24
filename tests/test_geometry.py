"""Camera geometry from the goal: pose, focal length, ground plane, fallbacks."""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.geometry import (
    CalibrationError,
    CameraCalibration,
    GeometryConfig,
    GoalModel,
    HeightScaleModel,
    build_scene_geometry,
    calibrate_from_goal,
    calibrate_from_ground_points,
)

SIZE = (1920, 1080)


def _camera(pos, target, hfov=65.0) -> CameraCalibration:
    w, h = SIZE
    f = 0.5 * w / np.tan(np.radians(hfov / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return CameraCalibration(K, R, (-R @ pos).reshape(3, 1), SIZE)


SIDE_ANGLE = (np.array([-9.0, 5.0, 2.5]), np.array([0.0, 4.0, 0.5]))


@pytest.mark.parametrize("hfov", [50.0, 65.0, 80.0])
def test_goal_corners_recover_pose_and_focal(hfov):
    true = _camera(*SIDE_ANGLE, hfov=hfov)
    goal = GoalModel(3.0, 2.0)
    calib = calibrate_from_goal(true.project(goal.mouth_corners()), SIZE, goal)
    assert calib.quality["focal_source"] == "estimated"
    assert calib.focal_px == pytest.approx(true.focal_px, rel=0.01)
    assert np.linalg.norm(calib.camera_position - SIDE_ANGLE[0]) < 0.05
    probe = np.array([[0.0, 6.0], [2.0, 10.0], [-3.0, 3.0]])
    back = calib.image_to_ground(true.ground_to_image(probe))
    assert np.abs(back - probe).max() < 0.05


def test_world_frame_convention():
    true = _camera(*SIDE_ANGLE)
    calib = calibrate_from_goal(true.project(GoalModel().mouth_corners()), SIZE)
    cam = calib.camera_position
    assert cam[1] > 0 and cam[2] > 0  # in front of the goal line, above the ground
    # The vertical scale at a spot is the image length of a 1 m pole standing there.
    foot = calib.ground_to_image([[1.0, 5.0]])
    assert calib.vertical_px_per_metre(foot)[0] == pytest.approx(
        np.linalg.norm(calib.project([[1.0, 5.0, 0.0]]) - calib.project([[1.0, 5.0, 1.0]])), rel=1e-6)


def test_uncertainty_is_reported_and_face_on_is_weak():
    goal = GoalModel()
    good = calibrate_from_goal(_camera(*SIDE_ANGLE).project(goal.mouth_corners()), SIZE, goal)
    assert good.quality["verdict"] in ("good", "usable")
    assert len(good.quality["probe_error_m_p90"]) == 5
    face_on = _camera(np.array([0.0, 15.0, 2.0]), np.array([0.0, 0.0, 1.0]))
    weak = calibrate_from_goal(face_on.project(goal.mouth_corners()), SIZE, goal)
    assert weak.quality["verdict"] == "weak"


def test_mirrored_corners_are_refused():
    true = _camera(*SIDE_ANGLE)
    corners = true.project(GoalModel().mouth_corners())[[1, 0, 3, 2]]
    with pytest.raises(CalibrationError):
        calibrate_from_goal(corners, SIZE)


def test_ground_rectangle_calibration():
    true = _camera(*SIDE_ANGLE)
    cones = np.array([[-4.0, 2.0], [4.0, 2.0], [4.0, 10.0], [-4.0, 10.0]])
    calib = calibrate_from_ground_points(cones, true.ground_to_image(cones), SIZE)
    probe = np.array([[0.0, 6.0]])
    assert np.abs(calib.image_to_ground(true.ground_to_image(probe)) - probe).max() < 0.05


def test_height_scale_fallback():
    # Box height linear in foot row: 60 px at y=300, 180 px at y=900.
    y = np.linspace(300, 900, 50)
    h = 60 + (y - 300) * 0.2 + np.random.default_rng(0).normal(0, 2, 50)
    model = HeightScaleModel.fit(y, h, person_height_m=1.8)
    assert model.px_per_metre(600)[()] == pytest.approx(120 / 1.8, rel=0.03)


def test_scene_prefers_goal_then_scale():
    true = _camera(*SIDE_ANGLE)
    kp = np.c_[true.project(GoalModel().mouth_corners()), np.ones(4)]
    scene = build_scene_geometry(GeometryConfig(), SIZE, goal_keypoints=[kp, kp + [0.5, -0.5, 0]])
    assert scene.method == "goal_pnp" and scene.goal_quad_source == "keypoints"
    assert scene.has_world
    boxes_only = build_scene_geometry(GeometryConfig(), SIZE, goal_boxes=[(100, 100, 300, 200)])
    assert boxes_only.method == "none" and boxes_only.goal_quad_source == "bbox"
    assert boxes_only.notes
