"""Camera pose from a known planar shape: the goal mouth, or a ground rectangle.

Plan §3: there are no pitch markings to lean on in a 1v1 drill, but there is a
goal, it is always in frame, and its size is known.  Its four mouth corners are
four known coplanar 3-D points, and that is all a planar PnP solve needs.

The focal length is solved too, when the view allows it.  A homography from a
known plane has eight degrees of freedom; pose takes six and a focal length
(square pixels, principal point at the centre) takes one, so one is spare.  It
fails when the plane is seen square-on, and it is noisy from only four points,
so every calibration carries a Monte-Carlo estimate of its own uncertainty: the
corners are jittered by a pixel, the solve repeated, and the spread of the
result reported in metres.  A calibration that cannot be trusted says so.

Ground-plane cones (the plan's "manual rectangle" fallback) go through exactly
the same solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from football_analysis.geometry.camera import CameraCalibration

__all__ = [
    "CalibrationError",
    "GoalModel",
    "estimate_focal",
    "calibrate_planar",
    "calibrate_from_goal",
    "calibrate_from_ground_points",
]


class CalibrationError(ValueError):
    """The points given cannot produce a physically sensible camera."""


@dataclass(frozen=True)
class GoalModel:
    width_m: float = 3.0
    height_m: float = 2.0
    depth_m: float = 1.0

    def mouth_corners(self) -> np.ndarray:
        """Goal mouth corners in world metres, in keypoint order.

        Left post base, right post base, right crossbar end, left crossbar end,
        where left and right are as seen in the image from the field.  See
        :mod:`football_analysis.geometry.camera` for why "image-left" is +X.
        """
        w, h = 0.5 * self.width_m, self.height_m
        return np.array([[w, 0, 0], [-w, 0, 0], [-w, 0, h], [w, 0, h]], dtype=float)

    def net_box(self) -> np.ndarray:
        """The eight corners of the volume behind the goal mouth (Y < 0)."""
        m = self.mouth_corners()
        back = m + np.array([0.0, -self.depth_m, 0.0])
        return np.vstack([m, back])


def estimate_focal(H: np.ndarray, image_size: tuple[int, int]) -> float | None:
    """Focal length (pixels) implied by a plane-to-image homography.

    Assumes square pixels, no skew and the principal point at the image centre.
    Returns ``None`` when the constraints are degenerate (the plane seen
    face-on) or disagree with a physically plausible lens.
    """
    w, h = image_size
    s = float(max(w, h))
    T = np.array([[1 / s, 0, -0.5 * w / s], [0, 1 / s, -0.5 * h / s], [0, 0, 1]])
    Hn = T @ H
    Hn = Hn / np.linalg.norm(Hn)
    h1, h2 = Hn[:, 0], Hn[:, 1]
    a1 = h1[0] * h2[0] + h1[1] * h2[1]
    b1 = h1[2] * h2[2]
    a2 = (h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)
    b2 = h1[2] ** 2 - h2[2] ** 2
    denom = a1 * a1 + a2 * a2
    if denom < 1e-12:
        return None
    inv_f2 = -(a1 * b1 + a2 * b2) / denom
    if not inv_f2 > 0:
        return None
    # How much the two constraints were actually able to say.
    if max(abs(b1), abs(b2)) < 1e-4:
        return None
    f = s / np.sqrt(inv_f2)
    fov = np.degrees(2 * np.arctan(0.5 * w / f))
    if not 15.0 <= fov <= 130.0:
        return None
    return float(f)


def _K(f: float, image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    return np.array([[f, 0, 0.5 * w], [0, f, 0.5 * h], [0, 0, 1.0]])


def _solve(
    world: np.ndarray,
    image: np.ndarray,
    image_size: tuple[int, int],
    plane: str,
    focal_px: float | None,
    estimate: bool,
    assumed_hfov_deg: float,
    require_front: bool,
) -> tuple[CameraCalibration, str]:
    axes = (0, 2) if plane == "goal" else (0, 1)
    plane2d = world[:, axes]
    H, _ = cv2.findHomography(plane2d, image, 0)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise CalibrationError("the points are degenerate: they do not span a quadrilateral")

    if focal_px:
        f, source = float(focal_px), "configured"
    else:
        f = estimate_focal(H, image_size) if estimate else None
        source = "estimated"
        if f is None:
            f = 0.5 * image_size[0] / np.tan(np.radians(0.5 * assumed_hfov_deg))
            source = "assumed"

    K = _K(f, image_size)
    # IPPE is solved in the plane's own frame (points at z = 0) and rotated
    # back: M maps plane coordinates (u, v, n) to world axes, and is a proper
    # rotation, so the goal plane's frame is (X, Z, -Y).
    M = np.eye(3) if plane == "ground" else np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    planar = np.c_[plane2d, np.zeros(len(plane2d))]
    n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        planar.astype(np.float64), image.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE
    )
    best = None
    for rvec_p, tvec in zip(rvecs, tvecs):
        R = cv2.Rodrigues(rvec_p)[0] @ M.T
        rvec = cv2.Rodrigues(R)[0]
        cam = (-R.T @ tvec).ravel()
        if cam[2] <= 0:
            continue  # below the ground
        if require_front and cam[1] <= 0:
            continue  # behind the goal line
        proj, _ = cv2.projectPoints(world, rvec, tvec, K, None)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image) ** 2, axis=1))))
        if best is None or err < best[0]:
            best = (err, R, tvec)
    if best is None:
        raise CalibrationError(
            "no camera position above the ground and in front of the points fits them; "
            "the corners are probably listed in the wrong order"
        )
    err, R, tvec = best
    calib = CameraCalibration(K=K, R=R, t=tvec.reshape(3, 1), image_size=tuple(image_size),
                              reprojection_error_px=err)
    return calib, source


def calibrate_planar(
    world: Sequence[Sequence[float]],
    image: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    *,
    plane: str,
    focal_px: float | None = None,
    estimate_focal_length: bool = True,
    assumed_hfov_deg: float = 65.0,
    require_front: bool = False,
    method: str = "planar",
    probe_points: Sequence[Sequence[float]] | None = None,
    jitter_px: float = 1.0,
    trials: int = 60,
    seed: int = 0,
) -> CameraCalibration:
    """Calibrate from four or more coplanar points of known position.

    ``plane`` is ``"goal"`` (points on ``Y = 0``) or ``"ground"`` (``Z = 0``).
    ``probe_points`` are ground ``(X, Y)`` points at which the uncertainty of
    the image-to-ground mapping is reported.
    """
    W = np.asarray(world, dtype=float)
    I = np.asarray(image, dtype=float)
    if W.shape[0] < 4 or W.shape[0] != I.shape[0]:
        raise CalibrationError("need at least four matching world and image points")
    calib, source = _solve(W, I, image_size, plane, focal_px, estimate_focal_length,
                           assumed_hfov_deg, require_front)
    calib.method = method

    # How far does the plane sit from being seen edge-on?  A goal mouth seen
    # from almost exactly the side is a sliver, and a sliver pins down little.
    normal = np.array([0.0, 1.0, 0.0]) if plane == "goal" else np.array([0.0, 0.0, 1.0])
    centre = W.mean(axis=0)
    view = centre - calib.camera_position
    view /= np.linalg.norm(view)
    incidence = float(np.degrees(np.arccos(min(1.0, abs(view @ normal)))))

    probes = np.asarray(probe_points if probe_points is not None else [], dtype=float).reshape(-1, 2)
    probe_px = calib.ground_to_image(probes) if len(probes) else np.zeros((0, 2))

    rng = np.random.default_rng(seed)
    focals, cams, probe_err = [], [], []
    for _ in range(trials):
        noisy = I + rng.normal(0.0, jitter_px, I.shape)
        try:
            c, _ = _solve(W, noisy, image_size, plane, focal_px, estimate_focal_length,
                          assumed_hfov_deg, require_front)
        except CalibrationError:
            continue
        focals.append(c.focal_px)
        cams.append(c.camera_position)
        if len(probes):
            back = c.image_to_ground(probe_px)
            probe_err.append(np.linalg.norm(back - probes, axis=1))
    quality: dict[str, Any] = {
        "focal_source": source,
        "view_incidence_deg": round(incidence, 1),
        "jitter_px": jitter_px,
        "stable_trials": f"{len(focals)}/{trials}",
    }
    if focals:
        quality["focal_std_px"] = round(float(np.std(focals)), 1)
        quality["camera_position_std_m"] = round(float(np.linalg.norm(np.std(cams, axis=0))), 3)
    if probe_err:
        pe = np.array(probe_err)
        quality["probe_points_m"] = probes.round(2).tolist()
        quality["probe_error_m_p90"] = np.round(np.nanpercentile(pe, 90, axis=0), 3).tolist()
    worst = max(quality.get("probe_error_m_p90", [0.0]) or [0.0])
    if len(focals) < 0.8 * trials or incidence > 80 or worst > 1.0:
        verdict = "weak"
    elif worst > 0.3:
        verdict = "usable"
    else:
        verdict = "good"
    quality["verdict"] = verdict
    calib.quality = quality
    return calib


def calibrate_from_goal(
    corners_px: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    goal: GoalModel = GoalModel(),
    *,
    focal_px: float | None = None,
    estimate_focal_length: bool = True,
    assumed_hfov_deg: float = 65.0,
) -> CameraCalibration:
    """Calibrate from the four goal-mouth corners (keypoint order, see :class:`GoalModel`)."""
    probes = [[0.0, 3.0], [0.0, 6.0], [0.0, 10.0], [3.0, 8.0], [-3.0, 8.0]]
    return calibrate_planar(
        goal.mouth_corners(), np.asarray(corners_px, dtype=float)[:, :2], image_size,
        plane="goal", focal_px=focal_px, estimate_focal_length=estimate_focal_length,
        assumed_hfov_deg=assumed_hfov_deg, require_front=True, method="goal_pnp",
        probe_points=probes,
    )


def calibrate_from_ground_points(
    ground_m: Sequence[Sequence[float]],
    image_px: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    *,
    focal_px: float | None = None,
    estimate_focal_length: bool = True,
    assumed_hfov_deg: float = 65.0,
) -> CameraCalibration:
    """Calibrate from four or more measured ground points (cones, line crossings)."""
    g = np.asarray(ground_m, dtype=float)[:, :2]
    world = np.c_[g, np.zeros(len(g))]
    lo, hi = g.min(axis=0), g.max(axis=0)
    mid = 0.5 * (lo + hi)
    probes = [mid.tolist(), [lo[0], mid[1]], [hi[0], mid[1]], [mid[0], lo[1]], [mid[0], hi[1]]]
    return calibrate_planar(
        world, np.asarray(image_px, dtype=float)[:, :2], image_size, plane="ground",
        focal_px=focal_px, estimate_focal_length=estimate_focal_length,
        assumed_hfov_deg=assumed_hfov_deg, method="ground_points", probe_points=probes,
    )
