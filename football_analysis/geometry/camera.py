"""A calibrated camera, and the image-to-world questions it answers.

World frame
-----------
Metres, right-handed, anchored on the goal:

* origin: the centre of the goal line, on the ground
* ``Z``: straight up
* ``Y``: perpendicular to the goal line, **into the field of play** -- so
  ``Y`` is simply "distance out from the goal line"
* ``X``: along the goal line.  Right-handedness then fixes its sign: it points
  toward the post that appears on the **left** of the image when the camera
  looks at the goal from the field.

A calibration from a ground rectangle instead (cones, see
:mod:`football_analysis.geometry.calibrate`) uses whatever ground coordinates
the rectangle was given in, with ``Z`` up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

__all__ = ["CameraCalibration"]


@dataclass
class CameraCalibration:
    K: np.ndarray
    """3x3 intrinsics."""
    R: np.ndarray
    """3x3 rotation, world -> camera."""
    t: np.ndarray
    """Translation, world -> camera, metres."""
    image_size: tuple[int, int]
    """``(width, height)`` the calibration applies to."""
    method: str = "unknown"
    reprojection_error_px: float = float("nan")
    quality: dict[str, Any] = field(default_factory=dict)

    # -- basics -------------------------------------------------------------

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    @property
    def camera_position(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return (-self.R.T @ self.t).ravel()

    @property
    def horizontal_fov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(0.5 * self.image_size[0] / self.focal_px)))

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        """World points ``(n, 3)`` to pixels ``(n, 2)``.  NaN behind the camera."""
        P = np.atleast_2d(np.asarray(points_xyz, dtype=float))
        cam = (self.R @ P.T).T + self.t.ravel()
        out = np.full((len(P), 2), np.nan)
        front = cam[:, 2] > 1e-6
        uvw = (self.K @ cam[front].T).T
        out[front] = uvw[:, :2] / uvw[:, 2:3]
        return out

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Unit viewing directions in world coordinates for pixels ``(n, 2)``."""
        px = np.atleast_2d(np.asarray(pixels, dtype=float))
        homo = np.c_[px, np.ones(len(px))]
        d_cam = (np.linalg.inv(self.K) @ homo.T).T
        d = (self.R.T @ d_cam.T).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def image_to_plane_z(self, pixels: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Where each pixel's ray meets the horizontal plane ``Z = z``.

        ``(n, 3)``; NaN where the ray points at or above the horizon.
        """
        d = self.rays(pixels)
        c = self.camera_position
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - c[2]) / d[:, 2]
        out = c[None, :] + s[:, None] * d
        out[~(s > 0)] = np.nan
        return out

    def image_to_ground(self, pixels: np.ndarray) -> np.ndarray:
        """Ground-plane ``(X, Y)`` in metres for pixels that touch the ground."""
        return self.image_to_plane_z(pixels, 0.0)[:, :2]

    def ground_to_image(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return self.project(np.c_[xy, np.zeros(len(xy))])

    # -- scale --------------------------------------------------------------

    def vertical_px_per_metre(self, foot_px: np.ndarray) -> np.ndarray:
        """Pixels per metre of *height* at a point standing on the ground.

        This is what a player's box height divides by to give their height in
        metres, and what gravity's pull looks like in pixels at that spot.
        """
        ground = self.image_to_plane_z(foot_px, 0.0)
        up = ground + np.array([0.0, 0.0, 1.0])
        a, b = self.project(ground), self.project(up)
        return np.linalg.norm(a - b, axis=1)

    def ground_px_per_metre(self, foot_px: np.ndarray) -> np.ndarray:
        """Geometric-mean pixels per metre along the ground at each pixel."""
        g = self.image_to_plane_z(foot_px, 0.0)
        a = self.project(g)
        bx = self.project(g + np.array([1.0, 0.0, 0.0]))
        by = self.project(g + np.array([0.0, 1.0, 0.0]))
        area = np.abs(np.cross(bx - a, by - a))
        return np.sqrt(area)

    # -- the ball -----------------------------------------------------------

    def ball_position(
        self, centre_px: np.ndarray, diameter_px: np.ndarray | None = None,
        ball_diameter_m: float = 0.22,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """World position of the ball, two ways.

        Returns ``(on_ground, from_size)``.  ``on_ground`` assumes the ball is
        resting on or rolling along the ground (its centre one radius up), which
        is exact for a ground ball and wrong for a lofted one.  ``from_size``
        uses the apparent diameter as a rangefinder, which works in the air but
        is noisy: at 20 px across, one pixel of error is 5% of the range.
        """
        centre_px = np.atleast_2d(np.asarray(centre_px, dtype=float))
        on_ground = self.image_to_plane_z(centre_px, 0.5 * ball_diameter_m)
        if diameter_px is None:
            return on_ground, None
        diam = np.atleast_1d(np.asarray(diameter_px, dtype=float))
        d = self.rays(centre_px)
        # Angular diameter -> range.
        with np.errstate(divide="ignore", invalid="ignore"):
            rng = ball_diameter_m * self.focal_px / diam
        from_size = self.camera_position[None, :] + rng[:, None] * d
        return on_ground, from_size

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        cam = self.camera_position
        return {
            "method": self.method,
            "image_size": list(self.image_size),
            "K": np.round(self.K, 4).tolist(),
            "R": np.round(self.R, 6).tolist(),
            "t": np.round(self.t.ravel(), 5).tolist(),
            "focal_px": round(self.focal_px, 2),
            "horizontal_fov_deg": round(self.horizontal_fov_deg, 2),
            "camera_position_m": np.round(cam, 3).tolist(),
            "camera_height_m": round(float(cam[2]), 3),
            "reprojection_error_px": round(float(self.reprojection_error_px), 3),
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraCalibration":
        return cls(
            K=np.array(d["K"], dtype=float),
            R=np.array(d["R"], dtype=float),
            t=np.array(d["t"], dtype=float).reshape(3, 1),
            image_size=tuple(d["image_size"]),
            method=d.get("method", "unknown"),
            reprojection_error_px=float(d.get("reprojection_error_px", float("nan"))),
            quality=dict(d.get("quality", {})),
        )

    def rvec(self) -> np.ndarray:
        return cv2.Rodrigues(self.R)[0]
