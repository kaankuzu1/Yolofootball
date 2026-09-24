"""Scale from the players themselves, when there is no calibration.

Plan §3's first fallback.  For a camera with little roll, the pixel height of a
person standing on flat ground is a linear function of the image row their
feet are on: ``h_px = a + b * y_foot``.  Fitting that line across every clear
player box in a clip, and dividing by a real height, gives pixels per metre
anywhere on the ground plane -- no goal, no markings, no clicks.

It is a scale, not a map: it does not say where on the ground a point is, and
a distance measured with it is only right across the image, not into it
(depth is foreshortened by the viewing angle, which this model cannot know).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["HeightScaleModel"]


@dataclass
class HeightScaleModel:
    a: float
    b: float
    person_height_m: float
    samples: int
    residual_px: float

    @classmethod
    def fit(
        cls,
        foot_y: Sequence[float],
        box_h: Sequence[float],
        person_height_m: float = 1.75,
        iterations: int = 20,
    ) -> "HeightScaleModel | None":
        """Robust (Huber-weighted) line fit.  ``None`` with too few samples."""
        y = np.asarray(foot_y, dtype=float)
        h = np.asarray(box_h, dtype=float)
        ok = np.isfinite(y) & np.isfinite(h) & (h > 0)
        y, h = y[ok], h[ok]
        if len(y) < 10 or np.ptp(y) < 1e-6:
            return None
        A = np.c_[np.ones_like(y), y]
        w = np.ones_like(y)
        coef = np.zeros(2)
        for _ in range(iterations):
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(A * sw[:, None], h * sw, rcond=None)
            r = h - A @ coef
            s = 1.4826 * np.median(np.abs(r)) + 1e-6
            k = 1.345 * s
            w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
        r = h - A @ coef
        return cls(float(coef[0]), float(coef[1]), float(person_height_m), int(len(y)),
                   float(1.4826 * np.median(np.abs(r))))

    def px_per_metre(self, foot_y: np.ndarray | float) -> np.ndarray:
        y = np.asarray(foot_y, dtype=float)
        return np.maximum(self.a + self.b * y, 1e-6) / self.person_height_m

    def to_dict(self) -> dict[str, Any]:
        return {"method": "player_height", "a": round(self.a, 4), "b": round(self.b, 6),
                "person_height_m": self.person_height_m, "samples": self.samples,
                "residual_px": round(self.residual_px, 2)}
