"""Kit colour as an identity signal.

Two players in a 1v1 almost always wear different colours, and colour is the
one cue that survives them tangling up: when their boxes separate again, the
kit says who is who.  Plan §2.1 notes that for two differently coloured kits an
HSV histogram is enough and far cheaper than a learned embedding.

The feature is a joint hue-saturation histogram plus a brightness histogram of
the shirt-and-shorts band of the box, with pitch-coloured pixels removed so
grass showing between the legs does not dominate.  Hue alone fails on white,
black and grey kits, which have no reliable hue; the brightness histogram is
what separates those.  Histograms are square-rooted (Hellinger), so Euclidean
distance between features behaves like a proper distance between distributions
and ordinary k-means works on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["GrassModel", "pitch_hue", "kit_feature", "FEATURE_DIM"]

_H_BINS, _S_BINS, _V_BINS = 16, 4, 8
FEATURE_DIM = _H_BINS * _S_BINS + _V_BINS


@dataclass(frozen=True)
class GrassModel:
    """What the pitch looks like in HSV: a hue band plus saturation and
    brightness ranges.  Masking on hue alone erases a lime or green kit along
    with the grass; the kit is far brighter and more saturated than the turf,
    so all three ranges together separate them."""

    hue: float
    hue_tol: float
    s_range: tuple[float, float]
    v_range: tuple[float, float]

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        dh = np.abs(hsv[:, 0] - self.hue)
        dh = np.minimum(dh, 180 - dh)
        return ((dh <= self.hue_tol)
                & (hsv[:, 1] >= self.s_range[0]) & (hsv[:, 1] <= self.s_range[1])
                & (hsv[:, 2] >= self.v_range[0]) & (hsv[:, 2] <= self.v_range[1]))


def pitch_hue(image: np.ndarray) -> GrassModel | None:
    """Model the dominant saturated colour of the frame: the grass, on any pitch.

    Returns ``None`` when the frame has no dominant saturated colour (an indoor
    court in grey, say), in which case nothing is masked.
    """
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    sat = (hsv[:, 1] > 50) & (hsv[:, 2] > 40)
    if sat.mean() < 0.2:
        return None
    hist = np.bincount(hsv[sat, 0].astype(int), minlength=180).astype(float)
    hist = np.convolve(np.r_[hist[-5:], hist, hist[:5]], np.ones(11) / 11, mode="valid")
    peak = int(np.argmax(hist))
    if hist[peak] * 11 / sat.sum() < 0.25:  # no single colour dominates
        return None
    dh = np.abs(hsv[:, 0] - peak)
    dh = np.minimum(dh, 180 - dh)
    grass = hsv[sat & (dh <= 12)]
    spread = 1.4826 * np.median(np.abs(grass[:, 0] - np.median(grass[:, 0])))
    s_lo, s_hi = np.percentile(grass[:, 1], [2, 98])
    v_lo, v_hi = np.percentile(grass[:, 2], [2, 98])
    return GrassModel(float(np.median(grass[:, 0])), float(np.clip(3 * spread, 4, 12)),
                      (float(s_lo) - 10, float(s_hi) + 10), (float(v_lo) - 15, float(v_hi) + 15))


def kit_feature(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    grass: "GrassModel | None",
    band: tuple[float, float] = (0.15, 0.6),
) -> np.ndarray | None:
    """Colour signature of one player box, or ``None`` if too little is visible."""
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # Central 60% of the width: the edges of a box are mostly background.
    cx1 = int(max(0, x1 + 0.2 * bw))
    cx2 = int(min(w_img, x2 - 0.2 * bw))
    cy1 = int(max(0, y1 + band[0] * bh))
    cy2 = int(min(h_img, y1 + band[1] * bh))
    if cx2 - cx1 < 2 or cy2 - cy1 < 3:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    keep = np.ones(len(hsv), dtype=bool)
    if grass is not None:
        keep &= ~grass.mask(hsv)
    if keep.sum() < 6:
        return None
    px = hsv[keep]
    chroma = px[:, 1] > 40
    hs = np.zeros(_H_BINS * _S_BINS)
    if chroma.any():
        hb = np.minimum(px[chroma, 0] * _H_BINS // 180, _H_BINS - 1)
        sb = np.minimum((px[chroma, 1] - 40) * _S_BINS // 216, _S_BINS - 1)
        hs = np.bincount(hb * _S_BINS + sb, minlength=_H_BINS * _S_BINS).astype(float)
    vb = np.minimum(px[:, 2] * _V_BINS // 256, _V_BINS - 1)
    vh = np.bincount(vb, minlength=_V_BINS).astype(float)
    # Weight the halves so neither dominates by pixel count alone.
    hs = hs / max(len(px), 1)
    vh = vh / max(len(px), 1)
    return np.sqrt(np.r_[hs, vh]).astype(np.float32)
