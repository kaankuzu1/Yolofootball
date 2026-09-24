"""The world-state records of one clip, as arrays.

Every feature in this package is a statement about how a joint or the ball
moved over a second or so, which is awkward over a list of dataclasses and
natural over arrays.  :class:`ClipSeries` is that conversion, done once.

Missing data is ``NaN``, never zero: a keypoint the pose model did not find,
a frame where a player was not tracked, a ball nobody saw.  Features are
written to tolerate ``NaN`` and the classifier is one that accepts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from football_analysis.state import FrameState

__all__ = ["KP", "PlayerSeries", "ClipSeries", "nanmean_stack"]

# COCO-17 indices, by name, for readability at the call sites.
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "l_wri": 9, "r_wri": 10,
    "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ank": 15, "r_ank": 16,
}
_COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_NAME_TO_INDEX = {n: i for i, n in enumerate(_COCO_NAMES)}


@dataclass
class PlayerSeries:
    player_id: str
    bbox: np.ndarray
    """``(N, 4)`` x1, y1, x2, y2; NaN where the player is absent."""
    kp: np.ndarray
    """``(N, 17, 3)`` x, y, confidence; NaN where pose is missing."""
    track_ids: set[int] = field(default_factory=set)

    @property
    def present(self) -> np.ndarray:
        return ~np.isnan(self.bbox[:, 0])

    @property
    def height(self) -> np.ndarray:
        return self.bbox[:, 3] - self.bbox[:, 1]

    @property
    def foot(self) -> np.ndarray:
        """``(N, 2)`` bottom centre of the box -- where the player meets the ground."""
        return np.stack([(self.bbox[:, 0] + self.bbox[:, 2]) / 2, self.bbox[:, 3]], axis=1)

    def joint(self, name: str, min_conf: float = 0.3) -> np.ndarray:
        """``(N, 2)`` one joint, NaN where missing or below ``min_conf``."""
        k = self.kp[:, KP[name]]
        xy = k[:, :2].copy()
        xy[~(k[:, 2] >= min_conf)] = np.nan
        return xy

    def mid(self, a: str, b: str, min_conf: float = 0.3) -> np.ndarray:
        """Midpoint of two joints, falling back to whichever one is present."""
        return nanmean_stack([self.joint(a, min_conf), self.joint(b, min_conf)])


def nanmean_stack(arrs: Sequence[np.ndarray]) -> np.ndarray:
    """Element-wise mean ignoring NaN; all-NaN gives NaN without a warning."""
    stack = np.stack(arrs)
    count = np.sum(~np.isnan(stack), axis=0)
    total = np.nansum(stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


@dataclass
class ClipSeries:
    t: np.ndarray
    """``(N,)`` seconds."""
    frame_index: np.ndarray
    players: dict[str, PlayerSeries]
    ball: np.ndarray
    """``(N, 2)`` ball centre; NaN where there is no ball position at all."""
    ball_observed: np.ndarray
    """``(N,)`` True only where the ball was actually detected, not coasted."""
    frame_size: tuple[int, int] = (0, 0)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def fps(self) -> float:
        if len(self.t) < 2:
            return 25.0
        dt = np.diff(self.t)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else 25.0

    def index_at(self, timestamp_s: float) -> int:
        return int(np.clip(np.searchsorted(self.t, timestamp_s), 0, len(self.t) - 1))

    @classmethod
    def from_states(cls, states: Sequence[FrameState]) -> "ClipSeries":
        states = sorted(states, key=lambda s: s.frame_index)
        n = len(states)
        t = np.array([s.timestamp_s for s in states], dtype=float)
        fi = np.array([s.frame_index for s in states], dtype=int)
        ball = np.full((n, 2), np.nan)
        observed = np.zeros(n, dtype=bool)
        players: dict[str, PlayerSeries] = {}
        size = states[0].frame_size if states else (0, 0)
        for k, s in enumerate(states):
            if s.ball is not None:
                ball[k] = (s.ball.position.x, s.ball.position.y)
                observed[k] = not s.ball.interpolated
            for p in s.players:
                ps = players.get(p.player_id)
                if ps is None:
                    ps = PlayerSeries(
                        player_id=p.player_id,
                        bbox=np.full((n, 4), np.nan),
                        kp=np.full((n, 17, 3), np.nan),
                    )
                    players[p.player_id] = ps
                ps.bbox[k] = p.bbox.as_tuple()
                ps.track_ids.add(int(p.track_id))
                for kpt in p.keypoints:
                    j = _NAME_TO_INDEX.get(kpt.name)
                    if j is not None:
                        ps.kp[k, j] = (kpt.x, kpt.y, kpt.confidence)
        return cls(t=t, frame_index=fi, players=players, ball=ball,
                   ball_observed=observed, frame_size=tuple(size))
