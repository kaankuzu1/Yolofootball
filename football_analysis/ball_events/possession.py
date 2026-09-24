"""Possession: who has the ball, frame by frame (plan §4.0).

Everything else in this package is a statement about how this signal changes,
so it is built to be conservative:

* A player *has* the ball only when it is in their foot region -- not merely
  near their box -- measured in their own box height, which is the right local
  ruler under a side-angle perspective.
* Proximity alone is not control.  A pass that flies past a player's boots is
  within reach for a frame or two, so a run of frames at someone's feet only
  counts when the ball is moving *with* them (low relative speed) over the run,
  or when the ball's velocity visibly changes there (a one-touch kick).
* When both players are within reach and neither is clearly closer, the frame
  is *contested*.  It credits nobody, and is recorded as such: a 50-50 is the
  tackle thread's business, not a possession.

The output is a list of :class:`Contact` records, which :mod:`.flights` turns
into possession spells and ball flights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from football_analysis.ball_events.config import BallEventConfig
from football_analysis.ball_events.state import ClipState, PlayerObs

__all__ = ["Contact", "PossessionTrack", "foot_distance", "compute_possession", "detect_breaks"]


@dataclass
class Contact:
    """A run of frames where one player had the ball at their feet."""

    player_id: str
    start: int
    """First frame (row in the clip state) of the run."""
    end: int
    """Last frame of the run, inclusive."""
    track_id: int | None = None
    touch_frames: list[int] = field(default_factory=list)
    """Frames inside the run where the ball's velocity changed abruptly."""
    reason: str = "control"
    """``control`` (ball moved with the player) or ``touch`` (a one-touch)."""

    @property
    def release_frame(self) -> int:
        """The frame the ball was played away on.

        The kick is the last touch, provided it is at the end of the run: a
        touch at the start is the reception, and one in the middle is a
        dribble.  Without such a touch, the last frame at the feet.
        """
        late = [t for t in self.touch_frames if t >= self.end - 3]
        return max(late) if late else self.end


@dataclass
class PossessionTrack:
    """Per-frame possession plus the contacts it was built from."""

    candidate: list[str | None]
    """Player whose feet the ball is at, per frame, before any smoothing."""
    distance: np.ndarray
    """Normalised distance (in the nearest player's heights) to the foot region."""
    contested: np.ndarray
    """True where two players were within reach and neither clearly closer."""
    contacts: list[Contact]
    breaks: np.ndarray
    possessor: list[str | None] = field(default_factory=list)
    """Filled by :mod:`.flights` once dribble touches are merged into spells."""

    def possessor_at(self, k: int) -> str | None:
        return self.possessor[k] if self.possessor else None


def foot_distance(ball_xy: np.ndarray, player: PlayerObs, cfg: BallEventConfig) -> float:
    """How far the ball is from ``player``'s foot region, in their box heights.

    The foot region is the box's width, from ``foot_band_top`` of the height
    above the box bottom to ``foot_band_below`` beneath it.  Zero means inside.
    """
    x1, y1, x2, y2 = player.bbox
    h = player.height
    top = y2 - cfg.foot_band_top * h
    bottom = y2 + cfg.foot_band_below * h
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    dx = max(0.0, x1 - bx, bx - x2)
    dy = max(0.0, top - by, by - bottom)
    return float(np.hypot(dx, dy) / h)


def detect_breaks(state: ClipState, ruler: float, cfg: BallEventConfig) -> np.ndarray:
    """Frames where the ball's velocity changes abruptly: a touch.

    The tracker's own kinematic breaks are taken as given.  On top of that, a
    turn of more than 35 degrees or a speed change of more than 1.8x between
    the three frames before and the three after is a break, provided the ball
    is moving at all.
    """
    n = len(state)
    out = state.ball_break.copy()
    v = state.ball_velocity
    speed = np.linalg.norm(v, axis=1)
    rest = cfg.rest_speed_d * ruler
    for k in range(3, n - 3):
        before = v[k - 3:k]
        after = v[k + 1:k + 4]
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            continue
        vb = before.mean(axis=0)
        va = after.mean(axis=0)
        sb, sa = float(np.linalg.norm(vb)), float(np.linalg.norm(va))
        if max(sb, sa) < 2 * rest:
            continue
        ratio = (sa + 1e-6) / (sb + 1e-6)
        cos = float(np.dot(vb, va) / (sb * sa)) if sb > rest and sa > rest else 1.0
        if ratio > 1.8 or ratio < 1 / 1.8 or cos < np.cos(np.radians(35)):
            out[k] = True
    # Collapse runs of adjacent break frames onto the frame with the largest
    # change, so one kick is one break.
    k = 0
    while k < n:
        if not out[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and out[j + 1]:
            j += 1
        if j > k:
            window = np.arange(k, j + 1)
            accel = np.array([
                np.linalg.norm(np.nan_to_num(v[min(i + 1, n - 1)] - v[max(i - 1, 0)]))
                for i in window
            ])
            keep = int(window[int(np.argmax(accel))])
            out[k:j + 1] = False
            out[keep] = True
        k = j + 1
    out |= _breaks_hidden_in_gaps(state, out, ruler, cfg)
    out &= state.present
    return out


def _observed_velocity(state: ClipState, idx: np.ndarray) -> np.ndarray | None:
    """Least-squares velocity over the detected frames ``idx``, px/s."""
    if len(idx) < 3:
        return None
    t = state.timestamps[idx] - state.timestamps[idx].mean()
    denom = float((t * t).sum())
    if denom <= 0:
        return None
    p = state.ball_xy[idx] - state.ball_xy[idx].mean(axis=0)
    return (t[:, None] * p).sum(axis=0) / denom


def _breaks_hidden_in_gaps(state: ClipState, known: np.ndarray, ruler: float,
                           cfg: BallEventConfig) -> np.ndarray:
    """Touches that happened while the detector lost the ball.

    A smoother bridges a gap with a gentle curve, so a kick inside the gap
    never shows as a step in velocity.  The detected frames either side of the
    gap still disagree, though: compare the velocity just before the gap with
    the velocity just after, and if they differ the way a touch does, place
    the break at the gap frame where the bridged ball comes closest to a
    player's feet (the middle of the gap if nobody is near).
    """
    n = len(state)
    out = np.zeros(n, dtype=bool)
    seen = state.observed
    rest = cfg.rest_speed_d * ruler
    max_gap = int(round(cfg.hidden_touch_max_gap_s * state.fps))
    if max_gap < 1:
        return out
    k = 0
    while k < n:
        if seen[k]:
            k += 1
            continue
        a = k
        while k < n and not seen[k]:
            k += 1
        b = k - 1  # gap is a..b inclusive
        if a == 0 or k >= n or b - a + 1 > max_gap:
            continue
        if known[max(0, a - 2):min(n, b + 3)].any():
            continue
        before = np.arange(max(0, a - 4), a)
        before = before[seen[before]]
        after = np.arange(b + 1, min(n, b + 5))
        after = after[seen[after]]
        vb, va = _observed_velocity(state, before), _observed_velocity(state, after)
        if vb is None or va is None:
            continue
        sb, sa = float(np.linalg.norm(vb)), float(np.linalg.norm(va))
        if max(sb, sa) < 2 * rest:
            continue
        ratio = (sa + 1e-6) / (sb + 1e-6)
        cos = float(np.dot(vb, va) / (sb * sa)) if sb > rest and sa > rest else 1.0
        if not (ratio > 1.8 or ratio < 1 / 1.8 or cos < np.cos(np.radians(35))):
            continue
        best, best_d = (a + b) // 2, np.inf
        for j in range(a, b + 1):
            if not state.present[j]:
                continue
            for p in state.players[j]:
                d = foot_distance(state.ball_xy[j], p, cfg)
                if d < best_d:
                    best, best_d = j, d
        out[best] = True
    return out


def _player_velocities(state: ClipState) -> dict[str, np.ndarray]:
    """Foot-point velocity per player id, px/s, NaN where unknown."""
    n = len(state)
    ids = state.player_ids()
    feet = {pid: np.full((n, 2), np.nan) for pid in ids}
    for k, row in enumerate(state.players):
        for p in row:
            feet[p.player_id][k] = p.foot
    out: dict[str, np.ndarray] = {}
    for pid, f in feet.items():
        v = np.full((n, 2), np.nan)
        for k in range(n):
            lo, hi = max(0, k - 2), min(n - 1, k + 2)
            if np.isfinite(f[lo]).all() and np.isfinite(f[hi]).all() and hi > lo:
                dt = state.timestamps[hi] - state.timestamps[lo]
                if dt > 0:
                    v[k] = (f[hi] - f[lo]) / dt
        out[pid] = v
    return out


def compute_possession(state: ClipState, cfg: BallEventConfig, ruler: float) -> PossessionTrack:
    n = len(state)
    candidate: list[str | None] = [None] * n
    track_of: list[int | None] = [None] * n
    height_of = np.full(n, np.nan)
    dist = np.full(n, np.inf)
    contested = np.zeros(n, bool)
    breaks = detect_breaks(state, ruler, cfg)

    # Who each touch belongs to: the player nearest the ball when it happened.
    # Without this, a kick made a frame before the ball reaches the defender's
    # feet would be credited to the defender it rolls past.
    break_owner: list[str | None] = [None] * n
    nearest_id: list[str | None] = [None] * n

    for k in range(n):
        if not state.present[k] or not state.players[k]:
            continue
        ball = state.ball_xy[k]
        scored = sorted(
            ((foot_distance(ball, p, cfg), p) for p in state.players[k]),
            key=lambda item: item[0],
        )
        best_d, best = scored[0]
        dist[k] = best_d
        nearest_id[k] = best.player_id
        if best_d > cfg.reach:
            continue
        if len(scored) > 1:
            second_d, second = scored[1]
            if second.player_id != best.player_id and second_d <= cfg.reach \
                    and second_d - best_d < cfg.contested_margin:
                contested[k] = True
                continue
        candidate[k] = best.player_id
        track_of[k] = best.track_id
        height_of[k] = best.height

    # A detected break can lag the foot contact by a frame or two, by which
    # time the ball may be nearer someone else: look back over that span.
    for k in np.nonzero(breaks)[0]:
        window = [j for j in range(max(0, k - 2), k + 1) if nearest_id[j] is not None]
        if window:
            j = min(window, key=lambda i: dist[i])
            if dist[j] <= 2 * cfg.reach:
                break_owner[k] = nearest_id[j]

    player_v = _player_velocities(state)
    contacts: list[Contact] = []
    k = 0
    while k < n:
        pid = candidate[k]
        if pid is None:
            k += 1
            continue
        j = k
        while j + 1 < n and candidate[j + 1] == pid:
            j += 1
        run = np.arange(k, j + 1)
        # A touch right at the edge of the run belongs to it: the kick frame is
        # often the last one the ball is still inside the foot region.
        touch_window = np.arange(max(0, k - 1), min(n, j + 2))
        touches = [int(i) for i in touch_window if breaks[i] and break_owner[i] == pid]
        rel = state.ball_velocity[run] - np.nan_to_num(player_v[pid][run])
        rel_speed = np.linalg.norm(rel, axis=1) / np.where(np.isfinite(height_of[run]),
                                                            height_of[run], np.inf)
        rel_speed = rel_speed[np.isfinite(rel_speed)]
        controlled = (
            len(run) >= cfg.possession_min_frames
            and rel_speed.size > 0
            and float(np.median(rel_speed)) <= cfg.control_speed_h
        )
        if controlled:
            contacts.append(Contact(pid, k, j, track_of[k], touches, "control"))
        elif touches:
            contacts.append(Contact(pid, k, j, track_of[k], touches, "touch"))
        k = j + 1

    return PossessionTrack(candidate, dist, contested, contacts, breaks)
