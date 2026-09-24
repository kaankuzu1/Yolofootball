"""The nutmeg, as an explicit rule (plan §4.3).

A nutmeg is geometric: the ball passes between the defender's feet, from the
attacker's side to the far side, and the defender does not come away with it.
That needs no classifier, and it is the trick people most want logged, so it
is a rule that can be read and checked rather than a learned score.

The trap on a side-angle camera is depth.  A ball rolling past a defender a
metre further from the camera can appear, in the image, to cross the gap
between their ankles.  So the ball's contact point with the ground must sit at
the height of the defender's feet in the image -- the same depth -- before a
crossing counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_analysis.body_events.features import _Body, _interp_ball, _nanmean
from football_analysis.body_events.series import ClipSeries

__all__ = ["NutmegConfig", "NutmegCall", "find_nutmegs"]


@dataclass
class NutmegConfig:
    ball_radius_h: float = 0.06
    """Ball radius in player heights (22 cm against 1.8 m)."""
    depth_tol_h: float = 0.06
    """How far the ball's ground point may sit from the feet line, in heights."""
    side_s: float = 0.3
    """The ball must be on the attacker's side this long before, and on the far
    side this long after."""
    had_ball_h: float = 0.75
    keep_h: float = 0.4
    """The defender 'kept' the ball if it stays this close to their feet."""
    min_conf: float = 0.3
    max_stance_offset_h: float = 0.12
    """The feet must straddle the hips: a lunge, with one foot far out in front,
    opens a gap too, and a ball through it is a failed tackle, not a nutmeg."""
    min_lead_frac: float = 0.5
    """On the way in, the attacker's foot must be nearer the ball than the
    defender's for at least this share of frames."""


@dataclass
class NutmegCall:
    attacker_id: str
    defender_id: str
    frame: int
    """Row of the clip where the ball was between the feet."""
    confidence: float
    inside_frames: int
    depth_error_h: float


def find_nutmegs(clip: ClipSeries, cfg: NutmegConfig | None = None) -> list[NutmegCall]:
    cfg = cfg or NutmegConfig()
    fps = clip.fps
    side = max(2, int(round(cfg.side_s * fps)))
    bodies = {pid: _Body(p, clip.t) for pid, p in clip.players.items()}
    ball = clip.ball
    bi = _interp_ball(clip)
    obs = clip.ball_observed
    calls: list[NutmegCall] = []
    for did, D in bodies.items():
        la, ra = D.p.joint("l_ank", cfg.min_conf), D.p.joint("r_ank", cfg.min_conf)
        hip = D.p.mid("l_hip", "r_hip", cfg.min_conf)
        H = D.h
        lo = np.fmin(la[:, 0], ra[:, 0])
        hi = np.fmax(la[:, 0], ra[:, 0])
        feet_y = np.fmax(la[:, 1], ra[:, 1])
        ground = ball[:, 1] + cfg.ball_radius_h * H
        straddle = np.abs(0.5 * (lo + hi) - hip[:, 0]) < cfg.max_stance_offset_h * H
        between = (
            obs
            & straddle
            & (ball[:, 0] > lo) & (ball[:, 0] < hi)
            & (ball[:, 1] > hip[:, 1])
            & (np.abs(ground - feet_y) < cfg.depth_tol_h * H)
        )
        # Which side of the defender the ball is on, with a dead zone the width
        # of the stance so a ball between the feet counts as neither.
        off = (bi[:, 0] - hip[:, 0]) / H
        rel = np.where(np.abs(off) > 0.12, np.sign(off), np.nan)
        k = 0
        n = len(clip)
        gap = max(1, int(round(0.08 * fps)))
        while k < n:
            if not between[k]:
                k += 1
                continue
            run_end = k
            while True:
                nxt = np.flatnonzero(between[run_end + 1:run_end + 2 + gap])
                if not len(nxt):
                    break
                run_end = run_end + 1 + int(nxt[-1])
            before = rel[max(0, k - side):k]
            after = rel[run_end + 1:run_end + 1 + side]
            b_side = _mode(before)
            a_side = _mode(after)
            crossed = b_side != 0 and a_side != 0 and b_side == -a_side
            if crossed:
                att = _attacker(bodies, did, bi, max(0, k - int(0.8 * fps)), k, cfg)
                collected = att is not None and _collects(
                    bodies[att], D, bi, run_end + 1, min(n, run_end + 1 + int(1.2 * fps)), a_side, hip, cfg)
                if collected:
                    mid = (k + run_end) // 2
                    seg = slice(k, run_end + 1)
                    inside = int(np.sum(between[seg]))
                    depth_err = float(np.nanmedian(np.abs(ground[seg] - feet_y[seg])[between[seg]] / H[seg][between[seg]]))
                    conf = 0.55 + 0.1 * min(inside, 3) + 0.15 * (1 - depth_err / cfg.depth_tol_h)
                    calls.append(NutmegCall(att, did, mid, float(min(0.95, conf)), inside, depth_err))
            k = run_end + 1
    return calls


def _mode(signs: np.ndarray) -> float:
    s = signs[~np.isnan(signs)]
    if len(s) < 2:
        return 0.0
    m = float(np.mean(s))
    return float(np.sign(m)) if abs(m) > 0.5 else 0.0


def _collects(A, D, bi: np.ndarray, a: int, b: int, far: float, hip: np.ndarray,
              cfg: NutmegConfig) -> bool:
    """The attacker gets to the ball on the far side before the defender does."""
    if b <= a:
        return False
    da = A.foot_to(bi)[a:b] / A.h[a:b]
    dd = D.foot_to(bi)[a:b] / D.h[a:b]
    far_side = np.sign(bi[a:b, 0] - hip[a:b, 0]) == far
    fps = 1.0 / max(1e-6, float(np.median(np.diff(A.t[a:b + 1])))) if b - a > 1 else 25.0
    held = 0
    for i in range(b - a):
        # The ball has just passed the defender's feet, so it is near them for
        # a frame or two whatever happens; only a ball that *stays* there was
        # stopped.
        held = held + 1 if dd[i] < cfg.keep_h and not (da[i] < dd[i]) else 0
        if held > 0.3 * fps:
            return False
        if da[i] < cfg.had_ball_h * 0.8 and far_side[i]:
            return True
    return False


def _attacker(bodies: dict, did: str, bi: np.ndarray, a: int, b: int, cfg: NutmegConfig) -> str | None:
    """Who played the ball through.  They must have been the one on the ball
    on the way in: in a shoulder duel both players are on top of it, and the
    ball rolling under the player who had it is not a nutmeg on them."""
    dd = bodies[did].foot_to(bi)[a:b] / bodies[did].h[a:b]
    best, best_d = None, np.inf
    for pid, B in bodies.items():
        if pid == did:
            continue
        d = B.foot_to(bi)[a:b] / B.h[a:b]
        m = np.nanmin(d) if np.any(~np.isnan(d)) else np.inf
        both = ~np.isnan(d) & ~np.isnan(dd)
        if not np.any(both) or np.mean(d[both] < dd[both]) < cfg.min_lead_frac:
            continue
        if m < cfg.had_ball_h and m < best_d:
            best, best_d = pid, m
    return best
