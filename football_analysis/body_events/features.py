"""Candidate windows and the features the classifier reads (plan §4.2, §4.3).

Two kinds of window, because the two events are found in different places:

``ContactWindow``  two players converge with the ball near them: a tackle, a
                   failed tackle, a shoulder duel or nothing.  One per episode
                   of closeness, centred on the moment the defender's foot got
                   nearest the ball (or the bodies got nearest each other).
``SkillWindow``    one player on the ball: an ordinary dribble or a trick.
                   A sliding window over every spell on the ball.

Every distance is in the player's own height, measured from their box, so a
player 80 pixels tall and one 400 pixels tall give the same numbers -- the
same ruler :mod:`football_analysis.ball_events.possession` uses.  Every
feature tolerates missing pose or ball and returns ``NaN`` rather than a
guess; the classifier accepts ``NaN``.

The feature lists are the contract with a saved model: a model records the
names it was trained on, and :func:`contact_vector` / :func:`skill_vector`
return values in that order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from football_analysis.body_events.series import ClipSeries, PlayerSeries, nanmean_stack

__all__ = [
    "CONTACT_FEATURES",
    "SKILL_FEATURES",
    "ContactWindow",
    "SkillWindow",
    "WindowConfig",
    "contact_windows",
    "skill_windows",
    "contact_features",
    "skill_features",
    "contact_vector",
    "skill_vector",
]


@dataclass
class WindowConfig:
    contact_gap_h: float = 0.9
    """Two players are 'in contact' below this separation, in heights."""
    contact_ball_h: float = 1.6
    """...and only when the ball is within this of one of them."""
    contact_half_s: float = 0.8
    """A contact window runs this long either side of its anchor."""
    merge_gap_s: float = 0.3
    on_ball_h: float = 0.75
    """A player is on the ball when it is within this of their feet."""
    skill_len_s: float = 1.2
    skill_stride_s: float = 0.2
    min_spell_s: float = 0.4
    depth_weight: float = 2.0
    """Foot-height differences count this much more than horizontal ones:
    a side-angle camera compresses depth into a little vertical offset."""


@dataclass
class ContactWindow:
    attacker_id: str
    defender_id: str
    start: int
    end: int
    anchor: int
    """Row the window is centred on -- the reported time of the event."""
    attacker_known: bool = True
    """False when the ball was not seen well enough to say who had it."""
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class SkillWindow:
    player_id: str
    start: int
    end: int
    opponent_id: str | None = None
    features: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Small helpers


def _smooth_height(p: PlayerSeries) -> np.ndarray:
    h = p.height.copy()
    ok = ~np.isnan(h)
    if not ok.any():
        return h
    idx = np.arange(len(h))
    h = np.interp(idx, idx[ok], h[ok])
    k = 7
    pad = np.pad(h, (k // 2, k // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)


def _runs(mask: np.ndarray, merge: int, min_len: int = 1) -> list[tuple[int, int]]:
    """Inclusive runs of True, merging gaps of up to ``merge`` rows."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e <= merge + 1:
            e = i
        else:
            runs.append((s, e))
            s = e = i
    runs.append((s, e))
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1])


def _nanmin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.any(~np.isnan(x)) else np.nan


def _nanmax(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.any(~np.isnan(x)) else np.nan


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else np.nan


def _nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.any(~np.isnan(x)) else np.nan


def _vel(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central-difference velocity, NaN-propagating, units per second."""
    v = np.full_like(xy, np.nan)
    if len(xy) < 3:
        return v
    dt = (t[2:] - t[:-2])[:, None]
    v[1:-1] = (xy[2:] - xy[:-2]) / np.where(dt > 0, dt, np.nan)
    return v


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.hypot(*u), np.hypot(*v)
    if not (nu > 1e-9 and nv > 1e-9) or np.isnan(nu) or np.isnan(nv):
        return np.nan
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1, 1))
    return float(np.degrees(np.arccos(c)))


def _path(xy: np.ndarray) -> float:
    """Summed frame-to-frame travel, skipping missing frames."""
    ok = ~np.isnan(xy[:, 0])
    pts = xy[ok]
    if len(pts) < 2:
        return np.nan
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def _interp_ball(clip: ClipSeries) -> np.ndarray:
    """Ball with short holes filled linearly, for proximity tests only."""
    b = clip.ball.copy()
    ok = ~np.isnan(b[:, 0])
    if ok.sum() < 2:
        return b
    idx = np.arange(len(b))
    for d in (0, 1):
        filled = np.interp(idx, idx[ok], b[ok, d])
        gap = np.abs(idx - idx[ok][np.clip(np.searchsorted(idx[ok], idx), 0, ok.sum() - 1)])
        b[:, d] = np.where(ok | (gap < clip.fps * 0.4), filled, np.nan)
    return b


class _Body:
    """Derived per-frame quantities for one player, in pixels and heights."""

    def __init__(self, p: PlayerSeries, t: np.ndarray):
        self.p = p
        self.h = _smooth_height(p)
        self.foot = p.foot
        self.hip = p.mid("l_hip", "r_hip")
        self.sho = p.mid("l_sho", "r_sho")
        self.ank = np.stack([p.joint("l_ank"), p.joint("r_ank")])  # (2, N, 2)
        self.knee = np.stack([p.joint("l_knee"), p.joint("r_knee")])
        self.nose = p.joint("nose")
        box_c = np.stack([(p.bbox[:, 0] + p.bbox[:, 2]) / 2,
                          p.bbox[:, 1] + 0.5 * (p.bbox[:, 3] - p.bbox[:, 1])], axis=1)
        # Where the body is, from pose if present, else the box.
        self.centre = np.where(np.isnan(self.hip), box_c, self.hip)
        self.t = t
        legs = p.kp[:, 11:17, 2]
        self.leg_conf = np.where(np.isnan(legs), 0.0, legs).mean(axis=1)
        self.leg_conf[~p.present] = np.nan
        self.has_pose = ~np.isnan(p.kp[:, 15, 0]) | ~np.isnan(p.kp[:, 16, 0])

    def ankle_rel_hip(self) -> np.ndarray:
        return self.ank - self.hip[None]

    def nearest_ankle_to(self, pt: np.ndarray) -> np.ndarray:
        d = np.stack([_dist(self.ank[0], pt), _dist(self.ank[1], pt)])
        with np.errstate(all="ignore"):
            return np.where(np.all(np.isnan(d), axis=0), np.nan, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=0))

    def foot_to(self, pt: np.ndarray) -> np.ndarray:
        """Distance from the feet (nearest ankle, else box bottom) to ``pt``."""
        a = self.nearest_ankle_to(pt)
        b = _dist(self.foot, pt)
        return np.where(np.isnan(a), b, np.fmin(a, b))

    def facing(self) -> np.ndarray:
        """+1 / -1 in x, from the nose against the shoulders; NaN if unknown."""
        dx = self.nose[:, 0] - self.sho[:, 0]
        out = np.sign(dx)
        out[np.abs(dx) < 0.01 * self.h] = np.nan
        return out


# ----------------------------------------------------------------------------
# Windows


def _pair_gap(a: _Body, b: _Body, cfg: WindowConfig) -> np.ndarray:
    h = 0.5 * (a.h + b.h)
    dx = np.abs(a.centre[:, 0] - b.centre[:, 0])
    dy = np.abs(a.foot[:, 1] - b.foot[:, 1]) * cfg.depth_weight
    return np.hypot(dx, dy) / h


def contact_windows(clip: ClipSeries, cfg: WindowConfig | None = None) -> list[ContactWindow]:
    cfg = cfg or WindowConfig()
    bodies = {pid: _Body(p, clip.t) for pid, p in clip.players.items()}
    ball = _interp_ball(clip)
    fps = clip.fps
    half = int(round(cfg.contact_half_s * fps))
    out: list[ContactWindow] = []
    ids = sorted(bodies)
    for i, pa in enumerate(ids):
        for pb in ids[i + 1:]:
            A, B = bodies[pa], bodies[pb]
            gap = _pair_gap(A, B, cfg)
            near_a = A.foot_to(ball) / A.h
            near_b = B.foot_to(ball) / B.h
            ball_near = (np.fmin(near_a, near_b) < cfg.contact_ball_h) | np.isnan(ball[:, 0])
            mask = (gap < cfg.contact_gap_h) & ball_near
            for s, e in _runs(mask, merge=int(cfg.merge_gap_s * fps)):
                # Who had the ball on the way in is the attacker.
                pre = slice(max(0, s - half), max(s, 1))
                da, db = _nanmean(near_a[pre]), _nanmean(near_b[pre])
                known = not (np.isnan(da) or np.isnan(db)) and abs(da - db) > 0.15
                att, dfd = (pa, pb) if (np.isnan(db) or (not np.isnan(da) and da <= db)) else (pb, pa)
                D = bodies[dfd]
                seg = slice(s, e + 1)
                ank_ball = (D.nearest_ankle_to(ball) / D.h)[seg]
                touching = np.flatnonzero(ank_ball < 0.25)
                if len(touching):
                    # The first touch is the tackle; what follows is the
                    # defender collecting the ball, not a second contact.
                    k0 = int(touching[0])
                    k1 = min(len(ank_ball), k0 + max(1, int(0.15 * fps)))
                    anchor = s + k0 + int(np.nanargmin(ank_ball[k0:k1]))
                elif np.any(~np.isnan(ank_ball)):
                    anchor = s + int(np.nanargmin(ank_ball))
                else:
                    anchor = s + int(np.nanargmin(np.where(np.isnan(gap[seg]), np.inf, gap[seg])))
                out.append(ContactWindow(
                    attacker_id=att, defender_id=dfd,
                    start=max(0, anchor - half), end=min(len(clip) - 1, anchor + half),
                    anchor=anchor, attacker_known=known,
                ))
    return out


def skill_windows(clip: ClipSeries, cfg: WindowConfig | None = None) -> list[SkillWindow]:
    cfg = cfg or WindowConfig()
    bodies = {pid: _Body(p, clip.t) for pid, p in clip.players.items()}
    ball = _interp_ball(clip)
    fps = clip.fps
    n = len(clip)
    if not bodies:
        return []
    dist = {pid: B.foot_to(ball) / B.h for pid, B in bodies.items()}
    stack = np.stack([np.where(np.isnan(d), np.inf, d) for d in dist.values()])
    nearest = np.array(list(dist))[np.argmin(stack, axis=0)]
    win = int(round(cfg.skill_len_s * fps))
    stride = max(1, int(round(cfg.skill_stride_s * fps)))
    pad = int(round(0.3 * fps))
    out: list[SkillWindow] = []
    for pid, d in dist.items():
        mask = (d < cfg.on_ball_h) & (nearest == pid)
        for s, e in _runs(mask, merge=int(cfg.merge_gap_s * fps), min_len=int(cfg.min_spell_s * fps)):
            s2, e2 = max(0, s - pad), min(n - 1, e + pad)
            starts = list(range(s2, max(s2 + 1, e2 - win + 2), stride))
            for a in starts:
                b = min(n - 1, a + win - 1)
                opp = _nearest_opponent(bodies, pid, a, b)
                out.append(SkillWindow(player_id=pid, start=a, end=b, opponent_id=opp))
    return out


def _nearest_opponent(bodies: dict[str, _Body], pid: str, a: int, b: int) -> str | None:
    me = bodies[pid]
    best, best_d = None, np.inf
    for oid, other in bodies.items():
        if oid == pid:
            continue
        d = _nanmedian(_dist(me.centre[a:b + 1], other.centre[a:b + 1]) / me.h[a:b + 1])
        if not np.isnan(d) and d < best_d:
            best, best_d = oid, d
    return best


# ----------------------------------------------------------------------------
# Contact features (plan §4.2)

CONTACT_FEATURES: tuple[str, ...] = (
    "min_gap_h",
    "min_torso_gap_h",
    "min_def_ankle_ball_h",
    "min_att_foot_ball_h_near",
    "def_ankle_speed_max",
    "def_ankle_toward_ball_max",
    "def_leg_ext_max",
    "def_leg_ext_delta",
    "def_leg_angle_max_deg",
    "def_hip_drop_h",
    "att_leg_ext_delta",
    "ball_speed_before",
    "ball_speed_after",
    "ball_heading_change_deg",
    "ball_accel_peak",
    "att_has_before",
    "att_has_after",
    "def_has_after",
    "none_has_after",
    "att_ball_dist_after",
    "def_ball_dist_after",
    "closing_speed",
    "ball_observed_frac",
    "def_leg_conf",
    "att_leg_conf",
    "pose_frac",
    "contact_duration_s",
    "attacker_known",
)


def contact_features(clip: ClipSeries, w: ContactWindow, cfg: WindowConfig | None = None) -> dict[str, float]:
    cfg = cfg or WindowConfig()
    A = _Body(clip.players[w.attacker_id], clip.t)
    D = _Body(clip.players[w.defender_id], clip.t)
    fps = clip.fps
    t = clip.t
    seg = slice(w.start, w.end + 1)
    c = w.anchor
    k04, k02, k01, k03 = (int(round(x * fps)) for x in (0.4, 0.2, 0.1, 0.3))
    near = slice(max(w.start, c - k04), min(w.end, c + k02) + 1)
    before = slice(w.start, max(w.start + 1, c - k01))
    after = slice(min(w.end, c + k03), w.end + 1)
    H = _nanmedian(np.concatenate([A.h[seg], D.h[seg]]))
    ball = clip.ball
    bi = _interp_ball(clip)

    gap = _pair_gap(A, D, cfg)
    f: dict[str, float] = {}
    f["min_gap_h"] = _nanmin(gap[seg])
    f["min_torso_gap_h"] = _nanmin(_dist(A.sho[seg], D.sho[seg])) / H
    f["min_def_ankle_ball_h"] = _nanmin(D.nearest_ankle_to(bi)[near]) / H
    f["min_att_foot_ball_h_near"] = _nanmin(A.foot_to(bi)[near]) / H

    rel = D.ankle_rel_hip()  # (2, N, 2)
    speeds, toward = [], []
    for leg in (0, 1):
        v = _vel(rel[leg], t)
        speeds.append(np.hypot(v[:, 0], v[:, 1]))
        to_ball = bi - D.ank[leg]
        norm = np.hypot(to_ball[:, 0], to_ball[:, 1])
        with np.errstate(all="ignore"):
            toward.append((v[:, 0] * to_ball[:, 0] + v[:, 1] * to_ball[:, 1]) / norm)
    f["def_ankle_speed_max"] = _nanmax(np.stack(speeds)[:, near]) / H
    f["def_ankle_toward_ball_max"] = _nanmax(np.stack(toward)[:, near]) / H

    ext = np.stack([_dist(D.ank[0], D.hip), _dist(D.ank[1], D.hip)]) / H
    f["def_leg_ext_max"] = _nanmax(ext[:, near])
    f["def_leg_ext_delta"] = f["def_leg_ext_max"] - _nanmedian(ext[:, seg])
    ang = []
    for leg in (0, 1):
        v = D.ank[leg] - D.hip
        with np.errstate(all="ignore"):
            ang.append(np.degrees(np.arctan2(np.abs(v[:, 0]), v[:, 1])))
    f["def_leg_angle_max_deg"] = _nanmax(np.stack(ang)[:, near])
    hip_h = (D.foot[:, 1] - D.hip[:, 1]) / H
    f["def_hip_drop_h"] = _nanmedian(hip_h[seg]) - _nanmin(hip_h[near])
    aext = np.stack([_dist(A.ank[0], A.hip), _dist(A.ank[1], A.hip)]) / H
    f["att_leg_ext_delta"] = _nanmax(aext[:, near]) - _nanmedian(aext[:, seg])

    # The ball, relative to the duel's own centre so a panning camera does not
    # read as ball speed.
    centre = nanmean_stack([A.foot, D.foot])
    vb = _vel(ball - centre, t) / H
    vb[~clip.ball_observed] = np.nan
    sp = np.hypot(vb[:, 0], vb[:, 1])
    pre_v = slice(max(w.start, c - k04), max(w.start + 1, c - 1))
    post_v = slice(min(w.end, c + 1), min(w.end, c + k04) + 1)
    f["ball_speed_before"] = _nanmedian(sp[pre_v])
    f["ball_speed_after"] = _nanmedian(sp[post_v])
    mv0 = np.array([_nanmean(vb[pre_v, 0]), _nanmean(vb[pre_v, 1])])
    mv1 = np.array([_nanmean(vb[post_v, 0]), _nanmean(vb[post_v, 1])])
    f["ball_heading_change_deg"] = _angle_between(mv0, mv1)
    dv = np.hypot(*np.diff(vb[near], axis=0).T) if (near.stop - near.start) > 1 else np.array([np.nan])
    f["ball_accel_peak"] = _nanmax(dv)

    da = A.foot_to(bi) / H
    dd = D.foot_to(bi) / H
    has = cfg.on_ball_h * 0.8
    f["att_has_before"] = _frac(da[before] < has, da[before])
    f["att_has_after"] = _frac(da[after] < has, da[after])
    f["def_has_after"] = _frac(dd[after] < has, dd[after])
    f["none_has_after"] = _frac((da[after] >= has) & (dd[after] >= has), da[after] + dd[after])
    f["att_ball_dist_after"] = _nanmedian(da[after])
    f["def_ball_dist_after"] = _nanmedian(dd[after])

    g = gap[before]
    tt = t[before]
    ok = ~np.isnan(g)
    f["closing_speed"] = float(-np.polyfit(tt[ok], g[ok], 1)[0]) if ok.sum() >= 3 else np.nan
    f["ball_observed_frac"] = float(np.mean(clip.ball_observed[near]))
    f["def_leg_conf"] = _nanmean(D.leg_conf[near])
    f["att_leg_conf"] = _nanmean(A.leg_conf[near])
    f["pose_frac"] = float(np.mean(D.has_pose[near] & A.has_pose[near]))
    in_contact = gap[seg] < cfg.contact_gap_h
    f["contact_duration_s"] = float(np.sum(in_contact)) / fps
    f["attacker_known"] = float(w.attacker_known)
    return f


def _frac(cond: np.ndarray, ref: np.ndarray) -> float:
    ok = ~np.isnan(ref)
    return float(np.mean(cond[ok])) if ok.any() else np.nan


# ----------------------------------------------------------------------------
# Skill features (plan §4.3)

SKILL_FEATURES: tuple[str, ...] = (
    "foot_path_rel_per_s",
    "body_disp_h",
    "ball_disp_h",
    "ball_path_h",
    "foot_per_ball",
    "foot_rel_per_ball",
    "ball_speed_mean",
    "ball_speed_min",
    "ball_heading_change_max_deg",
    "ball_reversal",
    "facing_flips",
    "facing_ball_mismatch",
    "hip_sway_h",
    "ankle_ball_crossings",
    "ankle_lift_max_h",
    "ankle_on_ball_frac",
    "dragged_frac",
    "ball_lead_s",
    "ball_foot_dist_mean",
    "touches",
    "opponent_dist_min_h",
    "lean_std_deg",
    "body_speed_mean",
    "leg_conf",
    "pose_frac",
    "ball_observed_frac",
)


def skill_features(clip: ClipSeries, w: SkillWindow, cfg: WindowConfig | None = None) -> dict[str, float]:
    cfg = cfg or WindowConfig()
    P = _Body(clip.players[w.player_id], clip.t)
    seg = slice(w.start, w.end + 1)
    t = clip.t[seg]
    dur = max(1e-6, float(t[-1] - t[0]))
    H = _nanmedian(P.h[seg])
    ball = clip.ball[seg]
    obs = clip.ball_observed[seg]
    b_full = _interp_ball(clip)
    bi = b_full[seg]
    f: dict[str, float] = {}

    rel = P.ankle_rel_hip()[:, seg]
    foot_rel = np.nansum([_path(rel[0]), _path(rel[1])]) / H
    foot_abs = np.nansum([_path(P.ank[0][seg]), _path(P.ank[1][seg])]) / H
    hip = P.centre[seg]
    f["foot_path_rel_per_s"] = foot_rel / dur
    f["body_disp_h"] = _net(hip) / H
    b_obs = np.where(obs[:, None], ball, np.nan)
    f["ball_disp_h"] = _net(bi) / H
    f["ball_path_h"] = _path(b_obs) / H
    f["foot_per_ball"] = foot_abs / ((f["ball_path_h"] if not np.isnan(f["ball_path_h"]) else 0.0) + 0.3)
    f["foot_rel_per_ball"] = foot_rel / ((f["ball_disp_h"] if not np.isnan(f["ball_disp_h"]) else 0.0) + 0.3)

    vb = _vel(b_obs, t) / H
    sp = np.hypot(vb[:, 0], vb[:, 1])
    f["ball_speed_mean"] = _nanmean(sp)
    f["ball_speed_min"] = _nanmin(_rolling(sp, max(3, int(0.2 * clip.fps))))
    # Heading change between consecutive quarter-second chunks of ball motion.
    chunk = max(2, int(0.25 * clip.fps))
    heads = []
    for a in range(0, len(t) - chunk, chunk // 2 or 1):
        v = np.array([_nanmean(vb[a:a + chunk, 0]), _nanmean(vb[a:a + chunk, 1])])
        heads.append(v if np.hypot(*v) > 0.4 else np.array([np.nan, np.nan]))
    changes = [_angle_between(heads[i], heads[i + 1]) for i in range(len(heads) - 1)]
    changes += [_angle_between(heads[i], heads[i + 2]) for i in range(len(heads) - 2)]
    f["ball_heading_change_max_deg"] = _nanmax(np.array(changes)) if changes else np.nan
    vx = _rolling(vb[:, 0], chunk)
    strong = np.abs(vx) > 0.5
    signs = np.sign(vx[strong & ~np.isnan(vx)])
    f["ball_reversal"] = float(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0.0

    face = P.facing()[seg]
    fs = face[~np.isnan(face)]
    f["facing_flips"] = float(np.sum(fs[1:] != fs[:-1])) if len(fs) > 1 else np.nan
    moving = strong & ~np.isnan(face)
    f["facing_ball_mismatch"] = float(np.mean(np.sign(vx[moving]) != face[moving])) if moving.any() else np.nan

    ok = ~np.isnan(hip[:, 0])
    if ok.sum() >= 4:
        tt = t[ok]
        res_x = hip[ok, 0] - np.polyval(np.polyfit(tt, hip[ok, 0], 1), tt)
        res_y = hip[ok, 1] - np.polyval(np.polyfit(tt, hip[ok, 1], 1), tt)
        f["hip_sway_h"] = float(np.std(np.hypot(res_x, res_y))) / H
    else:
        f["hip_sway_h"] = np.nan

    cross = 0.0
    lift = []
    on_ball = []
    for leg in (0, 1):
        ank = P.ank[leg][seg]
        dx = ank[:, 0] - bi[:, 0]
        close = _dist(ank, bi) < 0.35 * H
        s = np.sign(dx[close & ~np.isnan(dx)])
        cross += float(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0.0
        lift.append((P.foot[seg, 1] - ank[:, 1]) / H)
        on_ball.append(_dist(ank, bi) < 0.12 * H)
    f["ankle_ball_crossings"] = cross
    f["ankle_lift_max_h"] = _nanmax(np.stack(lift))
    f["ankle_on_ball_frac"] = float(np.mean(on_ball[0] | on_ball[1]))
    # A drag-back moves the ball *under* the sole: the ball is travelling and
    # a foot is on it at the same time.  An ordinary touch is a single frame.
    moving_ball = sp > 0.5
    f["dragged_frac"] = float(np.mean((on_ball[0] | on_ball[1]) & moving_ball))
    # Which reverses first, the ball or the body.  In a drag-back the ball
    # comes back while the body still faces the old way; in an ordinary turn
    # the body turns and the ball goes with it.
    body_vx = _rolling(_vel(hip, t)[:, 0] / H, chunk)
    f["ball_lead_s"] = _reversal_lead(vx, body_vx, t)
    f["ball_foot_dist_mean"] = _nanmean(P.foot_to(b_full)[seg] / H)

    acc = np.hypot(*np.diff(vb, axis=0).T)
    f["touches"] = float(np.sum(acc > 1.2 * max(1.0, _nanmedian(acc) * 3))) if np.any(~np.isnan(acc)) else np.nan
    if w.opponent_id and w.opponent_id in clip.players:
        O = _Body(clip.players[w.opponent_id], clip.t)
        f["opponent_dist_min_h"] = _nanmin(_dist(P.centre[seg], O.centre[seg])) / H
    else:
        f["opponent_dist_min_h"] = np.nan
    lean = P.sho[seg] - P.hip[seg]
    with np.errstate(all="ignore"):
        ang = np.degrees(np.arctan2(lean[:, 0], -lean[:, 1]))
    f["lean_std_deg"] = float(np.nanstd(ang)) if np.any(~np.isnan(ang)) else np.nan
    vh = _vel(hip, t) / H
    f["body_speed_mean"] = _nanmean(np.hypot(vh[:, 0], vh[:, 1]))
    f["leg_conf"] = _nanmean(P.leg_conf[seg])
    f["pose_frac"] = float(np.mean(P.has_pose[seg]))
    f["ball_observed_frac"] = float(np.mean(obs))
    return f


def _reversal_lead(ball_vx: np.ndarray, body_vx: np.ndarray, t: np.ndarray, floor: float = 0.4) -> float:
    """Seconds by which the ball's first strong x-reversal precedes the body's.

    NaN when the ball never reverses; the window length when the body never
    does (the ball came back and the body has not followed yet).
    """
    def first_flip(v: np.ndarray) -> int | None:
        strong = np.flatnonzero(np.abs(np.nan_to_num(v)) > floor)
        if len(strong) < 2:
            return None
        s0 = np.sign(v[strong[0]])
        flipped = strong[np.sign(v[strong]) == -s0]
        return int(flipped[0]) if len(flipped) else None

    bi = first_flip(ball_vx)
    if bi is None:
        return np.nan
    hi = first_flip(body_vx)
    if hi is None:
        return float(t[-1] - t[bi])
    return float(t[hi] - t[bi])


def _net(xy: np.ndarray) -> float:
    ok = ~np.isnan(xy[:, 0])
    if ok.sum() < 2:
        return np.nan
    pts = xy[ok]
    return float(np.hypot(*(pts[-1] - pts[0])))


def _rolling(x: np.ndarray, k: int) -> np.ndarray:
    """Centred rolling mean ignoring NaN."""
    ok = ~np.isnan(x)
    ker = np.ones(2 * (k // 2) + 1)
    total = np.convolve(np.where(ok, x, 0.0), ker, mode="same")
    count = np.convolve(ok.astype(float), ker, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def contact_vector(f: dict[str, float], names: Iterable[str] = CONTACT_FEATURES) -> np.ndarray:
    return np.array([f.get(n, np.nan) for n in names], dtype=float)


def skill_vector(f: dict[str, float], names: Iterable[str] = SKILL_FEATURES) -> np.ndarray:
    return np.array([f.get(n, np.nan) for n in names], dtype=float)
