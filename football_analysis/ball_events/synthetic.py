"""Scripted ball-and-player trajectories with a known answer.

The rules are built and tested against these before real tracks exist, and
kept honest by them afterwards.  Each scenario is a short 1v1 moment -- a goal,
a save, a post, a knock past the defender, a pass from a feeder -- built as the
per-frame record the rules read, with the event it should produce.

The scene is a side view at 25 fps: the goal on the right, its mouth 120 px
tall (a 2 m goal, so 60 px per metre and a 13 px ball at the goal's depth),
players 105 px tall standing on the same ground line as the goal.

These are fixtures for logic, not evidence of accuracy: every trajectory is
clean by construction.  Noise, misses and occlusions are added explicitly
where a scenario is about them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from football_analysis.ball_events.state import (
    ABSENT, BRIDGED, DETECTED, ClipState, GoalGeometry, PlayerObs, _fill_short_gaps,
)

__all__ = ["Scene", "SCENARIOS", "Scenario", "to_frame_states"]

GROUND_Y = 425.0
PLAYER_H = 105.0
PLAYER_W = 40.0
BALL_D = 13.0
GOAL_BOX = (1100.0, 305.0, 1180.0, GROUND_Y)

XY = tuple[float, float]


@dataclass
class Scene:
    duration_s: float
    fps: float = 25.0
    size: tuple[int, int] = (1280, 720)
    goal_box: tuple[float, float, float, float] | None = GOAL_BOX
    players: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    """player_id -> ``(t, foot_x)`` waypoints; everyone stands on the ground line."""
    ball: list[tuple[float, float, Callable[[float], XY]]] = field(default_factory=list)
    """``(t0, t1, f)`` -- the ball is at ``f(t)`` for ``t0 <= t < t1``."""
    hidden: list[tuple[float, float]] = field(default_factory=list)
    ball_diameter: Callable[[float], float] | None = None
    net_spikes: list[float] = field(default_factory=list)
    net_reading: bool = True
    noise_px: float = 0.0
    drop_rate: float = 0.0
    """Fraction of frames the detector misses the ball in, at random."""
    fill_s: float = 0.3
    """Gaps up to this long are bridged, as a raw per-frame tracker's are."""
    seed: int = 0

    # -- scripting ----------------------------------------------------------------

    def foot(self, pid: str, t: float) -> XY:
        pts = self.players[pid]
        ts = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        return (float(np.interp(t, ts, xs)), GROUND_Y)

    def at_feet(self, pid: str, ahead: float = 0.25, wobble: float = 0.0) -> Callable[[float], XY]:
        """The ball just ahead of a player's feet, in the direction they face."""
        def f(t: float) -> XY:
            x, y = self.foot(pid, t)
            direction = 1.0 if self._facing(pid, t) >= 0 else -1.0
            off = (ahead + wobble * np.sin(t * 9.0)) * PLAYER_W
            return (x + direction * (PLAYER_W / 2 + off), y - BALL_D / 2)
        return f

    def _facing(self, pid: str, t: float) -> float:
        a, b = self.foot(pid, t - 0.05)[0], self.foot(pid, t + 0.05)[0]
        return b - a if abs(b - a) > 1e-6 else 1.0

    def kick(self, t0: float, t1: float, p0: XY | Callable[[float], XY], p1: XY,
             arc: float = 0.0, ease: float = 0.0) -> Callable[[float], XY]:
        """Straight line from ``p0`` at ``t0`` to ``p1`` at ``t1``, lifted by
        ``arc`` px at the midpoint; ``ease`` > 0 slows it toward the end."""
        start = p0(t0) if callable(p0) else p0

        def f(t: float) -> XY:
            u = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
            if ease:
                u = (1 - ease) * u + ease * (1 - (1 - u) ** 2)
            x = start[0] + (p1[0] - start[0]) * u
            y = start[1] + (p1[1] - start[1]) * u - arc * 4 * u * (1 - u)
            return (x, y)
        return f

    @staticmethod
    def still(p: XY) -> Callable[[float], XY]:
        return lambda t: p

    # -- building ---------------------------------------------------------------

    def build(self) -> ClipState:
        rng = np.random.default_rng(self.seed)
        n = int(round(self.duration_s * self.fps))
        ts = np.arange(n) / self.fps
        xy = np.full((n, 2), np.nan)
        diam = np.full(n, np.nan)
        for k, t in enumerate(ts):
            if any(a <= t < b for a, b in self.hidden):
                continue
            for t0, t1, f in self.ball:
                if t0 <= t < t1:
                    xy[k] = f(t)
                    diam[k] = self.ball_diameter(t) if self.ball_diameter else BALL_D
                    break
        if self.noise_px:
            xy += rng.normal(0, self.noise_px, xy.shape)
            diam += rng.normal(0, 0.08 * BALL_D, n)
        if self.drop_rate:
            missed = rng.random(n) < self.drop_rate
            xy[missed] = np.nan
            diam[missed] = np.nan
        players = []
        for t in ts:
            row = []
            for i, pid in enumerate(self.players):
                x, y = self.foot(pid, t)
                row.append(PlayerObs(pid, (x - PLAYER_W / 2, y - PLAYER_H, x + PLAYER_W / 2, y),
                                     track_id=i + 1))
            players.append(row)
        # Leaving the frame means leaving it: no position outside the image.
        w, h = self.size
        outside = (xy[:, 0] < 0) | (xy[:, 0] > w) | (xy[:, 1] < 0) | (xy[:, 1] > h)
        xy[outside] = np.nan
        diam[outside] = np.nan
        src = np.where(np.isfinite(xy).all(axis=1), DETECTED, ABSENT)
        if self.fill_s:
            _fill_short_gaps(ts, xy, src, self.fill_s)
        goal = GoalGeometry.from_bbox(self.goal_box) if self.goal_box else None
        net = None
        if self.net_reading and goal is not None:
            net = 0.4 + 0.1 * rng.standard_normal(n)
            for t_spike in self.net_spikes:
                sel = (ts >= t_spike) & (ts < t_spike + 0.3)
                net[sel] += 6.0
        return ClipState(
            timestamps=ts, frame_indices=np.arange(n), ball_xy=xy,
            ball_source=src,
            ball_diameter=diam, players=players, frame_size=self.size,
            goal=goal, net_motion=net,
        )


@dataclass
class Scenario:
    name: str
    scene: Scene
    expect: dict
    """What the rules must produce:

    ``"<type>": n``               exactly ``n`` events of that type
    ``"<type>.<key>": v``         some event of that type has ``detail[key]``
                                  equal to ``v`` (or in ``v``, for a tuple)
    ``"<type>.receiver": pid``    some event of that type names ``pid`` second
    ``"<type>.confidence_max": c``  every event of that type is at most ``c``
    ``"needs_review": bool``      whether anything was flagged for review
    """
    rules: dict = field(default_factory=dict)
    """Overrides for :class:`BallEventConfig`."""

    def failures(self, result) -> list[str]:
        """The expectations ``result`` (a ``BallEventResult``) does not meet."""
        counts: dict[str, int] = {}
        for e in result.events:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        bad: list[str] = []
        for key, want in self.expect.items():
            if key == "needs_review":
                ok = bool(result.review) == want
            elif "." in key:
                etype, field_name = key.split(".", 1)
                evs = [e for e in result.events if e.type.value == etype]
                if field_name == "confidence_max":
                    ok = bool(evs) and all(e.confidence <= want for e in evs)
                elif field_name == "receiver":
                    ok = any(e.secondary_player_id == want for e in evs)
                else:
                    allowed = want if isinstance(want, tuple) else (want,)
                    ok = any(e.detail.get(field_name) in allowed for e in evs)
            else:
                ok = counts.get(key, 0) == want
            if not ok:
                bad.append(key)
        return bad


def _dribble_then(scene: Scene, pid: str, until: float) -> None:
    scene.ball.append((0.0, until, scene.at_feet(pid, wobble=0.2)))


def goal() -> Scenario:
    s = Scene(4.0)
    s.players = {"Player 1": [(0, 500), (2.0, 850), (4.0, 900)],
                 "Player 2": [(0, 980), (4.0, 960)]}
    _dribble_then(s, "Player 1", 2.0)
    s.ball.append((2.0, 2.36, s.kick(2.0, 2.36, s.at_feet("Player 1"), (1140, 360), arc=30)))
    s.ball.append((2.36, 2.6, s.kick(2.36, 2.6, (1140, 360), (1158, 412), ease=0.6)))
    s.ball.append((2.6, 4.0, s.still((1158, 412))))
    s.net_spikes = [2.40]
    return Scenario("goal", s, {"shot": 1, "goal": 1, "shot.outcome": "goal"})


def saved() -> Scenario:
    s = Scene(4.0)
    s.players = {"Player 1": [(0, 500), (2.0, 850), (4.0, 880)],
                 "Player 2": [(0, 1080), (4.0, 1080)]}
    _dribble_then(s, "Player 1", 2.0)
    catch = s.at_feet("Player 2", ahead=-0.9)
    s.ball.append((2.0, 2.3, s.kick(2.0, 2.3, s.at_feet("Player 1"), catch(2.3), arc=10)))
    s.ball.append((2.3, 4.0, s.at_feet("Player 2", ahead=-0.9)))
    return Scenario("saved", s, {"shot": 1, "goal": 0, "shot.outcome": "saved"})


def post() -> Scenario:
    s = Scene(4.0)
    s.players = {"Player 1": [(0, 500), (2.0, 850), (4.0, 870)],
                 "Player 2": [(0, 300), (4.0, 300)]}
    _dribble_then(s, "Player 1", 2.0)
    s.ball.append((2.0, 2.3, s.kick(2.0, 2.3, s.at_feet("Player 1"), (1103, 330), arc=20)))
    s.ball.append((2.3, 2.9, s.kick(2.3, 2.9, (1103, 330), (820, 380), arc=-10)))
    s.ball.append((2.9, 4.0, s.kick(2.9, 4.0, (820, 380), (760, 419), ease=0.9)))
    s.net_spikes = []
    return Scenario("post", s, {"shot": 1, "goal": 0, "shot.outcome": ("rebound", "woodwork")})


def push_past() -> Scenario:
    s = Scene(4.0)
    s.goal_box = GOAL_BOX
    s.players = {"Player 1": [(0, 400), (1.5, 600), (1.7, 600), (2.6, 880), (4.0, 1000)],
                 "Player 2": [(0, 720), (1.8, 720), (4.0, 760)]}
    _dribble_then(s, "Player 1", 1.6)
    p1 = s.at_feet("Player 1")
    s.ball.append((1.6, 2.5, s.kick(1.6, 2.5, p1, (905, GROUND_Y - BALL_D / 2), ease=0.5)))
    s.ball.append((2.5, 4.0, s.at_feet("Player 1", wobble=0.2)))
    return Scenario("push_past", s, {"push_past": 1, "pass": 0, "shot": 0},
                    rules={})


def feeder_pass() -> Scenario:
    s = Scene(4.0, goal_box=None)
    s.players = {"Feeder": [(0, 200), (4.0, 200)],
                 "Player 1": [(0, 650), (4.0, 700)],
                 "Player 2": [(0, 950), (4.0, 950)]}
    s.ball.append((0.0, 1.0, s.at_feet("Feeder", ahead=0.2)))
    s.ball.append((1.0, 1.9, s.kick(1.0, 1.9, s.at_feet("Feeder", ahead=0.2), (640, GROUND_Y - BALL_D / 2))))
    s.ball.append((1.9, 4.0, s.at_feet("Player 1", wobble=0.2)))
    return Scenario("feeder_pass", s, {"pass": 1, "pass.receiver": "Player 1"},
                    rules={"feeders": ["Feeder"]})


def turnover() -> Scenario:
    s = Scene(4.0)
    s.players = {"Player 1": [(0, 500), (4.0, 560)],
                 "Player 2": [(0, 820), (4.0, 800)]}
    _dribble_then(s, "Player 1", 1.5)
    s.ball.append((1.5, 2.2, s.kick(1.5, 2.2, s.at_feet("Player 1"), (785, GROUND_Y - BALL_D / 2))))
    s.ball.append((2.2, 4.0, s.at_feet("Player 2", ahead=-0.2)))
    return Scenario("turnover", s, {"pass": 0, "shot": 0, "possession_change": 1,
                                     "possession_change.cause": "interception"})


def wide() -> Scenario:
    s = Scene(3.5)
    s.players = {"Player 1": [(0, 500), (2.0, 850), (3.5, 870)],
                 "Player 2": [(0, 300), (3.5, 300)]}
    _dribble_then(s, "Player 1", 2.0)
    s.ball.append((2.0, 2.6, s.kick(2.0, 2.6, s.at_feet("Player 1"), (1300, 150), arc=40)))
    return Scenario("wide", s, {"shot": 1, "goal": 0, "shot.outcome": "off_target"})


def goal_in_gap() -> Scenario:
    sc = goal()
    s = sc.scene
    s.hidden = [(2.28, 4.0)]
    s.net_reading = False
    return Scenario("goal_in_gap", s, {"goal": 0, "shot": 1, "shot.outcome": "unresolved",
                                        "needs_review": True})


def hidden_touch() -> Scenario:
    """Player 2 redirects a pass while the detector has lost the ball."""
    s = Scene(4.0, goal_box=None, fill_s=0.5)
    s.players = {"Player 1": [(0, 400), (4.0, 420)], "Player 2": [(0, 800), (4.0, 800)]}
    s.ball.append((0.0, 1.2, s.at_feet("Player 1", wobble=0.2)))
    s.ball.append((1.2, 2.0, s.kick(1.2, 2.0, s.at_feet("Player 1"), (790, GROUND_Y - BALL_D / 2))))
    s.ball.append((2.0, 4.0, s.kick(2.0, 3.2, (790, GROUND_Y - BALL_D / 2), (1000, 150))))
    s.hidden = [(1.84, 2.2)]
    return Scenario("hidden_touch", s, {"possession_change": 1,
                                         "possession_change.cause": ("interception", "loose_ball")})


def goal_no_net() -> Scenario:
    sc = goal()
    sc.scene.net_reading = False
    return Scenario("goal_no_net", sc.scene, {"goal": 1, "shot": 1, "goal.confidence_max": 0.9})


def behind_goal() -> Scenario:
    """A ball struck across the face of the goal, far behind it: it overlaps
    the goal in the image but is half the size the goal's depth predicts."""
    sc = goal()
    s = sc.scene
    s.ball_diameter = lambda t: 6.5 if t >= 2.2 else BALL_D
    s.net_spikes = []
    return Scenario("behind_goal", s, {"goal": 0, "shot": 1})


def dribble() -> Scenario:
    s = Scene(4.0)
    s.players = {"Player 1": [(0, 300), (4.0, 800)],
                 "Player 2": [(0, 1000), (4.0, 1000)]}
    s.ball.append((0.0, 4.0, s.at_feet("Player 1", wobble=0.3)))
    return Scenario("dribble", s, {"pass": 0, "shot": 0, "goal": 0, "push_past": 0,
                                   "possession_change": 0})


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    f.__name__: f for f in (goal, saved, post, push_past, feeder_pass, turnover, wide,
                            goal_in_gap, hidden_touch, goal_no_net, behind_goal, dribble)
}


def to_frame_states(clip: ClipState) -> list:
    """The same clip as the pipeline's per-frame records, for end-to-end tests."""
    from football_analysis.state import BallState, FrameState, GoalState, PlayerState
    from football_analysis.types import BBox, Point

    goal_bbox = None
    if clip.goal is not None:
        q = clip.goal.quad
        goal_bbox = BBox(float(q[:, 0].min()), float(q[:, 1].min()),
                         float(q[:, 0].max()), float(q[:, 1].max()))
    out = []
    for k in range(len(clip)):
        ball = None
        if clip.present[k]:
            x, y = (float(v) for v in clip.ball_xy[k])
            d = clip.ball_diameter[k]
            r = (d if np.isfinite(d) else BALL_D) / 2
            ball = BallState(position=Point(x, y), confidence=0.8,
                             bbox=BBox(x - r, y - r, x + r, y + r),
                             interpolated=bool(clip.ball_source[k] in (BRIDGED,) or not np.isfinite(d)),
                             track_id=99)
        attrs = {}
        if clip.net_motion is not None and np.isfinite(clip.net_motion[k]):
            attrs["net_motion"] = float(clip.net_motion[k])
        out.append(FrameState(
            frame_index=int(clip.frame_indices[k]),
            timestamp_s=float(clip.timestamps[k]),
            players=[PlayerState(p.player_id, p.track_id or 0, BBox(*p.bbox), p.confidence)
                     for p in clip.players[k]],
            ball=ball,
            goal=GoalState(bbox=goal_bbox, confidence=0.9) if goal_bbox is not None else None,
            frame_size=clip.frame_size,
            attributes=attrs,
        ))
    return out
