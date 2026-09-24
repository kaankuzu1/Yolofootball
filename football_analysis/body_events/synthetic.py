"""Simulated 1v1 sequences with known tackles, duels, dribbles and tricks.

Why this exists.  The classifier needs labelled examples, and at the time of
writing there is no labelled side-angle 1v1 footage available to this project:
public tackle and trick footage lives on video sites this build cannot reach,
and the broadcast clips that can be reached show players too small and too
rarely in a 1v1 to label in any number.  So the first model is trained on
motion simulated from a kinematic stick figure, and this module is that
simulator.

What it is not.  It is not evidence of real-world accuracy.  A classifier
trained here learns the motion signatures written into these scripts -- the
plan's own hypotheses (§4.2, §4.3) about what a tackle and a trick look like --
and nothing about how real players actually move.  Numbers measured on this
data say the pipeline works and the features separate what they were designed
to separate; real accuracy comes only from labelled real windows (see
:mod:`football_analysis.body_events.dataset`).

How it works.  Each scenario scripts two bodies (attacker ``A`` and defender
``D``) and the ball over ~4 s in *world* units of one player height, then
renders them through a simple side-angle camera into
:class:`~football_analysis.state.FrameState` records carrying COCO-17
keypoints, boxes and a ball -- exactly what the real pipeline produces --
with pose noise, left/right swaps, dropouts, occlusion during contact, ball
detection misses (coasted and flagged ``interpolated``), camera pan, and a
spread of frame rates and player sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from football_analysis.state import BallState, FrameState, Keypoint, PlayerState
from football_analysis.types import BBox, Point

__all__ = [
    "SCENARIOS",
    "SENSOR",
    "HARD_SENSOR",
    "seeds_for",
    "CONTACT_SCENARIOS",
    "SKILL_SCENARIOS",
    "SyntheticSequence",
    "generate",
    "generate_many",
]

_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

THIGH = 0.245
SHIN = 0.245
TORSO = 0.29
BALL_R = 0.062  # a 22 cm ball against a 1.8 m player, as a radius
DEPTH_TO_Y = 0.35  # how much one height of depth moves a foot up the image


@dataclass
class SyntheticSequence:
    """One simulated clip and its ground truth."""

    states: list[FrameState]
    scenario: str
    contact_label: str | None
    """``tackle_won`` / ``tackle_failed`` / ``shoulder_duel`` / ``no_contact`` for
    the pair's contact episode, ``None`` when the scenario has no contact."""
    skill_label: str | None
    """``trick`` / ``dribble`` for the attacker's time on the ball."""
    sub_label: str | None
    """The named move, for when named classes are trained (``stepover`` ...)."""
    event_start_s: float
    event_end_s: float
    """The interval the labelled action occupies."""
    attacker_id: str
    defender_id: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# The body


@dataclass
class _Track:
    """Per-frame body parameters for one player, in world units."""

    n: int
    x: np.ndarray = None  # type: ignore[assignment]
    z: np.ndarray = None  # type: ignore[assignment]
    f: np.ndarray = None  # type: ignore[assignment]  facing, +1 / -1 in x
    lean: np.ndarray = None  # type: ignore[assignment]
    yaw: np.ndarray = None  # type: ignore[assignment]
    a: np.ndarray = None  # type: ignore[assignment]  (n, 2) thigh angles L, R
    k: np.ndarray = None  # type: ignore[assignment]  (n, 2) knee flexion L, R
    arm: np.ndarray = None  # type: ignore[assignment]  (n, 2) arm swing L, R
    foot_dz: np.ndarray = None  # type: ignore[assignment]  (n, 2) depth offset of each foot
    drop: np.ndarray = None  # type: ignore[assignment]  extra hip drop (slide)

    def __post_init__(self) -> None:
        n = self.n
        self.x = np.zeros(n)
        self.z = np.zeros(n)
        self.f = np.ones(n)
        self.lean = np.full(n, 0.08)
        self.yaw = np.full(n, 0.4)
        self.a = np.zeros((n, 2))
        self.k = np.full((n, 2), 0.1)
        self.arm = np.zeros((n, 2))
        self.foot_dz = np.zeros((n, 2))
        self.drop = np.zeros(n)


def _gait(tr: _Track, speed: np.ndarray, dt: float, rng: np.random.Generator,
          amp_scale: float = 1.0) -> None:
    """Fill leg and arm angles with a running gait at the given speeds."""
    phase = rng.uniform(0, 2 * np.pi)
    for i in range(tr.n):
        s = float(abs(speed[i]))
        cadence = 1.1 + 0.35 * min(s, 5.0)
        if s > 0.15:
            phase += 2 * np.pi * cadence * dt
        amp = amp_scale * min(0.62, 0.08 + 0.13 * s)
        sl = np.sin(phase)
        tr.a[i, 0] = amp * sl
        tr.a[i, 1] = -amp * sl
        swing = 0.25 + 0.14 * min(s, 5.0)
        tr.k[i, 0] = 0.12 + swing * max(0.0, np.cos(phase))
        tr.k[i, 1] = 0.12 + swing * max(0.0, -np.cos(phase))
        tr.arm[i, 0] = -0.9 * amp * sl
        tr.arm[i, 1] = 0.9 * amp * sl
        tr.lean[i] = 0.05 + 0.035 * min(s, 5.0)


def _skeleton(tr: _Track, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Joint positions (17, 2) relative to the ground point, y down, plus the
    per-joint depth offsets (17,) used to place near and far limbs."""
    f = tr.f[i]
    lean = tr.lean[i]
    yaw = tr.yaw[i]
    joints = np.zeros((17, 2))
    depth = np.zeros(17)

    ext = []
    for leg in (0, 1):
        a, k = tr.a[i, leg], tr.k[i, leg]
        ext.append(THIGH * np.cos(a) + SHIN * np.cos(a - k))
    hip_y = -(max(ext) + 0.03) + tr.drop[i]
    hip = np.array([0.0, hip_y])

    # Facing +x, the player's right side is toward the camera.
    near = {1: 1, -1: 0}[int(np.sign(f) or 1)]  # index (0=L, 1=R) of the near side
    half_hip = 0.06 * np.sin(yaw)
    half_sho = 0.10 * np.sin(yaw)

    for leg, (hi, ki, ai) in enumerate(((11, 13, 15), (12, 14, 16))):
        side = 1.0 if leg == near else -1.0
        a, k = tr.a[i, leg], tr.k[i, leg]
        h = hip + np.array([-f * side * half_hip * 0.3, 0.0])
        knee = h + THIGH * np.array([f * np.sin(a), np.cos(a)])
        ankle = knee + SHIN * np.array([f * np.sin(a - k), np.cos(a - k)])
        ankle[1] = min(ankle[1], -0.01)
        joints[hi], joints[ki], joints[ai] = h, knee, ankle
        dz = side * 0.05 * np.cos(yaw) + tr.foot_dz[i, leg]
        depth[hi], depth[ki], depth[ai] = side * 0.05, 0.5 * (side * 0.05 + dz), dz

    sho = hip + TORSO * np.array([f * np.sin(lean), -np.cos(lean)])
    for arm_i, (si, ei, wi) in enumerate(((5, 7, 9), (6, 8, 10))):
        side = 1.0 if arm_i == near else -1.0
        s = sho + np.array([-f * side * half_sho * 0.4, 0.0])
        b = tr.arm[i, arm_i]
        elbow = s + 0.16 * np.array([f * np.sin(b), np.cos(b)])
        wrist = elbow + 0.14 * np.array([f * np.sin(b + 0.5), np.cos(b + 0.5)])
        joints[si], joints[ei], joints[wi] = s, elbow, wrist
        depth[si] = depth[ei] = depth[wi] = side * 0.1 * np.cos(yaw)

    head = sho + 0.105 * np.array([f * np.sin(lean + 0.15), -np.cos(lean)])
    joints[0] = head + np.array([f * 0.035, 0.012])
    joints[1] = head + np.array([f * 0.026, -0.012])
    joints[2] = head + np.array([f * 0.022, -0.010])
    joints[3] = head + np.array([-f * 0.012, -0.004])
    joints[4] = head + np.array([-f * 0.016, -0.002])
    depth[[1, 3]] = 0.03 * (1 if near == 0 else -1)
    depth[[2, 4]] = -depth[1]
    return joints, depth


# --------------------------------------------------------------------------
# Ball kinematics


def _dribble_ball(tA: _Track, t: np.ndarray, d: float, rng: np.random.Generator,
                  t0: float, t1: float, bx: np.ndarray, bz: np.ndarray,
                  touches: list[float], lead_leg: int | None = None) -> None:
    """Ball kept ahead of the attacker between ``t0`` and ``t1`` with touches.

    Between touches the ball runs ahead and the player catches it up, so the
    ball's velocity relative to the player jumps at every touch -- the same
    signature a real dribble leaves.
    """
    reach = rng.uniform(0.25, 0.45)
    gap = rng.uniform(0.3, 0.65)
    ts = t0 + rng.uniform(0, 0.15)
    times = []
    while ts < t1:
        times.append(ts)
        ts += gap * rng.uniform(0.85, 1.15)
    # One frame past t1, so the frame a caller finds with searchsorted(t, t1)
    # already holds the ball; scripts continue from there.
    # A touch is never cut short: the last run of the ball lasts a full gap
    # even past t1 (callers overwrite what follows t1 anyway).
    times.append(max(t1 + (t[1] - t[0]) + 1e-6, times[-1] + gap))
    pre = (t >= t0) & (t < times[0])
    bx[pre] = tA.x[pre] + d * 0.14
    bz[pre] = tA.z[pre] + 0.04
    leg = lead_leg if lead_leg is not None else int(rng.integers(0, 2))
    for j in range(len(times) - 1):
        a, b = times[j], times[j + 1]
        m = (t >= a) & (t < b)
        tau = (t[m] - a) / max(1e-6, b - a)
        off = 0.14 + reach * np.sin(np.pi * tau)
        bx[m] = tA.x[m] + d * off
        bz[m] = tA.z[m] + 0.04
        touches.append(a)
        _kick(tA, t, a, leg if rng.random() < 0.7 else 1 - leg, rng.uniform(0.25, 0.45))


def _kick(tr: _Track, t: np.ndarray, at: float, leg: int, amp: float, width: float = 0.09) -> None:
    """A quick forward swing of one leg centred on ``at``."""
    w = np.exp(-0.5 * ((t - at) / width) ** 2)
    tr.a[:, leg] = (1 - w) * tr.a[:, leg] + w * amp
    tr.k[:, leg] = (1 - w) * tr.k[:, leg] + w * 0.15


def _roll(bx: np.ndarray, bz: np.ndarray, t: np.ndarray, i0: int, vx: float, vz: float,
          friction: float = 1.2) -> None:
    """Free rolling ball from frame ``i0`` on, decelerating."""
    x0, z0 = bx[i0], bz[i0]
    for i in range(i0, len(t)):
        s = t[i] - t[i0]
        decay = (1 - np.exp(-friction * s)) / friction
        bx[i] = x0 + vx * decay
        bz[i] = z0 + vz * decay


def _ease(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """0 before ``a``, 1 after ``b``, smooth in between."""
    u = np.clip((t - a) / max(1e-6, b - a), 0, 1)
    return u * u * (3 - 2 * u)


def _integrate(v: np.ndarray, dt: float, x0: float) -> np.ndarray:
    return x0 + np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])


# --------------------------------------------------------------------------
# Scenarios.  Each returns (tA, tD, bx, bz, meta) in world units.


def _setup(rng: np.random.Generator, n: int, t: np.ndarray):
    d = float(rng.choice([-1.0, 1.0]))
    vA = rng.uniform(1.2, 4.0)
    te = rng.uniform(1.6, 2.4)
    tA, tD = _Track(n), _Track(n),
    tA.f[:] = d
    tA.yaw[:] = rng.uniform(0.2, 0.9)
    tD.yaw[:] = rng.uniform(0.2, 0.9)
    tA.z[:] = rng.uniform(-0.3, 0.3)
    return d, vA, te, tA, tD


def _jockey(tD: _Track, tA: _Track, t: np.ndarray, d: float, dt: float,
            rng: np.random.Generator, gap_start: float, gap_end: float,
            t_close: float, facing: float | None = None) -> None:
    """Defender facing the attacker, closing the gap from ``gap_start`` to
    ``gap_end`` by ``t_close`` and backing off at the attacker's pace."""
    gap = gap_start + (gap_end - gap_start) * _ease(t, 0.0, t_close)
    tD.x[:] = tA.x + d * gap
    tD.f[:] = -d if facing is None else facing
    speed = np.gradient(tD.x, dt)
    _gait(tD, speed, dt, rng, amp_scale=0.8)


def _lunge(tD: _Track, t: np.ndarray, tc: float, leg: int, rng: np.random.Generator,
           slide: bool) -> float:
    """The tackling leg extends toward the ball, arriving at ``tc``.

    Returns how far ahead of the hip the foot reaches, in heights.
    """
    dur = rng.uniform(0.14, 0.3)
    a_max = rng.uniform(0.85, 1.35)
    up = _ease(t, tc - dur, tc)
    down = 1 - _ease(t, tc + 0.18, tc + 0.5)
    w = up * down
    other = 1 - leg
    tD.a[:, leg] = (1 - w) * tD.a[:, leg] + w * a_max
    tD.k[:, leg] = (1 - w) * tD.k[:, leg] + w * 0.04
    tD.a[:, other] = (1 - w) * tD.a[:, other] + w * rng.uniform(-0.45, -0.15)
    tD.k[:, other] = (1 - w) * tD.k[:, other] + w * rng.uniform(0.6, 1.1)
    tD.lean[:] = (1 - w) * tD.lean + w * rng.uniform(-0.25, 0.35)
    if slide:
        tD.a[:, other] = (1 - w) * tD.a[:, other] + w * 1.2
        tD.k[:, other] = (1 - w) * tD.k[:, other] + w * 1.4
        tD.drop[:] = w * rng.uniform(0.15, 0.3)
        tD.lean[:] = (1 - w) * tD.lean + w * -0.5
    return float(THIGH * np.sin(a_max) + SHIN * np.sin(a_max - 0.04))


def _attacker_run(tA: _Track, t: np.ndarray, dt: float, d: float, vA: float,
                  rng: np.random.Generator, speed_mod: np.ndarray | None = None) -> np.ndarray:
    speed = np.full(len(t), vA) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t))
    if speed_mod is not None:
        speed = speed * speed_mod
    tA.x[:] = _integrate(d * speed, dt, -d * vA * 2.0)
    _gait(tA, speed, dt, rng)
    return speed


def _sc_tackle(rng: np.random.Generator, t: np.ndarray, dt: float, won: bool):
    n = len(t)
    d, vA, tc, tA, tD = _setup(rng, n, t)
    slide = won and rng.random() < 0.2
    # The attacker is slowed by the contact, and stumbles if the tackle is won.
    mod = np.ones(n)
    if won:
        mod = 1 - 0.7 * _ease(t, tc, tc + 0.4)
    else:
        mod = 1 - 0.25 * _ease(t, tc - 0.3, tc) * (1 - _ease(t, tc + 0.1, tc + 0.6))
    _attacker_run(tA, t, dt, d, vA, rng, mod)
    if won:
        tA.lean[:] += _ease(t, tc, tc + 0.2) * (1 - _ease(t, tc + 0.4, tc + 0.9)) * rng.uniform(-0.3, 0.4)

    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    ic = int(np.searchsorted(t, tc))
    leg = int(rng.integers(0, 2))

    if won:
        _dribble_ball(tA, t, d, rng, 0.0, t[-1], bx, bz, touches)
        # Defender's foot meets the ball at tc.
        _jockey(tD, tA, t, d, dt, rng, rng.uniform(2.5, 4.0), 0.9, tc - 0.25)
        reach = _lunge(tD, t, tc, leg, rng, slide)
        tD.x[ic:] = tD.x[ic]  # planted at contact
        shift = bx[ic] - (tD.x[ic] + tD.f[ic] * reach)
        tD.x[:ic] += shift * _ease(t[:ic], tc - 0.6, tc)
        tD.x[ic:] += shift
        tD.z[:] = bz[ic] + rng.uniform(-0.12, 0.08)
        tD.foot_dz[:, leg] = (bz[ic] - tD.z[ic]) * _ease(t, tc - 0.3, tc)
        # Deflection: anywhere except on along the attacker's line.
        ang = rng.uniform(0.6, 2 * np.pi - 0.6)
        speed = rng.uniform(1.5, 6.0)
        vx = speed * np.cos(ang) * d
        vz = speed * np.sin(ang) * 0.6
        _roll(bx, bz, t, ic, vx, vz, friction=rng.uniform(0.6, 1.8))
        outcome = "defender" if rng.random() < 0.5 else "loose"
        if outcome == "defender":
            # Defender recovers and takes the ball away.
            it = min(n - 1, ic + int(rng.uniform(0.35, 0.6) / dt))
            follow = _ease(t, t[it], t[it] + 0.3)
            dir_x = np.sign(bx[-1] - tD.x[ic]) or 1.0
            tD.x[:] = (1 - follow) * tD.x + follow * (bx - dir_x * 0.2)
            tD.z[:] = (1 - follow) * tD.z + follow * bz
            tD.f[it:] = dir_x
        else:
            pass
        return tA, tD, bx, bz, dict(contact_s=tc, start=tc - 0.35, end=tc + 0.25,
                                    outcome=outcome, slide=slide, touches=touches, d=d)

    # Failed: the attacker plays the ball away just before the foot arrives.
    ta = tc - rng.uniform(0.08, 0.25)
    _dribble_ball(tA, t, d, rng, 0.0, ta, bx, bz, touches)
    ia = int(np.searchsorted(t, ta))
    _jockey(tD, tA, t, d, dt, rng, rng.uniform(2.5, 4.0), 0.9, tc - 0.25)
    # Where the ball would have been at tc had the attacker not played it.
    predicted = bx[ia] + d * vA * (tc - ta)
    reach = _lunge(tD, t, tc, leg, rng, False)
    ic = int(np.searchsorted(t, tc))
    target_x = tD.x[ic] + tD.f[ic] * reach
    shift = predicted - target_x
    tD.x[:ic] += shift * _ease(t[:ic], tc - 0.6, tc)
    tD.x[ic:] = tD.x[ic - 1] if ic > 0 else tD.x[ic]
    tD.z[:] = tA.z + rng.uniform(-0.12, 0.12)
    style = rng.choice(["around", "pullback"])
    if style == "around":
        # Knocked round the side of the lunge, so it is already off the
        # defender's line in depth by the time it passes them.
        vx = d * vA * rng.uniform(1.0, 1.5)
        vz = rng.choice([-1, 1]) * rng.uniform(1.6, 2.6)
    else:
        vx = -d * rng.uniform(0.8, 1.6)
        vz = rng.uniform(-0.4, 0.4)
    _roll(bx, bz, t, ia, vx, vz, friction=1.5)
    _kick(tA, t, ta, int(rng.integers(0, 2)), 0.4)
    # The attacker goes with the ball: rebuild the path after the touch.
    it = min(n - 1, ia + int(0.3 / dt))
    follow = _ease(t, t[ia], t[it])
    tA.x[:] = (1 - follow) * tA.x + follow * (bx - np.sign(vx) * 0.25)
    tA.z[:] = (1 - follow) * tA.z + follow * bz
    if style == "pullback":
        tA.f[it:] = np.sign(vx)
    # After the ball is regained, keep dribbling.
    after = t >= t[it] + 0.2
    if after.any():
        i2 = int(np.argmax(after))
        tail_speed = np.gradient(tA.x, dt)
        bx[i2:] = tA.x[i2:] + np.sign(tail_speed[i2:] + 1e-9) * 0.3
        bz[i2:] = tA.z[i2:] + 0.04
    return tA, tD, bx, bz, dict(contact_s=tc, start=tc - 0.35, end=tc + 0.25,
                                outcome="attacker", style=style, touches=touches, d=d)


def _sc_shoulder(rng: np.random.Generator, t: np.ndarray, dt: float):
    n = len(t)
    d, vA, tc, tA, tD = _setup(rng, n, t)
    _attacker_run(tA, t, dt, d, vA, rng)
    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    _dribble_ball(tA, t, d, rng, 0.0, t[-1], bx, bz, touches)
    dur = rng.uniform(0.4, 1.0)
    chase = rng.random() < 0.7
    if chase:
        # Running alongside, arriving at the attacker's shoulder.
        lag = rng.uniform(1.5, 3.0) * (1 - _ease(t, 0.0, tc)) + rng.uniform(-0.15, 0.2)
        tD.x[:] = tA.x - d * lag
        tD.f[:] = d
        tD.z[:] = tA.z + (rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)) * (1 - _ease(t, 0.0, tc)) \
            + rng.choice([-1, 1]) * rng.uniform(0.04, 0.12)
    else:
        # Body to body from the front, then shouldering past.
        gap = rng.uniform(2.5, 4.0) * (1 - _ease(t, 0.0, tc)) + rng.uniform(0.05, 0.25)
        tD.x[:] = tA.x + d * gap
        tD.f[:] = -d
        tD.z[:] = tA.z + rng.choice([-1, 1]) * rng.uniform(0.1, 0.2)
    # Someone wins the shoulder battle and the other falls away behind or
    # peels off in depth: a duel is an episode, not the rest of the clip.
    sep = _ease(t, tc + dur, tc + dur + 0.6)
    tD.x[:] -= d * sep * rng.uniform(0.8, 1.6) * (1.0 if chase else -1.0) * (t - tc - dur).clip(0) * 1.5
    tD.z[:] += sep * rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)
    # Jostle: both bodies shove at a few hertz while in contact.
    w = _ease(t, tc - 0.1, tc) * (1 - _ease(t, tc + dur, tc + dur + 0.2))
    jf = rng.uniform(2.5, 5.0)
    tD.x[:] += w * 0.04 * np.sin(2 * np.pi * jf * t)
    tA.x[:] -= w * 0.03 * np.sin(2 * np.pi * jf * t)
    speed = np.gradient(tD.x, dt)
    _gait(tD, speed, dt, rng)
    tD.lean[:] += w * rng.uniform(0.05, 0.25)
    tA.lean[:] += w * rng.uniform(-0.1, 0.15)
    return tA, tD, bx, bz, dict(contact_s=tc, start=tc, end=tc + dur, outcome="attacker",
                                touches=touches, d=d)


def _sc_no_contact(rng: np.random.Generator, t: np.ndarray, dt: float):
    n = len(t)
    d, vA, tc, tA, tD = _setup(rng, n, t)
    _attacker_run(tA, t, dt, d, vA, rng)
    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    _dribble_ball(tA, t, d, rng, 0.0, t[-1], bx, bz, touches)
    # The defender is passed at a distance, in depth or in time.
    tD.z[:] = tA.z + rng.choice([-1, 1]) * rng.uniform(0.45, 1.1)
    xd = tA.x[int(np.searchsorted(t, tc))] + rng.uniform(-0.3, 0.3)
    drift = rng.uniform(-0.6, 0.6)
    tD.x[:] = xd + drift * (t - tc)
    tD.f[:] = -d
    speed = np.gradient(tD.x, dt)
    _gait(tD, speed, dt, rng, amp_scale=0.7)
    if rng.random() < 0.4:
        # A half-hearted step toward the ball that never gets near it.
        _kick(tD, t, tc, int(rng.integers(0, 2)), rng.uniform(0.3, 0.55), width=0.12)
    return tA, tD, bx, bz, dict(contact_s=tc, start=tc - 0.3, end=tc + 0.3, outcome="attacker",
                                touches=touches, d=d)


def _far_defender(tD: _Track, tA: _Track, t: np.ndarray, d: float, dt: float,
                  rng: np.random.Generator) -> None:
    """A defender who stays out of reach: jockeying ahead at 1.3-3 heights."""
    gap_end = rng.uniform(1.3, 3.0)
    _jockey(tD, tA, t, d, dt, rng, gap_end + rng.uniform(0.5, 1.5), gap_end, t[-1] * 0.6)
    tD.z[:] = tA.z + rng.uniform(-0.4, 0.4)


def _sc_dribble(rng: np.random.Generator, t: np.ndarray, dt: float, turn: bool = False):
    n = len(t)
    d, vA, te, tA, tD = _setup(rng, n, t)
    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    if not turn:
        wobble = 1 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t + rng.uniform(0, 6))
        _attacker_run(tA, t, dt, d, vA, rng, wobble)
        _dribble_ball(tA, t, d, rng, 0.0, t[-1], bx, bz, touches)
        # Gentle curve in depth.
        curve = rng.uniform(-0.3, 0.3) * _ease(t, te - 0.5, te + 0.5)
        tA.z[:] += curve
        bz[:] += curve
        _far_defender(tD, tA, t, d, dt, rng)
        return tA, tD, bx, bz, dict(start=te - 0.6, end=te + 0.6, touches=touches, d=d,
                                    variant="straight")
    # An ordinary turn: slow down, turn the body, then take the ball back the
    # other way with one ordinary touch.  Reversal without a feint.
    v_in = vA * (1 - _ease(t, te - 0.5, te))
    v_out = vA * 0.8 * _ease(t, te + 0.15, te + 0.6)
    vel = d * v_in - d * v_out
    tA.x[:] = _integrate(vel, dt, -d * vA * 1.5)
    _gait(tA, np.abs(vel), dt, rng)
    ie = int(np.searchsorted(t, te))
    tA.f[:ie] = d
    tA.f[ie:] = -d
    tA.yaw[:] += 0.6 * _ease(t, te - 0.2, te) * (1 - _ease(t, te, te + 0.2))
    _dribble_ball(tA, t, d, rng, 0.0, te - 0.1, bx, bz, touches)
    i0 = int(np.searchsorted(t, te - 0.1))
    bx[i0:ie] = bx[i0]
    bz[i0:ie] = bz[i0]
    _kick(tA, t, te, int(rng.integers(0, 2)), 0.3)
    b2x, b2z = np.zeros(n), np.zeros(n)
    t2: list[float] = []
    _dribble_ball(tA, t, -d, rng, te, t[-1], b2x, b2z, t2)
    blend = _ease(t, te, te + 0.25)
    bx[ie:] = (1 - blend[ie:]) * bx[i0] + blend[ie:] * b2x[ie:]
    bz[ie:] = (1 - blend[ie:]) * bz[i0] + blend[ie:] * b2z[ie:]
    touches.extend(t2)
    _far_defender(tD, tA, t, d, dt, rng)
    return tA, tD, bx, bz, dict(start=te - 0.6, end=te + 0.6, touches=touches, d=d, variant="turn")


def _sc_stepover(rng: np.random.Generator, t: np.ndarray, dt: float):
    n = len(t)
    d, vA, te, tA, tD = _setup(rng, n, t)
    reps = int(rng.integers(1, 4))
    cyc = rng.uniform(0.3, 0.48)
    t_end = te + reps * cyc
    slow = 1 - 0.92 * _ease(t, te - 0.35, te) * (1 - _ease(t, t_end, t_end + 0.35))
    speed = _attacker_run(tA, t, dt, d, vA, rng, slow)
    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    _dribble_ball(tA, t, d, rng, 0.0, te - 0.2, bx, bz, touches)
    i0 = int(np.searchsorted(t, te - 0.2))
    i1 = int(np.searchsorted(t, t_end))
    ie = int(np.searchsorted(t, te))
    creep = d * rng.uniform(0.0, 0.25)
    # The ball is brought to rest just ahead of where the body stops, and the
    # body is held just behind it while the feet go round it.
    rest = tA.x[ie] + d * rng.uniform(0.18, 0.28)
    settle = _ease(t[i0:i1], te - 0.2, te)
    drift = rest + creep * np.clip((t[i0:i1] - te) / max(1e-6, t_end - te), 0, 1)
    bx[i0:i1] = (1 - settle) * bx[i0] + settle * drift
    bz[i0:i1] = bz[i0]
    tA.x[ie:i1] = drift[ie - i0:] - d * 0.22
    leg = int(rng.integers(0, 2))
    for r in range(reps):
        c0 = te + r * cyc
        u = np.clip((t - c0) / cyc, 0, 1)
        on = (t >= c0) & (t < c0 + cyc)
        arc = np.sin(np.pi * u)
        tA.a[on, leg] = 0.1 + 0.55 * arc[on]
        tA.k[on, leg] = 0.2 + 0.9 * arc[on]
        # The foot sweeps across the ball in depth: inside to outside.
        tA.foot_dz[on, leg] = 0.18 * np.sin(2 * np.pi * u[on]) * rng.uniform(0.7, 1.3)
        tA.a[on, 1 - leg] = -0.05
        tA.k[on, 1 - leg] = 0.35
        # Hips and shoulders sell the fake.
        sway = np.sin(np.pi * u[on]) * (1 if r % 2 == 0 else -1)
        tA.z[on] += 0.08 * sway
        tA.lean[on] = 0.1 + 0.12 * sway
        tA.arm[on, 1 - leg] = 0.8 * arc[on]
        leg = 1 - leg if rng.random() < 0.85 else leg
    # Explode away, either direction.
    out_d = d if rng.random() < 0.7 else -d
    b2x, b2z = np.zeros(n), np.zeros(n)
    t2: list[float] = []
    acc = _ease(t, t_end, t_end + 0.4)
    vel = np.gradient(tA.x, dt) * (1 - acc) + out_d * vA * acc
    x_after = _integrate(vel, dt, tA.x[0])
    tA.x[i1:] = x_after[i1:] - x_after[i1] + tA.x[i1 - 1]
    tA.f[i1:] = out_d
    _dribble_ball(tA, t, out_d, rng, t_end, t[-1], b2x, b2z, t2)
    bx[i1:] = b2x[i1:]
    bz[i1:] = b2z[i1:]
    _gait_tail(tA, t, dt, rng, i1)
    touches.extend(t2)
    _far_defender(tD, tA, t, d, dt, rng)
    return tA, tD, bx, bz, dict(start=te - 0.1, end=t_end + 0.1, reps=reps, touches=touches, d=d)


def _gait_tail(tr: _Track, t: np.ndarray, dt: float, rng: np.random.Generator, i0: int) -> None:
    """Re-run the gait after frame ``i0`` from the rebuilt path."""
    speed = np.abs(np.gradient(tr.x, dt))
    tmp = _Track(len(t))
    _gait(tmp, speed, dt, rng)
    tr.a[i0:], tr.k[i0:], tr.arm[i0:] = tmp.a[i0:], tmp.k[i0:], tmp.arm[i0:]
    tr.lean[i0:] = tmp.lean[i0:]


def _sc_dragback(rng: np.random.Generator, t: np.ndarray, dt: float):
    n = len(t)
    d, vA, te, tA, tD = _setup(rng, n, t)
    fake = rng.random() < 0.5
    stop = 1 - 0.95 * _ease(t, te - 0.4, te)
    speed = _attacker_run(tA, t, dt, d, vA, rng, stop)
    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    _dribble_ball(tA, t, d, rng, 0.0, te - 0.15, bx, bz, touches)
    i0 = int(np.searchsorted(t, te - 0.15))
    ie = int(np.searchsorted(t, te))
    leg = int(rng.integers(0, 2))
    if fake:
        # A shooting back-swing and forward swing that never touches the ball.
        tf = te - rng.uniform(0.25, 0.4)
        w = np.exp(-0.5 * ((t - tf) / 0.08) ** 2)
        tA.a[:, leg] = (1 - w) * tA.a[:, leg] + w * rng.uniform(0.7, 1.0)
        wb = np.exp(-0.5 * ((t - tf + 0.15) / 0.07) ** 2)
        tA.a[:, leg] = (1 - wb) * tA.a[:, leg] + wb * -0.6
        tA.k[:, leg] = (1 - wb) * tA.k[:, leg] + wb * 1.2
    # Sole on the ball, then drag it back under the body.
    pull = rng.uniform(0.3, 0.45)
    rest = tA.x[ie] + d * rng.uniform(0.15, 0.25)
    settle = _ease(t[i0:ie + 1], te - 0.15, te)
    bx[i0:ie + 1] = (1 - settle) * bx[i0] + settle * rest
    bz[i0:ie + 1] = bz[i0]
    on_ball = (t >= te - 0.1) & (t < te + pull)
    u = np.clip((t - te) / pull, 0, 1)
    tA.a[on_ball, leg] = 0.45 - 0.75 * u[on_ball]
    tA.k[on_ball, leg] = 0.55
    ball_back = rng.uniform(0.5, 0.9)
    ip = int(np.searchsorted(t, te + pull))
    bx[ie:ip] = bx[ie] - d * ball_back * u[ie:ip]
    bz[ie:ip] = bz[ie]
    tA.lean[on_ball] = -0.12
    # Turn and go the other way.
    tA.f[ip:] = -d
    tA.yaw[:] += 0.5 * _ease(t, te + pull - 0.1, te + pull) * (1 - _ease(t, te + pull, te + pull + 0.2))
    acc = _ease(t, te + pull, te + pull + 0.4)
    vel = -d * vA * 0.9 * acc
    tA.x[ip:] = tA.x[ip - 1] + _integrate(vel, dt, 0.0)[ip:] - _integrate(vel, dt, 0.0)[ip]
    b2x, b2z = np.zeros(n), np.zeros(n)
    t2: list[float] = []
    _dribble_ball(tA, t, -d, rng, te + pull, t[-1], b2x, b2z, t2)
    blend = _ease(t, te + pull, te + pull + 0.2)
    bx[ip:] = (1 - blend[ip:]) * bx[ip - 1] + blend[ip:] * b2x[ip:]
    bz[ip:] = (1 - blend[ip:]) * bz[ip - 1] + blend[ip:] * b2z[ip:]
    _gait_tail(tA, t, dt, rng, ip)
    touches.extend(t2)
    _far_defender(tD, tA, t, d, dt, rng)
    start = te - (0.45 if fake else 0.15)
    return tA, tD, bx, bz, dict(start=start, end=te + pull + 0.15, fake=fake, touches=touches, d=d)


def _sc_feint(rng: np.random.Generator, t: np.ndarray, dt: float):
    """Body swerve / scissors: the leg goes round the ball, the hips sell one
    way, the ball is taken the other."""
    n = len(t)
    d, vA, te, tA, tD = _setup(rng, n, t)
    dur = rng.uniform(0.35, 0.6)
    slow = 1 - 0.6 * _ease(t, te - 0.3, te) * (1 - _ease(t, te + dur, te + dur + 0.3))
    _attacker_run(tA, t, dt, d, vA, rng, slow)
    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    _dribble_ball(tA, t, d, rng, 0.0, t[-1], bx, bz, touches)
    side = rng.choice([-1.0, 1.0])
    sell = _ease(t, te, te + dur * 0.5) * (1 - _ease(t, te + dur * 0.5, te + dur))
    tA.z[:] += side * 0.25 * sell
    tA.lean[:] += 0.2 * sell
    on = (t >= te) & (t < te + dur * 0.6)
    u = np.clip((t - te) / (dur * 0.6), 0, 1)
    leg = int(rng.integers(0, 2))
    tA.a[on, leg] = 0.2 + 0.5 * np.sin(np.pi * u[on])
    tA.k[on, leg] = 0.3 + 0.7 * np.sin(np.pi * u[on])
    tA.foot_dz[on, leg] = side * 0.2 * np.sin(np.pi * u[on])
    # Ball goes the other way in depth.
    cut = -side * rng.uniform(0.35, 0.8) * _ease(t, te + dur * 0.6, te + dur + 0.4)
    tA.z[:] += cut
    bz[:] += cut
    _far_defender(tD, tA, t, d, dt, rng)
    return tA, tD, bx, bz, dict(start=te - 0.05, end=te + dur + 0.1, touches=touches, d=d)


def _sc_nutmeg(rng: np.random.Generator, t: np.ndarray, dt: float):
    n = len(t)
    d, vA, te, tA, tD = _setup(rng, n, t)
    slow = 1 - 0.4 * _ease(t, te - 0.5, te)
    _attacker_run(tA, t, dt, d, vA * 0.7, rng, slow)
    bx, bz = np.zeros(n), np.zeros(n)
    touches: list[float] = []
    tp = te - rng.uniform(0.2, 0.35)  # the push
    _dribble_ball(tA, t, d, rng, 0.0, tp, bx, bz, touches)
    ip = int(np.searchsorted(t, tp))
    # Defender set wide in front, facing the attacker.
    ie = int(np.searchsorted(t, te))
    push_v = rng.uniform(2.0, 4.0)
    _roll(bx, bz, t, ip, d * push_v, 0.0, friction=1.0)
    tD.x[:] = bx[ie] + rng.uniform(-0.04, 0.04)
    tD.z[:] = bz[ie] - 0.03
    tD.f[:] = -d
    spread = rng.uniform(0.3, 0.5)
    tD.a[:, 0] = spread
    tD.a[:, 1] = -spread
    tD.k[:, :] = rng.uniform(0.25, 0.45)
    tD.lean[:] = 0.15
    tD.x[:] += 0.02 * np.sin(2 * np.pi * 1.5 * t)
    _kick(tA, t, tp, int(rng.integers(0, 2)), 0.45)
    # Attacker goes round and collects.
    around = rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 0.7)
    w = _ease(t, tp, te) * (1 - _ease(t, te + 0.3, te + 0.7))
    tA.z[:] += around * w
    ic = min(n - 1, ie + int(0.45 / dt))
    follow = _ease(t, t[ic] - 0.2, t[ic])
    tA.x[:] = (1 - follow) * tA.x + follow * np.maximum(tA.x * d, (bx - d * 0.3) * d) * d
    return tA, tD, bx, bz, dict(start=tp - 0.1, end=te + 0.3, touches=touches, d=d,
                                nutmeg_s=te)


SCENARIOS: dict[str, tuple[Callable, str | None, str | None, str | None]] = {
    # name: (script, contact_label, skill_label, sub_label)
    "tackle_won": (lambda r, t, dt: _sc_tackle(r, t, dt, True), "tackle_won", None, None),
    "tackle_failed": (lambda r, t, dt: _sc_tackle(r, t, dt, False), "tackle_failed", None, None),
    "shoulder_duel": (_sc_shoulder, "shoulder_duel", None, None),
    "no_contact": (_sc_no_contact, "no_contact", None, None),
    "dribble": (_sc_dribble, None, "dribble", None),
    # An ordinary turn is the hard negative for the drag-back: the ball
    # reverses, but the body turns first and there is no sole on the ball.
    "turn": (lambda r, t, dt: _sc_dribble(r, t, dt, True), None, "dribble", None),
    "stepover": (_sc_stepover, None, "trick", "stepover"),
    "dragback": (_sc_dragback, None, "trick", "dragback"),
    "feint": (_sc_feint, None, "trick", "feint"),
    "nutmeg": (_sc_nutmeg, None, "trick", "nutmeg"),
}
CONTACT_SCENARIOS = ("tackle_won", "tackle_failed", "shoulder_duel", "no_contact")
SKILL_SCENARIOS = ("dribble", "turn", "stepover", "dragback", "feint", "nutmeg")


# --------------------------------------------------------------------------
# The camera and the sensor

SENSOR: dict[str, tuple[float, float]] = {
    "height_px": (55.0, 420.0),
    "kp_sigma": (0.006, 0.03),
    "swap_p": (0.0, 0.06),
    "drop_p": (0.02, 0.1),
    "miss_pose_p": (0.0, 0.1),
    "ball_obs_p": (0.55, 0.95),
}
"""Ranges each simulated clip draws its sensor quality from."""

HARD_SENSOR: dict[str, tuple[float, float]] = {
    "height_px": (40.0, 80.0),
    "kp_sigma": (0.03, 0.05),
    "swap_p": (0.05, 0.12),
    "drop_p": (0.1, 0.2),
    "miss_pose_p": (0.1, 0.25),
    "ball_obs_p": (0.35, 0.65),
}
"""Deliberately worse than anything trained on: small players, noisy pose,
a ball lost half the time.  Used only to measure how gracefully it degrades."""


def _render(tA: _Track, tD: _Track, bx: np.ndarray, bz: np.ndarray, t: np.ndarray,
            rng: np.random.Generator, ids: tuple[str, str],
            sensor: dict[str, tuple[float, float]] | None = None) -> tuple[list[FrameState], dict]:
    n = len(t)
    sensor = {**SENSOR, **(sensor or {})}
    lo, hi = sensor["height_px"]
    H = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    width, height = 1920, 1080
    # The target camera is fixed (plan §1); a slow drift on a few clips keeps
    # the model from depending on a perfectly still frame.
    pan_amp = rng.uniform(0, 0.8) if rng.random() < 0.15 else 0.0
    pan = pan_amp * H * np.sin(2 * np.pi * rng.uniform(0.03, 0.1) * t + rng.uniform(0, 6))
    cx = width / 2 - H * 0.5 * float(np.mean(tA.x) + np.mean(tD.x)) * 0.5
    ground_y = height * rng.uniform(0.6, 0.85)

    kp_sigma = rng.uniform(*sensor["kp_sigma"])
    swap_p = rng.uniform(*sensor["swap_p"])
    drop_p = rng.uniform(*sensor["drop_p"])
    miss_pose_p = rng.uniform(*sensor["miss_pose_p"])
    base_conf = rng.uniform(0.55, 0.9)
    ball_obs_p = rng.uniform(*sensor["ball_obs_p"])
    ball_sigma = rng.uniform(0.004, 0.015)

    def to_px(x: float, z: float, y: float, scale: float, i: int) -> tuple[float, float]:
        return (cx + x * H + pan[i] + 0.0, ground_y + z * H * DEPTH_TO_Y + y * H * scale)

    players_px: list[list[tuple[np.ndarray, np.ndarray, BBox] | None]] = [[], []]
    for pi, tr in enumerate((tA, tD)):
        for i in range(n):
            joints, depth = _skeleton(tr, i)
            scale = 1.0 + 0.1 * tr.z[i]
            xy = np.zeros((17, 2))
            for j in range(17):
                xy[j] = to_px(tr.x[i] + joints[j, 0] * scale, tr.z[i] + depth[j],
                              joints[j, 1], scale, i)
            x1, y1 = xy[:, 0].min(), xy[:, 1].min() - 0.07 * H * scale
            x2, y2 = xy[:, 0].max(), xy[:, 1].max() + 0.03 * H * scale
            pad = 0.04 * H * scale
            jit = rng.normal(0, 0.015 * H, 4)
            box = BBox(min(x1 - pad + jit[0], x2), min(y1 + jit[1], y2),
                       max(x2 + pad + jit[2], x1), max(y2 + jit[3], y1))
            players_px[pi].append((xy, np.full(17, base_conf), box))

    # Occlusion while the two bodies overlap: worse pose for both.
    overlap = np.array([
        players_px[0][i][2].iou(players_px[1][i][2]) for i in range(n)
    ])

    ball_px = np.zeros((n, 2))
    for i in range(n):
        ball_px[i] = to_px(bx[i], bz[i], -BALL_R, 1.0 + 0.1 * bz[i], i)
    ball_px += rng.normal(0, ball_sigma * H, ball_px.shape)
    observed = rng.random(n) < ball_obs_p
    # Misses come in bursts, and are likelier behind the legs in a duel.
    for i in range(1, n):
        if not observed[i - 1] and rng.random() < 0.5:
            observed[i] = False
        if overlap[i] > 0.15 and rng.random() < 0.35:
            observed[i] = False
    observed[0] = True

    states: list[FrameState] = []
    last_obs = 0
    obs_idx = np.flatnonzero(observed)
    for i in range(n):
        players: list[PlayerState] = []
        for pi in (0, 1):
            xy, conf, box = players_px[pi][i]
            occ = overlap[i] > 0.1
            kps: list[Keypoint] = []
            if rng.random() > miss_pose_p * (3 if occ else 1):
                sig = kp_sigma * H * (2.0 if occ else 1.0)
                noisy = xy + rng.normal(0, sig, xy.shape)
                c = np.clip(conf + rng.normal(0, 0.08, 17) - (0.25 if occ else 0.0), 0.01, 0.99)
                if rng.random() < swap_p * (3 if occ else 1):
                    for a, b in ((11, 12), (13, 14), (15, 16)):
                        noisy[[a, b]] = noisy[[b, a]]
                drops = rng.random(17) < drop_p
                noisy[drops] = [box.x1, box.y1] + rng.random((int(drops.sum()), 2)) * [box.width, box.height]
                c[drops] = rng.uniform(0.01, 0.25, int(drops.sum()))
                kps = [Keypoint(_NAMES[j], float(noisy[j, 0]), float(noisy[j, 1]), float(c[j]))
                       for j in range(17)]
            players.append(PlayerState(player_id=ids[pi], track_id=pi + 1, bbox=box,
                                       confidence=0.9, keypoints=kps))
        ball = None
        if observed[i]:
            last_obs = i
            ball = BallState(position=Point(*ball_px[i]), confidence=0.6)
        else:
            nxt = obs_idx[obs_idx > i]
            if len(nxt) and (nxt[0] - last_obs) * (t[1] - t[0]) < 1.0:
                j = int(nxt[0])
                u = (i - last_obs) / max(1, j - last_obs)
                p = (1 - u) * ball_px[last_obs] + u * ball_px[j]
                ball = BallState(position=Point(*p), confidence=0.3, interpolated=True,
                                 frames_since_seen=i - last_obs)
        states.append(FrameState(frame_index=i, timestamp_s=float(t[i]), players=players,
                                 ball=ball, frame_size=(width, height)))
    return states, dict(player_height_px=H, pan=pan_amp, kp_sigma=kp_sigma,
                        ball_obs_p=ball_obs_p)


def generate(scenario: str, seed: int, duration_s: float = 4.0,
             sensor: dict[str, tuple[float, float]] | None = None) -> SyntheticSequence:
    """One simulated clip of ``scenario``, reproducible from ``seed``.

    ``sensor`` overrides the ranges in :data:`SENSOR` -- e.g. ``HARD_SENSOR``
    for small players, noisy pose and a ball that is often lost.
    """
    rng = np.random.default_rng(seed)
    fps = float(rng.choice([25.0, 30.0, 50.0, 60.0]))
    dt = 1.0 / fps
    t = np.arange(int(duration_s * fps)) * dt
    script, contact_label, skill_label, sub_label = SCENARIOS[scenario]
    tA, tD, bx, bz, meta = script(rng, t, dt)
    ids = ("Player 1", "Player 2") if rng.random() < 0.5 else ("Player 2", "Player 1")
    states, cam = _render(tA, tD, bx, bz, t, rng, ids, sensor)
    meta = {k: v for k, v in meta.items() if k != "touches"} | cam | {"fps": fps}
    return SyntheticSequence(
        states=states,
        scenario=scenario,
        contact_label=contact_label,
        skill_label=skill_label,
        sub_label=sub_label,
        event_start_s=float(max(0.0, meta["start"])),
        event_end_s=float(min(t[-1], meta["end"])),
        attacker_id=ids[0],
        defender_id=ids[1],
        seed=seed,
        params=meta,
    )


def seeds_for(per_scenario: int, seed: int = 0,
              scenarios: tuple[str, ...] | None = None) -> list[tuple[str, int]]:
    """``(scenario, seed)`` pairs; disjoint for different ``seed`` values."""
    names = scenarios or tuple(SCENARIOS)
    return [(name, seed * 1_000_003 + si * 100_003 + k)
            for si, name in enumerate(names) for k in range(per_scenario)]


def generate_many(per_scenario: int, seed: int = 0,
                  scenarios: tuple[str, ...] | None = None,
                  sensor: dict[str, tuple[float, float]] | None = None) -> list[SyntheticSequence]:
    return [generate(name, s, sensor=sensor) for name, s in seeds_for(per_scenario, seed, scenarios)]
