"""The goal rule: four conditions, not a line call (plan §4.4).

One side camera cannot adjudicate whether a ball wholly crossed a line, and
this module does not try.  What it can establish is whether the ball went *into
the net*, from four independent pieces of evidence:

1. **Entry.**  The ball moves from outside the goal mouth to inside it, at a
   size consistent with being at the goal's depth.  The size check is what
   separates a ball in the net from one that merely passes in front of or
   behind the goal in the image: the posts are a ruler at the goal plane.
2. **Absorption.**  Within a short window the ball's speed collapses or its
   direction reverses: the net stopping it.
3. **No re-emergence.**  It does not come back out of the mouth and away
   within ~0.5 s.  This is what separates a goal from a post, a bar or a save.
4. **Net motion.**  With a fixed camera, the frame-difference energy over the
   net spikes when it takes the ball.  The pixels under the ball and the
   players are excluded when it is measured (see :mod:`.netmotion`), so the
   ball's own motion does not count as the net's.

Each condition reports ``pass``, ``fail`` or ``unknown`` -- unknown when the
evidence is missing (the ball was hidden, there are no pixels to difference),
which is not the same as the evidence being against.  :func:`judge` turns the
four into a verdict, and anything that is not clean is a verdict of
``possible_goal`` for a person to look at, never a silent call either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from football_analysis.ball_events.config import BallEventConfig
from football_analysis.ball_events.state import ClipState, GoalGeometry

__all__ = ["GoalCheck", "find_goal_entries", "check_goal", "net_energy_series", "judge"]

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


@dataclass
class GoalCheck:
    entry_frame: int
    conditions: dict[str, str]
    evidence: dict[str, Any] = field(default_factory=dict)
    verdict: str = "no_goal"
    """``goal`` | ``possible_goal`` | ``no_goal``."""
    confidence: float = 0.0
    review_reason: str | None = None


def _mouth_margin(goal: GoalGeometry, ruler: float) -> float:
    # Half a ball: the ball's centre can sit just outside the drawn quad while
    # the ball itself is over the line in the image.
    return 0.5 * ruler


def find_goal_entries(state: ClipState, cfg: BallEventConfig, ruler: float,
                      start: int = 0, end: int | None = None) -> list[int]:
    """Frames where the ball goes from outside the goal region to inside it."""
    goal = state.goal
    if goal is None:
        return []
    end = len(state) - 1 if end is None else end
    poly = goal.polygon
    margin = _mouth_margin(goal, ruler)
    entries: list[int] = []
    was_inside: bool | None = None
    for k in range(start, end + 1):
        if not state.present[k]:
            continue
        inside = goal.signed_distance(state.ball_xy[k], poly) >= -margin
        if inside and was_inside is False:
            entries.append(k)
        was_inside = inside
    return entries


def net_energy_series(state: ClipState, cfg: BallEventConfig, ruler: float) -> np.ndarray | None:
    """The per-frame net-motion reading, or ``None`` when nobody measured it.

    The reading itself is taken where the pixels are, by
    :class:`~football_analysis.ball_events.netmotion.NetMotionMeter`; the event
    rules only ever see the number.
    """
    if state.net_motion is None or state.goal is None:
        return None
    return state.net_motion


def _window(state: ClipState, k: int, seconds: float, forward: bool = True) -> np.ndarray:
    t0 = state.timestamps[k]
    if forward:
        sel = (state.timestamps > t0) & (state.timestamps <= t0 + seconds)
    else:
        sel = (state.timestamps < t0) & (state.timestamps >= t0 - seconds)
    return np.nonzero(sel)[0]


def check_goal(state: ClipState, entry: int, cfg: BallEventConfig, ruler: float,
               energy: np.ndarray | None) -> GoalCheck:
    goal = state.goal
    assert goal is not None
    conditions: dict[str, str] = {}
    evidence: dict[str, Any] = {"entry_frame_index": int(state.frame_indices[entry])}

    # 1. Entry, with the depth check.
    expected = goal.expected_ball_diameter(cfg.goal_height_m, cfg.ball_diameter_m)
    near = np.arange(max(0, entry - 3), min(len(state), entry + 4))
    sizes = state.ball_diameter[near]
    sizes = sizes[np.isfinite(sizes) & (sizes > 0) & state.observed[near]]
    observed_entry = bool(state.observed[max(0, entry - 1):entry + 2].any())
    if sizes.size:
        ratio = float(np.median(sizes) / expected)
        evidence["depth_ratio"] = round(ratio, 2)
        lo, hi = cfg.depth_ratio_range
        conditions["entry"] = PASS if lo <= ratio <= hi else FAIL
    else:
        # No size to check: entry is what the path says it is, but unproven in depth.
        conditions["entry"] = PASS if observed_entry else UNKNOWN
        evidence["depth_ratio"] = None
    evidence["entry_observed"] = observed_entry

    # 2. Absorption.
    speed = np.linalg.norm(state.ball_velocity, axis=1)
    before = np.arange(max(0, entry - 3), entry + 1)
    v_in = state.ball_velocity[before]
    v_in = v_in[np.isfinite(v_in).all(axis=1)]
    after = _window(state, entry, cfg.absorb_window_s)
    after = after[state.present[after]]
    if v_in.size and after.size:
        v_entry = v_in.mean(axis=0)
        s_entry = float(np.linalg.norm(v_entry))
        s_after = speed[after]
        s_after = s_after[np.isfinite(s_after)]
        va = state.ball_velocity[after]
        va = va[np.isfinite(va).all(axis=1)]
        reversed_ = bool(va.size and (va @ v_entry < -0.1 * s_entry * np.linalg.norm(va, axis=1)).any())
        slowed = bool(s_after.size and s_after.min() <= cfg.absorb_speed_ratio * s_entry)
        conditions["absorption"] = PASS if (slowed or reversed_) else FAIL
        evidence["entry_speed_d_s"] = round(s_entry / ruler, 1)
        evidence["min_speed_after_d_s"] = round(float(s_after.min()) / ruler, 1) if s_after.size else None
        evidence["reversed"] = reversed_
    else:
        conditions["absorption"] = UNKNOWN

    # 3. No re-emergence.
    window = _window(state, entry, cfg.reemerge_window_s)
    seen = window[state.present[window] & ~_is_bridge(state, window)]
    limit = cfg.reemerge_distance_h * goal.height_px
    poly = goal.polygon
    if seen.size:
        outside = [-goal.signed_distance(state.ball_xy[k], poly) for k in seen]
        far = max(outside)
        evidence["max_distance_outside_goal_h"] = round(far / goal.height_px, 2)
        if far > limit:
            conditions["no_reemergence"] = FAIL
        elif seen.size >= 2:
            conditions["no_reemergence"] = PASS
        else:
            conditions["no_reemergence"] = UNKNOWN
    else:
        conditions["no_reemergence"] = UNKNOWN
    # A ball that re-emerges later than the window but at speed straight out
    # of the mouth is still a rebound; look a little further if it was unseen.
    if conditions["no_reemergence"] == UNKNOWN:
        later = _window(state, entry, 2 * cfg.reemerge_window_s)
        later = later[state.observed[later]]
        if later.size and max(-goal.signed_distance(state.ball_xy[k], poly) for k in later) > limit:
            conditions["no_reemergence"] = FAIL

    # 4. Net motion.
    if energy is None:
        conditions["net_motion"] = UNKNOWN
    else:
        base_idx = _window(state, entry, cfg.net_baseline_s, forward=False)
        base_idx = base_idx[base_idx < entry - 1] if base_idx.size else base_idx
        base = energy[base_idx] if base_idx.size else np.array([])
        base = base[np.isfinite(base)]
        post_idx = np.concatenate([[entry], _window(state, entry, cfg.net_window_s)])
        post = energy[post_idx]
        post = post[np.isfinite(post)]
        if base.size >= 3 and post.size:
            med = float(np.median(base))
            mad = float(np.median(np.abs(base - med))) + 1e-3
            peak = float(post.max())
            threshold = max(med * cfg.net_spike_ratio, med + cfg.net_spike_mads * mad,
                            cfg.net_min_energy)
            conditions["net_motion"] = PASS if peak >= threshold else FAIL
            evidence["net_energy_baseline"] = round(med, 2)
            evidence["net_energy_peak"] = round(peak, 2)
            evidence["net_energy_threshold"] = round(threshold, 2)
        else:
            conditions["net_motion"] = UNKNOWN

    check = GoalCheck(entry, conditions, evidence)
    judge(check)
    return check


def _is_bridge(state: ClipState, idx: np.ndarray) -> np.ndarray:
    from football_analysis.ball_events.state import BRIDGED

    return state.ball_source[idx] == BRIDGED


def judge(check: GoalCheck) -> GoalCheck:
    """Four conditions in, one verdict out.

    ``goal``           entry and no re-emergence pass, at least one of
                       absorption and net motion passes, and nothing fails.
                       Confidence 0.95 with all four, 0.85 with three.
    ``possible_goal``  the ball entered and did not visibly come back out, but
                       the evidence is incomplete or split.  Flagged for review.
    ``no_goal``        entry failed (the depth check put the ball in front of or
                       behind the goal), or the ball came back out.
    """
    c = check.conditions
    fails = [name for name, v in c.items() if v == FAIL]
    passes = [name for name, v in c.items() if v == PASS]
    if c.get("entry") == FAIL:
        check.verdict, check.confidence = "no_goal", 0.0
        check.review_reason = None
        return check
    if c.get("no_reemergence") == FAIL:
        check.verdict, check.confidence = "no_goal", 0.0
        # A rebound with the net visibly moving and the ball stopped is the
        # one pattern a post hit and a goal-then-pushed-out share.  Say so.
        if c.get("net_motion") == PASS and c.get("absorption") == PASS:
            check.review_reason = "ball came back out, but the net moved and the ball was stopped"
        return check
    confirmed = (
        c.get("entry") == PASS
        and c.get("no_reemergence") == PASS
        and (c.get("absorption") == PASS or c.get("net_motion") == PASS)
        and not fails
    )
    if confirmed:
        check.verdict = "goal"
        check.confidence = 0.95 if len(passes) == 4 else 0.85
        check.review_reason = None
        return check
    check.verdict = "possible_goal"
    # Graded by how much speaks for it, but never above a confirmed goal.
    check.confidence = round(0.35 + 0.1 * len(passes) - 0.05 * len(fails), 2)
    missing = [name for name, v in c.items() if v != PASS]
    if fails:
        check.review_reason = "entered the goal but " + ", ".join(f"{m} failed" for m in fails)
    elif c.get("entry") == UNKNOWN or c.get("no_reemergence") == UNKNOWN:
        check.review_reason = "entered the goal during a gap in the ball track"
    else:
        check.review_reason = "entered the goal but no reading for " + ", ".join(missing)
    return check
