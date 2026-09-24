"""Flights: what the ball did between one player's contact and the next.

Every pass, shot and push-past is a flight -- the ball leaving a player -- and
what separates them is mostly how the flight *ended* (plan §4.1).  We are
offline, so for each release we can simply look forward and see: another
player's feet, the same player's feet again, the goal, the edge of the frame,
the ball coming to rest, or the track running out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from football_analysis.ball_events.config import BallEventConfig
from football_analysis.ball_events.possession import Contact
from football_analysis.ball_events.state import ClipState

__all__ = ["Flight", "build_flights"]

TERMINALS = ("player", "rest", "left_frame", "lost", "clip_end")


@dataclass
class Flight:
    releaser: str | None
    release_frame: int
    end_frame: int
    terminal: str
    receiver: str | None = None
    contact_index: int | None = None
    """Index of the releasing contact in the contact list."""
    next_contact_index: int | None = None
    release_velocity: tuple[float, float] = (float("nan"), float("nan"))
    release_speed_d: float = float("nan")
    displacement_px: float = 0.0
    path_length_px: float = 0.0
    max_gap_s: float = 0.0
    interpolated_fraction: float = 0.0
    goal_angle_deg: float | None = None
    """Angle between release direction and the release-to-goal vector."""
    min_goal_distance_h: float | None = None
    """Closest the actual path came to the goal mouth, in goal heights."""
    extrapolates_on_target: bool | None = None
    goal_entries: list[int] = field(default_factory=list)
    near_goal_break: int | None = None
    """A velocity break next to the goal frame without entry: woodwork."""
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.release_frame


def _last_present(state: ClipState, lo: int, hi: int) -> int | None:
    idx = np.nonzero(state.present[lo:hi + 1])[0]
    return int(lo + idx[-1]) if idx.size else None


def _flight_end(state: ClipState, release: int, limit: int, cfg: BallEventConfig,
                ruler: float) -> tuple[int, str, float]:
    """Walk forward from a release with nobody receiving it.  Returns the end
    frame, the terminal kind, and the longest gap seen."""
    ts = state.timestamps
    speed = np.linalg.norm(state.ball_velocity, axis=1)
    rest = cfg.rest_speed_d * ruler
    w, h = state.frame_size
    last_seen = release
    gap_start: int | None = None
    max_gap = 0.0
    rest_start: int | None = None
    for k in range(release + 1, limit + 1):
        if state.present[k]:
            if gap_start is not None:
                max_gap = max(max_gap, ts[k] - ts[gap_start - 1])
                gap_start = None
            last_seen = k
            if np.isfinite(speed[k]) and speed[k] < rest:
                rest_start = k if rest_start is None else rest_start
                if ts[k] - ts[rest_start] >= cfg.rest_min_s:
                    return rest_start, "rest", max_gap
            else:
                rest_start = None
        else:
            if gap_start is None:
                gap_start = k
            if ts[k] - ts[last_seen] > cfg.max_track_gap_s:
                return last_seen, _vanish_kind(state, last_seen, cfg), max(max_gap, ts[k] - ts[last_seen])
    if gap_start is not None:
        return last_seen, _vanish_kind(state, last_seen, cfg), max(max_gap, ts[limit] - ts[last_seen])
    return limit, "clip_end", max_gap


def _vanish_kind(state: ClipState, k: int, cfg: BallEventConfig) -> str:
    w, h = state.frame_size
    x, y = state.ball_xy[k]
    m = cfg.edge_margin_px
    # Moving outward near an edge = left the frame; otherwise just lost.
    vx, vy = np.nan_to_num(state.ball_velocity[k])
    if (x <= m and vx <= 0) or (x >= w - m and vx >= 0) or (y <= m and vy <= 0) or (y >= h - m and vy >= 0):
        return "left_frame"
    # Heading for the goal mouth, it went into the goal or behind a post as
    # far as anyone can tell, which is not the same as leaving the frame.
    goal = state.goal
    if goal is not None:
        tip = (x + vx * 0.3, y + vy * 0.3)
        if goal.segment_hits_mouth((x, y), tip, 0.1 * goal.height_px):
            return "lost"
    # Or it was heading there fast enough to reach the edge within the gap.
    for dt in (0.1, 0.2, 0.3):
        px, py = x + vx * dt, y + vy * dt
        if px < 0 or px > w or py < 0 or py > h:
            return "left_frame"
    return "lost"


def _describe(flight: Flight, state: ClipState, cfg: BallEventConfig, ruler: float) -> None:
    r, e = flight.release_frame, flight.end_frame
    ts = state.timestamps
    window = np.nonzero((ts > ts[r]) & (ts <= ts[r] + cfg.release_speed_window_s))[0]
    window = window[window <= max(e, r + 1)]
    v = state.ball_velocity[window] if window.size else np.empty((0, 2))
    v = v[np.isfinite(v).all(axis=1)] if v.size else v
    if v.size:
        vel = np.median(v, axis=0)
        flight.release_velocity = (float(vel[0]), float(vel[1]))
        flight.release_speed_d = float(np.linalg.norm(vel) / ruler)
    path_idx = np.arange(r, e + 1)
    path_idx = path_idx[state.present[path_idx]]
    if path_idx.size:
        pts = state.ball_xy[path_idx]
        flight.displacement_px = float(np.linalg.norm(pts[-1] - pts[0]))
        flight.path_length_px = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        span = np.arange(r, e + 1)
        flight.interpolated_fraction = float(state.interpolated[span].mean()) if span.size else 0.0

    goal = state.goal
    if goal is None or not path_idx.size:
        return
    p0 = state.ball_xy[path_idx[0]]
    to_goal = goal.centre - p0
    rv = np.asarray(flight.release_velocity)
    if np.isfinite(rv).all() and np.linalg.norm(rv) > 0 and np.linalg.norm(to_goal) > 0:
        cos = float(rv @ to_goal / (np.linalg.norm(rv) * np.linalg.norm(to_goal)))
        flight.goal_angle_deg = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        reach = 1.5 * float(np.linalg.norm(to_goal))
        tip = p0 + rv / np.linalg.norm(rv) * reach
        margin = cfg.shot_on_target_margin * goal.height_px
        flight.extrapolates_on_target = goal.segment_hits_mouth(p0, tip, margin)
    dists = [max(0.0, -goal.signed_distance(state.ball_xy[k])) for k in path_idx]
    flight.min_goal_distance_h = float(min(dists) / goal.height_px)
    # Woodwork: the ball is deflected (turned, not just slowed) right at the
    # goal frame, at a size that puts it at the goal's depth.
    breaks = set(flight.notes.get("breaks", [])) | {int(k) for k in path_idx if state.ball_break[k]}
    expected = goal.expected_ball_diameter(cfg.goal_height_m, cfg.ball_diameter_m)
    lo, hi = cfg.depth_ratio_range
    for k in sorted(breaks):
        if abs(goal.signed_distance(state.ball_xy[k], goal.quad)) > 0.25 * goal.height_px:
            continue
        before = state.ball_velocity[max(r, k - 3):k]
        after = state.ball_velocity[k + 1:k + 4]
        before = before[np.isfinite(before).all(axis=1)]
        after = after[np.isfinite(after).all(axis=1)]
        if not (before.size and after.size):
            continue
        vb, va = before.mean(axis=0), after.mean(axis=0)
        denom = float(np.linalg.norm(vb) * np.linalg.norm(va))
        if denom <= 0 or float(vb @ va) / denom > np.cos(np.radians(60)):
            continue
        near = np.arange(max(0, k - 3), min(len(state), k + 4))
        sizes = state.ball_diameter[near]
        sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
        if sizes.size and not lo <= float(np.median(sizes)) / expected <= hi:
            continue
        flight.near_goal_break = int(k)
        break


def build_flights(state: ClipState, contacts: list[Contact], cfg: BallEventConfig,
                  ruler: float, breaks: np.ndarray | None = None) -> list[Flight]:
    """One flight per contact: from its release to the next contact (or a
    terminal before it), plus a leading flight if the ball is already moving
    when the clip starts."""
    n = len(state)
    flights: list[Flight] = []
    if not n:
        return flights

    first_start = contacts[0].start if contacts else n
    first_seen = np.nonzero(state.present[:first_start])[0]
    if first_seen.size and first_start - first_seen[0] > 1:
        k0 = int(first_seen[0])
        if contacts:
            flights.append(Flight(None, k0, contacts[0].start, "player",
                                  receiver=contacts[0].player_id, next_contact_index=0))
        else:
            end, kind, gap = _flight_end(state, k0, n - 1, cfg, ruler)
            flights.append(Flight(None, k0, end, kind, max_gap_s=gap))

    for i, c in enumerate(contacts):
        release = min(c.release_frame, c.end + 1, n - 1)
        release = max(release, c.start)
        nxt = contacts[i + 1] if i + 1 < len(contacts) else None
        limit = nxt.start if nxt else n - 1
        if release >= limit:
            continue
        # Does the ball come to rest, leave, or get lost before anyone has it?
        end, kind, gap = _flight_end(state, release, limit, cfg, ruler)
        if nxt is not None and kind == "clip_end":
            end, kind = nxt.start, "player"
        flight = Flight(c.player_id, release, end, kind, contact_index=i, max_gap_s=gap)
        if kind == "player" and nxt is not None:
            flight.receiver = nxt.player_id
            flight.next_contact_index = i + 1
        elif nxt is not None:
            # Ended loose (rest / lost / out) and someone picked it up later.
            flight.notes["picked_up_by"] = nxt.player_id
            flight.notes["picked_up_frame"] = nxt.start
            flight.next_contact_index = i + 1
        if breaks is not None:
            flight.notes["breaks"] = [int(k) for k in np.nonzero(breaks[release + 1:end + 1])[0] + release + 1]
        flights.append(flight)

    for f in flights:
        _describe(f, state, cfg, ruler)
    return flights
