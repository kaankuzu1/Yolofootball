"""The Tier 1 rule engine: possession, pass, shot, goal, push-past.

:func:`detect_ball_events` is a pure function of a :class:`ClipState` and a
:class:`BallEventConfig`.  It returns events in the pipeline's schema plus
everything it reasoned from -- contacts, flights, goal checks, the per-frame
possessor -- so any call can be traced back to a threshold or a frame.

How a flight becomes an event
-----------------------------

=========================================  ==========================================
The ball, after leaving player A ...        Event
=========================================  ==========================================
enters the goal and all checks hold         ``shot`` (outcome ``goal``) + ``goal``
enters the goal, evidence incomplete        ``shot`` (outcome ``possible_goal``), flagged
enters the goal and comes back out          ``shot`` (outcome ``rebound``)
goes goalward fast, opponent collects it    ``shot`` (outcome ``saved``) near the goal,
near the goal / in its path                 ``blocked`` further out
goes goalward fast, misses / leaves frame   ``shot`` (outcome ``off_target``)
reaches B, and A->B counts as a pass        ``pass`` (completed)
reaches B, an opponent                      ``possession_change`` (``interception``)
reaches A again, having gone past the       ``push_past``
opponent
reaches A again, short                      nothing: a dribble touch
=========================================  ==========================================

"Goalward fast" is: release direction within ``shot_cone_deg`` of the goal or
extrapolating onto the mouth, and release speed at least ``shot_min_speed_d``.
The terminal state is decisive and the direction is necessary; speed is only a
floor, per plan §4.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from football_analysis.ball_events.config import BallEventConfig
from football_analysis.ball_events.flights import Flight, build_flights
from football_analysis.ball_events.goal import GoalCheck, check_goal, find_goal_entries, net_energy_series
from football_analysis.ball_events.possession import Contact, PossessionTrack, compute_possession
from football_analysis.ball_events.state import ClipState
from football_analysis.events import Event, EventType

__all__ = ["BallEventResult", "Spell", "detect_ball_events"]


@dataclass
class Spell:
    """A player's uninterrupted possession: contacts joined by dribble touches."""

    player_id: str
    start: int
    end: int
    contacts: list[int] = field(default_factory=list)


@dataclass
class BallEventResult:
    events: list[Event]
    possession: PossessionTrack
    spells: list[Spell]
    flights: list[Flight]
    goal_checks: list[GoalCheck]
    ruler_px: float
    review: list[dict[str, Any]] = field(default_factory=list)
    """Everything flagged for a person to look at, with the reason."""

    def possessor_at_time(self, state: ClipState, t: float) -> str | None:
        k = int(np.searchsorted(state.timestamps, t, side="right") - 1)
        if k < 0:
            return None
        return self.possession.possessor_at(k)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return {
            "ruler_px": round(self.ruler_px, 2),
            "contacts": len(self.possession.contacts),
            "spells": len(self.spells),
            "flights": len(self.flights),
            "goal_checks": len(self.goal_checks),
            "counts": counts,
            "needs_review": len(self.review),
        }


# -- helpers -------------------------------------------------------------------


def _event(state: ClipState, cfg: BallEventConfig, etype: EventType, k: int, confidence: float,
           rule: str, **kwargs: Any) -> Event:
    return Event(
        type=etype,
        timestamp_s=max(0.0, float(state.timestamps[k])),
        frame_index=int(state.frame_indices[k]),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        source=f"{cfg.source_name}.{rule}",
        **kwargs,
    )


def _team_relation(a: str, b: str, cfg: BallEventConfig, n_ids: int) -> str:
    """``team``, ``opponent`` or ``unknown`` for the pair."""
    if a in cfg.feeders or b in cfg.feeders:
        return "team"
    ta, tb = cfg.teams.get(a), cfg.teams.get(b)
    if ta is not None and tb is not None:
        return "team" if ta == tb else "opponent"
    if cfg.pass_mode == "auto" and n_ids <= 2 and not cfg.feeders:
        return "opponent"
    return "unknown"


def _is_pass(a: str, b: str, cfg: BallEventConfig, n_ids: int) -> tuple[bool, str]:
    rel = _team_relation(a, b, cfg, n_ids)
    if cfg.pass_mode == "never":
        return False, rel
    if cfg.pass_mode == "any":
        return True, rel
    if cfg.pass_mode == "teams":
        return rel == "team", rel
    # auto
    return rel in ("team", "unknown"), rel


def _opponents(pid: str | None, ids: list[str], cfg: BallEventConfig, n_ids: int) -> list[str]:
    if pid is None:
        return []
    return [o for o in ids if o != pid and o not in cfg.feeders
            and _team_relation(pid, o, cfg, n_ids) != "team"]


def _player_at(state: ClipState, k: int, pid: str):
    for p in state.players[k]:
        if p.player_id == pid:
            return p
    return None


def _nearest_row_with(state: ClipState, k: int, pid: str, span: int = 5):
    for d in range(span + 1):
        for j in (k - d, k + d):
            if 0 <= j < len(state):
                p = _player_at(state, j, pid)
                if p is not None:
                    return p
    return None


def _beaten_opponent(state: ClipState, flight: Flight, opponents: list[str],
                     cfg: BallEventConfig) -> tuple[str | None, dict[str, Any]]:
    """The opponent the ball was played past, if any.

    Along the ball's travel direction, the opponent was ahead of the ball at
    the release, within ``push_past_max_lateral_h`` of its line, and the ball
    ended beyond where they stood.
    """
    r, e = flight.release_frame, flight.end_frame
    if not (state.present[r] and state.present[e]):
        return None, {}
    p0, p1 = state.ball_xy[r], state.ball_xy[e]
    travel = p1 - p0
    length = float(np.linalg.norm(travel))
    if length <= 0:
        return None, {}
    u = travel / length
    normal = np.array([-u[1], u[0]])
    for opp in opponents:
        o = _nearest_row_with(state, r, opp)
        if o is None:
            continue
        foot = np.asarray(o.foot) - p0
        along = float(foot @ u)
        lateral = abs(float(foot @ normal)) / o.height
        if 0 < along < length and lateral <= cfg.push_past_max_lateral_h:
            return opp, {"defender_along_px": round(along, 1), "defender_lateral_h": round(lateral, 2)}
    return None, {}


# -- the engine ------------------------------------------------------------------


def detect_ball_events(state: ClipState, cfg: BallEventConfig | None = None) -> BallEventResult:
    cfg = (cfg or BallEventConfig()).validate()
    ruler = float(cfg.ball_diameter_px or state.ruler(cfg.ball_to_player_height))
    possession = compute_possession(state, cfg, ruler)
    contacts = possession.contacts
    flights = build_flights(state, contacts, cfg, ruler, possession.breaks)
    ids = state.player_ids()
    n_ids = len(ids)

    # Goal entries anywhere in the clip, then attached to the flight carrying them.
    goal_checks: list[GoalCheck] = []
    energy = net_energy_series(state, cfg, ruler) if state.goal is not None else None
    if state.goal is not None and state.goal.confidence >= cfg.goal_min_track_confidence:
        for entry in find_goal_entries(state, cfg, ruler):
            goal_checks.append(check_goal(state, entry, cfg, ruler, energy))
    checks_by_flight: dict[int, list[GoalCheck]] = {}
    orphan_checks: list[GoalCheck] = []
    for gc in goal_checks:
        owner = None
        for i, f in enumerate(flights):
            if f.release_frame <= gc.entry_frame <= max(f.end_frame, f.release_frame):
                owner = i
                break
        if owner is None:
            # The entry happened while someone "had" the ball (a player standing
            # in the mouth), or before any flight: attach to the latest flight.
            prior = [i for i, f in enumerate(flights) if f.release_frame <= gc.entry_frame]
            owner = prior[-1] if prior else None
        if owner is None:
            orphan_checks.append(gc)
        else:
            checks_by_flight.setdefault(owner, []).append(gc)
            flights[owner].goal_entries.append(gc.entry_frame)

    events: list[Event] = []
    review: list[dict[str, Any]] = []
    dribble_flights: set[int] = set()
    flight_labels: dict[int, str] = {}
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"ball_{counter:04d}"

    goal = state.goal
    for i, f in enumerate(flights):
        checks = checks_by_flight.get(i, [])
        interp_note = {"ball_interpolated_fraction": round(f.interpolated_fraction, 2),
                       "longest_track_gap_s": round(f.max_gap_s, 2)}
        common = {
            "release_speed_d_s": None if not np.isfinite(f.release_speed_d) else round(f.release_speed_d, 1),
            "speed_px_s": None if not np.isfinite(f.release_speed_d) else round(f.release_speed_d * ruler, 1),
            "terminal": f.terminal,
            **interp_note,
        }
        # Uncertainty in the track itself lowers every call made from this flight.
        track_penalty = 0.15 * f.interpolated_fraction + (0.15 if f.max_gap_s > cfg.max_track_gap_s else 0.0)

        goalward = bool(
            (f.goal_angle_deg is not None and f.goal_angle_deg <= cfg.shot_cone_deg)
            or f.extrapolates_on_target
        )
        fast = np.isfinite(f.release_speed_d) and f.release_speed_d >= cfg.shot_min_speed_d
        shot_like = f.releaser is not None and goal is not None and goalward and fast

        # 1. Into the goal.  An "entry" the depth check rejected was the ball
        #    passing in front of or behind the goal, so it is not an entry.
        checks = [c for c in checks if c.conditions.get("entry") != "fail"]
        decisive = [c for c in checks if c.verdict in ("goal", "possible_goal")]
        if checks:
            best = decisive[0] if decisive else checks[0]
            outcome = {"goal": "goal", "possible_goal": "possible_goal"}.get(best.verdict, "rebound")
            shot_id = next_id() if f.releaser is not None else None
            if f.releaser is not None:
                conf = 0.9 if outcome == "goal" else (0.75 if outcome == "possible_goal" else 0.7)
                detail = {
                    "on_target": True,
                    "outcome": outcome,
                    "distance_to_goal_px": _goal_distance_px(state, f.release_frame),
                    **common,
                    "goal_angle_deg": _r(f.goal_angle_deg),
                    "conditions": best.conditions,
                }
                if outcome == "possible_goal":
                    detail["needs_review"] = True
                    detail["review_reason"] = best.review_reason
                if outcome == "rebound" and f.near_goal_break is not None:
                    detail["woodwork_frame_index"] = int(state.frame_indices[f.near_goal_break])
                events.append(_event(state, cfg, EventType.SHOT, f.release_frame, conf - track_penalty, "shot",
                                     player_id=f.releaser, detail=detail, id=shot_id))
                flight_labels[i] = f"shot:{outcome}"
            if best.verdict == "goal":
                events.append(_event(
                    state, cfg, EventType.GOAL, best.entry_frame, best.confidence - track_penalty, "goal",
                    player_id=f.releaser,
                    detail={"scoring_player_id": f.releaser, "shot_event_id": shot_id,
                            "conditions": best.conditions, "evidence": best.evidence},
                    id=next_id(),
                ))
            for c in checks:
                if c.verdict == "possible_goal" or c.review_reason:
                    review.append({
                        "kind": "goal", "frame_index": int(state.frame_indices[c.entry_frame]),
                        "timestamp_s": round(float(state.timestamps[c.entry_frame]), 3),
                        "player_id": f.releaser, "verdict": c.verdict,
                        "reason": c.review_reason, "conditions": c.conditions,
                    })
            if f.releaser is not None:
                continue

        if f.releaser is None:
            continue

        receiver = f.receiver
        if receiver is None and f.terminal == "rest" and "picked_up_by" in f.notes:
            # A pass that rolled to a stop at the receiver's feet.
            pick = f.notes["picked_up_frame"]
            if state.timestamps[pick] - state.timestamps[f.end_frame] <= 1.0:
                receiver = f.notes["picked_up_by"]

        # 2. Goalward and fast, but stopped by an opponent: a save or a block.
        if shot_like and receiver is not None and receiver != f.releaser \
                and _team_relation(f.releaser, receiver, cfg, n_ids) != "team":
            d_goal = _dist_to_goal_h(state, f.end_frame)
            outcome = "saved" if d_goal is not None and d_goal <= cfg.shot_reception_near_goal_h else "blocked"
            conf = (0.75 if outcome == "saved" else 0.6) - track_penalty
            if f.extrapolates_on_target is False:
                conf -= 0.1
            events.append(_event(state, cfg, EventType.SHOT, f.release_frame, conf, "shot",
                                 player_id=f.releaser, secondary_player_id=receiver, id=next_id(),
                                 detail={"on_target": bool(f.extrapolates_on_target), "outcome": outcome,
                                         "blocked_by_player_id": receiver,
                                         "distance_to_goal_px": _goal_distance_px(state, f.release_frame),
                                         "goal_angle_deg": _r(f.goal_angle_deg), **common}))
            flight_labels[i] = f"shot:{outcome}"
            continue

        # 3. Goalward and fast, and nobody else got it: off target, off the
        #    woodwork, or on target and lost from view -- which is left for
        #    review, not called a goal.  Includes a rebound to the shooter.
        near = f.min_goal_distance_h is not None and f.min_goal_distance_h <= cfg.goal_near_h
        if receiver == f.releaser:
            # Back to the shooter only counts as a shot if it got to the goal
            # first; short of it, it was a knock past the defender.
            reached = f.near_goal_break is not None or (
                f.min_goal_distance_h is not None and f.min_goal_distance_h <= 0.5)
        else:
            reached = near or f.near_goal_break is not None or f.terminal in ("left_frame", "lost", "clip_end")
        if shot_like and (receiver is None or receiver == f.releaser) and reached:
            outcome = "off_target"
            detail = {"on_target": bool(f.extrapolates_on_target), "outcome": outcome,
                      "distance_to_goal_px": _goal_distance_px(state, f.release_frame),
                      "goal_angle_deg": _r(f.goal_angle_deg),
                      "closest_to_goal_h": _r(f.min_goal_distance_h), **common}
            conf = 0.7 if near else 0.55
            if f.near_goal_break is not None:
                detail["outcome"] = "woodwork"
                detail["woodwork_frame_index"] = int(state.frame_indices[f.near_goal_break])
            if f.extrapolates_on_target and f.terminal == "lost" and near:
                detail["outcome"] = "unresolved"
                detail["needs_review"] = True
                detail["review_reason"] = "on-target shot lost from view near the goal"
                review.append({"kind": "shot", "frame_index": int(state.frame_indices[f.release_frame]),
                               "timestamp_s": round(float(state.timestamps[f.release_frame]), 3),
                               "player_id": f.releaser, "reason": detail["review_reason"]})
            events.append(_event(state, cfg, EventType.SHOT, f.release_frame, conf - track_penalty, "shot",
                                 player_id=f.releaser, detail=detail, id=next_id()))
            flight_labels[i] = f"shot:{detail['outcome']}"
            continue

        # 4. To another player.
        if receiver is not None and receiver != f.releaser:
            is_pass, relation = _is_pass(f.releaser, receiver, cfg, n_ids)
            long_enough = f.displacement_px >= cfg.pass_min_distance_d * ruler
            quick_enough = (state.timestamps[f.end_frame] - state.timestamps[f.release_frame]
                            <= cfg.pass_max_duration_s)
            if is_pass and long_enough and quick_enough:
                conf = 0.8 if relation == "team" else 0.6
                if f.terminal == "rest":
                    conf -= 0.1
                events.append(_event(
                    state, cfg, EventType.PASS, f.release_frame, conf - track_penalty, "pass",
                    player_id=f.releaser, secondary_player_id=receiver, id=next_id(),
                    detail={"receiver_player_id": receiver, "completed": True,
                            "distance_px": round(f.displacement_px, 1),
                            "distance_d": round(f.displacement_px / ruler, 1),
                            "received_frame_index": int(state.frame_indices[f.end_frame]),
                            "team_relation": relation, "pass_mode": cfg.pass_mode, **common},
                ))
                flight_labels[i] = "pass"
            else:
                flight_labels[i] = "transfer"
            continue

        # 5. Back to the same player.
        if receiver == f.releaser:
            opps = _opponents(f.releaser, ids, cfg, n_ids)
            long_enough = f.displacement_px >= cfg.push_past_min_distance_d * ruler
            quick_enough = (state.timestamps[f.end_frame] - state.timestamps[f.release_frame]
                            <= cfg.push_past_max_duration_s)
            beaten, geo = (None, {})
            if long_enough and quick_enough:
                beaten, geo = _beaten_opponent(state, f, opps, cfg)
            if beaten is not None:
                events.append(_event(
                    state, cfg, EventType.PUSH_PAST, f.release_frame, 0.7 - track_penalty, "push_past",
                    player_id=f.releaser, secondary_player_id=beaten, id=next_id(),
                    detail={"beaten_player_id": beaten, "recollected": True,
                            "distance_px": round(f.displacement_px, 1),
                            "distance_d": round(f.displacement_px / ruler, 1),
                            "recollected_frame_index": int(state.frame_indices[f.end_frame]),
                            **geo, **common},
                ))
                flight_labels[i] = "push_past"
            elif f.displacement_px < cfg.min_flight_distance_d * ruler or not opps or True:
                dribble_flights.add(i)
                flight_labels[i] = "dribble"
            continue

        # 6. Loose ball out of the frame.
        if f.terminal == "left_frame" and cfg.emit_out_of_play:
            events.append(_event(state, cfg, EventType.OUT_OF_PLAY, f.end_frame, 0.6 - track_penalty,
                                 "out_of_play", player_id=f.releaser,
                                 detail={"last_touch_player_id": f.releaser, **interp_note}))
            flight_labels[i] = "out_of_play"

    for i, f in enumerate(flights):
        f.notes["label"] = flight_labels.get(i, "loose")

    # Possession spells: contacts of one player joined by their dribble touches.
    spells = _spells(contacts, flights, dribble_flights)
    possessor: list[str | None] = [None] * len(state)
    for s in spells:
        for k in range(s.start, s.end + 1):
            possessor[k] = s.player_id
    possession.possessor = possessor

    if cfg.emit_possession_changes:
        events.extend(_possession_changes(state, cfg, spells, flights, flight_labels, possession))

    for gc in orphan_checks:
        if gc.verdict == "goal":
            events.append(_event(state, cfg, EventType.GOAL, gc.entry_frame, gc.confidence * 0.8, "goal",
                                 detail={"scoring_player_id": None, "conditions": gc.conditions,
                                         "evidence": gc.evidence,
                                         "needs_review": True,
                                         "review_reason": "no shooter identified"}, id=next_id()))
        if gc.verdict == "goal" or gc.verdict == "possible_goal":
            review.append({"kind": "goal", "frame_index": int(state.frame_indices[gc.entry_frame]),
                           "timestamp_s": round(float(state.timestamps[gc.entry_frame]), 3),
                           "player_id": None, "verdict": gc.verdict,
                           "reason": gc.review_reason or "no shooter identified",
                           "conditions": gc.conditions})

    events.sort(key=lambda e: (e.timestamp_s, e.type.value))
    return BallEventResult(events, possession, spells, flights, goal_checks, ruler, review)


def _spells(contacts: list[Contact], flights: list[Flight], dribbles: set[int]) -> list[Spell]:
    by_contact = {f.contact_index: i for i, f in enumerate(flights) if f.contact_index is not None}
    spells: list[Spell] = []
    for ci, c in enumerate(contacts):
        prev_flight = by_contact.get(ci - 1)
        joins = (
            spells
            and spells[-1].player_id == c.player_id
            and prev_flight is not None
            and prev_flight in dribbles
        )
        if joins:
            spells[-1].end = c.end
            spells[-1].contacts.append(ci)
        else:
            spells.append(Spell(c.player_id, c.start, c.end, [ci]))
    return spells


def _possession_changes(state: ClipState, cfg: BallEventConfig, spells: list[Spell],
                        flights: list[Flight], labels: dict[int, str],
                        possession: PossessionTrack) -> list[Event]:
    out: list[Event] = []
    by_contact = {f.contact_index: i for i, f in enumerate(flights) if f.contact_index is not None}
    for prev, cur in zip(spells, spells[1:]):
        if prev.player_id == cur.player_id:
            continue
        fi = by_contact.get(prev.contacts[-1])
        label = labels.get(fi, "loose") if fi is not None else "loose"
        between = slice(prev.end, cur.start + 1)
        contested = bool(possession.contested[between].any())
        if label == "pass":
            cause = "pass"
        elif label.startswith("shot"):
            cause = "shot_" + label.split(":", 1)[1]
        elif contested or (cur.start - prev.end) <= 2:
            cause = "contest"
        elif label == "transfer":
            cause = "interception"
        else:
            cause = "loose_ball"
        out.append(_event(
            state, cfg, EventType.POSSESSION_CHANGE, cur.start, 0.6, "possession",
            player_id=cur.player_id, secondary_player_id=prev.player_id,
            detail={"from_player_id": prev.player_id, "cause": cause,
                    "lost_frame_index": int(state.frame_indices[prev.end])},
        ))
    return out


def _dist_to_goal_h(state: ClipState, k: int) -> float | None:
    goal = state.goal
    if goal is None or not state.present[k]:
        return None
    return max(0.0, -goal.signed_distance(state.ball_xy[k])) / goal.height_px


def _goal_distance_px(state: ClipState, k: int) -> float | None:
    goal = state.goal
    if goal is None or not state.present[k]:
        return None
    return round(float(np.linalg.norm(state.ball_xy[k] - goal.centre)), 1)


def _r(x: float | None, nd: int = 1) -> float | None:
    return None if x is None else round(float(x), nd)
