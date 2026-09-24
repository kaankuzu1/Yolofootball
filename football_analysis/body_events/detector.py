"""Tier 2 as a pipeline stage: tackles and tricks from how the bodies moved.

:class:`BodyEventDetector` satisfies
:class:`~football_analysis.interfaces.EventDetector`.  It buffers the records
and decides everything in ``finalize``, because both events need the frames
after them: a tackle is only 'won' once you see who came away with the ball.

What it emits
-------------
``tackle``  ``player_id`` is the tackler, ``secondary_player_id`` the player
            tackled; ``timestamp_s`` is the moment the tackler's foot got
            nearest the ball.  ``detail``:

            ``outcome``            ``won`` / ``failed``
            ``winner_player_id``, ``loser_player_id``
            ``contact_point``      ``{x, y}`` in frame pixels
            ``probabilities``      all four contact classes
            ``evidence``           the features furthest from a shoulder duel
            ``window``             ``{start_s, end_s}`` the call was made over

``trick``   ``player_id`` is who did it, ``secondary_player_id`` the nearest
            opponent.  ``timestamp_s`` .. ``end_timestamp_s`` is the move.
            ``detail``:

            ``trick_name``         ``nutmeg`` (from the explicit rule), else
                                   ``skill_move`` -- named moves wait for
                                   named labels (plan §4.3)
            ``signature``          which plan signal fired: ``feet_round_ball``,
                                   ``direction_change``, ``body_feint``,
                                   ``ball_through_legs``
            ``touch_count``, ``probability``, ``evidence``

Both carry ``needs_review`` and ``review_reasons``: a call is flagged rather
than dropped when the two best classes were close, the players were too small
for pose to be trusted, pose was mostly missing, the ball was not seen at the
moment that mattered, or who had the ball could not be told.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from football_analysis.body_events.features import (
    ContactWindow,
    SkillWindow,
    WindowConfig,
    _Body,
    contact_features,
    contact_windows,
    skill_features,
    skill_windows,
)
from football_analysis.body_events.model import BodyEventModel, default_model_path
from football_analysis.body_events.rules import NutmegConfig, find_nutmegs
from football_analysis.body_events.series import ClipSeries
from football_analysis.events import Event, EventType
from football_analysis.state import FrameState

logger = logging.getLogger(__name__)

__all__ = ["BodyEventConfig", "BodyEventDetector", "detect_body_events"]


@dataclass
class BodyEventConfig:
    model_path: str | None = None
    """Saved model; ``None`` means :func:`default_model_path`."""
    tackle_threshold: float = 0.5
    """Minimum P(tackle_won) + P(tackle_failed) to report a tackle."""
    trick_threshold: float = 0.85
    """Set high on purpose: at 0.6 an ordinary sharp turn is called a trick
    about half the time on held-out simulated clips; at 0.85 about 5%, for a
    few points of trick recall."""
    trick_min_windows: int = 3
    """A real move is covered by several overlapping windows; one window on
    its own above threshold is usually a noisy frame, not a trick."""
    review_margin: float = 0.2
    """Flag a call when the runner-up class is within this of the winner."""
    min_player_height_px: float = 45.0
    """Below this, pose is too coarse to read a leg from; calls are flagged."""
    min_pose_frac: float = 0.5
    min_ball_seen_frac: float = 0.3
    suppress_s: float = 0.8
    """Two calls for the same player closer than this are one call."""
    ball_seen_near_s: float = 0.4
    """A tackle also needs the ball actually detected (not interpolated) near
    the pair within this many seconds of the contact.  On broadcast footage a
    false ball far away, interpolated across a gap, walked past a goalkeeper
    and a referee standing still and made a confident tackle out of them."""
    min_evidence: int = 2
    """A tackle is only reported when at least this many of three things held
    around the contact: the ball was seen, pose covered both players, and it
    was clear who had the ball.  With fewer the classifier is reading noise
    (a crowded broadcast frame gave dozens of confident calls on nothing), so
    the stage abstains rather than flag a call nobody can check."""
    windows: WindowConfig = field(default_factory=WindowConfig)
    nutmeg: NutmegConfig = field(default_factory=NutmegConfig)


class BodyEventDetector:
    """Tackles and tricks over pose, as an ``EventDetector``."""

    def __init__(self, config: Any = None, *, body_config: BodyEventConfig | None = None,
                 model: BodyEventModel | None = None) -> None:
        # ``config`` is the run's Config (what default_event_stage passes); the
        # body-event settings are their own object so this stays importable
        # without touching the shared config schema.
        self.run_config = config
        self.config = body_config or BodyEventConfig()
        self._model = model
        self._states: list[FrameState] = []
        self.last_summary: dict[str, Any] | None = None

    @property
    def model(self) -> BodyEventModel:
        if self._model is None:
            path = Path(self.config.model_path) if self.config.model_path else default_model_path()
            self._model = BodyEventModel.load(path)
        return self._model

    def reset(self) -> None:
        self._states = []

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"name": type(self).__name__}
        try:
            info["model"] = {k: v for k, v in self.model.meta.items() if k != "evaluation"}
        except Exception as exc:  # pragma: no cover - reported, not fatal
            info["model_error"] = str(exc)
        if self.last_summary is not None:
            info["summary"] = self.last_summary
        return info

    def update(self, state: FrameState) -> list[Event]:
        self._states.append(state)
        return []

    def finalize(self) -> list[Event]:
        summary: dict[str, Any] = {}
        events = detect_body_events(self._states, self.model, self.config, summary=summary)
        self.last_summary = summary
        self._states = []
        return events


def detect_body_events(states: list[FrameState], model: BodyEventModel,
                       cfg: BodyEventConfig | None = None, *,
                       summary: dict[str, Any] | None = None) -> list[Event]:
    """The whole of Tier 2 as a pure function of a clip's records.

    Pass a dict as ``summary`` to get back what the stage looked at, so that
    "no tackles" can be told apart from "nothing usable to read": frames,
    players, how much pose and ball there was, windows cut, and how many
    calls cleared the threshold but were dropped, and why.
    """
    cfg = cfg or BodyEventConfig()
    s = summary if summary is not None else {}
    s.clear()
    s.update(frames=len(states), players=0, abstained_reason=None,
             contact={"windows": 0, "above_threshold": 0, "merged_duplicates": 0,
                      "dropped_ball_not_seen_near": 0, "dropped_low_evidence": 0, "reported": 0},
             skill={"windows": 0, "windows_above_threshold": 0, "groups_too_short": 0,
                    "reported": 0},
             nutmeg={"rule_hits": 0, "folded_into_tackle": 0, "merged_with_skill_call": 0,
                     "reported": 0})
    if len(states) < 5:
        s["abstained_reason"] = "too_few_frames"
        return []
    clip = ClipSeries.from_states(states)
    s["players"] = len(clip.players)
    s["pose_coverage"] = round(float(np.mean([np.mean(np.any(p.kp[:, :, 2] > 0.3, axis=1))
                                              for p in clip.players.values()])), 3) if clip.players else 0.0
    s["ball_observed_frac"] = round(float(np.mean(clip.ball_observed)), 3) if len(clip) else 0.0
    if len(clip.players) < 2:
        s["abstained_reason"] = "fewer_than_two_players"
        return []
    tackles = _tackles(clip, model, cfg, s["contact"])
    tricks = _tricks(clip, model, cfg, tackles, s["skill"], s["nutmeg"])
    return tackles + tricks


# ----------------------------------------------------------------------------


def _review(reasons: list[str], probs: dict[str, float], cfg: BodyEventConfig) -> list[str]:
    top = sorted(probs.values(), reverse=True)
    if len(top) > 1 and top[0] - top[1] < cfg.review_margin:
        reasons.append("classes_close")
    return reasons


def _height_px(clip: ClipSeries, pid: str, a: int, b: int) -> float:
    h = clip.players[pid].height[a:b + 1]
    return float(np.nanmedian(h)) if np.any(~np.isnan(h)) else float("nan")


def _tracks(clip: ClipSeries, *pids: str) -> list[int]:
    out: list[int] = []
    for pid in pids:
        if pid and pid in clip.players:
            out.extend(sorted(clip.players[pid].track_ids))
    return out


def _ball_seen_near(clip: ClipSeries, w: ContactWindow, cfg: BodyEventConfig) -> bool:
    k = max(1, int(round(cfg.ball_seen_near_s * clip.fps)))
    seg = slice(max(0, w.anchor - k), w.anchor + k + 1)
    obs = clip.ball_observed[seg]
    if not np.any(obs):
        return False
    ball = clip.ball[seg]
    near = np.full(len(obs), np.inf)
    for pid in (w.attacker_id, w.defender_id):
        B = _Body(clip.players[pid], clip.t)
        d = (B.foot_to(clip.ball) / B.h)[seg]
        near = np.fmin(near, np.where(np.isnan(d), np.inf, d))
    return bool(np.any(obs & ~np.isnan(ball[:, 0]) & (near < cfg.windows.contact_ball_h)))


def _tackles(clip: ClipSeries, model: BodyEventModel, cfg: BodyEventConfig,
             stats: dict[str, int]) -> list[Event]:
    windows = contact_windows(clip, cfg.windows)
    stats["windows"] = len(windows)
    if not windows:
        return []
    feats = [contact_features(clip, w, cfg.windows) for w in windows]
    proba = model.contact.predict_proba(feats)
    classes = model.contact.classes
    cands = []
    for w, f, p in zip(windows, feats, proba):
        probs = {c: float(v) for c, v in zip(classes, p)}
        p_tackle = probs.get("tackle_won", 0.0) + probs.get("tackle_failed", 0.0)
        if p_tackle >= cfg.tackle_threshold:
            cands.append((p_tackle, w, f, probs))
    stats["above_threshold"] = len(cands)
    cands.sort(key=lambda c: -c[0])
    kept: list[tuple[float, ContactWindow, dict, dict]] = []
    for c in cands:
        t = clip.t[c[1].anchor]
        pair = {c[1].attacker_id, c[1].defender_id}
        if any(pair == {k[1].attacker_id, k[1].defender_id}
               and abs(clip.t[k[1].anchor] - t) < cfg.suppress_s for k in kept):
            stats["merged_duplicates"] += 1
            continue
        kept.append(c)

    events = []
    for p_tackle, w, f, probs in kept:
        won = probs.get("tackle_won", 0.0) >= probs.get("tackle_failed", 0.0)
        reasons: list[str] = []
        h = min(_height_px(clip, w.attacker_id, w.start, w.end),
                _height_px(clip, w.defender_id, w.start, w.end))
        if not h >= cfg.min_player_height_px:
            reasons.append("players_small_for_pose")
        if not f.get("pose_frac", 0.0) >= cfg.min_pose_frac:
            reasons.append("pose_sparse")
        if not f.get("ball_observed_frac", 0.0) >= cfg.min_ball_seen_frac:
            reasons.append("ball_unseen_at_contact")
        if not w.attacker_known:
            reasons.append("possession_unclear")
        if not _ball_seen_near(clip, w, cfg):
            stats["dropped_ball_not_seen_near"] += 1
            continue
        missing = {"pose_sparse", "ball_unseen_at_contact", "possession_unclear"} & set(reasons)
        if 3 - len(missing) < cfg.min_evidence:
            stats["dropped_low_evidence"] += 1
            continue
        stats["reported"] += 1
        _review(reasons, {"won": probs.get("tackle_won", 0.0), "failed": probs.get("tackle_failed", 0.0),
                          "not_tackle": 1.0 - p_tackle}, cfg)
        ball = clip.ball[w.anchor]
        if np.isnan(ball[0]):
            d = clip.players[w.defender_id]
            ball = np.array([np.nanmean(d.bbox[w.anchor, [0, 2]]), d.bbox[w.anchor, 3]])
        events.append(Event(
            type=EventType.TACKLE,
            timestamp_s=float(clip.t[w.anchor]),
            frame_index=int(clip.frame_index[w.anchor]),
            confidence=float(np.clip(p_tackle, 0.0, 1.0)),
            player_id=w.defender_id,
            secondary_player_id=w.attacker_id,
            track_ids=_tracks(clip, w.defender_id, w.attacker_id),
            source="body_events.contact_classifier",
            detail={
                "outcome": "won" if won else "failed",
                "winner_player_id": w.defender_id if won else w.attacker_id,
                "loser_player_id": w.attacker_id if won else w.defender_id,
                "contact_point": {"x": round(float(ball[0]), 1), "y": round(float(ball[1]), 1)},
                "probabilities": {k: round(v, 3) for k, v in probs.items()},
                "evidence": model.contact.explain(f, "shoulder_duel"),
                "window": {"start_s": round(float(clip.t[w.start]), 3),
                           "end_s": round(float(clip.t[w.end]), 3)},
                "needs_review": bool(reasons),
                "review_reasons": reasons,
            },
        ))
    return events


_SIGNATURES = {
    "feet_round_ball": ("foot_rel_per_ball", "ankle_ball_crossings", "foot_per_ball", "ankle_lift_max_h"),
    "direction_change": ("ball_reversal", "ball_heading_change_max_deg", "ankle_on_ball_frac"),
    "body_feint": ("hip_sway_h", "lean_std_deg", "facing_ball_mismatch"),
}


def _signature(model: BodyEventModel, f: dict[str, float]) -> str:
    ref = model.skill.reference.get("dribble", {})
    best, best_z = "skill_move", 0.0
    for sig, names in _SIGNATURES.items():
        zs = []
        for n in names:
            v = f.get(n, np.nan)
            if n in ref and v is not None and not np.isnan(v):
                med, spread = ref[n]
                zs.append((v - med) / spread)
        if zs and max(zs) > best_z:
            best, best_z = sig, max(zs)
    return best


def _tricks(clip: ClipSeries, model: BodyEventModel, cfg: BodyEventConfig,
            tackles: list[Event], stats: dict[str, int], nstats: dict[str, int]) -> list[Event]:
    windows = skill_windows(clip, cfg.windows)
    stats["windows"] = len(windows)
    feats = [skill_features(clip, w, cfg.windows) for w in windows]
    proba = model.skill.predict_proba(feats) if windows else np.zeros((0, 2))
    ti = model.skill.classes.index("trick") if "trick" in model.skill.classes else None
    events: list[Event] = []

    # A window over a tackle belongs to the contact head.
    busy = [(e.timestamp_s - 0.4, e.timestamp_s + 0.4, {e.player_id, e.secondary_player_id})
            for e in tackles]

    def in_tackle(pid: str, t0: float, t1: float) -> bool:
        return any(pid in who and t0 < b and t1 > a for a, b, who in busy)

    if ti is not None:
        by_player: dict[str, list[tuple[SkillWindow, dict, float]]] = {}
        for w, f, p in zip(windows, feats, proba):
            if p[ti] >= cfg.trick_threshold and not in_tackle(w.player_id, clip.t[w.start], clip.t[w.end]):
                by_player.setdefault(w.player_id, []).append((w, f, float(p[ti])))
                stats["windows_above_threshold"] += 1
        for pid, hits in by_player.items():
            hits.sort(key=lambda h: h[0].start)
            groups: list[list[tuple[SkillWindow, dict, float]]] = []
            for h in hits:
                if groups and h[0].start <= groups[-1][-1][0].end:
                    groups[-1].append(h)
                else:
                    groups.append([h])
            for g in groups:
                if len(g) < cfg.trick_min_windows:
                    stats["groups_too_short"] += 1
                    continue
                stats["reported"] += 1
                best = max(g, key=lambda h: h[2])
                # The move is what every positive window covers; if they do not
                # share a span, fall back to the most confident window.
                a = max(h[0].start for h in g)
                b = min(h[0].end for h in g)
                if b <= a or clip.t[b] - clip.t[a] < 0.3:
                    a, b = best[0].start, best[0].end
                w, f, p = best
                reasons: list[str] = []
                if not _height_px(clip, pid, a, b) >= cfg.min_player_height_px:
                    reasons.append("players_small_for_pose")
                if not f.get("pose_frac", 0.0) >= cfg.min_pose_frac:
                    reasons.append("pose_sparse")
                if not f.get("ball_observed_frac", 0.0) >= cfg.min_ball_seen_frac:
                    reasons.append("ball_unseen")
                _review(reasons, {"trick": p, "dribble": 1 - p}, cfg)
                touches = f.get("touches", np.nan)
                events.append(Event(
                    type=EventType.TRICK,
                    timestamp_s=float(clip.t[a]),
                    end_timestamp_s=float(clip.t[b]),
                    frame_index=int(clip.frame_index[a]),
                    end_frame_index=int(clip.frame_index[b]),
                    confidence=float(np.clip(p, 0, 1)),
                    player_id=pid,
                    secondary_player_id=w.opponent_id,
                    track_ids=_tracks(clip, pid),
                    source="body_events.skill_classifier",
                    detail={
                        "trick_name": "skill_move",
                        "signature": _signature(model, f),
                        "touch_count": None if np.isnan(touches) else int(touches),
                        "probability": round(p, 3),
                        "evidence": model.skill.explain(f, "dribble"),
                        "windows_agreeing": len(g),
                        "needs_review": bool(reasons),
                        "review_reasons": reasons,
                    },
                ))

    for call in find_nutmegs(clip, cfg.nutmeg):
        nstats["rule_hits"] += 1
        t = float(clip.t[call.frame])
        reasons = []
        if in_tackle(call.attacker_id, t - 0.3, t + 0.3):
            # A lunge opens the legs and the ball often squeezes through; that
            # moment is the tackle, already logged.  Note it there instead.
            for e in tackles:
                if ({e.player_id, e.secondary_player_id} >= {call.attacker_id, call.defender_id}
                        and abs(e.timestamp_s - t) < 0.7):
                    e.detail["ball_through_legs"] = True
            nstats["folded_into_tackle"] += 1
            continue
        if not _height_px(clip, call.defender_id, call.frame, call.frame) >= cfg.min_player_height_px:
            reasons.append("players_small_for_pose")
        nutmeg = Event(
            type=EventType.TRICK,
            timestamp_s=max(0.0, t - 0.2),
            end_timestamp_s=t + 0.2,
            frame_index=int(clip.frame_index[call.frame]),
            confidence=call.confidence,
            player_id=call.attacker_id,
            secondary_player_id=call.defender_id,
            track_ids=_tracks(clip, call.attacker_id, call.defender_id),
            source="body_events.nutmeg_rule",
            detail={
                "trick_name": "nutmeg",
                "signature": "ball_through_legs",
                "frames_between_feet": call.inside_frames,
                "depth_error_h": round(call.depth_error_h, 3),
                "needs_review": bool(reasons),
                "review_reasons": reasons,
            },
        )
        # A nutmeg the classifier also saw is one trick, named by the rule.
        merged = False
        for e in events:
            if (e.player_id == call.attacker_id and e.detail.get("trick_name") == "skill_move"
                    and e.timestamp_s - 0.3 <= t <= (e.end_timestamp_s or e.timestamp_s) + 0.3):
                e.detail.update(trick_name="nutmeg", signature="ball_through_legs",
                                classifier_probability=e.detail.get("probability"))
                e.confidence = max(e.confidence, call.confidence)
                e.source = "body_events.nutmeg_rule+skill_classifier"
                merged = True
        if merged:
            nstats["merged_with_skill_call"] += 1
        else:
            nstats["reported"] += 1
            events.append(nutmeg)
    return events
