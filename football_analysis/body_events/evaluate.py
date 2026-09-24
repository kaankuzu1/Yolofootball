"""Event-level scoring: did the detector call the tackle, at the right time,
on the right player -- and how often did it call one that was not there?

Window accuracy flatters a detector (most windows are easy dribbles), so this
scores what a user would see: one list of calls per clip against one list of
true events.  A call matches a true event of the same type, on the same
player, within ``tolerance_s``; each true event matches at most one call.

The two failure rates the plan asks for directly are reported as their own
numbers, per clip type:

* ``tackle_on_shoulder_duel``  share of shoulder-duel clips given a tackle
* ``trick_on_dribble``         share of ordinary-dribble clips given a trick
* ``trick_on_turn``            share of ordinary-turn clips given a trick
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from football_analysis.body_events.detector import BodyEventConfig, detect_body_events
from football_analysis.body_events.model import BodyEventModel
from football_analysis.events import Event, EventType

__all__ = ["TruthEvent", "ClipResult", "score", "evaluate_synthetic", "truth_from_synthetic"]


@dataclass
class TruthEvent:
    type: EventType
    start_s: float
    end_s: float
    player_id: str
    outcome: str | None = None
    """``won`` / ``failed`` for tackles."""


@dataclass
class ClipResult:
    clip: str
    kind: str
    truth: list[TruthEvent]
    calls: list[Event]
    matches: list[tuple[int, int]] = field(default_factory=list)


def truth_from_synthetic(seq: Any) -> list[TruthEvent]:
    if seq.contact_label in ("tackle_won", "tackle_failed"):
        tc = float(seq.params["contact_s"])
        return [TruthEvent(EventType.TACKLE, tc, tc, seq.defender_id,
                           "won" if seq.contact_label == "tackle_won" else "failed")]
    if seq.skill_label == "trick":
        return [TruthEvent(EventType.TRICK, seq.event_start_s, seq.event_end_s, seq.attacker_id)]
    return []


def _match(truth: Sequence[TruthEvent], calls: Sequence[Event], tol: float) -> list[tuple[int, int]]:
    pairs = []
    used: set[int] = set()
    for ti, te in enumerate(truth):
        best, best_d = None, np.inf
        for ci, c in enumerate(calls):
            if ci in used or c.type != te.type or c.player_id != te.player_id:
                continue
            c0 = c.timestamp_s
            c1 = c.end_timestamp_s if c.end_timestamp_s is not None else c0
            d = max(0.0, te.start_s - c1, c0 - te.end_s)
            if d <= tol and d < best_d:
                best, best_d = ci, d
        if best is not None:
            used.add(best)
            pairs.append((ti, best))
    return pairs


def score(results: Iterable[ClipResult], tolerance_s: float = 0.5) -> dict[str, Any]:
    results = list(results)
    out: dict[str, Any] = {"clips": len(results), "tolerance_s": tolerance_s}
    for etype in (EventType.TACKLE, EventType.TRICK):
        tp = fp = fn = 0
        outcome_ok = outcome_n = 0
        timing = []
        for r in results:
            r.matches = _match(r.truth, r.calls, tolerance_s)
            truth = [i for i, t in enumerate(r.truth) if t.type == etype]
            calls = [i for i, c in enumerate(r.calls) if c.type == etype]
            hit_t = {ti for ti, _ in r.matches if r.truth[ti].type == etype}
            hit_c = {ci for _, ci in r.matches if r.calls[ci].type == etype}
            tp += len(hit_t)
            fn += len(truth) - len(hit_t)
            fp += len(calls) - len(hit_c)
            for ti, ci in r.matches:
                if r.truth[ti].type != etype:
                    continue
                timing.append(abs(r.calls[ci].timestamp_s - r.truth[ti].start_s))
                if etype is EventType.TACKLE:
                    outcome_n += 1
                    outcome_ok += r.calls[ci].detail.get("outcome") == r.truth[ti].outcome
        p = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * rc / (p + rc) if p + rc > 0 else float("nan")
        block = {"tp": tp, "fp": fp, "fn": fn, "precision": _r(p), "recall": _r(rc), "f1": _r(f1),
                 "median_timing_error_s": _r(float(np.median(timing))) if timing else None}
        if etype is EventType.TACKLE:
            block["outcome_accuracy"] = _r(outcome_ok / outcome_n) if outcome_n else None
        out[etype.value] = block

    per_kind: dict[str, dict[str, float]] = {}
    for r in results:
        k = per_kind.setdefault(r.kind, {"clips": 0, "with_tackle_call": 0, "with_trick_call": 0,
                                          "flagged_calls": 0, "calls": 0})
        k["clips"] += 1
        k["with_tackle_call"] += any(c.type is EventType.TACKLE for c in r.calls)
        k["with_trick_call"] += any(c.type is EventType.TRICK for c in r.calls)
        k["calls"] += len(r.calls)
        k["flagged_calls"] += sum(bool(c.detail.get("needs_review")) for c in r.calls)
    for k in per_kind.values():
        n = max(1, k["clips"])
        k["tackle_rate"] = _r(k["with_tackle_call"] / n)
        k["trick_rate"] = _r(k["with_trick_call"] / n)
    out["per_clip_type"] = per_kind
    out["tackle_on_shoulder_duel"] = per_kind.get("shoulder_duel", {}).get("tackle_rate")
    out["tackle_on_no_contact"] = per_kind.get("no_contact", {}).get("tackle_rate")
    out["trick_on_dribble"] = per_kind.get("dribble", {}).get("trick_rate")
    out["trick_on_turn"] = per_kind.get("turn", {}).get("trick_rate")
    nut = [r for r in results if r.kind == "nutmeg"]
    if nut:
        named = sum(any(c.detail.get("trick_name") == "nutmeg" for c in r.calls) for r in nut)
        out["nutmeg_named_rate"] = _r(named / len(nut))
    wrong_nut = [r for r in results if r.kind != "nutmeg"]
    if wrong_nut:
        out["nutmeg_on_other_clips"] = _r(
            sum(any(c.detail.get("trick_name") == "nutmeg" for c in r.calls) for r in wrong_nut)
            / len(wrong_nut))
    return out


def _r(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def evaluate_synthetic(model: BodyEventModel, seqs: Iterable[Any],
                       cfg: BodyEventConfig | None = None, tolerance_s: float = 0.5) -> tuple[dict, list[ClipResult]]:
    results = []
    for seq in seqs:
        calls = detect_body_events(seq.states, model, cfg)
        results.append(ClipResult(f"{seq.scenario}:{seq.seed}", seq.scenario,
                                  truth_from_synthetic(seq), calls))
    return score(results, tolerance_s), results
