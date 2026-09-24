"""Score a run's events against hand labels (plan §7, "How we know it's right").

Per event type, a system event and a label *match* when they are within the
temporal tolerance (±1 s by default, SoccerNet's mAP@1 convention).  Matching
is one-to-one and maximises the number of matches first, then minimises the
total time error, so two events 0.3 s apart can't both claim one label.

After matching, every scored type reports:

- right (true positives), missed (labels nothing matched), invented (system
  events that matched nothing);
- precision, recall, F1, and AP (average precision ranking the system's events
  by confidence, the number published work reports as mAP);
- the plan's v1 target for that type and whether it is met.

Shots and goals are also scored at ±0.5 s, where the plan says precision is cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from football_analysis.evaluation.labels import Label, LabelSet

__all__ = ["TARGETS", "TypeScore", "Report", "load_events", "match", "score", "score_many"]

DEFAULT_TOLERANCE_S = 1.0
TIGHT_TOLERANCE_S = 0.5
TIGHT_TYPES = ("shot", "goal")
SMALL_SAMPLE = 10

# Plan §7 v1 targets.  Goals need both precision and recall; the rest are F1.
TARGETS: dict[str, dict[str, float]] = {
    "goal": {"precision": 0.95, "recall": 0.95},
    "shot": {"f1": 0.85},
    "pass": {"f1": 0.80},
    "tackle": {"f1": 0.75},
    "trick": {"f1": 0.60},
}


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Events from a run's ``events.json`` (an EventTimeline, or any ``{"events": [...]}``)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload["events"] if isinstance(payload, dict) else payload
    return [e for e in events if "type" in e and "timestamp_s" in e]


def match(times: list[float], labels: list[Label], tolerance_s: float) -> list[tuple[int, int]]:
    """One-to-one (event index, label index) pairs within the tolerance.

    Maximum number of pairs first, least total error second: every feasible
    pair costs its error, every infeasible one more than all feasible errors
    together, so the assignment never trades a match away to shave error.
    """
    if not times or not labels:
        return []
    cost = np.array([[lab.distance_s(t) for lab in labels] for t in times], dtype=float)
    feasible = cost <= tolerance_s + 1e-9
    big = tolerance_s * (len(times) + len(labels)) + 1.0
    rows, cols = linear_sum_assignment(np.where(feasible, cost, big))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if feasible[r, c]]


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p == 0 or r == 0:
        return 0.0
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r)


def average_precision(
    events: list[dict[str, Any]], labels: list[Label], ambiguous: list[Label], tolerance_s: float
) -> float | None:
    """AP with events ranked by confidence, each taking the nearest free label.

    The SoccerNet way of computing it: walk the events from most to least
    confident; one within tolerance of an unclaimed label is right, one within
    tolerance of only an ambiguous label is skipped, anything else is wrong.
    """
    if not labels:
        return None
    order = sorted(events, key=lambda e: -float(e.get("confidence", 1.0)))
    free = list(range(len(labels)))
    tp = fp = 0
    points: list[tuple[float, float]] = []
    for e in order:
        t = float(e["timestamp_s"])
        near = [(labels[i].distance_s(t), i) for i in free if labels[i].distance_s(t) <= tolerance_s + 1e-9]
        if near:
            free.remove(min(near)[1])
            tp += 1
        elif any(a.distance_s(t) <= tolerance_s + 1e-9 for a in ambiguous):
            continue
        else:
            fp += 1
        points.append((tp / len(labels), tp / (tp + fp)))
    ap, prev_recall = 0.0, 0.0
    for k, (recall, _) in enumerate(points):
        if recall > prev_recall:
            ap += (recall - prev_recall) * max(p for _, p in points[k:])
            prev_recall = recall
    return ap


@dataclass
class TypeScore:
    """Everything one type's score says, including which moments went wrong."""

    type: str
    tolerance_s: float
    labelled: int
    ambiguous: int
    predicted: int
    right: int
    missed: list[Label]
    invented: list[dict[str, Any]]
    ignored: int
    errors_s: list[float]
    ap: float | None
    player_checked: int = 0
    player_same: int = 0
    tight: "TypeScore | None" = None

    @property
    def precision(self) -> float | None:
        return _ratio(self.right, self.right + len(self.invented))

    @property
    def recall(self) -> float | None:
        return _ratio(self.right, self.labelled)

    @property
    def f1(self) -> float | None:
        return _f1(self.precision, self.recall)

    @property
    def mean_abs_error_s(self) -> float | None:
        return float(np.mean(self.errors_s)) if self.errors_s else None

    def target_status(self) -> str:
        target = TARGETS.get(self.type)
        if target is None:
            return "no target"
        if self.labelled == 0:
            return "no labels"
        values = {"precision": self.precision, "recall": self.recall, "f1": self.f1}
        met = all((values[k] or 0.0) >= v for k, v in target.items())
        return "met" if met else "not met"

    def to_dict(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        out: dict[str, Any] = {
            "tolerance_s": self.tolerance_s,
            "labelled": self.labelled,
            "ambiguous_labels": self.ambiguous,
            "predicted": self.predicted,
            "right": self.right,
            "missed": len(self.missed),
            "invented": len(self.invented),
            "ignored_on_ambiguous": self.ignored,
            "precision": r(self.precision),
            "recall": r(self.recall),
            "f1": r(self.f1),
            "ap": r(self.ap),
            "mean_abs_error_s": r(self.mean_abs_error_s),
            "target": TARGETS.get(self.type),
            "target_status": self.target_status(),
            "small_sample": self.labelled < SMALL_SAMPLE,
            "missed_at": [lab.to_dict() for lab in self.missed],
            "invented_at": [
                {k: e.get(k) for k in ("timestamp_s", "confidence", "player_id", "source") if e.get(k) is not None}
                for e in self.invented
            ],
        }
        if self.player_checked:
            out["player_agreement"] = {"checked": self.player_checked, "same": self.player_same}
        if self.tight is not None:
            out["tight"] = self.tight.to_dict()
        return out


def _score_type(
    event_type: str, events: list[dict[str, Any]], labels: list[Label], tolerance_s: float
) -> TypeScore:
    real = [lab for lab in labels if not lab.ambiguous]
    fuzzy = [lab for lab in labels if lab.ambiguous]
    times = [float(e["timestamp_s"]) for e in events]
    pairs = match(times, real, tolerance_s)
    used_e = {i for i, _ in pairs}
    used_l = {j for _, j in pairs}
    rest = [i for i in range(len(events)) if i not in used_e]
    soaked = {rest[i] for i, _ in match([times[i] for i in rest], fuzzy, tolerance_s)}
    checked = same = 0
    for i, j in pairs:
        who, truth = events[i].get("player_id"), real[j].player_id
        if who is not None and truth is not None:
            checked += 1
            same += who == truth
    return TypeScore(
        type=event_type,
        tolerance_s=tolerance_s,
        labelled=len(real),
        ambiguous=len(fuzzy),
        predicted=len(events),
        right=len(pairs),
        missed=[real[j] for j in range(len(real)) if j not in used_l],
        invented=[events[i] for i in rest if i not in soaked],
        ignored=len(soaked),
        errors_s=[real[j].distance_s(times[i]) for i, j in pairs],
        ap=average_precision(events, real, fuzzy, tolerance_s),
        player_checked=checked,
        player_same=same,
    )


@dataclass
class Report:
    """The scorecard for one or more clips."""

    tolerance_s: float
    types: dict[str, TypeScore]
    unscored: dict[str, int]
    confusions: list[dict[str, Any]]
    clips: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tolerance_s": self.tolerance_s,
            "tight_tolerance_s": TIGHT_TOLERANCE_S,
            "clips": self.clips,
            "types": {t: s.to_dict() for t, s in self.types.items()},
            "unscored_system_events": self.unscored,
            "type_confusions": self.confusions,
        }

    def to_markdown(self) -> str:
        def f(x: float | None) -> str:
            return "–" if x is None else f"{x:.2f}"

        def target(t: str) -> str:
            spec = TARGETS.get(t)
            return ", ".join(f"{k[0].upper() if k != 'f1' else 'F1'} ≥ {v:.2f}" for k, v in spec.items()) if spec else "–"

        lines = [
            f"Scored at ±{self.tolerance_s:g} s on {', '.join(self.clips) or 'one clip'}.",
            "",
            "| Type | Labelled | Found | Right | Missed | Invented | Precision | Recall | F1 | AP | Target | Status |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t, s in self.types.items():
            status = s.target_status() + (" (few labels)" if s.labelled and s.labelled < SMALL_SAMPLE and t in TARGETS else "")
            lines.append(
                f"| {t} | {s.labelled} | {s.predicted} | {s.right} | {len(s.missed)} | {len(s.invented)} | "
                f"{f(s.precision)} | {f(s.recall)} | {f(s.f1)} | {f(s.ap)} | {target(t)} | {status} |"
            )
        who = [s for s in self.types.values() if s.player_checked]
        if who:
            lines += ["", "Right player on the right events: " + ", ".join(
                f"{s.type} {s.player_same}/{s.player_checked}" for s in who)]
        tight = [s.tight for s in self.types.values() if s.tight is not None and s.tight.labelled]
        if tight:
            lines += ["", f"At ±{TIGHT_TOLERANCE_S:g} s:"]
            for s in tight:
                lines.append(f"- {s.type}: precision {f(s.precision)}, recall {f(s.recall)}, F1 {f(s.f1)}")
        wrong = [(s.type, lab, None) for s in self.types.values() for lab in s.missed]
        wrong += [(s.type, None, e) for s in self.types.values() for e in s.invented]
        if wrong:
            lines += ["", "What went wrong:"]
            for t, lab, e in sorted(wrong, key=lambda w: w[1].start_s if w[1] else float(w[2]["timestamp_s"])):
                if lab is not None:
                    when = f"{lab.start_s:.2f}–{lab.end_s:.2f} s" if lab.is_window else f"{lab.start_s:.2f} s"
                    lines.append(f"- missed {t} at {when}" + (f" ({lab.note})" if lab.note else ""))
                else:
                    lines.append(f"- invented {t} at {float(e['timestamp_s']):.2f} s (confidence {float(e.get('confidence', 0)):.2f})")
        if self.confusions:
            lines += ["", "Right moment, wrong type:"]
            for c in self.confusions:
                lines.append(f"- system said {c['predicted_type']} at {c['predicted_s']:.2f} s, label says {c['label_type']} at {c['label_s']:.2f} s")
        if self.unscored:
            lines += ["", "Not scored (the labels don't cover these types): "
                      + ", ".join(f"{n} {t}" for t, n in sorted(self.unscored.items()))]
        return "\n".join(lines) + "\n"


def score_many(
    runs: Iterable[tuple[list[dict[str, Any]], LabelSet]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> Report:
    """Pool several clips: counts add up across clips, matching never crosses them."""
    per_type: dict[str, list[TypeScore]] = {}
    tight: dict[str, list[TypeScore]] = {}
    unscored: dict[str, int] = {}
    confusions: list[dict[str, Any]] = []
    clips: list[str] = []
    for events, labelset in runs:
        clips.append(labelset.clip or "clip")
        events = [e for e in events if labelset.in_span(float(e["timestamp_s"]))]
        scored = list(labelset.labelled_types)
        for e in events:
            if e["type"] not in scored:
                unscored[e["type"]] = unscored.get(e["type"], 0) + 1
        this_clip: dict[str, TypeScore] = {}
        for t in scored:
            mine = [e for e in events if e["type"] == t]
            this_clip[t] = _score_type(t, mine, labelset.of_type(t), tolerance_s)
            per_type.setdefault(t, []).append(this_clip[t])
            if t in TIGHT_TYPES:
                tight.setdefault(t, []).append(
                    _score_type(t, mine, labelset.of_type(t), min(tolerance_s, TIGHT_TOLERANCE_S))
                )
        confusions += _confusions(this_clip, tolerance_s)
    types = {t: _pool(scores) for t, scores in per_type.items()}
    for t, scores in tight.items():
        types[t].tight = _pool(scores)
    order = [t for t in ("goal", "shot", "tackle", "pass", "trick") if t in types]
    order += sorted(t for t in types if t not in order)
    return Report(tolerance_s, {t: types[t] for t in order}, unscored, confusions, clips)


def score(
    events: list[dict[str, Any]], labels: LabelSet, tolerance_s: float = DEFAULT_TOLERANCE_S
) -> Report:
    return score_many([(events, labels)], tolerance_s)


def _confusions(latest: dict[str, TypeScore], tolerance_s: float) -> list[dict[str, Any]]:
    """Invented events sitting on a missed label of another type (a pass that was a turnover)."""
    out = []
    for t, s in latest.items():
        for e in s.invented:
            ts = float(e["timestamp_s"])
            for other, o in latest.items():
                if other == t:
                    continue
                for lab in o.missed:
                    if lab.distance_s(ts) <= tolerance_s:
                        out.append({"predicted_type": t, "predicted_s": ts,
                                    "label_type": other, "label_s": lab.timestamp_s})
    return out


def _pool(scores: list[TypeScore]) -> TypeScore:
    if len(scores) == 1:
        return scores[0]
    first = scores[0]
    labelled = sum(s.labelled for s in scores)
    # AP does not pool by summing; weight each clip's AP by its label count.
    aps = [(s.ap, s.labelled) for s in scores if s.ap is not None]
    return TypeScore(
        type=first.type,
        tolerance_s=first.tolerance_s,
        labelled=labelled,
        ambiguous=sum(s.ambiguous for s in scores),
        predicted=sum(s.predicted for s in scores),
        right=sum(s.right for s in scores),
        missed=[lab for s in scores for lab in s.missed],
        invented=[e for s in scores for e in s.invented],
        ignored=sum(s.ignored for s in scores),
        errors_s=[x for s in scores for x in s.errors_s],
        ap=sum(a * n for a, n in aps) / sum(n for _, n in aps) if aps else None,
        player_checked=sum(s.player_checked for s in scores),
        player_same=sum(s.player_same for s in scores),
    )
