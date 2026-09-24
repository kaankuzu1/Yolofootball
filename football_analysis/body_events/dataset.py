"""Labelled windows: the on-disk dataset, and labels for simulated clips.

The dataset is designed to grow when real footage arrives, so it stores what
the pipeline *produced*, not what a feature function computed from it:

    assets/body_events/
      labels.csv              one row per labelled window (see LABEL_COLUMNS)
      windows/<window_id>.jsonl
                              the world-state records covering the window plus
                              half a second either side, in the state-cache
                              format (football_analysis.state)

Features are recomputed from the stored records at training time, so a change
to a feature never invalidates a label.  A window is labelled once, by a
person looking at the clip; ``propose`` (see ``__main__``) cuts candidate
windows and a review sheet for each so that person only has to fill in the
``label`` column.

Label vocabulary
----------------
``kind = contact``  ``tackle_won`` / ``tackle_failed`` / ``shoulder_duel`` /
                    ``no_contact``  -- plan §4.2's four classes
``kind = skill``    ``trick`` / ``dribble``, with the named move (``stepover``,
                    ``nutmeg``, ``dragback``, ``feint`` ...) in ``sub_label``
                    when known -- plan §4.3's binary first step
``label = skip``    looked at, unusable (occluded, off-frame, not a 1v1)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from football_analysis.body_events.features import (
    ContactWindow,
    SkillWindow,
    WindowConfig,
    contact_features,
    contact_windows,
    skill_features,
    skill_windows,
)
from football_analysis.body_events.series import ClipSeries
from football_analysis.state import STATE_SCHEMA_VERSION, FrameState

__all__ = [
    "CONTACT_LABELS",
    "SKILL_LABELS",
    "LABEL_COLUMNS",
    "Sample",
    "WindowDataset",
    "synthetic_samples",
]

CONTACT_LABELS = ("tackle_won", "tackle_failed", "shoulder_duel", "no_contact")
SKILL_LABELS = ("dribble", "trick")
LABEL_COLUMNS = (
    "window_id", "kind", "label", "sub_label", "source_clip", "start_s", "end_s",
    "anchor_s", "player_id", "other_player_id", "origin", "labeller", "notes",
)
MARGIN_S = 0.5


@dataclass
class Sample:
    """One labelled window, as the classifier sees it."""

    kind: str
    label: str
    features: dict[str, float]
    group: str
    """Windows from the same clip share a group, so a split never puts one
    clip on both sides of a train/test line."""
    origin: str = "synthetic"
    sub_label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Simulated clips


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def synthetic_samples(seq: Any, cfg: WindowConfig | None = None) -> list[Sample]:
    """Every candidate window the pipeline would cut from a simulated clip,
    labelled from the script's ground truth.

    Windows are cut by the same code that cuts them from real footage, so the
    training distribution is the inference distribution.  Windows that only
    partly cover the action are dropped rather than forced into a class.
    """
    cfg = cfg or WindowConfig()
    clip = ClipSeries.from_states(seq.states)
    t = clip.t
    e0, e1 = seq.event_start_s, seq.event_end_s
    group = f"syn:{seq.scenario}:{seq.seed}"
    out: list[Sample] = []

    for w in contact_windows(clip, cfg):
        ta = float(t[w.anchor])
        pair = {w.attacker_id, w.defender_id} == {seq.attacker_id, seq.defender_id}
        if seq.contact_label is not None and pair and e0 - 0.35 <= ta <= e1 + 0.35:
            label = seq.contact_label
        elif seq.contact_label is not None and ta > e1 and ta < e1 + 1.2:
            continue  # the aftermath of the same duel; neither clean nor independent
        else:
            label = "no_contact"
        out.append(Sample("contact", label, contact_features(clip, w, cfg), group,
                          meta={"scenario": seq.scenario, "anchor_s": ta,
                                "attacker_ok": w.attacker_id == seq.attacker_id}))

    for w in skill_windows(clip, cfg):
        w0, w1 = float(t[w.start]), float(t[w.end])
        ov = _overlap(w0, w1, e0, e1)
        ev_len = max(1e-6, e1 - e0)
        mine = w.player_id == seq.attacker_id
        if seq.skill_label == "trick" and mine:
            if ov >= min(0.6 * ev_len, 0.6 * (w1 - w0)):
                label = "trick"
            elif ov == 0.0:
                label = "dribble"
            else:
                continue
        elif seq.contact_label is not None and ov > 0:
            continue  # a window over a duel is the contact head's business
        else:
            label = "dribble"
        out.append(Sample("skill", label, skill_features(clip, w, cfg), group,
                          sub_label=seq.sub_label if label == "trick" else None,
                          meta={"scenario": seq.scenario, "start_s": w0, "end_s": w1}))
    return out


# ----------------------------------------------------------------------------
# Real, labelled windows on disk


class WindowDataset:
    """The labelled-window store under ``assets/body_events``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.labels_path = self.root / "labels.csv"
        self.windows_dir = self.root / "windows"

    def rows(self) -> list[dict[str, str]]:
        if not self.labels_path.exists():
            return []
        with self.labels_path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def write_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.labels_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(LABEL_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in LABEL_COLUMNS})

    def add_window(self, window_id: str, states: Iterable[FrameState], row: dict[str, Any]) -> Path:
        """Store the records for one window and append (or replace) its row."""
        self.windows_dir.mkdir(parents=True, exist_ok=True)
        path = self.windows_dir / f"{window_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"schema_version": STATE_SCHEMA_VERSION, "record": "header"}) + "\n")
            for s in states:
                fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        rows = [r for r in self.rows() if r["window_id"] != window_id]
        rows.append({**row, "window_id": window_id})
        self.write_rows(rows)
        return path

    def load_states(self, window_id: str) -> list[FrameState]:
        from football_analysis.state import read_state_cache

        return list(read_state_cache(self.windows_dir / f"{window_id}.jsonl"))

    def samples(self, cfg: WindowConfig | None = None) -> list[Sample]:
        """Every labelled window, with features recomputed from its records."""
        cfg = cfg or WindowConfig()
        out: list[Sample] = []
        for row in self.rows():
            label = (row.get("label") or "").strip()
            kind = (row.get("kind") or "").strip()
            if not label or label == "skip":
                continue
            valid = CONTACT_LABELS if kind == "contact" else SKILL_LABELS
            if label not in valid:
                raise ValueError(f"{row['window_id']}: label {label!r} is not one of {valid}")
            clip = ClipSeries.from_states(self.load_states(row["window_id"]))
            i0 = clip.index_at(float(row["start_s"]))
            i1 = clip.index_at(float(row["end_s"]))
            if kind == "contact":
                anchor = clip.index_at(float(row.get("anchor_s") or row["start_s"]))
                w = ContactWindow(attacker_id=row["player_id"], defender_id=row["other_player_id"],
                                  start=i0, end=i1, anchor=anchor)
                feats = contact_features(clip, w, cfg)
            else:
                w = SkillWindow(player_id=row["player_id"], start=i0, end=i1,
                                opponent_id=row.get("other_player_id") or None)
                feats = skill_features(clip, w, cfg)
            out.append(Sample(kind, label, feats, group=f"real:{row.get('source_clip', '')}",
                              origin=row.get("origin") or "real",
                              sub_label=row.get("sub_label") or None,
                              meta={"window_id": row["window_id"]}))
        return out

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.rows():
            key = f"{row.get('kind')}:{row.get('label') or 'unlabelled'}"
            out[key] = out.get(key, 0) + 1
        return out


def feature_matrix(samples: Sequence[Sample], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[s.features.get(n, np.nan) for n in names] for s in samples], dtype=float)
    y = np.array([s.label for s in samples])
    g = np.array([s.group for s in samples])
    return X, y, g
