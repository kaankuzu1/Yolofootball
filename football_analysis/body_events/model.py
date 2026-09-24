"""The two classifiers, and how they are saved.

``contact``  tackle_won / tackle_failed / shoulder_duel / no_contact
``skill``    trick / dribble

Both are gradient-boosted trees over the hand-built features in
:mod:`.features`.  The plan (§4) sketched a small 1D-CNN or GRU over raw
keypoint sequences; this uses trees over engineered features instead because
the labelled data this has to learn from is tiny (tens of real windows at
best, for now), trees need far less of it, they accept missing values
natively, and a tree over named features can say *why* it called a tackle.
A sequence model becomes worth trying once a few hundred real labelled
windows exist.

A saved model is one ``joblib`` file holding both classifiers, the feature
names each was trained on (so a feature added later cannot silently shift the
columns), per-class reference statistics used to explain a call, and a
metadata block recording what it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from football_analysis.body_events.dataset import Sample, feature_matrix
from football_analysis.body_events.features import CONTACT_FEATURES, SKILL_FEATURES

__all__ = ["Head", "BodyEventModel", "default_model_path", "train_head"]

MODEL_VERSION = "body_events-1"


def default_model_path() -> Path:
    """``assets/body_events/model/body_events.joblib`` in the project checkout."""
    return Path(__file__).resolve().parents[2] / "assets" / "body_events" / "model" / "body_events.joblib"


@dataclass
class Head:
    """One classifier and what it needs to be used and explained."""

    estimator: Any
    features: tuple[str, ...]
    classes: tuple[str, ...]
    reference: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    """``reference[class][feature] = (median, spread)`` over that class's
    training windows, so a call can be explained as 'this feature was far from
    what an ordinary dribble looks like'."""

    def predict_proba(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, len(self.classes)))
        X = np.array([[r.get(n, np.nan) for n in self.features] for r in rows], dtype=float)
        proba = self.estimator.predict_proba(X)
        # Reorder to self.classes whatever order the estimator keeps.
        order = [list(self.estimator.classes_).index(c) for c in self.classes]
        return proba[:, order]

    def explain(self, row: dict[str, float], against: str, top: int = 4) -> list[dict[str, Any]]:
        """The features that most separate ``row`` from class ``against``."""
        ref = self.reference.get(against, {})
        scored = []
        for name, (med, spread) in ref.items():
            v = row.get(name, np.nan)
            if v is None or np.isnan(v) or spread <= 0:
                continue
            scored.append((abs(v - med) / spread, name, v, med))
        scored.sort(reverse=True)
        return [
            {"feature": n, "value": round(float(v), 3), f"typical_{against}": round(float(m), 3)}
            for _, n, v, m in scored[:top]
        ]


def _reference(samples: Sequence[Sample], names: Sequence[str]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for label in sorted({s.label for s in samples}):
        X, _, _ = feature_matrix([s for s in samples if s.label == label], names)
        stats = {}
        for j, n in enumerate(names):
            col = X[:, j][~np.isnan(X[:, j])]
            if len(col) >= 5:
                q1, med, q3 = np.percentile(col, [25, 50, 75])
                stats[n] = (float(med), float(max(q3 - q1, 1e-6)))
        out[label] = stats
    return out


def train_head(samples: Sequence[Sample], names: Sequence[str], classes: Sequence[str],
               sample_weight: np.ndarray | None = None, seed: int = 0) -> Head:
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y, _ = feature_matrix(samples, names)
    est = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=350, l2_regularization=0.5,
        min_samples_leaf=15, class_weight="balanced", random_state=seed,
    )
    est.fit(X, y, sample_weight=sample_weight)
    present = tuple(c for c in classes if c in set(y))
    return Head(est, tuple(names), present, _reference(samples, names))


@dataclass
class BodyEventModel:
    contact: Head
    skill: Head
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": MODEL_VERSION, "contact": self.contact,
                     "skill": self.skill, "meta": self.meta}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BodyEventModel":
        import joblib

        blob = joblib.load(path)
        if blob.get("version") != MODEL_VERSION:
            raise ValueError(f"{path} is model format {blob.get('version')!r}, "
                             f"this build reads {MODEL_VERSION!r}; retrain it")
        return cls(blob["contact"], blob["skill"], blob.get("meta", {}))

    @classmethod
    def train(cls, samples: Sequence[Sample], *, real_weight: float = 5.0, seed: int = 0,
              meta: dict[str, Any] | None = None) -> "BodyEventModel":
        """Fit both heads.  Real labelled windows count ``real_weight`` times a
        simulated one, so a few dozen real windows can pull the model toward
        real motion without being drowned by thousands of simulated ones."""
        from football_analysis.body_events.dataset import CONTACT_LABELS, SKILL_LABELS

        heads = {}
        counts: dict[str, dict[str, int]] = {}
        for kind, names, classes in (("contact", CONTACT_FEATURES, CONTACT_LABELS),
                                     ("skill", SKILL_FEATURES, SKILL_LABELS)):
            chosen = [s for s in samples if s.kind == kind]
            w = np.array([real_weight if s.origin != "synthetic" else 1.0 for s in chosen])
            heads[kind] = train_head(chosen, names, classes, w, seed)
            c: dict[str, int] = {}
            for s in chosen:
                key = f"{s.origin}:{s.label}"
                c[key] = c.get(key, 0) + 1
            counts[kind] = c
        info = {"model_version": MODEL_VERSION, "trained_on": counts, **(meta or {})}
        return cls(heads["contact"], heads["skill"], info)
