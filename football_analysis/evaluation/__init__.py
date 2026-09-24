"""Evaluation harness (plan §7): hand-labelled events and a scorer.

``load_labels`` / ``LabelSet``  the label file format, one JSON file per clip
``draft_labels``               seed a label file from a run, to correct by hand
``score`` / ``score_many``     match a run's events to labels within a tolerance
                               and report right / missed / invented per type,
                               next to the plan's v1 targets

Command line: ``python -m football_analysis.evaluation {score,draft,check}``.
"""

from football_analysis.evaluation.labels import Label, LabelSet, draft_labels, load_labels
from football_analysis.evaluation.score import TARGETS, Report, TypeScore, load_events, match, score, score_many

__all__ = [
    "TARGETS",
    "Label",
    "LabelSet",
    "Report",
    "TypeScore",
    "draft_labels",
    "load_events",
    "load_labels",
    "match",
    "score",
    "score_many",
]
