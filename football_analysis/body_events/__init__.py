"""Tier 2 events: tackles and tricks, read from how the bodies moved (plan §4.2, §4.3).

A tackle and a shoulder bump can leave the ball in the same place; the
difference is in the legs.  A trick and a dribble can cover the same ground;
the difference is how much the feet moved for how little the ball did.  So
this package reads pose keypoints from the world-state record, cuts candidate
windows, computes the plan's features over them, and classifies:

``BodyEventDetector``  the pipeline stage (buffers, decides in ``finalize``)
``detect_body_events`` the same as a pure function of a clip's records
``BodyEventModel``     the two classifiers, saved as one file
``WindowDataset``      the labelled-window store under ``assets/body_events``
``synthetic``          the motion simulator the first model was trained on
``find_nutmegs``       the nutmeg, as an explicit geometric rule

Pose itself comes from :mod:`football_analysis.pose`.

Command line (``python -m football_analysis.body_events --help``): ``train``,
``evaluate``, ``propose`` (cut windows from a clip for labelling), ``label``,
``detect``.
"""

from football_analysis.body_events.dataset import (
    CONTACT_LABELS,
    SKILL_LABELS,
    Sample,
    WindowDataset,
    synthetic_samples,
)
from football_analysis.body_events.detector import (
    BodyEventConfig,
    BodyEventDetector,
    detect_body_events,
)
from football_analysis.body_events.features import WindowConfig
from football_analysis.body_events.model import BodyEventModel, default_model_path
from football_analysis.body_events.rules import NutmegConfig, find_nutmegs

__all__ = [
    "BodyEventConfig",
    "BodyEventDetector",
    "BodyEventModel",
    "CONTACT_LABELS",
    "NutmegConfig",
    "SKILL_LABELS",
    "Sample",
    "WindowConfig",
    "WindowDataset",
    "default_model_path",
    "detect_body_events",
    "find_nutmegs",
    "synthetic_samples",
]
