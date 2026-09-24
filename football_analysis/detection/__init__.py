"""Object detection for football analysis.

Exposes a single stable entry point, :class:`Detector`, whose ``detect(frame)``
returns a list of :class:`Detection` for one BGR frame.  The rest of the
pipeline should not need to know which backend or weights are in use.
"""

from .types import Detection, BALL, PLAYER, GOALKEEPER, REFEREE, GOAL
from .detector import Detector, DetectorConfig

__all__ = [
    "Detection",
    "Detector",
    "DetectorConfig",
    "BALL",
    "PLAYER",
    "GOALKEEPER",
    "REFEREE",
    "GOAL",
]
