"""Tier 1 ball events: possession, pass, shot, goal (plan §4.0, §4.1, §4.4).

Geometry and physics over the tracks, not a trained model, so every call is
interpretable and can be debugged from a single clip.  Every event carries the
rule that fired (``source``), the conditions it checked, and whether the ball
track was interpolated through it.

``BallEventDetector``  the pipeline stage (buffers, decides in ``finalize``)
``detect_ball_events`` the same rules as a pure function of a ``ClipState``
``BallEventConfig``    every threshold, in ball diameters or player heights
``NetMotionMeter``     the one pixel measurement the goal rule needs
"""

from football_analysis.ball_events.config import BallEventConfig
from football_analysis.ball_events.detector import BallEventDetector
from football_analysis.ball_events.engine import BallEventResult, Spell, detect_ball_events
from football_analysis.ball_events.netmotion import NetMotionMeter
from football_analysis.ball_events.state import ClipState, GoalGeometry, PlayerObs

__all__ = [
    "BallEventConfig",
    "BallEventDetector",
    "BallEventResult",
    "ClipState",
    "GoalGeometry",
    "NetMotionMeter",
    "PlayerObs",
    "Spell",
    "detect_ball_events",
]
