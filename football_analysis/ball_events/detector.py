"""The ball-event rules as a pipeline stage.

:class:`BallEventDetector` satisfies
:class:`~football_analysis.interfaces.EventDetector`.  It buffers the per-frame
record and decides everything in :meth:`finalize`, because the rules need to
look forward: whether a release was a pass or a shot depends on where the ball
ended up (plan §4.1), and a goal is only confirmed once the ball has failed to
come back out (plan §4.4).  ``update`` therefore returns nothing.

The last run's full reasoning is kept on :attr:`BallEventDetector.last_result`
-- contacts, flights, goal checks, the per-frame possessor and the review list
-- for the overlay, for debugging, and for the tackle and trick logic, which
reads possession from here rather than re-deriving it.
"""

from __future__ import annotations

from typing import Any

from football_analysis.ball_events.config import BallEventConfig
from football_analysis.ball_events.engine import BallEventResult, detect_ball_events
from football_analysis.ball_events.state import ClipState
from football_analysis.events import Event

__all__ = ["BallEventDetector"]


class BallEventDetector:
    def __init__(self, config: Any | None = None, *, rules: BallEventConfig | None = None) -> None:
        self.config = config
        self.rules = (rules or BallEventConfig.from_config(config)).validate()
        self._states: list[Any] = []
        self.last_state: ClipState | None = None
        self.last_result: BallEventResult | None = None

    def reset(self) -> None:
        self._states = []
        self.last_state = None
        self.last_result = None

    def update(self, state: Any) -> list[Event]:
        self._states.append(state)
        return []

    def finalize(self) -> list[Event]:
        if not self._states:
            return []
        clip = ClipState.from_frame_states(
            self._states, goal_min_confidence=self.rules.goal_min_track_confidence
        )
        result = detect_ball_events(clip, self.rules)
        self.last_state, self.last_result = clip, result
        self._states = []
        return list(result.events)

    def possessor_at(self, timestamp_s: float) -> str | None:
        """Who had the ball at ``timestamp_s`` in the last run, if anyone."""
        if self.last_state is None or self.last_result is None:
            return None
        return self.last_result.possessor_at_time(self.last_state, timestamp_s)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": type(self).__name__,
            "emits": ["shot", "goal", "pass", "push_past", "possession_change", "out_of_play"],
            "pass_mode": self.rules.pass_mode,
        }
        if self.last_result is not None:
            out["summary"] = self.last_result.summary()
            if self.last_result.review:
                out["needs_review"] = self.last_result.review
        return out
