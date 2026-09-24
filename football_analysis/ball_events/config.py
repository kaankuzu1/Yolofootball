"""Thresholds for the ball-event rules.

Every distance here is in one of two rulers, never in raw pixels, so the same
settings work at any resolution and any camera distance:

``D``   the ball's own diameter in pixels (the clip's median detected size).
        One ball diameter is ~0.22 m, so 45 D/s is ~10 m/s.  This is the same
        ruler :mod:`football_analysis.track` uses.
``H``   the height of the player box in question.  Used for "is the ball at
        this player's feet", where a *local* ruler matters: under a side angle
        a metre near the camera is many more pixels than a metre far from it,
        and the player standing right there is the best local ruler we have.

:meth:`BallEventConfig.from_config` maps the handful of pixel-denominated keys
in the shared :class:`football_analysis.config.EventConfig` onto these only
when a caller explicitly asks it to; the defaults below are what the rules are
tuned to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

__all__ = ["BallEventConfig", "PassMode"]

PassMode = str
"""``"auto"`` | ``"teams"`` | ``"any"`` | ``"never"``.  See :attr:`BallEventConfig.pass_mode`."""

_PASS_MODES = ("auto", "teams", "any", "never")


@dataclass
class BallEventConfig:
    # -- rulers ---------------------------------------------------------------

    ball_diameter_px: float | None = None
    """Override the ruler.  ``None`` = measure it from the clip."""

    ball_to_player_height: float = 0.125
    """Fallback ruler when the ball was never sized: a 0.22 m ball next to a
    ~1.75 m player."""

    # -- possession (plan §4.0) -------------------------------------------------

    foot_band_top: float = 0.30
    """The foot region starts this fraction of H above the box bottom."""

    foot_band_below: float = 0.15
    """... and extends this fraction of H below it.  A ball nearer the camera
    than the player projects lower in the image than their feet."""

    reach: float = 0.30
    """Ball within this many H of the foot region is at that player's feet."""

    possession_min_frames: int = 3
    """Consecutive frames at one player's feet before they are said to have it."""

    control_speed_h: float = 1.5
    """Median ball speed relative to the player, in H/s, below which the ball
    is being controlled rather than rolling past their feet (~2.6 m/s)."""

    contested_margin: float = 0.10
    """When two players are both within reach and their normalised distances
    differ by less than this, the frame is contested and credits nobody."""

    # -- flights: the ball between two possessions ------------------------------

    release_speed_window_s: float = 0.16
    """Release velocity is the median over this long after the ball leaves."""

    min_flight_distance_d: float = 6.0
    """A flight shorter than this (~1.3 m) returning to the same player is a
    dribble touch and is folded into the possession."""

    max_track_gap_s: float = 0.6
    """A ball unseen for longer than this mid-flight ends the flight as ``lost``."""

    rest_speed_d: float = 3.0
    """Below this many D/s the ball has stopped."""

    rest_min_s: float = 0.5
    """... and must stay stopped this long for a flight to end at ``rest``."""

    edge_margin_px: float = 12.0
    """A ball last seen within this of the frame edge left the frame."""

    # -- passes (plan §4.1) -----------------------------------------------------

    pass_mode: PassMode = "auto"
    """Who a transfer between two players counts as a pass for.

    ``auto``   in a clip with exactly two identities and no feeder, a transfer
               is a turnover (the defender won it), never a pass; with a feeder
               configured, or three or more identities, a transfer is a pass
               unless ``teams`` says the two are opponents.
    ``teams``  a pass only between players ``teams`` puts on the same side, or
               to/from a feeder.
    ``any``    every completed transfer is a pass.
    ``never``  no passes at all; transfers are turnovers.
    """

    feeders: list[str] = field(default_factory=list)
    """``player_id``\\ s with the feeder/server role (plan §4.1's third player)."""

    teams: dict[str, str] = field(default_factory=dict)
    """``player_id`` -> side, e.g. ``{"Player 1": "A", "Feeder": "A"}``."""

    pass_min_distance_d: float = 8.0
    """A transfer shorter than this (~1.8 m) is a touch or a duel, not a pass."""

    pass_max_duration_s: float = 4.0

    # -- push past (plan §4.1 default for a strict 1v1) ------------------------

    push_past_min_distance_d: float = 8.0
    push_past_max_duration_s: float = 3.0
    push_past_max_lateral_h: float = 2.5
    """The defender must be within this many of their own heights of the ball's
    line to have been 'passed'."""

    # -- shots ------------------------------------------------------------------

    shot_min_speed_d: float = 30.0
    """Release speed floor, D/s (~6.6 m/s).  A tiebreaker, not the main test."""

    shot_cone_deg: float = 35.0
    """Release direction within this of the release-to-goal vector."""

    shot_on_target_margin: float = 0.10
    """Extrapolated path passing within this fraction of the goal's size of the
    goal mouth still counts as on target."""

    goal_near_h: float = 1.5
    """'Near the goal' = within this many goal heights of the goal mouth."""

    shot_reception_near_goal_h: float = 2.5
    """An opponent collecting a goalward shot within this many goal heights of
    the goal made a save or a block."""

    # -- goals (plan §4.4) --------------------------------------------------------

    goal_height_m: float = 2.0
    ball_diameter_m: float = 0.22

    depth_ratio_range: tuple[float, float] = (0.6, 1.8)
    """Ball diameter / expected diameter at the goal plane.  Outside this, the
    ball is in front of or behind the goal and only overlaps it in the image."""

    absorb_window_s: float = 0.4
    absorb_speed_ratio: float = 0.5
    """Speed after entry below this fraction of the entry speed = absorbed."""

    reemerge_window_s: float = 0.5
    reemerge_distance_h: float = 0.8
    """Ball further than this many goal heights outside the mouth within the
    window has come back out (a post, a bar, a save)."""

    net_margin: float = 0.25
    """The net region is the goal quad grown by this fraction of its size."""

    net_baseline_s: float = 2.0
    net_window_s: float = 0.6
    net_spike_ratio: float = 2.0
    net_spike_mads: float = 4.0
    net_min_pixels: int = 64
    """Fewer usable pixels than this in the net region = no reading."""

    net_min_energy: float = 1.0
    """A spike must also clear this mean grey-level change, so a perfectly
    still baseline does not turn sensor noise into a ripple."""

    goal_min_track_confidence: float = 0.2

    # -- output -----------------------------------------------------------------

    emit_possession_changes: bool = True
    emit_out_of_play: bool = True

    source_name: str = "ball_events"

    def validate(self) -> "BallEventConfig":
        if self.pass_mode not in _PASS_MODES:
            raise ValueError(f"pass_mode must be one of {_PASS_MODES}, got {self.pass_mode!r}")
        if self.possession_min_frames < 1:
            raise ValueError("possession_min_frames must be >= 1")
        lo, hi = self.depth_ratio_range
        if not 0 < lo < hi:
            raise ValueError("depth_ratio_range must be (lo, hi) with 0 < lo < hi")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BallEventConfig":
        payload = dict(payload or {})
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown ball_events keys: {sorted(unknown)}")
        if "depth_ratio_range" in payload:
            payload["depth_ratio_range"] = tuple(payload["depth_ratio_range"])
        return cls(**payload).validate()

    @classmethod
    def from_config(cls, config: Any | None) -> "BallEventConfig":
        """Build from the shared :class:`~football_analysis.config.Config`.

        Reads only keys that already exist there and are unit-free, so a run
        configured the ordinary way stays consistent: the minimum possession
        frames, the shot cone and the pass duration.  The pixel keys in that
        config predate the ball-diameter ruler and are not converted.
        """
        out = cls()
        events = getattr(config, "events", None)
        if events is not None:
            out.possession_min_frames = int(
                getattr(events, "possession_min_frames", out.possession_min_frames)
            )
            out.shot_cone_deg = float(getattr(events, "shot_goal_cone_deg", out.shot_cone_deg))
            out.pass_max_duration_s = float(
                getattr(events, "pass_max_duration_s", out.pass_max_duration_s)
            )
        return out.validate()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out
