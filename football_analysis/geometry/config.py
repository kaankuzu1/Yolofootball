"""Settings for the geometry layer: the goal's size, and how to calibrate.

Kept beside the code rather than in :mod:`football_analysis.config` so this
layer can be built while that module is owned elsewhere; it is a plain
dataclass and can be nested into the shared config as a ``geometry`` section.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["GeometryConfig"]


@dataclass
class GeometryConfig:
    goal_width_m: float = 3.0
    """Inside width of the goal mouth, post to post.  3 x 2 m is a futsal or
    small-sided goal; a full-size goal is 7.32 x 2.44 m."""

    goal_height_m: float = 2.0
    """Ground to the underside of the crossbar."""

    goal_depth_m: float = 1.0
    """How far the net runs back behind the goal line, for 'inside the net'."""

    goal_classes: tuple[str, ...] = ("goal",)

    goal_corners_px: list[list[float]] | None = None
    """Hand-clicked goal mouth corners, in processed-frame pixels: left post
    base, right post base, right crossbar end, left crossbar end (left and
    right as seen in the image).  Overrides anything the detector finds."""

    ground_points_px: list[list[float]] | None = None
    """Hand-clicked ground points (cones), four or more, in pixels ..."""

    ground_points_m: list[list[float]] | None = None
    """... and the same points' measured ground coordinates in metres, in the
    same order, listed counter-clockwise as seen from above."""

    focal_px: float | None = None
    """The camera's focal length in pixels, if known.  Otherwise it is
    estimated from the goal's shape, falling back to ``assumed_hfov_deg``."""

    estimate_focal: bool = True
    assumed_hfov_deg: float = 65.0
    """Horizontal field of view assumed when focal length can be neither read
    nor estimated.  65 degrees is typical of a phone's main camera in video."""

    ball_diameter_m: float = 0.22
    """Size 5 football.  A futsal (size 4) ball is about 0.20 m."""

    player_height_m: float = 1.75
    """Used by the player-height fallback when there is no calibration."""

    player_heights_m: dict[str, float] = field(default_factory=dict)
    """Per-player heights by label, if known; overrides ``player_height_m``."""

    min_goal_keypoint_confidence: float = 0.3
