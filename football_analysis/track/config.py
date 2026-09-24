"""Settings for the tracking layer.

These are separate dataclasses rather than new keys on
:class:`football_analysis.config.TrackingConfig` so this layer can be built and
tested while the config module is owned elsewhere.  :meth:`from_config` reads
the handful of keys the shared config already carries (the number of players,
their labels, the lost-track budget), so a run configured the ordinary way
still behaves consistently.

Distances in :class:`BallFilterConfig` are in **ball diameters**, not pixels.
The ball is the one object whose real size is fixed, so its median detected
size is a ruler that is always available, before any camera calibration, and
it makes every threshold here independent of the clip's resolution and of how
far the camera stands from play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["BallFilterConfig", "PlayerTrackConfig", "TrackLayerConfig"]


@dataclass
class BallFilterConfig:
    """The physics-gated Kalman filter and smoother for the ball (plan §2.2)."""

    ball_classes: tuple[str, ...] = ("ball",)

    min_candidate_confidence: float = 0.05
    """Detections below this are ignored outright.  The detector is run low
    on purpose (plan §1.3.6); this filter, not the detector, supplies precision."""

    init_confidence: float = 0.5
    """A lone detection this confident may start a track without confirmation."""

    measurement_sigma: float = 0.2
    """Detector centre noise, in ball diameters."""

    jerk_density: float = 2.0e3
    """White-jerk spectral density, in diameters^2 / s^5, while the ball is free.
    Loose enough to follow rolling friction, bounces and perspective curvature."""

    contact_jerk_scale: float = 50.0
    """Multiplier on ``jerk_density`` while the ball is within reach of a
    player's feet, where a touch can change its velocity in one frame."""

    gate_chi2: float = 13.8
    """Mahalanobis gate (2 dof).  13.8 keeps 99.9% of true detections."""

    init_accel_sigma: float = 25.0
    """Prior spread of a new track's acceleration, diameters / s^2.  Gravity on a
    0.22 m ball is ~45; most of the time the ball is rolling and it is ~0."""

    coast_min_score: float = 0.15
    """After two missed frames the gate has widened; from then on a candidate
    must score at least this to be accepted, so a coasting track cannot drift
    onto low-grade clutter."""

    break_innovation_chi2: float = 9.0
    """An accepted detection this surprising marks a velocity change (a touch),
    and the smoother is split there rather than blurring the kick."""

    max_speed: float = 180.0
    """Physical ceiling, diameters per second.  ~40 m/s for a 0.22 m ball."""

    break_min_confidence: float = 0.25
    """A detection outside the gate but inside physical reach is taken as a
    touch (velocity reset) only when it is at least this confident."""

    max_coast_s: float = 0.4
    """Coast on the prediction this long through missed frames (plan: ~0.4 s)."""

    confirm_hits: int = 3
    """A tentative ball track needs this many hits to take over."""

    confirm_window: int = 5
    """... within this many frames."""

    bridge_max_s: float = 1.0
    """Gaps between two confirmed track segments up to this long are bridged
    with an interpolated path, when the implied speed is physically possible."""

    head_zone_fraction: float = 0.35
    """Top fraction of a player box where ball-like false positives (heads)
    cluster.  Candidates there are down-weighted, not dropped: headers exist."""

    head_zone_penalty: float = 0.5

    fixed_camera: bool = True
    """Enables the static-clutter map: a 'ball' that sits on the same pixels for
    a large part of the clip is almost certainly background.  Turn off for a
    panning camera, where static pixels do not mean a static object."""

    clutter_min_fraction: float = 0.2
    """A cell is clutter when candidates appear there in this fraction of frames."""

    clutter_penalty: float = 0.2

    reach_diameters: float = 4.0
    """How far past a player's feet the ball can be and still be touchable."""


@dataclass
class PlayerTrackConfig:
    """ByteTrack association plus the global two-identity lock (plan §2.1)."""

    player_classes: tuple[str, ...] = ("player", "goalkeeper", "referee")
    """Everything person-shaped.  A 1v1 has no goalkeeper or referee, so when a
    model says either it is almost always mislabelling one of the two players."""

    num_identities: int = 2
    labels: list[str] = field(default_factory=lambda: ["Player 1", "Player 2"])

    high_confidence: float = 0.45
    """ByteTrack's first-stage threshold."""

    low_confidence: float = 0.1
    """Detections between low and high only extend existing tracks."""

    new_track_confidence: float = 0.5
    match_iou: float = 0.2
    """Minimum IoU to associate a high-confidence detection."""

    low_match_iou: float = 0.4
    max_lost_frames: int = 30

    overlap_iou: float = 0.05
    """Two player boxes overlapping this much are treated as an occlusion
    episode; tracklets are cut at both ends of it and re-identified after."""

    merge_memory_frames: int = 5
    """A box is also treated as an occlusion when a player seen within this many
    frames has vanished inside it (the detector merged two bodies into one)."""

    switch_penalty: float = 8.0
    """Inside an overlap, the cost (in units of the colour clusters' own spread)
    of a tracklet changing identity between frames.  Higher trusts the
    frame-to-frame tracker more; lower trusts each crop's colours more."""

    unassigned_cost: float = 30.0
    """Cost of leaving an overlapping detection without an identity (a third
    person, or a duplicate box), against giving it the nearest free one."""

    min_centroid_distance: float = 0.25
    """Absolute floor on how different the two kits' colour signatures must be
    (Euclidean distance between square-rooted histograms)."""

    min_separation: float = 1.5
    """Below this Fisher ratio between the two appearance clusters, the players
    are judged to look alike and identity falls back to motion continuity,
    with a warning (plan §2.1 fallback)."""

    max_fill_gap_s: float = 0.5
    """A player missing for up to this long has their box interpolated."""

    torso_band: tuple[float, float] = (0.15, 0.6)
    """Vertical slice of the box used for kit colour: shirt and shorts."""


@dataclass
class TrackLayerConfig:
    ball: BallFilterConfig = field(default_factory=BallFilterConfig)
    players: PlayerTrackConfig = field(default_factory=PlayerTrackConfig)

    @classmethod
    def from_config(cls, config: Any | None) -> "TrackLayerConfig":
        """Build from the shared :class:`football_analysis.config.Config`.

        Reads only keys that already exist there, and tolerates their absence.
        """
        layer = cls()
        tracking = getattr(config, "tracking", None)
        if tracking is not None:
            layer.players.num_identities = int(
                getattr(tracking, "max_players", layer.players.num_identities)
            )
            labels = getattr(tracking, "player_labels", None)
            if labels:
                layer.players.labels = list(labels)
            layer.players.max_lost_frames = int(
                getattr(tracking, "max_age_frames", layer.players.max_lost_frames)
            )
        return layer
