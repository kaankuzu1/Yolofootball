"""Tracking: who is who, where the ball is, and the per-frame record (plan §2).

``WorldStateTracker``     the pipeline stage; ``finalize_states()`` after the
                          last frame gives the smoothed, identity-locked
                          :class:`~football_analysis.state.FrameState` records
``BallTrajectoryFilter``  the ball: gated Kalman, RTS smoother, gap bridging
``ByteTracker``           frame-to-frame player association
``IdentityLock``          exactly N player identities for the whole clip
"""

from football_analysis.track.ball import BallGapSpan, BallTrack, BallTrajectoryFilter, SOURCE_NAMES
from football_analysis.track.config import BallFilterConfig, PlayerTrackConfig, TrackLayerConfig
from football_analysis.track.players import ByteTracker, IdentityLock, IdentityResult, PlayerFrameObs
from football_analysis.track.tracker import TrackingOutput, WorldStateTracker

__all__ = [
    "BallFilterConfig",
    "BallGapSpan",
    "BallTrack",
    "BallTrajectoryFilter",
    "ByteTracker",
    "IdentityLock",
    "IdentityResult",
    "PlayerFrameObs",
    "PlayerTrackConfig",
    "SOURCE_NAMES",
    "TrackLayerConfig",
    "TrackingOutput",
    "WorldStateTracker",
]
