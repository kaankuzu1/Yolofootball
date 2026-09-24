"""Pose keypoints for the players (plan §5, ``pose/``).

``PoseEstimator``   YOLO pose on player crops, one COCO-17 skeleton per box
``PoseTracker``     wraps any tracker so its player tracks carry keypoints,
                    which :func:`~football_analysis.state.state_from_tracks`
                    then copies into the world-state record
``attach_pose``     fills keypoints into finished ``FrameState`` records by
                    re-reading the clip, for trackers that only settle their
                    boxes after the last frame

Keypoints are in the processed frame's pixel space, like every other
coordinate in the record.  Names follow COCO: ``left_ankle``, ``right_hip`` ...
"""

from football_analysis.pose.estimator import (
    COCO_KEYPOINTS,
    PoseConfig,
    PoseEstimator,
    PoseTracker,
    attach_pose,
)

__all__ = ["COCO_KEYPOINTS", "PoseConfig", "PoseEstimator", "PoseTracker", "attach_pose"]
