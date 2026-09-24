"""The tracking layer end to end: detections in, FrameState records out."""

from __future__ import annotations

import json

import numpy as np

from football_analysis.geometry import GeometryConfig, GoalModel
from football_analysis.state import FrameState
from football_analysis.track import WorldStateTracker
from football_analysis.types import BBox, Detection

from test_geometry import SIDE_ANGLE, SIZE, _camera


def _clip():
    """Two players walking, a ball rolled between them, a goal clicked once."""
    cam = _camera(*SIDE_ANGLE)
    goal = GoalModel(3.0, 2.0)
    frames = []
    for k in range(60):
        t = k / 25.0
        dets = []
        for x in (-2.0 + 0.5 * t, 2.0 - 0.5 * t):
            foot = cam.ground_to_image([[x, 6.0]])[0]
            head = cam.project([[x, 6.0, 1.8]])[0]
            h = foot[1] - head[1]
            dets.append(Detection("player", BBox(foot[0] - 0.2 * h, head[1], foot[0] + 0.2 * h, foot[1]), 0.9))
        if k % 3:  # the ball is missed every third frame
            b = cam.project([[-1.5 + 1.2 * t, 5.0, 0.11]])[0]
            dets.append(Detection("ball", BBox(b[0] - 6, b[1] - 6, b[0] + 6, b[1] + 6), 0.6))
        frames.append((k, t, dets))
    corners = cam.project(goal.mouth_corners()).round(1).tolist()
    return frames, corners


def test_states_carry_identities_ball_and_world_coordinates():
    frames, corners = _clip()
    tracker = WorldStateTracker(geometry=GeometryConfig(goal_corners_px=corners))
    for k, t, dets in frames:
        tracker.observe(index=k, timestamp_s=t, image=None, detections=dets, frame_size=SIZE)
    out = tracker.finalize()
    states = out.states
    assert len(states) == 60 and all(isinstance(s, FrameState) for s in states)
    assert {p.player_id for p in states[10].players} == {"Player 1", "Player 2"}
    # Every frame has a ball; the missed ones are marked as not observed.
    assert all(s.ball is not None for s in states[2:])
    assert states[3].ball.interpolated and not states[4].ball.interpolated
    # Geometry: goal PnP, feet on the ground at Y = 6 m, ball speed ~1.2 m/s.
    assert out.geometry.method == "goal_pnp"
    feet = np.array([p.attributes["foot_m"] for p in states[30].players])
    assert np.allclose(feet[:, 1], 6.0, atol=0.1)
    assert abs(states[30].ball.attributes["speed_mps"] - 1.2) < 0.3
    assert states[30].goal is not None and len(states[30].goal.quad) == 4
    assert "ball_to_feet_m" in states[30].attributes
    # The record and the summary both serialise.
    json.dumps([s.to_dict() for s in states])
    json.dumps(out.summary)
    FrameState.from_dict(states[30].to_dict())
