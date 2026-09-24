"""The tracking layer as a pipeline stage, and the per-frame record it produces.

:class:`WorldStateTracker` satisfies the :class:`~football_analysis.interfaces.Tracker`
protocol, so it drops into :class:`~football_analysis.pipeline.AnalysisPipeline`
as is.  But the best answers it can give need the whole clip -- the identity
lock clusters every player crop at once, and the ball smoother runs backward
from the future -- so it works in two phases:

``update(frame, detections)``
    Online.  Associates players frame to frame (ByteTrack), records kit colour
    features and all candidates, and returns *provisional* tracks so a per-frame
    consumer has something to draw.  Provisional player ids are ByteTrack
    tracklet ids, which can change across an overlap.

``finalize_states()``
    Offline, after the last frame.  Runs the identity lock, the ball filter and
    smoother, and the geometry calibration, and returns one
    :class:`~football_analysis.state.FrameState` per processed frame.  **This is
    the record the event logic should read.**  ``summary()`` then carries what
    belongs to the clip rather than to any frame: the calibration, the identity
    report, and every ball gap.

What the record carries beyond the shared schema, in ``attributes``:

PlayerState
    ``identity_confidence`` (0-1), ``interpolated`` (box filled across a short
    miss), ``occluded`` (overlapping the other player), ``foot_m`` (ground
    ``[X, Y]`` in metres, calibrated runs only), ``speed_mps``.

BallState
    ``source`` (``detected`` / ``smoothed`` / ``bridged``), ``kinematic_break``
    (velocity reset here: a touch, a deflection), ``diameter_px``,
    ``speed_mps``, ``ground_m`` (``[X, Y, Z]`` assuming the ball is on the
    ground), ``range_m`` (``[X, Y, Z]`` from its apparent size, noisy but valid
    in the air), ``gap`` (index into ``summary()["ball_gaps"]`` while hidden).

GoalState
    ``quad`` is the mouth in keypoint order; ``attributes["source"]`` says
    whether it came from configuration, keypoints or only a box, and
    ``net_px`` outlines mouth and net together when calibrated.

FrameState
    ``ball_to_feet_m``: each player's scale-corrected distance from the ball
    (plan §4.0's possession input), and ``geometry``: the calibration method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from football_analysis.geometry.config import GeometryConfig
from football_analysis.geometry.scene import SceneGeometry, build_scene_geometry
from football_analysis.interfaces import BaseTracker
from football_analysis.state import BallState, FrameState, GoalState, PlayerState
from football_analysis.track.appearance import kit_feature, pitch_hue
from football_analysis.track.ball import SOURCE_NAMES, BallTrack, BallTrajectoryFilter, DETECTED
from football_analysis.track.config import TrackLayerConfig
from football_analysis.track.observations import Obs, to_obs_list
from football_analysis.track.players import ByteTracker, IdentityLock, IdentityResult, PlayerFrameObs
from football_analysis.types import BALL, BBox, GOAL, PLAYER, Point, Track, VideoFrame

__all__ = ["WorldStateTracker", "TrackingOutput"]


@dataclass
class _FrameRecord:
    index: int
    timestamp_s: float
    size: tuple[int, int]
    scale: float
    exact: bool
    players: list[PlayerFrameObs]
    balls: list[Obs]
    goals: list[Obs]


@dataclass
class TrackingOutput:
    states: list[FrameState]
    summary: dict[str, Any]
    ball: BallTrack
    identities: IdentityResult
    geometry: SceneGeometry
    labels: dict[int, str] = field(default_factory=dict)


class WorldStateTracker(BaseTracker):
    """ByteTrack + identity lock for players, Kalman/RTS for the ball, goal geometry."""

    def __init__(
        self,
        config: Any | None = None,
        *,
        layer: TrackLayerConfig | None = None,
        geometry: GeometryConfig | None = None,
    ) -> None:
        self.layer = layer or TrackLayerConfig.from_config(config)
        self.geometry_config = geometry or GeometryConfig()
        self._bytetrack = ByteTracker(self.layer.players)
        self._frames: list[_FrameRecord] = []
        self._output: TrackingOutput | None = None

    def reset(self) -> None:
        self._bytetrack.reset()
        self._frames = []
        self._output = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": "WorldStateTracker",
            "players": "ByteTrack + global kit-colour identity lock",
            "ball": "constant-acceleration Kalman, Mahalanobis gate, RTS smoother",
            "geometry": "goal-mouth PnP, ground-point and player-height fallbacks",
            "num_identities": self.layer.players.num_identities,
        }

    # -- online ------------------------------------------------------------------

    def update(self, frame: VideoFrame, detections: list[Any]) -> list[Track]:
        obs = to_obs_list(detections)
        self.observe(
            index=frame.index, timestamp_s=frame.timestamp_s, image=frame.image,
            detections=obs, scale=frame.scale, exact=frame.timestamp_is_exact,
        )
        return self._provisional_tracks(frame)

    def observe(
        self,
        *,
        index: int,
        timestamp_s: float,
        image: np.ndarray | None,
        detections: Sequence[Any],
        frame_size: tuple[int, int] | None = None,
        scale: float = 1.0,
        exact: bool = True,
    ) -> None:
        """Record one frame.  ``image`` may be ``None`` (no kit colours then)."""
        pcfg = self.layer.players
        obs = to_obs_list(detections)
        if image is not None:
            size = (image.shape[1], image.shape[0])
        elif frame_size is not None:
            size = tuple(frame_size)
        else:
            raise ValueError("observe() needs an image or a frame_size")
        people = [o for o in obs if o.cls in pcfg.player_classes and o.conf >= pcfg.low_confidence]
        ids = self._bytetrack.step([o.xyxy() for o in people], [o.conf for o in people])
        hue = pitch_hue(image) if image is not None else None
        players = []
        for o, tid in zip(people, ids):
            feature = kit_feature(image, o.xyxy(), hue, pcfg.torso_band) if image is not None else None
            edge = o.x1 <= 2 or o.y1 <= 2 or o.x2 >= size[0] - 2 or o.y2 >= size[1] - 2
            players.append(PlayerFrameObs(o.xyxy(), o.conf, tid, feature, edge))
        self._frames.append(_FrameRecord(
            index=index, timestamp_s=float(timestamp_s), size=size, scale=scale, exact=exact,
            players=players,
            balls=[o for o in obs if o.cls in self.layer.ball.ball_classes],
            goals=[o for o in obs if o.cls in self.geometry_config.goal_classes],
        ))
        self._output = None

    def _provisional_tracks(self, frame: VideoFrame) -> list[Track]:
        rec = self._frames[-1]
        tracks: list[Track] = []
        for p in rec.players:
            if p.tracklet is None:
                continue
            tracks.append(Track(
                track_id=p.tracklet, class_name=PLAYER, bbox=BBox(*p.box), confidence=p.conf,
                frame_index=frame.index, timestamp_s=frame.timestamp_s,
                attributes={"provisional": True},
            ))
        if rec.balls:
            b = max(rec.balls, key=lambda o: o.conf)
            if b.conf >= 0.3:
                tracks.append(Track(
                    track_id=0, class_name=BALL, bbox=BBox(*b.xyxy()), confidence=b.conf,
                    frame_index=frame.index, timestamp_s=frame.timestamp_s,
                    attributes={"provisional": True, "interpolated": False},
                ))
        if rec.goals:
            g = max(rec.goals, key=lambda o: o.conf)
            tracks.append(Track(
                track_id=-1, class_name=GOAL, bbox=BBox(*g.xyxy()), confidence=g.conf,
                frame_index=frame.index, timestamp_s=frame.timestamp_s,
            ))
        return tracks

    # -- offline -------------------------------------------------------------------

    def finalize_states(self) -> list[FrameState]:
        return self.finalize().states

    def summary(self) -> dict[str, Any]:
        return self.finalize().summary

    def finalize(self) -> TrackingOutput:
        if self._output is None:
            self._output = self._build()
        return self._output

    def _build(self) -> TrackingOutput:
        frames = self._frames
        n = len(frames)
        pcfg = self.layer.players
        times = [f.timestamp_s for f in frames]

        identities = IdentityLock(pcfg).run([f.players for f in frames])
        labels = self._labels(identities)

        player_boxes = [[p.box for p in f.players if p.conf >= 0.3] for f in frames]
        ball = BallTrajectoryFilter(self.layer.ball).run(times, [f.balls for f in frames], player_boxes)

        # Geometry: every frame is another look at the same static goal.
        feet_y, heights = [], []
        for k, f in enumerate(frames):
            for i, p in enumerate(f.players):
                if (identities.assign[k][i] is not None and not identities.occluded[k][i]
                        and not p.at_edge and p.conf >= 0.5):
                    feet_y.append(p.box[3]); heights.append(p.box[3] - p.box[1])
        goal_kps = [g.keypoints for f in frames for g in f.goals
                    if g.keypoints is not None and g.conf >= 0.3]
        goal_boxes = [g.xyxy() for f in frames for g in f.goals if g.conf >= 0.3]
        size = frames[0].size if frames else (0, 0)
        scene = build_scene_geometry(
            self.geometry_config, size, goal_keypoints=goal_kps, goal_boxes=goal_boxes,
            player_samples=(feet_y, heights) if feet_y else None,
        )

        players_by_frame = self._player_series(frames, identities, labels, scene)
        gaps = self._gap_records(ball, times, players_by_frame, scene)
        gap_of = np.full(n, -1)
        for gi, g in enumerate(gaps):
            gap_of[g["start_frame_pos"]:g["end_frame_pos"] + 1] = gi

        states: list[FrameState] = []
        goal_state = self._goal_state(scene)
        last_seen = -10**9
        for k, f in enumerate(frames):
            if ball.source[k] == DETECTED:
                last_seen = k
            ball_state = self._ball_state(ball, k, k - last_seen, scene, gap_of[k])
            fs = FrameState(
                frame_index=f.index, timestamp_s=f.timestamp_s, players=players_by_frame[k],
                ball=ball_state, goal=goal_state, frame_size=f.size, scale=f.scale,
                timestamp_is_exact=f.exact, attributes={"geometry": scene.method},
            )
            if ball_state is not None and fs.players:
                fs.attributes["ball_to_feet_m"] = {
                    p.player_id: _round(self._feet_distance_m(p, ball_state, scene), 3)
                    for p in fs.players
                }
            states.append(fs)

        summary = {
            "schema": "football_analysis.tracking_summary/1",
            "frames": n,
            "players": {labels[j]: {"identity": j} for j in sorted(labels)},
            "identity": identities.report,
            "ball": ball.stats,
            "ball_gaps": [{k: v for k, v in g.items() if not k.endswith("_pos")} for g in gaps],
            "geometry": scene.to_dict(),
        }
        return TrackingOutput(states, summary, ball, identities, scene, labels)

    # -- players -------------------------------------------------------------------

    def _labels(self, identities: IdentityResult) -> dict[int, str]:
        """Name identities in order of first appearance, left to right on ties."""
        pcfg = self.layer.players
        first: dict[int, tuple[int, float]] = {}
        for k, f in enumerate(self._frames):
            for i, p in enumerate(f.players):
                j = identities.assign[k][i]
                if j is not None and j not in first:
                    first[j] = (k, 0.5 * (p.box[0] + p.box[2]))
        order = sorted(first, key=lambda j: first[j])
        names = list(pcfg.labels) + [f"Player {i + 1}" for i in range(len(pcfg.labels), 64)]
        return {j: names[r] for r, j in enumerate(order)}

    def _player_series(self, frames, identities, labels, scene) -> list[list[PlayerState]]:
        n = len(frames)
        pcfg = self.layer.players
        times = np.array([f.timestamp_s for f in frames])
        out: list[list[PlayerState]] = [[] for _ in range(n)]
        for j, label in labels.items():
            boxes = np.full((n, 4), np.nan)
            conf = np.zeros(n)
            idconf = np.zeros(n)
            occluded = np.zeros(n, bool)
            tracklet = np.full(n, -1)
            for k in range(n):
                for i, p in enumerate(frames[k].players):
                    if identities.assign[k][i] == j:
                        boxes[k] = p.box
                        conf[k] = p.conf
                        idconf[k] = identities.confidence[k][i]
                        occluded[k] = identities.occluded[k][i]
                        tracklet[k] = p.tracklet if p.tracklet is not None else -1
            seen = ~np.isnan(boxes[:, 0])
            filled = np.zeros(n, bool)
            idx = np.where(seen)[0]
            for a, b in zip(idx[:-1], idx[1:]):
                if b - a > 1 and times[b] - times[a] <= pcfg.max_fill_gap_s:
                    for k in range(a + 1, b):
                        s = (times[k] - times[a]) / (times[b] - times[a])
                        boxes[k] = (1 - s) * boxes[a] + s * boxes[b]
                        filled[k] = True
                        idconf[k] = min(idconf[a], idconf[b])
                        tracklet[k] = tracklet[a]
            have = seen | filled
            foot = np.c_[0.5 * (boxes[:, 0] + boxes[:, 2]), boxes[:, 3]]
            vel = _smoothed_velocity(foot, times, have)
            foot_m = scene.ground_xy(np.where(have[:, None], foot, 0.0)) if scene.has_world else None
            speeds = scene.speed_mps(np.where(have[:, None], foot, 0.0), np.nan_to_num(vel))
            for k in np.where(have)[0]:
                attrs: dict[str, Any] = {
                    "identity_confidence": round(float(idconf[k]), 3),
                    "interpolated": bool(filled[k]),
                    "occluded": bool(occluded[k]),
                }
                if foot_m is not None and np.all(np.isfinite(foot_m[k])):
                    attrs["foot_m"] = np.round(foot_m[k], 3).tolist()
                if np.all(np.isfinite(vel[k])) and np.isfinite(speeds[k]):
                    attrs["speed_mps"] = round(float(speeds[k]), 2)
                out[k].append(PlayerState(
                    player_id=label, track_id=int(j), bbox=BBox(*boxes[k]),
                    confidence=float(conf[k]),
                    velocity=Point(float(vel[k][0]), float(vel[k][1])) if np.all(np.isfinite(vel[k])) else None,
                    attributes=attrs,
                ))
        for row in out:
            row.sort(key=lambda p: p.player_id)
        return out

    # -- ball -----------------------------------------------------------------------

    def _ball_state(self, ball: BallTrack, k: int, since: int, scene: SceneGeometry, gap: int):
        if not ball.present[k]:
            return None
        x, y = ball.xy[k]
        d = float(ball.diameter_px[k]) if np.isfinite(ball.diameter_px[k]) else ball.reference_diameter_px
        v = ball.velocity[k]
        attrs: dict[str, Any] = {
            "source": SOURCE_NAMES[int(ball.source[k])],
            "kinematic_break": bool(ball.kinematic_break[k]),
            "diameter_px": round(d, 2),
        }
        if gap >= 0:
            attrs["gap"] = int(gap)
        # Measured at the ball's contact point: the ground is one radius below its centre.
        if np.all(np.isfinite(v)):
            speed = float(scene.speed_mps(np.array([[x, y + 0.5 * d]]), v[None])[0])
            if np.isfinite(speed):
                attrs["speed_mps"] = round(speed, 2)
        if scene.has_world:
            on_ground, from_size = scene.ball_world(np.array([[x, y]]), np.array([d]))
            if np.all(np.isfinite(on_ground[0])):
                attrs["ground_m"] = np.round(on_ground[0], 3).tolist()
            if from_size is not None and np.all(np.isfinite(from_size[0])):
                attrs["range_m"] = np.round(from_size[0], 2).tolist()
        observed = ball.source[k] == DETECTED
        return BallState(
            position=Point(float(x), float(y)),
            confidence=float(ball.confidence[k]) if observed else 0.0,
            bbox=BBox(x - d / 2, y - d / 2, x + d / 2, y + d / 2),
            velocity=Point(float(v[0]), float(v[1])) if np.all(np.isfinite(v)) else None,
            interpolated=not observed,
            track_id=int(ball.segment[k]) if ball.segment[k] >= 0 else None,
            frames_since_seen=int(max(0, since)) if since < 10**8 else 0,
            attributes=attrs,
        )

    def _feet_distance_m(self, p: PlayerState, b: BallState, scene: SceneGeometry) -> float:
        """Ball to the player's foot region (bottom quarter of the box), in metres."""
        x1, y1, x2, y2 = p.bbox.as_tuple()
        top = y2 - 0.25 * (y2 - y1)
        bx, by = b.position.x, b.position.y
        dx = max(x1 - bx, 0.0, bx - x2)
        dy = max(top - by, 0.0, by - y2)
        scale = float(scene.px_per_metre(np.array([[0.5 * (x1 + x2), y2]]))[0])
        if not np.isfinite(scale) or scale <= 0:
            return float("nan")
        return float(np.hypot(dx, dy) / scale)

    def _gap_records(self, ball: BallTrack, times, players_by_frame, scene) -> list[dict[str, Any]]:
        """Every ball gap with where it happened and who was nearest (plan §2.2)."""
        out = []
        for g in ball.gaps:
            rec: dict[str, Any] = {
                "start_frame": self._frames[g.start].index,
                "end_frame": self._frames[g.end].index,
                "start_frame_pos": g.start,
                "end_frame_pos": g.end,
                "start_s": round(times[g.start], 3),
                "duration_s": round((times[g.after] if g.after is not None else times[g.end])
                                    - (times[g.before] if g.before is not None else times[g.start]), 3),
                "kind": g.kind,
                "filled": g.filled,
            }
            for side, k in (("before", g.before), ("after", g.after)):
                if k is None:
                    continue
                xy = ball.xy[k]
                v = ball.velocity[k]
                rec[f"{side}_xy"] = np.round(xy, 1).tolist()
                if np.all(np.isfinite(v)):
                    rec[f"{side}_velocity_px_s"] = np.round(v, 1).tolist()
                near = None
                for p in players_by_frame[k]:
                    d = p.position.distance_to(Point(*xy))
                    if near is None or d < near[1]:
                        near = (p.player_id, d)
                if near is not None:
                    rec[f"{side}_nearest_player"] = near[0]
                    rec[f"{side}_nearest_px"] = round(near[1], 1)
            vb, va = rec.get("before_velocity_px_s"), rec.get("after_velocity_px_s")
            if vb is not None and va is not None:
                sb, sa = float(np.hypot(*vb)), float(np.hypot(*va))
                rec["speed_ratio"] = round(sa / sb, 3) if sb > 1e-6 else None
                if sb > 1e-6 and sa > 1e-6:
                    cosang = float(np.dot(vb, va) / (sb * sa))
                    rec["heading_change_deg"] = round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 1)
            out.append(rec)
        return out

    # -- goal ------------------------------------------------------------------------

    def _goal_state(self, scene: SceneGeometry) -> GoalState | None:
        if scene.goal_quad_px is None:
            return None
        q = scene.goal_quad_px
        attrs: dict[str, Any] = {"source": scene.goal_quad_source}
        net = scene.goal_net_polygon_px() if scene.has_world else None
        if net is not None:
            attrs["net_px"] = np.round(net, 1).tolist()
        return GoalState(
            bbox=BBox(float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())),
            quad=[Point(float(x), float(y)) for x, y in q],
            confidence=1.0 if scene.goal_quad_source in ("configured", "keypoints") else 0.5,
            attributes=attrs,
        )


def _smoothed_velocity(xy: np.ndarray, times: np.ndarray, valid: np.ndarray, half: int = 2) -> np.ndarray:
    """Least-squares slope over a small window: robust to single-frame box jitter."""
    n = len(xy)
    out = np.full((n, 2), np.nan)
    for k in np.where(valid)[0]:
        lo, hi = max(0, k - half), min(n, k + half + 1)
        w = np.arange(lo, hi)
        w = w[valid[w]]
        if len(w) < 3:
            continue
        t = times[w] - times[k]
        if np.ptp(t) <= 0:
            continue
        A = np.c_[t, np.ones_like(t)]
        coef, *_ = np.linalg.lstsq(A, xy[w], rcond=None)
        out[k] = coef[0]
    return out


def _round(v: float, nd: int) -> float | None:
    return round(v, nd) if np.isfinite(v) else None
