"""What the ball-event rules read: the whole clip, one row per frame.

This is the event side of plan §5's world-state record, reduced to the fields
the Tier 1 rules actually use.  It is built once per clip and never mutated by
the rules, which makes a run a pure function of it.  That in turn means a
cached state can be re-run in milliseconds while the thresholds are tuned.

Three ways in:

* :meth:`ClipState.from_frame_states` takes the pipeline's per-frame records
  (:class:`football_analysis.state.FrameState`), including the extras the
  tracking layer puts on them (ball provenance, kinematic breaks, the goal's
  net outline) and the net-motion reading.  This is the normal path.
* :meth:`ClipState.from_ball_track` takes the tracking layer's whole-clip
  :class:`~football_analysis.track.ball.BallTrack` directly, for experiments
  that bypass the pipeline.
* :meth:`ClipState.from_frames` takes plain per-frame dicts; the other two
  are built on it.

Everything is in the pixel space of the processed frame, as the stage contract
requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

__all__ = [
    "ABSENT",
    "DETECTED",
    "SMOOTHED",
    "BRIDGED",
    "PlayerObs",
    "GoalGeometry",
    "ClipState",
]

# Provenance codes, identical to football_analysis.track.ball's.
ABSENT, DETECTED, SMOOTHED, BRIDGED = 0, 1, 2, 3
_SOURCE_CODES = {"absent": ABSENT, "detected": DETECTED, "smoothed": SMOOTHED,
                 "bridged": BRIDGED, "interpolated": SMOOTHED, "predicted": SMOOTHED}

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class PlayerObs:
    """One player in one frame."""

    player_id: str
    bbox: Box
    track_id: int | None = None
    confidence: float = 1.0
    interpolated: bool = False

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def foot(self) -> tuple[float, float]:
        return (0.5 * (self.bbox[0] + self.bbox[2]), self.bbox[3])


@dataclass
class GoalGeometry:
    """The goal mouth as a quad in image space.

    Corner order follows :mod:`football_analysis.track.observations`: left post
    base, right post base, right crossbar end, left crossbar end, where left
    and right are as they appear in the image.
    """

    quad: np.ndarray
    confidence: float = 1.0
    back_quad: np.ndarray | None = None
    """Back of the net, when geometry supplies it."""
    net_outline: np.ndarray | None = None
    """Mouth and net together as one image polygon, when calibration gives it
    (the tracking layer's ``net_px``).  Takes precedence over ``back_quad``."""

    def __post_init__(self) -> None:
        self.quad = np.asarray(self.quad, dtype=float).reshape(4, 2)
        if self.back_quad is not None:
            self.back_quad = np.asarray(self.back_quad, dtype=float).reshape(4, 2)
        if self.net_outline is not None:
            self.net_outline = np.asarray(self.net_outline, dtype=float).reshape(-1, 2)

    @classmethod
    def from_bbox(cls, bbox: Sequence[float], confidence: float = 1.0) -> "GoalGeometry":
        """A detector that only gives a box: its corners stand in for the posts."""
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return cls(np.array([[x1, y2], [x2, y2], [x2, y1], [x1, y1]]), confidence)

    @property
    def polygon(self) -> np.ndarray:
        """The region the ball can be 'in the goal' in, in image space."""
        if self.net_outline is not None and len(self.net_outline) >= 3:
            pts = np.vstack([self.quad, self.net_outline]).astype(np.float32)
            return cv2.convexHull(pts).reshape(-1, 2).astype(float)
        if self.back_quad is None:
            return self.quad
        pts = np.vstack([self.quad, self.back_quad]).astype(np.float32)
        return cv2.convexHull(pts).reshape(-1, 2).astype(float)

    @property
    def centre(self) -> np.ndarray:
        return self.quad.mean(axis=0)

    @property
    def height_px(self) -> float:
        """Mean post length -- the 2 m (or whatever) ruler at the goal plane."""
        left = np.linalg.norm(self.quad[3] - self.quad[0])
        right = np.linalg.norm(self.quad[2] - self.quad[1])
        return float(max(1.0, 0.5 * (left + right)))

    @property
    def width_px(self) -> float:
        bottom = np.linalg.norm(self.quad[1] - self.quad[0])
        top = np.linalg.norm(self.quad[2] - self.quad[3])
        return float(max(1.0, 0.5 * (bottom + top)))

    def signed_distance(self, xy: Sequence[float], polygon: np.ndarray | None = None) -> float:
        """Positive inside, negative outside, in pixels."""
        poly = (self.polygon if polygon is None else polygon).astype(np.float32).reshape(-1, 1, 2)
        return float(cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), True))

    def grown(self, fraction: float) -> np.ndarray:
        """The mouth polygon scaled about its centre by ``1 + fraction``."""
        poly = self.polygon
        c = poly.mean(axis=0)
        return c + (poly - c) * (1.0 + fraction)

    def expected_ball_diameter(self, goal_height_m: float, ball_diameter_m: float) -> float:
        return self.height_px * ball_diameter_m / goal_height_m

    def segment_hits_mouth(self, p0: Sequence[float], p1: Sequence[float], margin_px: float) -> bool:
        """Whether the image segment p0->p1 passes through the mouth quad."""
        poly = self.quad
        if self.signed_distance(p1, poly) >= -margin_px or self.signed_distance(p0, poly) >= -margin_px:
            return True
        n = 24
        for t in np.linspace(0.0, 1.0, n):
            p = (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)
            if self.signed_distance(p, poly) >= -margin_px:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in self.quad],
            "confidence": round(float(self.confidence), 3),
        }
        if self.back_quad is not None:
            out["back_quad"] = [[round(float(x), 1), round(float(y), 1)] for x, y in self.back_quad]
        return out


@dataclass
class ClipState:
    """The clip as the ball-event rules see it."""

    timestamps: np.ndarray
    frame_indices: np.ndarray
    ball_xy: np.ndarray
    """``(n, 2)``; NaN where the ball's position is unknown."""
    ball_source: np.ndarray
    """Provenance codes (:data:`ABSENT` ... :data:`BRIDGED`)."""
    players: list[list[PlayerObs]]
    frame_size: tuple[int, int]
    ball_velocity: np.ndarray | None = None
    """``(n, 2)`` px/s.  Estimated from ``ball_xy`` when not supplied."""
    ball_diameter: np.ndarray | None = None
    ball_confidence: np.ndarray | None = None
    ball_break: np.ndarray | None = None
    """True where the tracker saw the ball's velocity change abruptly."""
    goal: GoalGeometry | None = None
    net_motion: np.ndarray | None = None
    """``(n,)`` frame-difference energy over the net per frame, NaN where there
    was no reading.  See :mod:`football_analysis.ball_events.netmotion`."""
    reference_diameter_px: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.frame_indices = np.asarray(self.frame_indices, dtype=int)
        self.ball_xy = np.asarray(self.ball_xy, dtype=float).reshape(n, 2)
        self.ball_source = np.asarray(self.ball_source, dtype=int).reshape(n)
        # A position with no provenance is taken at face value as detected.
        known = np.isfinite(self.ball_xy).all(axis=1)
        self.ball_source = np.where(known & (self.ball_source == ABSENT), DETECTED, self.ball_source)
        self.ball_source = np.where(~known, ABSENT, self.ball_source)
        if self.ball_diameter is None:
            self.ball_diameter = np.full(n, np.nan)
        self.ball_diameter = np.asarray(self.ball_diameter, dtype=float).reshape(n)
        if self.ball_confidence is None:
            self.ball_confidence = np.where(self.ball_source == DETECTED, 1.0, 0.0)
        self.ball_confidence = np.asarray(self.ball_confidence, dtype=float).reshape(n)
        if self.ball_break is None:
            self.ball_break = np.zeros(n, dtype=bool)
        self.ball_break = np.asarray(self.ball_break, dtype=bool).reshape(n)
        if self.ball_velocity is None:
            self.ball_velocity = estimate_velocity(self.timestamps, self.ball_xy, self.ball_break)
        self.ball_velocity = np.asarray(self.ball_velocity, dtype=float).reshape(n, 2)
        if self.net_motion is not None:
            self.net_motion = np.asarray(self.net_motion, dtype=float).reshape(n)
        if len(self.players) != n:
            raise ValueError(f"players has {len(self.players)} rows for {n} frames")
        if len(self.frame_indices) != n:
            raise ValueError("frame_indices and timestamps differ in length")

    # -- derived ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def present(self) -> np.ndarray:
        return self.ball_source != ABSENT

    @property
    def observed(self) -> np.ndarray:
        return self.ball_source == DETECTED

    @property
    def interpolated(self) -> np.ndarray:
        return (self.ball_source == SMOOTHED) | (self.ball_source == BRIDGED)

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 25.0
        dt = np.diff(self.timestamps)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if dt.size else 25.0

    def player_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.players:
            for p in row:
                seen.setdefault(p.player_id, None)
        return list(seen)

    def ruler(self, ball_to_player_height: float = 0.125) -> float:
        """Ball diameter in pixels: supplied, measured, or inferred from players."""
        if self.reference_diameter_px and np.isfinite(self.reference_diameter_px):
            return float(self.reference_diameter_px)
        d = self.ball_diameter[self.observed & np.isfinite(self.ball_diameter)]
        d = d[d > 0]
        if d.size >= 3:
            return float(np.median(d))
        heights = [p.height for row in self.players for p in row]
        if heights:
            return float(np.median(heights) * ball_to_player_height)
        return max(6.0, self.frame_size[0] / 120.0)

    def frame_of_index(self, frame_index: int) -> int | None:
        hits = np.nonzero(self.frame_indices == frame_index)[0]
        return int(hits[0]) if hits.size else None

    # -- construction -------------------------------------------------------------

    @classmethod
    def from_ball_track(
        cls,
        ball_track: Any,
        *,
        timestamps: Sequence[float],
        players: Sequence[Sequence[PlayerObs]],
        frame_size: tuple[int, int],
        frame_indices: Sequence[int] | None = None,
        goal: GoalGeometry | None = None,
        net_motion: Sequence[float] | None = None,
    ) -> "ClipState":
        """Adapt :class:`football_analysis.track.ball.BallTrack` as it stands.

        Read by attribute so this module does not import the tracking layer.
        """
        n = len(timestamps)
        ref = getattr(ball_track, "reference_diameter_px", None)
        return cls(
            timestamps=np.asarray(timestamps, float),
            frame_indices=np.arange(n) if frame_indices is None else np.asarray(frame_indices),
            ball_xy=np.asarray(ball_track.xy, float),
            ball_source=np.asarray(ball_track.source, int),
            ball_velocity=np.asarray(ball_track.velocity, float),
            ball_diameter=np.asarray(getattr(ball_track, "diameter_px", np.full(n, np.nan)), float),
            ball_confidence=np.asarray(getattr(ball_track, "confidence", np.zeros(n)), float),
            ball_break=np.asarray(getattr(ball_track, "kinematic_break", np.zeros(n, bool)), bool),
            players=[list(row) for row in players],
            frame_size=frame_size,
            goal=goal,
            net_motion=None if net_motion is None else np.asarray(net_motion, float),
            reference_diameter_px=float(ref) if ref is not None and np.isfinite(ref) else None,
        )

    @classmethod
    def from_frames(
        cls,
        rows: Sequence[dict[str, Any]],
        *,
        frame_size: tuple[int, int],
        goal: GoalGeometry | None = None,
        net_motion: Sequence[float] | None = None,
        max_fill_s: float = 0.3,
    ) -> "ClipState":
        """Build from per-frame dicts, as buffered by the pipeline detector.

        Each row: ``frame_index``, ``timestamp_s``, ``players`` (list of
        :class:`PlayerObs`), and optionally ``ball_xy``, ``ball_diameter``,
        ``ball_confidence``, ``ball_source`` (name or code), ``ball_break``.

        A raw per-frame tracker gives no smoothing, so short gaps (up to
        ``max_fill_s``) between two sightings are filled linearly and marked
        ``bridged``; nothing longer is invented.
        """
        n = len(rows)
        xy = np.full((n, 2), np.nan)
        diam = np.full(n, np.nan)
        conf = np.zeros(n)
        src = np.zeros(n, int)
        brk = np.zeros(n, bool)
        vel = np.full((n, 2), np.nan)
        have_vel = False
        for k, row in enumerate(rows):
            b = row.get("ball_xy")
            if b is not None and np.all(np.isfinite(b)):
                xy[k] = b
                raw = row.get("ball_source", DETECTED)
                src[k] = _SOURCE_CODES.get(raw, DETECTED) if isinstance(raw, str) else int(raw)
                if src[k] == ABSENT:
                    src[k] = DETECTED
            if row.get("ball_diameter") is not None:
                diam[k] = float(row["ball_diameter"])
            conf[k] = float(row.get("ball_confidence") or 0.0)
            brk[k] = bool(row.get("ball_break", False))
            v = row.get("ball_velocity")
            if v is not None and np.all(np.isfinite(v)):
                vel[k] = v
                have_vel = True
        ts = np.asarray([float(r["timestamp_s"]) for r in rows])
        _fill_short_gaps(ts, xy, src, max_fill_s)
        return cls(
            timestamps=ts,
            frame_indices=np.asarray([int(r["frame_index"]) for r in rows]),
            ball_xy=xy,
            ball_source=src,
            ball_velocity=vel if have_vel and np.isfinite(vel[src != ABSENT]).all() else None,
            ball_diameter=diam,
            ball_confidence=conf,
            ball_break=brk,
            players=[list(r.get("players", [])) for r in rows],
            frame_size=frame_size,
            goal=goal,
            net_motion=None if net_motion is None else np.asarray(net_motion, float),
        )


    @classmethod
    def from_frame_states(cls, states: Sequence[Any], *, max_fill_s: float = 0.3,
                          goal_min_confidence: float = 0.2) -> "ClipState":
        """Build from the pipeline's per-frame records
        (:class:`football_analysis.state.FrameState`), read by attribute.

        * The ball's ``interpolated`` flag becomes ``smoothed`` provenance, so
          no rule mistakes a coasted position for a sighting.  A tracker that
          says more (``attributes["source"]``, ``attributes["kinematic_break"]``)
          is taken at its word.
        * The goal is static (the camera is fixed), so one goal is made for the
          whole clip: the median of the solved mouth quads if geometry supplied
          them, else the median of the confident goal boxes.
        * ``attributes["net_motion"]`` on the frame (or on its goal) is the net
          ripple reading.
        """
        rows: list[dict[str, Any]] = []
        quads: list[np.ndarray] = []
        backs: list[np.ndarray] = []
        boxes: list[tuple[float, float, float, float, float]] = []
        net: list[float] = []
        net_outline: np.ndarray | None = None
        frame_size = (0, 0)
        for s in states:
            if frame_size == (0, 0) and getattr(s, "frame_size", (0, 0)) != (0, 0):
                frame_size = tuple(int(v) for v in s.frame_size)
            row: dict[str, Any] = {
                "frame_index": int(s.frame_index),
                "timestamp_s": float(s.timestamp_s),
                "players": [
                    PlayerObs(
                        player_id=str(p.player_id),
                        bbox=_box(p.bbox),
                        track_id=getattr(p, "track_id", None),
                        confidence=float(getattr(p, "confidence", 1.0) or 0.0),
                        interpolated=bool((getattr(p, "attributes", None) or {}).get("interpolated", False)),
                    )
                    for p in (s.players or [])
                ],
            }
            ball = getattr(s, "ball", None)
            if ball is not None and ball.position is not None:
                attrs = getattr(ball, "attributes", None) or {}
                row["ball_xy"] = (float(ball.position.x), float(ball.position.y))
                row["ball_source"] = attrs.get("source") or ("smoothed" if ball.interpolated else "detected")
                row["ball_confidence"] = float(ball.confidence or 0.0)
                row["ball_break"] = bool(attrs.get("kinematic_break", False))
                if not ball.interpolated and row["ball_source"] in ("detected", DETECTED):
                    if attrs.get("diameter_px") is not None:
                        row["ball_diameter"] = float(attrs["diameter_px"])
                    elif ball.bbox is not None:
                        b = _box(ball.bbox)
                        row["ball_diameter"] = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
            goal = getattr(s, "goal", None)
            reading = (getattr(s, "attributes", None) or {}).get("net_motion")
            if goal is not None:
                gattrs = getattr(goal, "attributes", None) or {}
                if reading is None:
                    reading = gattrs.get("net_motion")
                conf = float(goal.confidence or 0.0)
                if len(goal.quad or []) == 4:
                    quads.append(np.array([[p.x, p.y] for p in goal.quad], float))
                    if gattrs.get("back_quad") and len(gattrs["back_quad"]) == 4:
                        backs.append(np.array([_xy(p) for p in gattrs["back_quad"]], float))
                    if gattrs.get("net_px") and net_outline is None:
                        net_outline = np.array([_xy(p) for p in gattrs["net_px"]], float)
                elif goal.bbox is not None and conf >= goal_min_confidence:
                    boxes.append((*_box(goal.bbox), conf))
            net.append(float(reading) if reading is not None else np.nan)
            rows.append(row)

        goal_geom: GoalGeometry | None = None
        if quads:
            goal_geom = GoalGeometry(np.median(np.stack(quads), axis=0), 1.0,
                                     np.median(np.stack(backs), axis=0) if backs else None,
                                     net_outline)
        elif boxes:
            arr = np.asarray(boxes)
            goal_geom = GoalGeometry.from_bbox(np.median(arr[:, :4], axis=0), float(np.median(arr[:, 4])))
        net_arr = np.asarray(net, float)
        return cls.from_frames(
            rows, frame_size=frame_size, goal=goal_geom,
            net_motion=net_arr if np.isfinite(net_arr).any() else None,
            max_fill_s=max_fill_s,
        )



def _box(b: Any) -> Box:
    if hasattr(b, "x1"):
        return (float(b.x1), float(b.y1), float(b.x2), float(b.y2))
    x1, y1, x2, y2 = b
    return (float(x1), float(y1), float(x2), float(y2))


def _xy(p: Any) -> tuple[float, float]:
    if hasattr(p, "x"):
        return (float(p.x), float(p.y))
    if isinstance(p, dict):
        return (float(p["x"]), float(p["y"]))
    return (float(p[0]), float(p[1]))

def _fill_short_gaps(ts: np.ndarray, xy: np.ndarray, src: np.ndarray, max_fill_s: float) -> None:
    known = np.nonzero(np.isfinite(xy).all(axis=1))[0]
    for a, b in zip(known[:-1], known[1:]):
        if b - a <= 1 or ts[b] - ts[a] > max_fill_s:
            continue
        for k in range(a + 1, b):
            t = (ts[k] - ts[a]) / max(1e-9, ts[b] - ts[a])
            xy[k] = (1 - t) * xy[a] + t * xy[b]
            src[k] = BRIDGED


def estimate_velocity(ts: np.ndarray, xy: np.ndarray, breaks: np.ndarray | None = None,
                      half_window: int = 2) -> np.ndarray:
    """Local least-squares slope of position against time, px/s.

    Fitted over up to ``half_window`` frames either side, never across a gap
    in the track or across a kinematic break, so a kick shows up as a clean
    step in velocity rather than being smeared over the frames around it.
    """
    n = len(ts)
    out = np.full((n, 2), np.nan)
    known = np.isfinite(xy).all(axis=1)
    run_id = np.full(n, -1)
    rid = -1
    for k in range(n):
        if not known[k]:
            continue
        starts_new = k == 0 or not known[k - 1] or (breaks is not None and breaks[k])
        if starts_new:
            rid += 1
        run_id[k] = rid
    for k in range(n):
        if run_id[k] < 0:
            continue
        lo, hi = max(0, k - half_window), min(n, k + half_window + 1)
        idx = [j for j in range(lo, hi) if run_id[j] == run_id[k]]
        if len(idx) < 2:
            # A run of one frame: borrow the neighbour across the break, if any.
            nb = [j for j in (k + 1, k - 1) if 0 <= j < n and known[j]]
            if not nb:
                continue
            j = nb[0]
            dt = ts[j] - ts[k]
            if dt != 0:
                out[k] = (xy[j] - xy[k]) / dt
            continue
        t = ts[idx] - ts[idx].mean()
        denom = float((t * t).sum())
        if denom <= 0:
            continue
        p = xy[idx] - xy[idx].mean(axis=0)
        out[k] = (t[:, None] * p).sum(axis=0) / denom
    return out
