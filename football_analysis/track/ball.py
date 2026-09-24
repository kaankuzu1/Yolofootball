"""The ball's trajectory: a physics-gated Kalman filter, smoothed both ways.

This is plan §2.2.  The detector is run at a deliberately low confidence so it
misses as little as possible; this module is where precision comes back.  A
detection is believed only if it fits the ball's motion, and the motion model
is also what carries the ball through the frames where nobody saw it.

How a clip is processed
-----------------------
1. **A ruler.**  The median detected ball size becomes the unit for every
   threshold (see :mod:`football_analysis.track.config`), so the same settings
   work at any resolution and camera distance.
2. **Candidate scoring.**  Each detection's confidence is discounted where false
   positives are known to live: on a player's head, and (fixed camera only) on
   pixels where "a ball" is reported for a large part of the whole clip.
3. **Forward pass.**  A constant-acceleration Kalman filter in image space.
   Detections are accepted inside a Mahalanobis gate.  Near a player's feet the
   process noise is raised, because a touch can happen there.  A confident
   detection outside the gate but inside physical reach is taken as a touch:
   the velocity is reset and a *kinematic break* is recorded.  With no
   acceptable detection the filter coasts for up to ``max_coast_s``, then the
   track is declared lost.  Meanwhile unclaimed detections build tentative
   tracks, and one that is confirmed takes over when the main track is lost or
   coasting -- that is how the filter recovers from locking onto the wrong
   thing, or from a kick far from any player.
4. **Backward pass.**  A Rauch-Tung-Striebel smoother over each continuous
   segment, so frames the ball was hidden in are reconstructed from both sides
   of the occlusion rather than extrapolated from one.  Segments are split at
   kinematic breaks so the smoother never blurs a kick into a curve.
5. **Bridging.**  A short gap between two segments is filled with a Hermite
   curve matched to the velocities on either side, when the speed that would
   imply is physically possible.

Every frame's position says where it came from (``detected``, ``smoothed``,
``bridged``), and every gap is kept as a record, because a ball that vanishes
at a player's feet and reappears moving differently is often exactly where a
tackle or a trick happened.

On gravity.  The plan suggests gravity as a known bias on the vertical axis.
The acceleration is instead a state the filter estimates: most of a 1v1 is a
rolling ball, whose image-space vertical acceleration is zero, and imposing g
would drag every ground-ball prediction downward.  For a ball in flight the
filter learns the parabola's curvature within a few frames and carries it
through a gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from football_analysis.track.config import BallFilterConfig
from football_analysis.track.kalman import H_POS, KalmanStep, ca_matrices, rts_smooth
from football_analysis.track.observations import Obs

__all__ = ["BallTrack", "BallGapSpan", "BallTrajectoryFilter", "SOURCE_NAMES"]

# Per-frame provenance codes.
ABSENT, DETECTED, SMOOTHED, BRIDGED = 0, 1, 2, 3
SOURCE_NAMES = {ABSENT: "absent", DETECTED: "detected", SMOOTHED: "smoothed", BRIDGED: "bridged"}

Box = tuple[float, float, float, float]


@dataclass
class BallGapSpan:
    """A run of frames with no accepted detection."""

    start: int
    """First frame without an observation."""
    end: int
    """Last frame without an observation (inclusive)."""
    before: int | None
    """Last observed frame before the gap, if any."""
    after: int | None
    """First observed frame after it, if any."""
    filled: bool
    """Whether a position was reconstructed for the gap's frames."""
    kind: str
    """``occlusion`` (the filter coasted through), ``lost`` (between segments),
    ``clip_start`` or ``clip_end``."""


@dataclass
class BallTrack:
    """The ball through a whole clip, one row per frame."""

    xy: np.ndarray
    """``(n, 2)`` best position in pixels; NaN where unknown."""
    velocity: np.ndarray
    """``(n, 2)`` pixels per second; NaN where unknown."""
    raw_xy: np.ndarray
    """``(n, 2)`` the accepted detection's centre, NaN when none was accepted."""
    diameter_px: np.ndarray
    confidence: np.ndarray
    """Detector confidence of the accepted detection; 0 when none."""
    source: np.ndarray
    """Codes from :data:`SOURCE_NAMES`."""
    kinematic_break: np.ndarray
    """True where the velocity was reset: a touch, a deflection, a re-acquisition."""
    segment: np.ndarray
    """Which continuous filter segment the frame belongs to; -1 for none."""
    gaps: list[BallGapSpan] = field(default_factory=list)
    reference_diameter_px: float = float("nan")
    stats: dict = field(default_factory=dict)

    @property
    def interpolated(self) -> np.ndarray:
        return (self.source == SMOOTHED) | (self.source == BRIDGED)

    @property
    def present(self) -> np.ndarray:
        return self.source != ABSENT


# -- internal bookkeeping ------------------------------------------------------


@dataclass
class _Segment:
    frames: list[int] = field(default_factory=list)
    steps: list[KalmanStep] = field(default_factory=list)
    observed: list[bool] = field(default_factory=list)
    starts_with_break: bool = False
    breaks: list[int] = field(default_factory=list)
    """Step indices where an accepted detection was surprising enough to call
    a velocity change.  The smoother is not run across them."""

    @property
    def last_observed_frame(self) -> int | None:
        for f, o in zip(reversed(self.frames), reversed(self.observed)):
            if o:
                return f
        return None

    def coast_run(self) -> int:
        run = 0
        for o in reversed(self.observed):
            if o:
                break
            run += 1
        return run


@dataclass
class _Tentative:
    hits: list[tuple[int, float, float, float]] = field(default_factory=list)
    """``(frame, x, y, score)``"""

    def velocity(self, times: Sequence[float]) -> tuple[float, float]:
        if len(self.hits) < 2:
            return (0.0, 0.0)
        (f0, x0, y0, _), (f1, x1, y1, _) = self.hits[-2], self.hits[-1]
        dt = times[f1] - times[f0]
        if dt <= 0:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)


# -- the filter ----------------------------------------------------------------


class BallTrajectoryFilter:
    """Offline ball tracking over a whole clip's candidate detections."""

    def __init__(self, config: BallFilterConfig | None = None) -> None:
        self.cfg = config or BallFilterConfig()

    # public ---------------------------------------------------------------

    def run(
        self,
        timestamps: Sequence[float],
        candidates: Sequence[Sequence[Obs]],
        players: Sequence[Sequence[Box]] | None = None,
    ) -> BallTrack:
        """Track the ball.

        ``candidates[k]`` are the ball detections of frame ``k`` (already
        restricted to ball classes); ``players[k]`` are that frame's player
        boxes, used to know where a touch is possible and where heads are.
        """
        cfg = self.cfg
        n = len(timestamps)
        times = [float(t) for t in timestamps]
        players = players if players is not None else [[] for _ in range(n)]
        cands = [
            [c for c in frame if c.conf >= cfg.min_candidate_confidence] for frame in candidates
        ]

        track = _empty_track(n)
        sizes = [c.diameter for frame in cands for c in frame if c.diameter > 0]
        if not sizes:
            track.stats = _stats(track, times)
            return track
        confident = [c.diameter for frame in cands for c in frame if c.conf >= 0.3 and c.diameter > 0]
        D = float(np.median(confident if len(confident) >= 5 else sizes))
        D = max(D, 2.0)
        track.reference_diameter_px = D

        scores = self._score(cands, players, D, n)
        segments, used = self._forward(times, cands, scores, players, D)
        self._assemble(track, times, cands, segments, used, D)
        track.stats = _stats(track, times)
        return track

    # scoring --------------------------------------------------------------

    def _score(self, cands, players, D: float, n: int) -> list[list[float]]:
        cfg = self.cfg
        clutter: set[tuple[int, int]] = set()
        if cfg.fixed_camera and n >= 50:
            cell = 1.5 * D
            counts: dict[tuple[int, int], set[int]] = {}
            for k, frame in enumerate(cands):
                for c in frame:
                    counts.setdefault((int(c.cx // cell), int(c.cy // cell)), set()).add(k)
            limit = cfg.clutter_min_fraction * n
            clutter = {key for key, frames in counts.items() if len(frames) >= limit}

        out: list[list[float]] = []
        for k, frame in enumerate(cands):
            row = []
            for c in frame:
                s = c.conf
                if clutter and (int(c.cx // (1.5 * D)), int(c.cy // (1.5 * D))) in clutter:
                    s *= cfg.clutter_penalty
                for (x1, y1, x2, y2) in players[k]:
                    if x1 <= c.cx <= x2 and y1 <= c.cy <= y1 + cfg.head_zone_fraction * (y2 - y1):
                        s *= cfg.head_zone_penalty
                        break
                row.append(max(s, 1e-6))
            out.append(row)
        self._clutter_cells = len(clutter)
        return out

    # forward --------------------------------------------------------------

    def _near_player(self, x: float, y: float, boxes: Sequence[Box], D: float) -> bool:
        reach = self.cfg.reach_diameters * D
        for (x1, y1, x2, y2) in boxes:
            top = y1 + 0.5 * (y2 - y1)  # legs: the lower half of the box
            dx = max(x1 - x, 0.0, x - x2)
            dy = max(top - y, 0.0, y - y2)
            if dx * dx + dy * dy <= reach * reach:
                return True
        return False

    def _init_state(self, x: float, y: float, v: tuple[float, float] | None, D: float):
        cfg = self.cfg
        sigma = cfg.measurement_sigma * D
        v_sigma = (0.5 * cfg.max_speed if v is None else 0.15 * cfg.max_speed) * D
        state = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
        if v is not None:
            state[2:4] = v
        P = np.diag([sigma**2, sigma**2, v_sigma**2, v_sigma**2, (cfg.init_accel_sigma * D) ** 2, (cfg.init_accel_sigma * D) ** 2])
        return state, P

    def _forward(self, times, cands, scores, players, D):
        cfg = self.cfg
        n = len(times)
        R = np.eye(2) * (cfg.measurement_sigma * D) ** 2
        max_speed_px = cfg.max_speed * D

        segments: list[_Segment] = []
        used: dict[int, int] = {}  # frame -> index of accepted candidate
        cur: _Segment | None = None
        x = P = None
        tentatives: list[_Tentative] = []

        def start_segment(frame: int, state, cov, is_break: bool) -> _Segment:
            seg = _Segment(starts_with_break=is_break)
            seg.frames.append(frame)
            seg.steps.append(KalmanStep(state.copy(), cov.copy(), state.copy(), cov.copy(), np.eye(6)))
            seg.observed.append(True)
            segments.append(seg)
            return seg

        for k in range(n):
            claimed: int | None = None
            if cur is not None:
                dt = max(times[k] - times[cur.frames[-1]], 1e-3)
                near = self._near_player(x[0], x[1], players[k], D)
                q = cfg.jerk_density * D * D * (cfg.contact_jerk_scale if near else 1.0)
                F, Q = ca_matrices(dt, q)
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
                S = H_POS @ P_pred @ H_POS.T + R
                S_inv = np.linalg.inv(S)

                min_score = cfg.coast_min_score if cur.coast_run() >= 2 else 0.0
                best, best_cost, best_d2 = None, math.inf, 0.0
                for i, c in enumerate(cands[k]):
                    r = np.array([c.cx, c.cy]) - x_pred[:2]
                    d2 = float(r @ S_inv @ r)
                    if d2 > cfg.gate_chi2 or scores[k][i] < min_score:
                        continue
                    cost = d2 - 2.0 * math.log(scores[k][i])
                    if cost < best_cost:
                        best, best_cost, best_d2 = i, cost, d2

                if best is not None:
                    c = cands[k][best]
                    z = np.array([c.cx, c.cy])
                    K = P_pred @ H_POS.T @ S_inv
                    x = x_pred + K @ (z - x_pred[:2])
                    P = (np.eye(6) - K @ H_POS) @ P_pred
                    cur.frames.append(k)
                    cur.steps.append(KalmanStep(x_pred, P_pred, x.copy(), P.copy(), F))
                    cur.observed.append(True)
                    if best_d2 > cfg.break_innovation_chi2:
                        cur.breaks.append(len(cur.frames) - 1)
                    claimed = best
                else:
                    # Outside the gate.  Is there a confident detection the ball
                    # could physically have reached -- a touch?
                    last_obs = cur.last_observed_frame
                    last_step = cur.steps[cur.frames.index(last_obs)] if last_obs is not None else None
                    brk = None
                    if last_step is not None:
                        since = max(times[k] - times[last_obs], 1e-3)
                        lx, ly = last_step.x[0], last_step.x[1]
                        touch_zone = near or self._near_player(lx, ly, players[last_obs], D)
                        best_s = 0.0
                        for i, c in enumerate(cands[k]):
                            s = scores[k][i]
                            need = cfg.break_min_confidence if touch_zone else max(0.5, cfg.break_min_confidence)
                            if s < need:
                                continue
                            dist = math.hypot(c.cx - lx, c.cy - ly)
                            if dist <= max_speed_px * since + 2.0 * D and s > best_s:
                                brk, best_s = i, s
                    if brk is not None:
                        c = cands[k][brk]
                        since = max(times[k] - times[last_obs], 1e-3)
                        v = ((c.cx - last_step.x[0]) / since, (c.cy - last_step.x[1]) / since)
                        x, P = self._init_state(c.cx, c.cy, v, D)
                        cur = start_segment(k, x, P, True)
                        claimed = brk
                    else:
                        x, P = x_pred, P_pred
                        cur.frames.append(k)
                        cur.steps.append(KalmanStep(x_pred, P_pred, x.copy(), P.copy(), F))
                        cur.observed.append(False)
                        coast_s = times[k] - times[last_obs] if last_obs is not None else math.inf
                        if coast_s > cfg.max_coast_s:
                            cur = None

            if claimed is not None:
                used[k] = claimed

            # Tentative tracks from everything the main track did not claim.
            self._update_tentatives(tentatives, k, times, cands, scores, claimed, D, max_speed_px)

            # Hand over to a confirmed tentative when the main track is lost or
            # has been coasting.
            coasting = cur is not None and cur.coast_run() >= 2
            if cur is None or coasting:
                chosen = self._pick_tentative(tentatives, k, times, D)
                if chosen is None and cur is None:
                    # A single, confident detection may start a track outright.
                    free = [i for i in range(len(cands[k])) if i != claimed]
                    if free:
                        i = max(free, key=lambda j: scores[k][j])
                        if scores[k][i] >= cfg.init_confidence:
                            c = cands[k][i]
                            x, P = self._init_state(c.cx, c.cy, None, D)
                            cur = start_segment(k, x, P, False)
                            used[k] = i
                elif chosen is not None:
                    tentatives.remove(chosen)
                    x, P, cur = self._adopt(chosen, times, cands, used, segments, cur, D, R)
        return segments, used

    def _update_tentatives(self, tentatives, k, times, cands, scores, claimed, D, max_speed_px):
        cfg = self.cfg
        tentatives[:] = [t for t in tentatives if k - t.hits[-1][0] <= cfg.confirm_window]
        pairs = []
        for ti, t in enumerate(tentatives):
            f, tx, ty, _ = t.hits[-1]
            if f == k:
                continue
            dt = max(times[k] - times[f], 1e-3)
            vx, vy = t.velocity(times)
            px, py = tx + vx * dt, ty + vy * dt
            radius = max(2.0 * D, 0.35 * max_speed_px * dt)
            for i, c in enumerate(cands[k]):
                if i == claimed:
                    continue
                d = math.hypot(c.cx - px, c.cy - py)
                if d <= radius:
                    pairs.append((d, ti, i))
        pairs.sort()
        taken: set[int] = set()
        extended: set[int] = set()
        for _, ti, i in pairs:
            if ti in extended or i in taken:
                continue
            c = cands[k][i]
            t = tentatives[ti]
            t.hits.append((k, c.cx, c.cy, scores[k][i]))
            del t.hits[:-4 * cfg.confirm_window]
            taken.add(i)
            extended.add(ti)
        for i, c in enumerate(cands[k]):
            if i != claimed and i not in taken:
                tentatives.append(_Tentative([(k, c.cx, c.cy, scores[k][i])]))

    def _pick_tentative(self, tentatives, k, times, D) -> _Tentative | None:
        cfg = self.cfg
        best, best_score = None, 0.0
        for t in tentatives:
            if t.hits[-1][0] != k:
                continue
            recent = [h for h in t.hits if k - h[0] < cfg.confirm_window]
            if len(recent) < cfg.confirm_hits:
                continue
            mean_s = sum(h[3] for h in recent) / len(recent)
            span = math.hypot(recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2])
            if span < 0.5 * D and mean_s < cfg.init_confidence:
                continue  # a motionless blob needs to be convincing on its own
            score = sum(h[3] for h in recent)
            if score > best_score:
                best, best_score = t, score
        return best

    def _adopt(self, tent: _Tentative, times, cands, used, segments, cur, D, R):
        """Start a new segment from a confirmed tentative, replaying its hits."""
        cfg = self.cfg
        floor = cur.last_observed_frame if cur is not None else -1
        hits = [h for h in tent.hits if h[0] > (floor if floor is not None else -1)]
        f0, x0, y0, _ = hits[0]
        x, P = self._init_state(x0, y0, None, D)
        seg = _Segment(starts_with_break=cur is not None)
        seg.frames.append(f0)
        seg.steps.append(KalmanStep(x.copy(), P.copy(), x.copy(), P.copy(), np.eye(6)))
        seg.observed.append(True)
        hit_at = {h[0]: h for h in hits}
        for f in range(f0 + 1, hits[-1][0] + 1):
            dt = max(times[f] - times[seg.frames[-1]], 1e-3)
            F, Q = ca_matrices(dt, cfg.jerk_density * D * D * cfg.contact_jerk_scale)
            x_pred, P_pred = F @ x, F @ P @ F.T + Q
            if f in hit_at:
                z = np.array(hit_at[f][1:3])
                S = H_POS @ P_pred @ H_POS.T + R
                K = P_pred @ H_POS.T @ np.linalg.inv(S)
                x = x_pred + K @ (z - x_pred[:2])
                P = (np.eye(6) - K @ H_POS) @ P_pred
                observed = True
            else:
                x, P, observed = x_pred, P_pred, False
            seg.frames.append(f)
            seg.steps.append(KalmanStep(x_pred, P_pred, x.copy(), P.copy(), F))
            seg.observed.append(observed)
        # Record which candidate each replayed hit was.
        for f, hx, hy, _ in hits:
            for i, c in enumerate(cands[f]):
                if abs(c.cx - hx) < 1e-6 and abs(c.cy - hy) < 1e-6:
                    used[f] = i
                    break
        # The old segment's coasted tail now belongs to the gap, not to it.
        if cur is not None:
            while cur.observed and not cur.observed[-1]:
                cur.frames.pop(); cur.steps.pop(); cur.observed.pop()
        segments.append(seg)
        return x, P, seg

    # assembly -------------------------------------------------------------

    def _assemble(self, track: BallTrack, times, cands, segments, used, D):
        cfg = self.cfg
        n = len(times)
        for k, i in used.items():
            c = cands[k][i]
            track.raw_xy[k] = (c.cx, c.cy)
            track.confidence[k] = c.conf
            track.diameter_px[k] = c.diameter

        for sid, seg in enumerate(segments):
            # Drop the tail the filter only extrapolated into.
            while seg.observed and not seg.observed[-1]:
                seg.frames.pop(); seg.steps.pop(); seg.observed.pop()
            if not seg.frames:
                continue
            # A track that started from nothing and never gathered enough
            # observations to be confirmed was a false start.  (A segment that
            # begins at a break continues a trusted trajectory and is kept.)
            if not seg.starts_with_break and sum(seg.observed) < cfg.confirm_hits:
                for f, obs in zip(seg.frames, seg.observed):
                    if obs and f in used:
                        del used[f]
                        track.raw_xy[f] = np.nan
                        track.confidence[f] = 0.0
                        track.diameter_px[f] = np.nan
                continue
            cuts = [0] + [b for b in seg.breaks if 0 < b < len(seg.steps)] + [len(seg.steps)]
            xs: list[np.ndarray] = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                part, _ = rts_smooth(seg.steps[a:b])
                xs.extend(part)
            for b in cuts[1:-1]:
                track.kinematic_break[seg.frames[b]] = True
            for f, xsm, obs in zip(seg.frames, xs, seg.observed):
                if track.segment[f] >= 0 and track.source[f] == DETECTED:
                    continue  # frame already owned by an earlier segment's detection
                track.xy[f] = xsm[:2]
                track.velocity[f] = xsm[2:4]
                track.source[f] = DETECTED if obs and f in used else SMOOTHED
                track.segment[f] = sid
            if seg.starts_with_break:
                track.kinematic_break[seg.frames[0]] = True

        # Gaps: every run of frames without an accepted detection.
        observed = np.array([k in used and track.source[k] == DETECTED for k in range(n)])
        k = 0
        while k < n:
            if observed[k]:
                k += 1
                continue
            start = k
            while k < n and not observed[k]:
                k += 1
            end = k - 1
            before = start - 1 if start > 0 else None
            after = k if k < n else None
            if before is None:
                kind = "clip_start"
            elif after is None:
                kind = "clip_end"
            elif track.segment[before] == track.segment[after] and not track.kinematic_break[after]:
                kind = "occlusion"
            else:
                kind = "lost"
            filled = kind == "occlusion" and bool(np.all(track.source[start:end + 1] == SMOOTHED))
            if kind == "lost":
                filled = self._bridge(track, times, before, after, D)
            if kind != "occlusion" or not filled:
                # Positions the smoother left in a gap it could not close are guesses.
                if not filled:
                    track.xy[start:end + 1] = np.nan
                    track.velocity[start:end + 1] = np.nan
                    track.source[start:end + 1] = ABSENT
            track.gaps.append(BallGapSpan(start, end, before, after, filled, kind))

        # Ball size: observed where seen, carried across gaps, the ruler otherwise.
        diam = track.diameter_px
        seen = ~np.isnan(diam)
        if seen.any():
            idx = np.arange(n)
            track.diameter_px = np.interp(idx, idx[seen], diam[seen])
        else:
            track.diameter_px[:] = D
        track.diameter_px[~track.present] = np.nan

    def _bridge(self, track: BallTrack, times, before: int, after: int, D: float) -> bool:
        cfg = self.cfg
        T = times[after] - times[before]
        if T <= 0 or T > cfg.bridge_max_s:
            return False
        p0, p1 = track.xy[before], track.xy[after]
        chord = float(np.hypot(*(p1 - p0)))
        if chord / T > cfg.max_speed * D:
            return False
        v0, v1 = track.velocity[before].copy(), track.velocity[after].copy()
        limit = 2.0 * chord / T + 2.0 * D / T
        for v in (v0, v1):
            speed = float(np.hypot(*v))
            if not np.isfinite(speed):
                v[:] = (p1 - p0) / T
            elif speed > limit:
                v *= limit / speed
        for f in range(before + 1, after):
            s = (times[f] - times[before]) / T
            h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
            h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
            track.xy[f] = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
            d00, d10 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1
            d01, d11 = -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
            track.velocity[f] = (d00 * p0 + d01 * p1) / T + d10 * v0 + d11 * v1
            track.source[f] = BRIDGED
            track.segment[f] = -1
        return True


def _empty_track(n: int) -> BallTrack:
    nan2 = np.full((n, 2), np.nan)
    return BallTrack(
        xy=nan2.copy(),
        velocity=nan2.copy(),
        raw_xy=nan2.copy(),
        diameter_px=np.full(n, np.nan),
        confidence=np.zeros(n),
        source=np.zeros(n, dtype=int),
        kinematic_break=np.zeros(n, dtype=bool),
        segment=np.full(n, -1, dtype=int),
    )


def _stats(track: BallTrack, times) -> dict:
    n = len(track.source)
    if n == 0:
        return {"frames": 0}
    detected = int(np.sum(track.source == DETECTED))
    present = int(np.sum(track.present))
    gap_lengths = [g.end - g.start + 1 for g in track.gaps]
    return {
        "frames": n,
        "detected_frames": detected,
        "present_frames": present,
        "presence_before_interpolation": round(detected / n, 4),
        "presence_after_interpolation": round(present / n, 4),
        "gaps": len(track.gaps),
        "gaps_filled": sum(1 for g in track.gaps if g.filled),
        "longest_gap_frames": max(gap_lengths) if gap_lengths else 0,
        "kinematic_breaks": int(track.kinematic_break.sum()),
        "reference_diameter_px": round(float(track.reference_diameter_px), 2)
        if np.isfinite(track.reference_diameter_px) else None,
    }
