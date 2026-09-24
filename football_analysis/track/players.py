"""Two players, two identities, for the whole clip.

Plan §2.1: ByteTrack for frame-to-frame association, then a *global* identity
lock.  In a 1v1 the number of people is known in advance and never changes, so
"who is this?" stops being a frame-to-frame association problem and becomes a
two-way classification with a constraint -- no identity may appear twice in
one frame.

How it works
------------
1. :class:`ByteTracker` links detections frame to frame into *tracklets*: a
   high-confidence matching pass, then a second pass that lets low-confidence
   detections extend tracks that already exist (ByteTrack's central idea: a
   partly hidden player scores low but is still that player).
2. :class:`IdentityLock` then works on the whole clip at once:

   * Wherever two player boxes overlap, both are marked *occluded*, and every
     tracklet is cut at the start and end of each such episode.  Overlaps are
     where a frame-to-frame tracker swaps people, so no identity is trusted to
     survive one on motion alone.
   * Kit colour features from the clear (unoccluded, not cut off by the frame
     edge) boxes are clustered into as many groups as there are players.
   * Each clear piece of tracklet is assigned to the cluster its colours match,
     most decisive first, never giving one identity two boxes in the same frame.
   * Pieces inside an overlap inherit the identity of the same tracklet just
     before it, unless that would clash.
   * If the two kits are too alike to tell apart, identity falls back to motion
     continuity and the result says so loudly, because every event downstream
     depends on knowing who is who.

Identity switches should be zero.  The report counts the swaps this corrected
(a tracklet whose identity differs either side of an overlap), which is how
often plain ByteTrack would have got it wrong on this clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from football_analysis.track.config import PlayerTrackConfig

__all__ = ["ByteTracker", "IdentityLock", "IdentityResult", "PlayerFrameObs", "box_iou"]

Box = tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    out = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i, j] = box_iou(x, y)
    return out


# -- ByteTrack ---------------------------------------------------------------------


@dataclass
class _BoxTrack:
    tid: int
    x: np.ndarray  # cx, cy, w, h, vcx, vcy, vw, vh
    P: np.ndarray
    lost: int = 0
    hits: int = 1

    def box(self) -> Box:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.hstack([np.eye(4), np.zeros((4, 4))])


def _xywh(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


class ByteTracker:
    """Frame-to-frame association (ByteTrack), one frame at a time."""

    def __init__(self, config: PlayerTrackConfig | None = None) -> None:
        self.cfg = config or PlayerTrackConfig()
        self.tracks: list[_BoxTrack] = []
        self._next = 1

    def reset(self) -> None:
        self.tracks = []
        self._next = 1

    def _predict(self, t: _BoxTrack) -> None:
        h = max(t.x[3], 1.0)
        q_pos, q_vel = (0.05 * h) ** 2, (0.00625 * h) ** 2
        Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_vel])
        t.x = _F @ t.x
        t.P = _F @ t.P @ _F.T + Q

    def _update(self, t: _BoxTrack, box: Box) -> None:
        h = max(t.x[3], 1.0)
        R = np.eye(4) * (0.05 * h) ** 2
        z = _xywh(box)
        S = _H @ t.P @ _H.T + R
        K = t.P @ _H.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - _H @ t.x)
        t.P = (np.eye(8) - K @ _H) @ t.P
        t.lost = 0
        t.hits += 1

    def _new(self, box: Box) -> _BoxTrack:
        z = _xywh(box)
        h = max(z[3], 1.0)
        P = np.diag([(0.1 * h) ** 2] * 4 + [(0.1 * h) ** 2] * 4)
        t = _BoxTrack(self._next, np.r_[z, np.zeros(4)], P)
        self._next += 1
        self.tracks.append(t)
        return t

    def step(self, boxes: Sequence[Box], confs: Sequence[float]) -> list[int | None]:
        """Associate one frame; return the tracklet id for each detection."""
        cfg = self.cfg
        for t in self.tracks:
            self._predict(t)
        out: list[int | None] = [None] * len(boxes)
        high = [i for i, c in enumerate(confs) if c >= cfg.high_confidence]
        low = [i for i, c in enumerate(confs) if cfg.low_confidence <= c < cfg.high_confidence]

        def match(track_idx: list[int], det_idx: list[int], min_iou: float):
            if not track_idx or not det_idx:
                return [], track_idx, det_idx
            iou = _iou_matrix([self.tracks[i].box() for i in track_idx], [boxes[j] for j in det_idx])
            rows, cols = linear_sum_assignment(-iou)
            pairs, used_t, used_d = [], set(), set()
            for r, c in zip(rows, cols):
                if iou[r, c] >= min_iou:
                    pairs.append((track_idx[r], det_idx[c]))
                    used_t.add(track_idx[r]); used_d.add(det_idx[c])
            return (pairs, [i for i in track_idx if i not in used_t],
                    [j for j in det_idx if j not in used_d])

        all_tracks = list(range(len(self.tracks)))
        pairs1, rest_tracks, rest_high = match(all_tracks, high, cfg.match_iou)
        active = [i for i in rest_tracks if self.tracks[i].lost == 0]
        pairs2, _, _ = match(active, low, cfg.low_match_iou)
        matched = set()
        for ti, di in pairs1 + pairs2:
            self._update(self.tracks[ti], boxes[di])
            out[di] = self.tracks[ti].tid
            matched.add(ti)
        for i, t in enumerate(self.tracks):
            if i not in matched:
                t.lost += 1
        for di in rest_high:
            if confs[di] >= cfg.new_track_confidence:
                out[di] = self._new(boxes[di]).tid
        self.tracks = [t for t in self.tracks if t.lost <= cfg.max_lost_frames]
        return out


# -- the identity lock ------------------------------------------------------------


@dataclass
class PlayerFrameObs:
    """One player detection, as the identity lock sees it."""

    box: Box
    conf: float
    tracklet: int | None
    feature: np.ndarray | None = None
    at_edge: bool = False


@dataclass
class IdentityResult:
    """Per frame: ``assign[k][i]`` is the identity (0..N-1) of detection ``i``, or None."""

    assign: list[list[int | None]]
    confidence: list[list[float]]
    occluded: list[list[bool]]
    centroids: np.ndarray | None
    report: dict = field(default_factory=dict)


@dataclass
class _Piece:
    tracklet: int
    occluded: bool
    frames: list[int] = field(default_factory=list)
    dets: list[int] = field(default_factory=list)
    identity: int | None = None
    confidence: float = 0.0
    costs: np.ndarray | None = None


def _kmeans(X: np.ndarray, k: int, restarts: int = 8, iters: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        # k-means++ initialisation
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            p = d / d.sum() if d.sum() > 0 else None
            c.append(X[rng.choice(len(X), p=p)])
        C = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


class IdentityLock:
    """Assigns every player detection in a clip to one of N fixed identities."""

    def __init__(self, config: PlayerTrackConfig | None = None) -> None:
        self.cfg = config or PlayerTrackConfig()

    def run(self, frames: Sequence[Sequence[PlayerFrameObs]]) -> IdentityResult:
        cfg = self.cfg
        N = cfg.num_identities
        n = len(frames)
        occluded = self._occlusions(frames)

        pieces = self._pieces(frames, occluded)
        clear_feats = [
            frames[k][i].feature
            for k, obs in enumerate(frames)
            for i, o in enumerate(obs)
            if o.feature is not None and not occluded[k][i] and not o.at_edge and o.tracklet is not None
        ]
        report: dict = {"num_identities": N, "tracklets": len({p.tracklet for p in pieces}),
                        "pieces": len(pieces), "clear_features": len(clear_feats), "warnings": []}

        centroids = None
        separation = 0.0
        centroid_distance = 0.0
        if len(clear_feats) >= 5 * N and N >= 2:
            X = np.array(clear_feats, dtype=float)
            centroids, labels = _kmeans(X, N)
            within = np.mean([((X[labels == j] - centroids[j]) ** 2).sum(1).mean()
                              for j in range(N) if np.any(labels == j)])
            between = min(((centroids[a] - centroids[b]) ** 2).sum()
                          for a in range(N) for b in range(a + 1, N))
            separation = float(between / max(within, 1e-9))
            centroid_distance = float(np.sqrt(between))
            report["centroid_distance"] = round(centroid_distance, 3)
            report["cluster_sizes"] = [int(np.sum(labels == j)) for j in range(N)]
            sigma2 = max(within, 0.01)  # a floor, so near-identical crops do not look decisive
        report["separation"] = round(separation, 3)

        distinct = separation >= cfg.min_separation and centroid_distance >= cfg.min_centroid_distance
        if centroids is not None and distinct:
            report["method"] = "appearance"
            pieces = self._split_on_colour(pieces, frames, centroids)
            report["pieces"] = len(pieces)
            self._assign_by_appearance(pieces, frames, occluded, centroids, sigma2)
        else:
            report["method"] = "motion_continuity"
            if N >= 2:
                report["warnings"].append(
                    "The players' kits are too similar to tell apart reliably "
                    f"(separation {separation:.2f}, colour distance {centroid_distance:.2f}); identities follow "
                    "motion continuity and may swap after the players overlap."
                )
            self._assign_by_motion(pieces, frames)

        resolved = self._assign_occluded(pieces, frames, centroids, sigma2 if centroids is not None else 1.0)

        assign = [[None] * len(f) for f in frames]
        conf = [[0.0] * len(f) for f in frames]
        for p in pieces:
            for k, i in zip(p.frames, p.dets):
                assign[k][i] = p.identity
                conf[k][i] = p.confidence if p.identity is not None else 0.0
        for (k, i), (j, c) in resolved.items():
            assign[k][i], conf[k][i] = j, c

        # Swaps corrected: a tracklet whose identity changes along its length is
        # a place where frame-to-frame association alone would have been wrong.
        seq: dict[int, list[int]] = {}
        for k, f in enumerate(frames):
            for i, o in enumerate(f):
                if o.tracklet is not None and assign[k][i] is not None:
                    seq.setdefault(o.tracklet, []).append(assign[k][i])
        report["swaps_corrected"] = sum(
            sum(1 for a, b in zip(ids, ids[1:]) if a != b) for ids in seq.values())
        report["overlap_frames"] = int(sum(any(row) for row in occluded))
        report["unassigned_detections"] = int(sum(
            1 for k, f in enumerate(frames) for i, o in enumerate(f)
            if assign[k][i] is None and o.conf >= cfg.high_confidence))
        return IdentityResult(assign, conf, occluded, centroids, report)

    # -- occlusion ------------------------------------------------------------

    def _occlusions(self, frames) -> list[list[bool]]:
        """Mark detections that overlap another player, or that have swallowed one.

        Two visible boxes overlapping is the easy case.  The harder one is a
        merge: the detector reports one box for two bodies, and the other
        player's track simply stops.  So a box also counts as occluded when a
        player who was visible in the last few frames, and is not visible now,
        was last seen inside it.
        """
        cfg = self.cfg
        occluded = [[False] * len(f) for f in frames]
        last_seen: dict[int, tuple[int, Box]] = {}
        for k, obs in enumerate(frames):
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    if box_iou(obs[i].box, obs[j].box) > cfg.overlap_iou:
                        occluded[k][i] = occluded[k][j] = True
            present = {o.tracklet for o in obs if o.tracklet is not None}
            for i, o in enumerate(obs):
                for tid, (kf, box) in last_seen.items():
                    if tid == o.tracklet or tid in present or k - kf > cfg.merge_memory_frames:
                        continue
                    if box_iou(o.box, box) > cfg.overlap_iou:
                        occluded[k][i] = True
                        break
            for o in obs:
                if o.tracklet is not None:
                    last_seen[o.tracklet] = (k, o.box)
        return occluded

    # -- pieces ---------------------------------------------------------------

    def _pieces(self, frames, occluded) -> list[_Piece]:
        open_piece: dict[int, _Piece] = {}
        pieces: list[_Piece] = []
        for k, obs in enumerate(frames):
            for i, o in enumerate(obs):
                if o.tracklet is None:
                    continue
                occ = occluded[k][i]
                p = open_piece.get(o.tracklet)
                if p is None or p.occluded != occ:
                    p = _Piece(o.tracklet, occ)
                    pieces.append(p)
                    open_piece[o.tracklet] = p
                p.frames.append(k)
                p.dets.append(i)
        return pieces

    # -- assignment -----------------------------------------------------------

    def _busy(self, identity_frames: list[set[int]], identity: int, frames: list[int]) -> bool:
        s = identity_frames[identity]
        return any(f in s for f in frames)

    def _split_on_colour(self, pieces, frames, C, run: int = 3) -> list[_Piece]:
        """Cut a clear piece where its kit colour changes for good.

        Frame-to-frame trackers mostly swap people inside overlaps, which are
        already cut out, but not only there.  A piece whose crops match one kit
        and then, for ``run`` frames or more in a row, the other, is two people.
        """
        out: list[_Piece] = []
        for p in pieces:
            if p.occluded or len(p.frames) < 2 * run:
                out.append(p)
                continue
            labels = []
            for k, i in zip(p.frames, p.dets):
                f = frames[k][i].feature
                labels.append(-1 if f is None else int(np.argmin(((f[None] - C) ** 2).sum(1))))
            cuts, current, streak_start, streak_label = [0], None, 0, None
            for idx, lab in enumerate(labels):
                if lab < 0:
                    continue
                if current is None:
                    current = lab
                if lab == current:
                    streak_label = None
                    continue
                if streak_label != lab:
                    streak_label, streak_start = lab, idx
                if idx - streak_start + 1 >= run:
                    cuts.append(streak_start)
                    current, streak_label = lab, None
            cuts.append(len(labels))
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b > a:
                    out.append(_Piece(p.tracklet, False, p.frames[a:b], p.dets[a:b]))
        return out

    def _assign_by_appearance(self, pieces, frames, occluded, C, sigma2) -> None:
        N = len(C)
        taken: list[set[int]] = [set() for _ in range(N)]
        clear = []
        for p in pieces:
            if p.occluded:
                continue
            feats = [frames[k][i].feature for k, i in zip(p.frames, p.dets)
                     if frames[k][i].feature is not None and not frames[k][i].at_edge]
            if not feats:
                continue
            F = np.array(feats, dtype=float)
            costs = (((F[:, None, :] - C[None]) ** 2).sum(-1) / sigma2).mean(0)
            p.costs = costs
            order = np.sort(costs)
            margin = float(order[1] - order[0]) if N > 1 else 1.0
            clear.append((margin * np.sqrt(len(feats)), p))
        clear.sort(key=lambda item: -item[0])
        for _, p in clear:
            for j in np.argsort(p.costs):
                if not self._busy(taken, int(j), p.frames):
                    p.identity = int(j)
                    logits = -0.5 * p.costs * min(len(p.frames), 25) / 5.0
                    probs = np.exp(logits - logits.max())
                    probs /= probs.sum()
                    p.confidence = float(probs[j])
                    taken[j].update(p.frames)
                    break

    def _assign_by_motion(self, pieces, frames) -> None:
        """Fallback when kits cannot be told apart: continue each identity along
        its own velocity, and give each new piece to the identity whose
        predicted position it starts closest to.  Every piece is used here,
        overlaps included, since there is no colour to resolve them later."""
        N = self.cfg.num_identities
        taken: list[set[int]] = [set() for _ in range(N)]
        last: list[tuple[int, float, float, float, float] | None] = [None] * N

        def foot(k, i):
            b = frames[k][i].box
            return (b[0] + b[2]) / 2, b[3]

        for p in sorted(pieces, key=lambda p: p.frames[0]):
            k0 = p.frames[0]
            x0, y0 = foot(k0, p.dets[0])
            best, best_cost = None, np.inf
            for j in range(N):
                if self._busy(taken, j, p.frames):
                    continue
                if last[j] is None:
                    cost = 1e6 + j  # an unused identity: take it only if nothing continues
                else:
                    kf, xf, yf, vx, vy = last[j]
                    gap = max(1, k0 - kf)
                    cost = np.hypot(x0 - (xf + vx * gap), y0 - (yf + vy * gap)) / gap ** 0.5
                if cost < best_cost:
                    best, best_cost = j, cost
            if best is None:
                continue
            p.identity, p.confidence = best, 0.5 if not p.occluded else 0.3
            taken[best].update(p.frames)
            kl = p.frames[-1]
            xl, yl = foot(kl, p.dets[-1])
            back = max(0, len(p.frames) - 6)
            kb = p.frames[back]
            xb, yb = foot(kb, p.dets[back])
            span = max(1, kl - kb)
            vx, vy = ((xl - xb) / span, (yl - yb) / span) if kl > kb else (0.0, 0.0)
            last[best] = (kl, xl, yl, vx, vy)

    def _assign_occluded(self, pieces, frames, C, sigma2) -> dict[tuple[int, int], tuple[int | None, float]]:
        """Resolve every detection the clear pieces left open, frame by frame.

        Inside an overlap a frame-to-frame tracker can swap people back and
        forth, so a whole occluded piece cannot simply inherit one identity.
        Instead each run of frames with open detections is decoded with a small
        Viterbi: the states are the ways of giving that frame's open detections
        distinct free identities, the cost of a state is how badly each crop's
        colours fit the identity it is given, and changing a tracklet's identity
        from one frame to the next -- or from what it was just before the
        overlap, or will be just after -- costs ``switch_penalty``.  A partly
        hidden kit is still mostly the right colour, and comparing the two ways
        round is far more robust than judging either crop alone.
        """
        from itertools import product

        cfg = self.cfg
        N = cfg.num_identities
        lam = cfg.switch_penalty
        n = len(frames)
        fixed: dict[tuple[int, int], int] = {}
        open_dets: list[list[int]] = [[] for _ in range(n)]
        for p in pieces:
            for k, i in zip(p.frames, p.dets):
                if p.identity is not None:
                    fixed[(k, i)] = p.identity
                else:
                    open_dets[k].append(i)
        by_tracklet: dict[int, list[tuple[int, int]]] = {}
        for (k, i), j in fixed.items():
            t = frames[k][i].tracklet
            by_tracklet.setdefault(t, []).append((k, j))
        for v in by_tracklet.values():
            v.sort()

        def known(tracklet, k, direction):
            seq = by_tracklet.get(tracklet, [])
            if direction < 0:
                prior = [j for kk, j in seq if kk < k]
                return prior[-1] if prior else None
            later = [j for kk, j in seq if kk > k]
            return later[0] if later else None

        def emission(k, i, j):
            # Relative to the crop's best fit, so a crop muddied by the other
            # body is not also pushed toward "unassigned" by its absolute misfit.
            f = frames[k][i].feature
            if C is None or f is None:
                return 0.0
            d = ((f[None, :] - C) ** 2).sum(1) / sigma2
            return float(min(d[j] - d.min(), lam))

        out: dict[tuple[int, int], tuple[int | None, float]] = {}
        k = 0
        while k < n:
            if not open_dets[k]:
                k += 1
                continue
            run = []
            while k < n and open_dets[k]:
                run.append(k)
                k += 1
            layers = []
            for kk in run:
                taken_here = {fixed[(kk, i)] for i in range(len(frames[kk])) if (kk, i) in fixed}
                free = [j for j in range(N) if j not in taken_here]
                dets = open_dets[kk]
                # Each free identity goes to one open detection or to none;
                # detections left over stay unassigned.  (d + 1) ** N states.
                states = []
                for pick in product(*[[None] + list(range(len(dets))) for _ in free]):
                    chosen = [q for q in pick if q is not None]
                    if len(chosen) != len(set(chosen)):
                        continue
                    st = [None] * len(dets)
                    for j, q in zip(free, pick):
                        if q is not None:
                            st[q] = j
                    states.append(tuple(st))
                costs = []
                for st in states:
                    c = sum(cfg.unassigned_cost if j is None else emission(kk, i, j)
                            for i, j in zip(dets, st))
                    costs.append(c)
                layers.append((kk, dets, states, np.array(costs)))

            def boundary(kk, dets, st, direction):
                c = 0.0
                for i, j in zip(dets, st):
                    ref = known(frames[kk][i].tracklet, kk, direction)
                    if ref is not None and j != ref:
                        c += lam
                return c

            kk0, dets0, states0, e0 = layers[0]
            acc = e0 + np.array([boundary(kk0, dets0, st, -1) for st in states0])
            back = []
            for (kp, dp, sp, _), (kc, dc, sc, ec) in zip(layers[:-1], layers[1:]):
                trans = np.zeros((len(sp), len(sc)))
                tp = [frames[kp][i].tracklet for i in dp]
                tc = [frames[kc][i].tracklet for i in dc]
                for a, sa in enumerate(sp):
                    ida = dict(zip(tp, sa))
                    for b, sb in enumerate(sc):
                        trans[a, b] = sum(lam for t, j in zip(tc, sb) if t in ida and ida[t] != j)
                tot = acc[:, None] + trans
                back.append(np.argmin(tot, axis=0))
                acc = tot.min(axis=0) + ec
            kl, dl, sl, _ = layers[-1]
            acc = acc + np.array([boundary(kl, dl, st, +1) for st in sl])
            best = int(np.argmin(acc))
            path = [best]
            for bp in reversed(back):
                path.append(int(bp[path[-1]]))
            path.reverse()
            for (kk, dets, states, costs), si in zip(layers, path):
                order = np.sort(costs)
                gap = float(order[1] - order[0]) if len(order) > 1 else 0.0
                conf = float(np.clip(0.3 + 0.05 * gap, 0.3, 0.8))
                for i, j in zip(dets, states[si]):
                    out[(kk, i)] = (j, conf if j is not None else 0.0)
        return out
