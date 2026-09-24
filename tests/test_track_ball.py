"""The ball filter: rejects impossible detections, survives occlusion, sees kicks."""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.track.ball import BRIDGED, DETECTED, SMOOTHED, BallTrajectoryFilter
from football_analysis.track.config import BallFilterConfig
from football_analysis.track.observations import Obs

D = 20.0
FPS = 30.0


def _ball(x: float, y: float, conf: float) -> Obs:
    return Obs("ball", x - D / 2, y - D / 2, x + D / 2, y + D / 2, conf)


def _scenario(seed: int):
    """Roll, kick beside a player, fly, rebound off a wall, then a long occlusion."""
    rng = np.random.default_rng(seed)
    n = int(6 * FPS)
    t = np.arange(n) / FPS
    truth = np.zeros((n, 2))
    for k, tk in enumerate(t):
        if tk < 2.0:
            truth[k] = (200 + 150 * tk - 10 * tk**2, 600)
        elif tk < 3.2:
            s = tk - 2.0
            truth[k] = (460 + 900 * s, 600 - 700 * s + 583 * s * s)
        else:
            truth[k] = (1540 - 300 * (tk - 3.2), 600)
    hidden = set(range(75, 82)) | set(range(120, 140))
    cands, players = [], []
    for k in range(n):
        px = truth[min(k, 60)][0]
        players.append([(px - 40, 480, px + 10, 610)])
        c = []
        if k not in hidden and rng.random() > 0.1:
            x, y = truth[k] + rng.normal(0, 2, 2)
            c.append(_ball(x, y, float(rng.uniform(0.2, 0.8))))
        if rng.random() < 0.3:  # a false positive anywhere
            c.append(_ball(rng.uniform(0, 1900), rng.uniform(0, 1000), float(rng.uniform(0.05, 0.4))))
        if rng.random() < 0.6:  # static clutter, e.g. a white mark on a hoarding
            c.append(_ball(1510, 310, 0.3))
        cands.append(c)
    return t, cands, players, truth


@pytest.mark.parametrize("seed", range(6))
def test_track_follows_the_ball_and_rejects_false_positives(seed):
    t, cands, players, truth = _scenario(seed)
    track = BallTrajectoryFilter().run(t, cands, players)
    err = np.hypot(*(track.xy - truth).T)
    assert np.nanmedian(err[track.source == DETECTED]) < 2.5
    # Nothing present is ever far from the truth: no locking onto clutter.
    assert np.nanmax(err[track.present]) < 30
    assert track.stats["presence_after_interpolation"] > 0.97


def test_occlusion_is_filled_and_flagged():
    t, cands, players, truth = _scenario(0)
    track = BallTrajectoryFilter().run(t, cands, players)
    hidden = np.arange(75, 82)
    assert np.all(np.isin(track.source[hidden], [SMOOTHED, BRIDGED]))
    assert np.all(track.interpolated[hidden])
    assert np.nanmax(np.hypot(*(track.xy[hidden] - truth[hidden]).T)) < 10
    assert any(g.start <= 75 and g.end >= 81 for g in track.gaps)


def test_rebound_is_a_kinematic_break():
    t, cands, players, truth = _scenario(0)
    track = BallTrajectoryFilter().run(t, cands, players)
    breaks = np.where(track.kinematic_break)[0]
    assert any(94 <= b <= 100 for b in breaks)  # the wall rebound at 3.2 s


def test_long_gap_is_not_invented():
    """A ball missing longer than the bridge limit stays missing."""
    n = 120
    t = np.arange(n) / FPS
    cands = [[_ball(100 + 5 * k, 500, 0.6)] if (k < 30 or k >= 90) else [] for k in range(n)]
    track = BallTrajectoryFilter(BallFilterConfig(bridge_max_s=1.0)).run(t, cands)
    assert not track.present[40:80].any()
    gap = next(g for g in track.gaps if g.start <= 40 <= g.end)
    assert not gap.filled and gap.kind == "lost"


def test_no_candidates_gives_an_empty_track():
    track = BallTrajectoryFilter().run([0.0, 0.04], [[], []])
    assert not track.present.any()
