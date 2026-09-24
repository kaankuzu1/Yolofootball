"""The identity lock: two players, two identities, through every overlap."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import pytest

from football_analysis.track.appearance import kit_feature, pitch_hue
from football_analysis.track.players import ByteTracker, IdentityLock, PlayerFrameObs, box_iou


def _render(kits, n=160, seed=0):
    """Two players crossing twice on grass; fully overlapping boxes merge into one."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        img = np.zeros((360, 640, 3), np.uint8)
        img[:] = (60, 130, 55)
        s = k / n
        pos = [(100 + 430 * s, 250), (540 - 430 * s + 50 * np.sin(12 * s), 245)]
        boxes = []
        for j, (x, y) in enumerate(pos):
            b = (x - 14, y - 75, x + 14, y)
            cv2.rectangle(img, (int(b[0]), int(b[1] + 10)), (int(b[2]), int(b[1] + 40)), kits[j], -1)
            cv2.rectangle(img, (int(b[0] + 4), int(b[1] + 40)), (int(b[2] - 4), int(b[3])), (30, 30, 30), -1)
            boxes.append(b)
        if box_iou(boxes[0], boxes[1]) > 0.5:
            b = tuple(np.r_[np.minimum(boxes[0][:2], boxes[1][:2]), np.maximum(boxes[0][2:], boxes[1][2:])])
            dets = [(b, 0.6, None)]
        else:
            dets = [(tuple(v + rng.normal(0, 1.5) for v in b), 0.8, j) for j, b in enumerate(boxes)]
        rng.shuffle(dets)
        out.append((img, dets))
    return out


def _run(kits, swap_tracklets=False):
    frames = _render(kits)
    bt = ByteTracker()
    obs = []
    for k, (img, dets) in enumerate(frames):
        ids = bt.step([d[0] for d in dets], [d[1] for d in dets])
        if swap_tracklets and k >= len(frames) // 2:
            # Simulate a frame-to-frame tracker that swapped the two people.
            ids = [({1: 2, 2: 1}.get(i, i) if i is not None else None) for i in ids]
        grass = pitch_hue(img)
        obs.append([PlayerFrameObs(d[0], d[1], tid, kit_feature(img, d[0], grass))
                    for d, tid in zip(dets, ids)])
    result = IdentityLock().run(obs)
    pairs = [(result.assign[k][i], d[2]) for k, (_, dets) in enumerate(frames)
             for i, d in enumerate(dets) if d[2] is not None and result.assign[k][i] is not None]
    mapping = {}
    for (a, b), _ in Counter(pairs).most_common():
        if a not in mapping and b not in mapping.values():
            mapping[a] = b
    wrong = sum(1 for a, b in pairs if mapping.get(a) != b)
    return result, wrong, len(pairs)


@pytest.mark.parametrize("kits", [[(40, 40, 200), (200, 120, 40)], [(240, 240, 240), (20, 20, 20)],
                                  [(60, 230, 190), (235, 235, 235)]])  # red/blue, white/black, lime/white
def test_no_identity_errors_with_distinct_kits(kits):
    result, wrong, total = _run(kits)
    assert result.report["method"] == "appearance"
    assert total > 250
    assert wrong == 0


def test_a_tracker_swap_is_corrected():
    result, wrong, _ = _run([(40, 40, 200), (200, 120, 40)], swap_tracklets=True)
    assert wrong == 0
    assert result.report["swaps_corrected"] >= 1


def test_lookalike_kits_fall_back_and_warn():
    result, _, _ = _run([(40, 40, 200), (40, 40, 205)])
    assert result.report["method"] == "motion_continuity"
    assert result.report["warnings"]


def test_never_two_boxes_for_one_identity():
    result, _, _ = _run([(40, 40, 200), (200, 120, 40)])
    for row in result.assign:
        ids = [j for j in row if j is not None]
        assert len(ids) == len(set(ids))


def test_lime_kit_survives_grass_masking():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (60, 130, 55)
    cv2.rectangle(img, (80, 40), (120, 120), (60, 240, 180), -1)  # fluorescent green shirt
    f = kit_feature(img, (80, 30, 120, 180), pitch_hue(img))
    assert f is not None and f[:64].sum() > 0.5  # chroma bins, not just brightness
