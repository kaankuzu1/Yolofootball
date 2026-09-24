"""Review material for one window: a contact sheet and a short zoomed clip.

A label is only as good as the look someone took before writing it, and a
tackle is over in a third of a second.  So every candidate window gets two
things a person can decide from quickly: eight frames across the window on
one image, and a short clip cropped to the players involved, both with the
pose skeletons and the ball drawn on, so what the model saw is visible next
to what happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from football_analysis.state import FrameState

__all__ = ["render_review"]

_EDGES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
          (11, 13), (13, 15), (12, 14), (14, 16))
_COLOURS = ((80, 200, 255), (255, 140, 60), (120, 255, 120), (220, 120, 255))


def _crop_box(states: Sequence[FrameState], ids: Sequence[str], size: tuple[int, int]) -> tuple[int, int, int, int]:
    xs, ys, hs = [], [], []
    for s in states:
        for p in s.players:
            if p.player_id in ids:
                xs += [p.bbox.x1, p.bbox.x2]
                ys += [p.bbox.y1, p.bbox.y2]
                hs.append(p.bbox.height)
        if s.ball is not None:
            xs.append(s.ball.position.x)
            ys.append(s.ball.position.y)
    w, h = size
    if not xs:
        return 0, 0, w, h
    pad = 0.6 * (np.median(hs) if hs else 100)
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    # 16:9, so the sheet and the clip read naturally.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw = max(x1 - x0, (y1 - y0) * 16 / 9, 160)
    bh = bw * 9 / 16
    x0, x1 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
    y0, y1 = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
    return x0, y0, x1, y1


def _draw(image: np.ndarray, state: FrameState, ids: Sequence[str]) -> None:
    for p in state.players:
        if p.player_id not in ids:
            continue
        col = _COLOURS[list(ids).index(p.player_id) % len(_COLOURS)]
        b = p.bbox.as_int_tuple()
        cv2.rectangle(image, b[:2], b[2:], col, 1)
        cv2.putText(image, p.player_id, (b[0], max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
        kp = p.keypoints
        if len(kp) == 17:
            for a, c in _EDGES:
                if kp[a].confidence >= 0.3 and kp[c].confidence >= 0.3:
                    cv2.line(image, (int(kp[a].x), int(kp[a].y)), (int(kp[c].x), int(kp[c].y)),
                             col, 2, cv2.LINE_AA)
            for j in (15, 16):
                if kp[j].confidence >= 0.3:
                    cv2.circle(image, (int(kp[j].x), int(kp[j].y)), 3, (0, 0, 255), -1)
    if state.ball is not None:
        colour = (255, 255, 255) if not state.ball.interpolated else (140, 140, 140)
        cv2.circle(image, (int(state.ball.position.x), int(state.ball.position.y)), 6, colour, 2)


def render_review(video_path: str | Path, states: Sequence[FrameState], ids: Sequence[str],
                  start_s: float, end_s: float, out_stem: str | Path, *,
                  title: str = "", anchor_s: float | None = None,
                  resize: tuple[int, int] | None = None, clip_scale: int = 720) -> dict[str, str]:
    """Write ``<out_stem>.jpg`` (contact sheet) and ``<out_stem>.mp4`` (clip).

    ``states`` must be in the same pixel space as the frames once ``resize``
    (``(width, height)``) has been applied; ``None`` means the clip's own size.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    by_index = {s.frame_index: s for s in states}
    window = [s for s in states if start_s - 1e-6 <= s.timestamp_s <= end_s + 1e-6]
    if not window:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = window[0].frame_size if window[0].frame_size != (0, 0) else (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    x0, y0, x1, y1 = _crop_box(window, ids, size)
    first, last = window[0].frame_index, window[-1].frame_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    crops: list[tuple[int, float, np.ndarray]] = []
    for fi in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        state = by_index.get(fi)
        if state is not None:
            _draw(frame, state, ids)
        crop = frame[y0:y1, x0:x1]
        scale = clip_scale / max(1, crop.shape[0])
        crop = cv2.resize(crop, (int(crop.shape[1] * scale) // 2 * 2, clip_scale // 2 * 2),
                          interpolation=cv2.INTER_CUBIC)
        t = state.timestamp_s if state is not None else fi / fps
        stamp = f"{t:7.2f}s"
        if anchor_s is not None and abs(t - anchor_s) < 0.5 / fps:
            stamp += "  <- call"
        cv2.putText(crop, stamp, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        crops.append((fi, t, crop))
    cap.release()
    if not crops:
        return {}

    # Contact sheet: eight frames, the call frame among them when there is one.
    picks = list(np.linspace(0, len(crops) - 1, 8).round().astype(int))
    if anchor_s is not None:
        k = int(np.argmin([abs(c[1] - anchor_s) for c in crops]))
        nearest = int(np.argmin([abs(p - k) for p in picks]))
        picks[nearest] = k
    cells = [cv2.resize(crops[i][2], (400, 225)) for i in sorted(set(picks))]
    while len(cells) < 8:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[:4]), np.hstack(cells[4:8])])
    if title:
        bar = np.full((34, sheet.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        sheet = np.vstack([bar, sheet])
    jpg = out_stem.with_suffix(".jpg")
    cv2.imwrite(str(jpg), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    mp4 = out_stem.with_suffix(".mp4")
    h, w = crops[0][2].shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _, _, crop in crops:
        writer.write(cv2.resize(crop, (w, h)))
    writer.release()
    return {"sheet": str(jpg), "clip": str(mp4)}
