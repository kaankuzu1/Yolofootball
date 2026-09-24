#!/usr/bin/env python3
"""Render a synthetic 1v1 clip, for trying the pipeline without real footage.

    python scripts/make_sample_clip.py out/sample.mp4 --duration 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_analysis.io.synthetic import make_sample_clip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="where to write the clip")
    parser.add_argument("--duration", type=float, default=6.0, help="seconds")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    truth = make_sample_clip(
        args.output,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(
        f"wrote {args.output}: {truth.frame_count} frames, "
        f"{truth.duration_s:.1f}s @ {truth.fps:g}fps, {truth.size[0]}x{truth.size[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
