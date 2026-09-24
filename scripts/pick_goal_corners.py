#!/usr/bin/env python3
"""Click the goal's four corners on one frame and print the --goal-corners value.

    python scripts/pick_goal_corners.py match.mp4 --at 5

A window opens on the frame at 5 s. Click, in this order, as seen in the image:
the bottom of the left post, the bottom of the right post, the right end of the
crossbar, the left end of the crossbar. Esc cancels. The corners are printed at
the clip's own resolution, which is what ``--goal-corners`` expects by default:

    football-analyse match.mp4 --goal-corners 980,700,1420,700,1420,420,980,420

Do this once per camera setup. With no display (``--save-frame`` only), the
frame is written to an image instead, so the corners can be read off it in any
image viewer that shows pixel coordinates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from football_analysis.detection.goal import StaticGoal  # noqa: E402


def read_frame(path: str, at_s: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame at {at_s}s in {path}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="the clip to read a frame from")
    parser.add_argument("--at", type=float, default=0.0, help="seconds into the clip (default 0)")
    parser.add_argument("--save", metavar="PATH", help="also save the corners as JSON, for reuse")
    parser.add_argument(
        "--save-frame", metavar="PATH",
        help="write the frame to an image and exit, without opening a window",
    )
    args = parser.parse_args()

    frame = read_frame(args.clip, args.at)
    if args.save_frame:
        cv2.imwrite(args.save_frame, frame)
        print(f"frame at {args.at}s -> {args.save_frame} ({frame.shape[1]}x{frame.shape[0]})")
        return 0

    try:
        goal = StaticGoal.from_click(frame)
    except ValueError:
        raise SystemExit("cancelled: all four corners are needed")
    finally:
        cv2.destroyAllWindows()
    if args.save:
        goal.save(args.save)
    value = ",".join(f"{v:.0f}" for point in goal.corners for v in point)
    print(f"--goal-corners {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
