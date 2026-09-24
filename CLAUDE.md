# CLAUDE.md

Timestamped tackles, tricks, passes, shots and goals from a 1v1 football clip
filmed by one still, side-angled camera. Python 3.11+, Ultralytics YOLO,
OpenCV, package `football_analysis`. The owner (Kaan) often writes in Turkish:
answer in the language he used. The desktop app's UI is Turkish.

## Commands

```bash
pip install -r requirements-app.txt && pip install -e .   # everything, app included
python -m pytest                    # whole suite, no network; weight-dependent tests skip
ruff check .                        # lint, line length 100
football-analyse clip.mp4 --goal-corners X1,Y1,...,X4,Y4 --goal-size 3 2 \
  -o clip.events.json --render clip.annotated.mp4 --state-cache clip.states.jsonl
football-analyse --replay clip.states.jsonl -o out.json   # re-run the rules, no inference
python -m football_analysis.app     # the desktop app (PySide6)
```

Use `python -m pytest`, not a bare `pytest` from another environment. If
torch and scipy crash with `Intel oneMKL FATAL ERROR`, set
`MKL_THREADING_LAYER=GNU`. App tests need `QT_QPA_PLATFORM=offscreen` on a
machine with no display.

## Layout

| Where | What |
| --- | --- |
| `io/` | video reader with trustworthy timestamps, writer |
| `detection/` | YOLO for players, ball and goal; `goal.py` holds the corner order |
| `track/`, `geometry/` | identities, ball Kalman filter, camera pose from the goal (PnP) |
| `pose/` | keypoints, run as a `StateRefiner` after tracking settles |
| `ball_events/` | possession, pass, push_past, shot, goal: geometric rules |
| `body_events/` | tackle and trick: a small classifier over pose |
| `evaluation/` | scorer against hand labels |
| `app/` | the desktop app: camera picker, live boxes, recording, analysis |
| `pipeline.py`, `interfaces.py`, `state.py`, `events.py`, `types.py` | the run and the shared contracts |
| `cli/`, `config/`, `viz/` | command line, config dataclasses, drawing |
| `docs/system-plan.md` | why the design is what it is; `docs/mac-app.md` is the app guide (Turkish) |

## Rules the code relies on

- **Event logic never sees pixels.** It reads one `FrameState` per frame
  (`state.py`). A rule that needs pixels gets a `FrameObserver` that writes onto
  `state.attributes`, so the reading also lands in the state cache and
  `--replay` stays honest.
- **The run is two-phase.** The tracker's `finalize_states()` refines the whole
  clip before the rules run; rules emit headline events from `finalize()`; the
  annotated clip is drawn in a second decode pass. So events cannot be produced
  live, and the app lists them only after a recording.
- **Boxes are in processed-frame pixels** (`video.resize_width`, 1280 by
  default), not source pixels. Goal corners are clicked at the source
  resolution and rescaled; `geometry.goal_corners_px` is processed space.
- **Goal corner order:** left post base, right post base, right crossbar end,
  left crossbar end, as seen in the image.
- **Weights are never downloaded implicitly.** They live in `assets/models/`
  (gitignored, some exceed GitHub's 100 MB). A missing checkpoint falls back
  to a placeholder with a loud warning; `stats.stages.detector` in the output
  says what actually ran. Check it before trusting a result.
- **Adding an event type never changes the event envelope** (`events.py`).
- **The camera is assumed still.** The ball's gravity term, the net-ripple goal
  check and the one-time goal corners all depend on it.
- A stage with its own vocabulary is wrapped by an adapter (`adapters.py`),
  not rewritten.

## Working here

- Kaan's own clips are the held-out test. Never tune thresholds on them, or
  the score flatters the system.
- Claims about accuracy carry what they were measured on. Nothing has been
  measured yet on real side-angle 1v1 footage: the numbers in the README come
  from a broadcast clip and simulation.
- No GPU was available while this was built. Timings in the docs are CPU
  numbers, and nothing in `app/` has been run on a real Mac.
- Keep the README's "How far to trust it today" table current when numbers
  change.
