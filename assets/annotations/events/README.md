# Hand-labelled events

One `<clip>.labels.json` per clip in `assets/clips/`. The scorer in
`football_analysis/evaluation/` checks a run's `events.json` against these.

## Label a clip

1. Run the system on it: `football-analyse assets/clips/my_1v1.mp4 -o out/my_1v1.events.json`
2. Seed a label file from the run (much faster than starting blank):
   `python -m football_analysis.evaluation draft out/my_1v1.events.json -o assets/annotations/events/my_1v1.labels.json`
3. Open the clip in any player that shows the time to the millisecond or the frame number,
   and go through it start to end. Fix every drafted label's time, delete the ones that
   didn't happen, add the ones the system missed, and remove each `unchecked` note as you go.
4. `python -m football_analysis.evaluation check assets/annotations/events/my_1v1.labels.json`

The format is documented at the top of `football_analysis/evaluation/labels.py`. The rules
that change a score:

- **`labelled_types`** promises that every event of those types in `span_s` is labelled.
  A system event of a listed type that matches no label counts as invented. Types not
  listed are not scored at all.
- **Times** go in `timestamp_s`, `clock` (`"01:13.400"`), `frame_index` (needs `fps`),
  or `window_s: [start, end]` when you know it happened but not the exact frame.
- **`ambiguous: true`** for a moment you can't call. A system event on it is neither right
  nor wrong, and not finding it isn't a miss.
- Time an event at the moment of contact: the kick for a pass or shot, the ball crossing
  the line for a goal, the first contact for a tackle, the moment of the touch for a trick.

## Score a run

    python -m football_analysis.evaluation score out/my_1v1.events.json \
        assets/annotations/events/my_1v1.labels.json --json out/evaluation/my_1v1.score.json

Matching is one-to-one within ±1 s (SoccerNet's convention; `--tolerance` changes it),
shots and goals are also scored at ±0.5 s, and each type is reported next to the plan's
v1 target. Pool several clips with `--pair EVENTS LABELS` once per clip. Because the event
rules can be replayed from a state cache (`--state-cache` once, then `--replay`), a
threshold change can be re-scored in well under a second.

## Files here

- `broadcast_08fd33_4.labels.json`: the 30 s 11-a-side broadcast clip. Built from the ball
  events thread's frame-by-frame check of its own output, not an independent pass, and
  there are no shots, goals, tackles or tricks in it. It proves the harness works; it says
  nothing about the 1v1 targets. Kaan's own clip, labelled exhaustively, is the real test.
