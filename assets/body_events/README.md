# Tackle and trick dataset

Labelled windows for the Tier 2 classifier in `football_analysis/body_events/`
(plan §4.2 and §4.3). Built to grow: every window stores the world-state
records the pipeline produced, not precomputed features, so features can change
without relabelling anything.

```
assets/body_events/
  labels.csv                  one row per candidate window
  windows/<window_id>.jsonl   state records for the window plus 0.5 s either side
  review/<window_id>.jpg      eight frames across the window, pose drawn on
  review/<window_id>.mp4      the window as a short zoomed clip
  model/body_events.joblib    the trained classifiers
  model/report.json           what the current model scored, and on what
  real/<clip>.states.jsonl    full-clip records of real clips proposed from
```

## labels.csv

| column | meaning |
| --- | --- |
| `window_id` | `<clip>_<c or s>_<start seconds>_<player>` |
| `kind` | `contact` (two players converging) or `skill` (one player on the ball) |
| `label` | see below; empty = not labelled yet |
| `sub_label` | named move when known: `stepover`, `nutmeg`, `dragback`, `feint`, `roulette`, `elastico`, `other` |
| `source_clip` | the video the window came from |
| `start_s`, `end_s` | the window, seconds in the source clip |
| `anchor_s` | contact windows: the moment the call is made at |
| `player_id` | contact: the player who had the ball (attacker); skill: the player on the ball |
| `other_player_id` | contact: the defender; skill: the nearest opponent |
| `origin` | `real` for footage; `synthetic` never appears here (simulated windows are regenerated from seeds) |
| `labeller`, `notes` | who looked, and anything worth knowing |

### Labels

`kind = contact`

- `tackle_won`: the defender's foot reached the ball and the attacker lost it
  (to the defender or loose)
- `tackle_failed`: the defender went for the ball with a leg (lunge, poke,
  slide) and the attacker kept it
- `shoulder_duel`: bodies met, no leg went for the ball
- `no_contact`: they came close, nothing happened

`kind = skill`

- `trick`: a deliberate skill move (feet round the ball, drag-back, feint,
  nutmeg, turn with a fake). Put the move's name in `sub_label` when you know it.
- `dribble`: ordinary running with the ball, including an ordinary turn

`skip`: looked at and unusable (occluded, off frame, not a two-player moment).

**The one call that needs a policy:** is a sharp ordinary turn a trick? The
current labels say no (`dribble`), and a drag-back or Cruyff turn yes. If your
footage says otherwise, label it that way; the model follows the labels.

## Adding footage

```bash
# 1. run the pipeline on a clip and keep its records
football-analyse clip.mp4 --state-cache out/clip.states.jsonl
# 2. cut candidate windows and review sheets
python -m football_analysis.body_events propose clip.mp4 --states out/clip.states.jsonl
# 3. watch review/<window_id>.mp4, then label (or fill the column in labels.csv)
python -m football_analysis.body_events label clip_c_0012.40_Player2 tackle_won
# 4. retrain; real windows count 5x a simulated one
python -m football_analysis.body_events train
python -m football_analysis.body_events evaluate --out assets/body_events/model/report.json
```

If the records have no pose keypoints (a pipeline run without the pose stage),
`football_analysis.pose.attach_pose` fills them in from the clip.

## What is in here now

- `real/broadcast_08fd33_4.states.jsonl`: a 30 s Bundesliga broadcast clip,
  the only real footage reachable from the build environment. Wide shot,
  panning camera, players 30 to 95 px tall, the ball seen in about one frame
  in five. The stage makes no calls on it. Before the evidence gate
  (`BodyEventConfig.min_evidence`, `ball_seen_near_s`) it made 63, nearly all
  on crowded moments with no ball seen, and the two with full evidence were a
  goalkeeper and a referee standing still near a false ball
  (`examples/real_broadcast_rejected_26s.*`). It is a sanity check on real
  pose, not a benchmark: no labelled windows come from it.
- No side-angle 1v1 footage yet. The model is trained on simulated motion
  (`football_analysis/body_events/synthetic.py`); see the module docstring for
  what that does and does not show. `model/report.json` holds the held-out
  scores; `examples/` holds rendered calls (`render_examples.py` remakes them).
