# football-analysis

Analyse a 1v1 football clip shot from a side angle and get back a timestamped
list of what happened: tackles, tricks, passes, shots and goals, from
detections of the ball, the two players and the goal.

A clip goes in one end:

```bash
football-analyse match.mp4 -o events.json --render annotated.mp4
```

and a JSON timeline comes out the other:

```json
{
  "schema_version": "1.0",
  "source": { "path": "match.mp4", "width": 1920, "height": 1080, "fps": 29.97 },
  "players": [{ "player_id": "Player 1", "label": "Player 1", "track_id": 1 }],
  "counts": { "pass": 12, "tackle": 3, "shot": 4, "goal": 1 },
  "events": [
    {
      "id": "evt_0007",
      "type": "tackle",
      "timestamp_s": 73.44,
      "clock": "01:13.440",
      "confidence": 0.81,
      "player_id": "Player 2",
      "secondary_player_id": "Player 1",
      "detail": { "winner_player_id": "Player 2" }
    }
  ]
}
```

## Start here: your own clip in four steps

This is the whole system: YOLO finds the two players, the ball and the goal,
a tracker follows them, and two rule engines turn that into a timeline of
tackles, tricks, passes, shots and goals. Everything below the Install section
is reference for changing it.

**1. Install** (Python 3.11 or newer; a GPU is used automatically if there is one)

```bash
pip install -r requirements-detect.txt
pip install -e .
```

**2. Mark the goal once per camera setup.** Film from a tripod or a clamped
phone with the whole goal in view. Then click its four corners on any frame:

```bash
python scripts/pick_goal_corners.py my_1v1.mp4 --at 5
```

It prints a `--goal-corners ...` value to paste into step 3. Without the goal
corners the system still logs passes, tackles and tricks, but it cannot call a
shot on target or a goal.

**3. Run it**

```bash
football-analyse my_1v1.mp4 --goal-corners <from step 2> --goal-size 3 2 \
  -o my_1v1.events.json --render my_1v1.annotated.mp4 --state-cache my_1v1.states.jsonl
```

`--goal-size` is the goal mouth's width and height in metres (a full-size goal
is `7.32 2.44`). You get the timeline as JSON, the clip with boxes, ball trail
and event captions drawn on, and a record that lets you re-run the event rules
in under a second after changing a setting (`football-analyse --replay
my_1v1.states.jsonl -o ...`). Events a rule was unsure of are listed under
`needs_review` at the top of the JSON, so check those first.

**4. Check it and make it better on your footage.** Correct a label file drafted
from the run, then score the system against it
(`assets/annotations/events/README.md` has the steps). The tackle and trick
model retrains on your labelled moments with one command
(`assets/body_events/README.md`).

### Or use the Mac app

A window instead of a terminal: pick the camera (the Mac's front camera by
default; USB cameras and an iPhone appear by name when connected), click the
goal's corners once, record, and get the event list beside the video with
each event one click from its moment. Double-click
`scripts/mac/1v1 Analiz.command` to set up and start it, or run
`python -m football_analysis.app`. Steps in Turkish: `docs/mac-app.md`.

### How far to trust it today

| What | Checked on | Result |
| --- | --- | --- |
| Players | 25 hand-labelled frames of real broadcast footage | every player found, no crowd false positives |
| Ball | same frames | 82.6% found, 86.4% of finds correct; the tracker fills the gaps (100% of frames placed on a 30 s clip, longest filled gap 0.48 s) |
| Goal geometry | simulation, 10 m from a 3 x 2 m goal | positions on the ground to within 0.1 to 0.4 m |
| Possession changes | 30 s real broadcast clip | 10 of 11 found, none invented |
| Passes | same clip | 1 of 4 found, 1 invented; this is 11-a-side play, not a 1v1 |
| Shots and goals | 11 simulated scenes only | not yet checked on real footage |
| Tackles | 400 simulated clips | F1 0.96 in a clean view, 0.91 with small, noisy players (target 0.75) |
| Tricks | same | F1 0.93 and 0.85 (target 0.60); an ordinary turn is called a trick 5% of the time in a clean view, 37% with small players |

No side-angle 1v1 footage was reachable while this was built, so nothing above
has been measured on the kind of clip it is for. The first real test is your
own clip, scored in step 4. That clip is kept as the test and never used to
tune settings, or the score would flatter the system.

**Speed.** On a CPU with no GPU, a run processes about 1.4 frames a second
(measured on 10 s of 1920 x 1080 footage, pose included), so a 3-minute clip at
25 fps takes around 55 minutes. `--stride 2` halves that at the cost of
timing precision. A GPU should be many times faster, but that was not
measured here.

### Choices made for you, and how to change them

| Question | Default | To change it |
| --- | --- | --- |
| Does the camera move? | It stays still. A panning camera breaks the ball's gravity model, the net-ripple check and the one-time goal corners. | Tell us; it needs code changes, not a setting |
| Goal size | 3 x 2 m | `--goal-size W H` |
| What is a pass in a 1v1? | With two players, a knock past the defender to run onto is logged as `push_past`, and the ball changing feet between them is a turnover. A third "feeder" player who serves the ball in makes passes possible, but that setup has not been tried on real footage. | the `ball_events:` and `tracking:` sections of `configs/default.yaml` explain both |
| Is a sharp ordinary turn a trick? | No. A drag-back or Cruyff turn is. | Label turns as tricks in step 4 and retrain |
| Trick sensitivity | `body_events.trick_threshold: 0.85`, set high so ordinary turns are rarely called | lower it to catch more tricks and accept more false ones |

### What's in the box

| Part | Where |
| --- | --- |
| Video input with trustworthy timestamps | `football_analysis/io/` |
| YOLO detection (players, ball, goal) | `football_analysis/detection/`, weights in `assets/models/` |
| Tracking, identities and camera geometry | `football_analysis/track/`, `football_analysis/geometry/` |
| Pose keypoints | `football_analysis/pose/` |
| Possession, pass, shot and goal rules | `football_analysis/ball_events/` |
| Tackle and trick classifier | `football_analysis/body_events/`, model in `assets/body_events/model/` |
| Accuracy scorecard | `football_analysis/evaluation/` |
| How the design was decided | `docs/system-plan.md`, `docs/detection-report.md` |

## Install

Python 3.11 or newer.

```bash
pip install -r requirements.txt       # io, config, viz, CLI, event logic
pip install -r requirements-detect.txt  # adds ultralytics + torch
pip install -e .                      # puts `football-analyse` on your PATH
```

The detector's dependencies are a separate file on purpose: everything except
detection installs, imports and tests without a multi-gigabyte torch download.

### Model weights

Weights and video clips are kept out of git (`*.pt` and `*.mp4` are in
`.gitignore`), except the small goal model, which was trained for this project
and exists nowhere else. A run looks for weights under `assets/models/` and
never downloads them on its own, so put these three files there first:

| File | What it does | Where to get it |
|---|---|---|
| `forzasys_soccer.pt` | players and ball (the default detector) | the football checkpoint from [forzasys-students/SportsVision-YOLO](https://github.com/forzasys-students/SportsVision-YOLO), saved under this name |
| `yolo11s-pose.pt` | body keypoints for tackles and tricks | [Ultralytics release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt) |
| `goal_yolo11n.pt` | finds the goal | in this repository |

The project's delivery zip (`football-analysis.zip`) carries all three plus the
sample broadcast clip. Without `forzasys_soccer.pt` a run falls back to a
placeholder detector and warns loudly; check `stats.stages.detector` in the
output to see what actually ran.

## Try it without footage

```bash
python scripts/make_sample_clip.py out/sample.mp4 --duration 8
python -m football_analysis.cli out/sample.mp4 \
  -o out/events.json --render out/annotated.mp4 \
  --events ball_touch,possession_change --min-event-confidence 0.3
```

That renders a synthetic side-angle 1v1 -- a pitch, a goal, two players and a
ball that is dribbled, passed and shot -- runs it through the whole pipeline,
and writes both a timeline and an annotated clip. It is a fixture for
exercising the plumbing, not training data.

## Usage

```
football-analyse CLIP [-o events.json] [--render annotated.mp4]

  -c, --config PATH       YAML or JSON config layered over the defaults
  --start / --end SECS    analyse only part of the clip
  --stride N              process every Nth frame
  --resize-width PX       resize before analysis (default 1280)
  --no-resize             analyse at the clip's own resolution
  --model PATH            detector weights (a bare name is looked for in
                          assets/models too)
  --device DEV            auto | cpu | cuda | cuda:0 | mps
  --events LIST           which event types to report
  --state-cache PATH      write the per-frame world-state record, for --replay
  --replay PATH           re-run event logic over a cached record, no clip needed
  --goal-corners LIST     the four goal-mouth corners, annotated once per setup
  --goal-size W H         goal mouth in metres (default 3 2)
  --print-config          show the resolved settings and exit
```

Settings resolve in three layers, most specific last: defaults, then the
config file, then the flags. The resolved config is copied into the output
document, so any result says how it was produced.

From Python:

```python
from football_analysis import analyze

result = analyze("match.mp4")
for event in result.timeline.primary():
    print(event.clock, event.type.value, event.player_id, event.confidence)
```

## The event schema

One document per run: source metadata, the players it refers to, run stats,
the config, and a flat list of events sorted by time. Every event carries:

| Field | Meaning |
| --- | --- |
| `id` | stable within a run, so events can reference each other |
| `type` | `tackle`, `trick`, `pass`, `shot`, `goal`, plus supporting types |
| `timestamp_s` | seconds from the start of the source clip |
| `clock` | the same time as `MM:SS.mmm`, for reading |
| `confidence` | 0..1, from the stage that called it |
| `frame_index` | source frame the call was made on |
| `end_timestamp_s` | for events with duration; absent means instantaneous |
| `player_id` | whose event it is -- the tackler, the passer, the shooter |
| `secondary_player_id` | the other party -- the tackled, the receiver |
| `track_ids` | the tracks that evidenced it, for drawing and debugging |
| `detail` | type-specific payload (see below) |
| `source` | which stage emitted it, for triage |

Two conventions worth knowing. A stage that assigns its own event `id` (like
`ball_0003`) is stating that it has already de-duplicated; the pipeline then
leaves those events alone rather than applying `min_gap_s`, so a shot and the
rebound the same player puts away 0.2s later both survive. And a rule that is
unsure sets `detail["needs_review"]` with a `review_reason`; those events are
listed under `needs_review` at the top of the document and by
`timeline.needs_review()`, so uncertainty is surfaced rather than hidden
behind a confident-looking number.

`detail` is where each event type puts what is specific to it: a pass records
`receiver_player_id` and `distance_px`, a shot records `on_target` and
`speed_px_s`, a trick records `trick_name`. Adding a type means adding an
`EventType` member and documenting its `detail` keys -- never changing the
envelope, because everything downstream reads the envelope.

Alongside the five headline types the schema carries `push_past`,
`ball_touch`, `possession_change` and `out_of_play`.

`push_past` is reported by default. A strict 1v1 has no teammate, so a
deliberate forward knock beyond the defender to re-collect is a real event of
its own, and calling it a pass would be wrong.

The other three are opt-in via `--events`. They are cheap to emit and they
make a timeline debuggable -- the headline events are usually derived from
them -- but they are intermediate signal rather than something to report.
`timeline.primary()` filters down to the five headline types whatever is
enabled.


## Which detector a run actually uses

The real model is used whenever its weights are on disk. `--model` takes a
path, or a bare name that is also looked for under `assets/models`, so the
bundled weights can be named without a path:

```bash
football-analyse match.mp4 --model forzasys_soccer.pt
```

A relative search path is tried against the working directory first and then
against the project the package was installed from, so running this from your
clips folder finds the same weights as running it from the checkout. Without
that, the working directory silently decided which detector you got.

Weights are never downloaded implicitly -- a run that quietly pulls a model off
the internet is a surprise, and one that hangs trying to is worse. When the
weights are not there, the run falls back to the placeholder detector and says
so at WARNING level, because a run that analyses motion blobs while reporting
success looks like it worked and is wrong about everything. The document also
always records which detector actually ran, under `stats.stages.detector`.

Backend-specific knobs go in `detection.options`, which is passed straight
through -- `tile_ball: true` runs a second pass over crops so the ball is seen
nearer its native scale.

With no `-o`, the timeline goes to stdout and nothing else does: anything a
model backend prints is redirected to stderr, so `football-analyse clip.mp4 |
jq` works.

## Telling it where the goal is

No detector reliably finds a goal, so the goal is annotated once per camera
setup. Read the four mouth corners off your clip and pass them in the order
left post base, right post base, right crossbar end, left crossbar end (left
and right as seen in the image):

```bash
football-analyse match.mp4 --goal-corners 980,700,1420,700,1420,420,980,420 \
                           --goal-size 3 2
```

Corners are taken at the clip's **own** resolution by default and rescaled for
you if `--resize-width` applies -- because people read coordinates off their
footage, not off a resized frame they never see, and silently mixing the two
puts the goal in the wrong place and every goal call with it. Pass
`--goal-corners-space processed` if you really do mean processed-frame pixels.

## The world-state record

Detection, tracking, pose and geometry write one `FrameState` per processed
frame. Event logic reads only that record and never touches the video:

```python
@dataclass
class FrameState:
    frame_index: int
    timestamp_s: float
    players: list[PlayerState]   # id, box, confidence, velocity, pose keypoints
    ball: BallState | None       # position, confidence, box, velocity, interpolated
    goal: GoalState | None       # box, and the four mouth corners once solved
    possessor_id: str | None
    frame_size: tuple[int, int]  # the space every coordinate above is in
    scale: float
```

The field that is easiest to get wrong is **`ball.interpolated`**. A ball
tracker coasts through occlusions -- a player's legs hide the ball for five
frames and the tracker keeps predicting where it must be. Those positions are
not observations, and a rule that treats them as observations will call passes
and shots off a position nobody ever saw. The flag travels with the position;
`ball.is_observed` is the readable form, and `ball.frames_since_seen` says how
stale it is.

`state_from_tracks` builds the record from tracks and picks up anything a
stage put on a track's `attributes` -- `keypoints` from a pose stage, `quad`
from geometry, `interpolated` from a ball tracker that knows it is coasting.
So those stages need no change to the pipeline to get their data into the
record.

### When a rule needs pixels

Some decisions cannot be made from boxes. Whether the net rippled is the
obvious one: no amount of geometry tells you, and the goal rule is much weaker
without it. Event logic is still denied the frame, so the reading is taken
where the frame exists and left on the record:

```python
class NetMotionMeter:
    def measure(self, image, state) -> float | None: ...

pipeline = AnalysisPipeline(
    config, detector=..., tracker=..., event_detector=...,
    observers=[MeasurementObserver(NetMotionMeter(), "net_motion")],
)
```

The rule then reads `state.attributes["net_motion"]`. Because the reading
lands on the record, it goes into the state cache too, so a replay sees
exactly what the live run saw.

A meter returning `None` means *no reading on this frame*, and the key stays
absent -- which is deliberately different from a reading of zero. "The net did
not move" and "nobody was looking at the net" are different facts, and a rule
that cannot tell them apart will draw the wrong conclusion.

### When a reading needs the settled boxes

An observer runs inside the loop, one frame at a time, before the tracker has
settled anything. Some readings cannot be taken then. Pose keypoints are the
case: reading a skeleton is expensive, and it is only worth spending on the
players who turn out to be real, on the boxes the tracker finally chose.

So there is a second seam, a `StateRefiner`, which runs *after*
`finalize_states()` and *before* the rules, and is handed the finished records
plus the clip it may re-read:

```python
pipeline = AnalysisPipeline(
    config, detector=..., tracker=..., event_detector=...,
    refiners=[PoseRefiner(PoseEstimator(), target_width=config.video.resize_width)],
)
```

On a crowded clip that is the difference between running pose on two players
and running it on twenty provisional boxes, half of which the tracker is about
to merge or drop. Measured on 60 frames of broadcast footage: pose filled 117
of 120 player records and cost about 0.15 s a frame on CPU, next to 0.7 s for
detection.

Refiners run before the state cache is written, so what they add survives into
a `--replay`. A refiner that raises is reported and skipped rather than taking
the run down: pose weights that will not load should cost the tackles, not the
goals. Which refiners ran is recorded in the timeline's
`stats.stages.refiners`, because "no tackles" and "no pose" look identical
from the timeline alone.

Pose weights, like detector weights, are never downloaded implicitly. A bare
name is looked for under `detection.weights_search_paths`; if it is not there
the step is skipped with a warning naming the path it wanted.

### Running several rule engines

Event logic is split by what it reasons about -- ball flight in one place,
bodies in another. `CompositeEventDetector` fans each record out to all of
them and merges what they call:

```python
from football_analysis.adapters import CompositeEventDetector

event_detector = CompositeEventDetector(BallEventDetector(config), TackleDetector(config))
```

`default_event_stage(config)` builds this for you -- the real rule engines
when they are importable, each with whatever observers its rules need, and the
placeholder only as a fallback so a partly-built checkout still runs.

### Caching and replay

Because the record is the whole interface, a run can write it to disk and the
event rules can then be re-run against it without any inference:

```bash
football-analyse match.mp4 -o events.json --state-cache state.jsonl
football-analyse --replay state.jsonl -o events.json --events tackle,pass,shot,goal
```

On a 100-frame segment of the broadcast clip in `assets/`, inference takes
about 26 seconds on CPU and replaying the same frames from the record takes
0.003 seconds. Tuning a tackle threshold is a file read, not a re-run.

The cache is JSON Lines with a schema-version header, so an interrupted run
still leaves a readable file, a long clip never has to be held in memory, and
a cache written by a different schema is refused rather than misread.

## The two-phase run

A tracker's best answers often need the whole clip: an identity lock clusters
every player crop at once, a ball smoother runs backward from the future. So a
run has two phases. The loop reads frames, detects, tracks and takes any pixel
measurements; then, once the clip has ended, the tracker refines its records
and *those* are what the rules read.

A tracker that can do this exposes `finalize_states()`; one that works frame by
frame does not, and its records pass straight through. Both work.

Two details that matter more than they look:

- **Observer readings are carried across the refinement.** They were written
  onto the provisional records during the loop, and a refined record knows
  nothing about them. Taking the new list wholesale would drop every one, and
  the only symptom would be a goal rule quietly getting less sure of itself.
- **The annotated clip is drawn from the refined records too.** Otherwise the
  video captions someone `player#21` while the JSON calls them `Player 1`, and
  the two outputs contradict each other -- worse than either being wrong alone.

The tracker's clip-level summary (calibration, identity report, every ball gap)
goes into the state cache's header under `tracking`, nested rather than merged
so a summary key called `record` cannot corrupt the header.

## Why the annotated clip is rendered in a second pass

Event logic emits its headline events from `finalize()`, not per frame, and it
has to: whether a ball leaving a player's foot was a pass or a shot depends on
where it ended up, which is only known later. So drawing while analysing would
silently omit exactly the events worth watching -- the passes, the shots, the
goals -- and leave an annotated clip showing only the cheap ones.

So the run analyses first, lets `finalize()` complete the timeline, and only
then decodes the clip a second time to draw. Decoding twice costs a fraction of
what running the models once costs, and per-frame boxes are held in memory in
between, which is small.

Each event is drawn on the frame nearest its timestamp, with half a frame of
tolerance so an event landing a hair past the final frame is still shown. An
event genuinely timed beyond the clip stays in the JSON and is logged as
undrawable, rather than disappearing without a word.

## Why the video reader is not a `cap.read()` loop

Phone footage is the normal case here, and phone footage lies. A clip recorded
at "30fps" routinely delivers frames 28ms apart in good light and 45ms apart in
bad, because the container's frame rate is an average, not a rate. Deriving an
event's time as `index / fps` on such a clip drifts -- and a drifting timestamp
is a wrong answer to the only question this system is being asked.

So `VideoReader` prefers the container's own presentation timestamp for every
frame and falls back to the nominal rate only when that value is missing,
stalled or moving backwards, recording per frame which of the two it used
(`VideoFrame.timestamp_is_exact`, and `stats.inexact_timestamps` in the
output). It also watches the gaps between presentation times and flags the clip
as variable frame rate when they disagree with the nominal rate, so nothing
downstream trusts `fps` for arithmetic.

Timestamps stay anchored to the source clip throughout. Analysing a segment
with `--start 30` reports events at their real time in the original footage,
not rebased to the segment, and `--stride` skips frames without changing any
frame's time.

## Layout

```
football_analysis/
  types.py        BBox, Point, VideoFrame, Detection, Track
  events.py       the output schema: EventType, Event, EventTimeline
  interfaces.py   the contracts a detector, tracker and event logic satisfy
  pipeline.py     the loop that joins the stages
  state.py        the per-frame world-state record, its cache and its reader
  stubs.py        placeholder stages, so a run works before the real ones land
  adapters.py     seams between this contract and a stage with its own vocabulary
  config/         one object describing a whole run, and its loader
  io/             reading clips with trustworthy timestamps; writing annotated ones
  viz/            boxes, ball trail, event captions
  cli/            the `football-analyse` entry point
  detection/      the YOLO stage
  track/          identities across frames; geometry/ the camera pose
  pose/           COCO-17 keypoints per player, on the settled boxes
  ball_events/    possession, pass, shot, goal, over the record
  body_events/    tackles and tricks, over pose
  evaluation/     the accuracy scorecard against hand labels
tests/            the suite below
scripts/          make_sample_clip.py
configs/          a worked example config
```

All bounding boxes live in the coordinate space of `VideoFrame.image` -- after
any resizing the reader applied. `VideoFrame.scale` maps back to source pixels.
One space everywhere removes a class of off-by-a-resize bugs.

## Implementing a stage

A stage does not import a base class; it just needs the right methods
(`interfaces.py` defines them as `Protocol`s, so conformance is structural).

```python
class YoloDetector:
    def detect(self, frame: VideoFrame) -> list[Detection]: ...
    def reset(self) -> None: ...
    def describe(self) -> dict: ...       # goes in the run's provenance record

class MyTracker:
    def update(self, frame: VideoFrame, detections: list[Detection]) -> list[Track]: ...

class MyEventLogic:
    def update(self, state: FrameState) -> list[Event]: ...
    def finalize(self) -> list[Event]: ...  # events only confirmable in hindsight
```

Then inject them:

```python
from football_analysis.pipeline import AnalysisPipeline
from football_analysis.config import load_config

pipeline = AnalysisPipeline(
    load_config("configs/default.yaml"),
    detector=YoloDetector(...), tracker=MyTracker(), event_detector=MyEventLogic(),
)
result = pipeline.run("match.mp4")
```

Four rules the pipeline relies on:

- **Boxes are in `frame.image` space**, not source pixels.
- **A tracker returns snapshots**, not objects it will keep mutating -- event
  logic routinely holds a track to compare against a later frame.
- **A detector returns `[]` for a frame with nothing in it**, rather than raising.
- **Event logic reads the record, never the video** -- see below.

`tests/test_interfaces.py` is the executable version of this contract: an
implementation that passes it drops in unchanged.

### Stages that already have their own vocabulary

A stage does not have to be rewritten to fit. `adapters.py` holds the seam:
`FrameDetectorAdapter` wraps a detector that takes raw pixels and returns its
own record type, and presents it as a `Detector`. It converts boxes, clamps
them to the frame, drops degenerate ones rather than letting a zero-area box
reach code that will divide by it, and can rename or filter classes.

That is how the YOLO stage in `football_analysis/detection/` is wired in --
it speaks `label` / `confidence` / `xyxy` and takes an `ndarray`, and neither
side had to change:

```python
from football_analysis.adapters import adapt_detector
from football_analysis.detection import Detector, DetectorConfig

raw = Detector(DetectorConfig(weights="assets/models/hayati_football.pt", imgsz=1280))
detector = adapt_detector(raw, label_map={"goalkeeper": "player"}, keep=["player", "ball"])
```

`adapt_detector` returns a detector that already speaks this contract
untouched, so it is safe to call on anything.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

`python -m pytest` rather than a bare `pytest`, so the tests run in the same
Python the package was installed into.

The suite needs no network and no model weights. The foundation's own layers
(io, config, viz, cli, state, pipeline, adapters) carry 224 of the tests; the
detection, tracking, event and scorecard modules add their own alongside them,
for 373 in total.

Two environment snags worth knowing:

- `Intel oneMKL FATAL ERROR: Cannot load libtorch_cpu.so` is scipy and torch
  racing over the MKL threading layer, not your clip. Set
  `MKL_THREADING_LAYER=GNU` and run it again.
- On a network-mounted working copy, reading a cached `.pyc` can hang
  indefinitely and the run simply stops with no error. Set
  `PYTHONDONTWRITEBYTECODE=1`, or work from a local copy. They render a real video
file and read it back, so the reader, the writer, the overlay and the CLI are
tested against actual decoding rather than mocks. The container behaviour that
cannot be provoked from a file OpenCV wrote -- a missing presentation time, a
stalled one, a nonsense frame rate -- is tested against the timing logic with a
stand-in capture.

The foundation has also been run for real, end to end: the YOLO stage from
`football_analysis/detection/` on `assets/clips/broadcast_08fd33_4.mp4` at
1280px, writing both a timeline and an annotated clip, at roughly 4.5 frames
per second on CPU. That clip is broadcast footage rather than the side-angle
1v1 the system targets, so it exercises the plumbing, not the 1v1 assumptions:
with `max_players: 2` the tracker holds two identities out of twenty-two, which
is the configured behaviour rather than a detection result.
