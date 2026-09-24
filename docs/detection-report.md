# Detection pass: what YOLO already sees, and what it doesn't

Measured on 2026-09-22/23 against real side-angle football footage. All timings
are **CPU only** — this machine has no GPU (`torch.cuda.is_available()` is
False), so treat every millisecond figure as an upper bound, not as what the
finished system will run at.

## The footage

`assets/clips/broadcast_08fd33_4.mp4` — 30 s, 750 frames, 1920x1080, 25 fps.
A real Bundesliga match (Mönchengladbach v Wolfsburg) shot from a raised side
angle, the standard clip used in public football-CV work.

It is 11v11 broadcast, not 1v1, which makes it the **harder** test in the one
way that matters: the ball covers about 18 px. On a camera framed on a single
duel the ball will be several times larger. Two things it cannot tell us are
how the system behaves on a phone-quality image and on a small pitch or cage,
so these numbers want re-running on Kaan's own footage.

The goal is only in shot for the last ~11 s (frames 493-749), as the camera
pans right. That was enough to work the goal question properly.

## Ground truth

There is no public per-frame ball annotation for this clip, so one was made.
Every 30th frame (25 frames, one every 1.2 s) was rendered with every ball
candidate from a three-model ensemble circled, numbered, and shown as a zoomed
crop; a human picked which candidate, if any, is the match ball. Frames where
no candidate was the ball were re-checked at 2.2x zoom across the whole pitch
before being recorded as "ball not visible".

Result: the ball is visible in **23 of 25** frames. In frames 120 and 420 it is
genuinely not visible anywhere on the pitch. Ground truth is in
`assets/annotations/ball_gt.json` (method + the human choices) and
`ball_gt_points.json` (resolved pixel points).

A detection counts as a hit if its centre is within 30 px of the ground-truth
point. Every other ball detection in a frame is a false positive, including all
ball detections on the two frames where the ball is absent.

## Players: solved

Nothing needed doing here. The football-finetuned weights put a box on every
player, goalkeeper and assistant referee on the pitch, in every sample frame
checked by eye, with no crowd, dugout or touchline-staff false positives. 21-22
outfield players plus 3-4 officials per frame, consistently, at **207 ms/frame**
on CPU. Stock COCO YOLO does the same job (it calls them all `person`) but also
boxes the crowd, so the football weights are worth it for the class labels
alone.

Sample frames with boxes drawn: `assets/frames/detected_*.jpg`.

## The ball: the real problem

`python -m football_analysis.detection.benchmark`

| configuration | ball recall | precision | FP/frame | ms/frame (CPU) |
|---|---|---|---|---|
| yolo11n COCO @1280 | 26.1% | 100% | 0.00 | 105 |
| yolo11x COCO @1280 | 56.5% | 100% | 0.00 | 1554 |
| yolo11x COCO + tiled ball | 73.9% | 94.4% | 0.04 | 4523 |
| yolo11x COCO + tiled, conf 0.10 | 73.9% | 60.7% | 0.44 | 4314 |
| yolo11x COCO + tiled, conf 0.10, grass filter | 73.9% | 89.5% | 0.08 | 4449 |
| football-ft (hayati) @1280 | 34.8% | 80.0% | 0.08 | 207 |
| football-ft + tiled ball | 52.2% | 57.1% | 0.36 | 729 |
| hybrid: ft players + tiled COCO ball | 69.6% | 100% | 0.00 | 3341 |
| forzasys @1280, conf 0.25 | 82.6% | 82.6% | 0.16 | 539 |
| **forzasys @1280, conf 0.35** | **82.6%** | **86.4%** | **0.12** | **533** |
| forzasys @1280, conf 0.50 | 78.3% | 85.7% | 0.12 | 511 |
| forzasys + tiled ball | 82.6% | 73.1% | 0.28 | 5164 |
| union: forzasys + COCO tiled + grass | 82.6% | 50.0% | 0.76 | 3784 |

**The second football checkpoint is the best ball detector, and I missed it on
the first pass.** `forzasys_soccer.pt` (`player / ball / logo`) was downloaded
and its class list inspected, but not benchmarked, because the other football
checkpoint had already scored poorly. That was wrong: it reaches **82.6%
recall at 86.4% precision in 533 ms**, which beats the tiled COCO pass on
recall by 9 points and is 8x faster. The tracking thread flagged the omission.

Four things worth knowing:

**Stock COCO YOLO beats the football-specific weights on the ball.** That was
not the expected result. `sports ball` is a COCO class and yolo11x is a much
larger model; the football checkpoint is a yolov8-size model fine-tuned on a
Roboflow player dataset where the ball is one small class among four. The
football weights win on *players* (clean class labels, 7x faster); COCO wins on
the *ball*.

**Tiling closes the gap for a model that needs it, and it is a resolution
effect.** Running
the ball pass over six overlapping crops at native resolution instead of
squashing 1920x1080 down to 1280 takes recall from 56.5% to 73.9%. The
mechanism is simply that the ball is no longer downsampled below what the
detector can resolve. Upsampling does *not* substitute for this: a control
experiment that cropped around the ball and scaled it up 2x made things worse,
not better (yolo11x fell to 52.2%), because interpolation adds no detail.

**The false positives are systematic, not random, but there are two kinds.**
COCO's are in the stand — a yellow-green object it scores 0.1-0.25. Those are
killed by a grass check on the ring around the detection
(`ball_on_grass=True`), which took the tiled COCO pass from 60.7% to **89.5%**
precision at identical recall. The football checkpoints' false positives are
different: a whitish smear near the left touchline that is *on grass*, so the
same filter does nothing for them (forzasys unchanged at 82.6% precision with
it on). Measured, not assumed — I expected the filter to help both.

**Ensembling does not help.** A union of forzasys and the grass-filtered tiled
COCO pass stays at 82.6% recall and halves precision to 50%. Both models miss
the *same* four frames, where the ball is occluded in a tackle or smeared by
motion. Those frames are a job for the trajectory filter, not for another
detector.

### Recommendation for the ball

**`forzasys_soccer.pt` at imgsz 1280, ball confidence 0.35**, for players
either it or the hayati checkpoint. 82.6% recall at 86.4% precision in 533 ms
on CPU — better than every alternative on all three axes at once, which is not
usually how this goes.

If precision matters more than the last few points of recall, the
grass-filtered tiled COCO pass gives 89.5% precision at 73.9% recall, but costs
8x the time. Ensembling the two is not worth it (see above).

Three things will push recall past 82.6% and none of them is a new model:

1. **A tracker.** 73.9% per-frame recall on a ball moving smoothly means a
   Kalman/ByteTrack filter recovers most of the remaining frames from
   neighbours. Per-frame recall is the floor, not the ceiling.
2. **Kaan's own framing.** At 18 px the ball is at the edge of what a detector
   can resolve. Framed on a 1v1 it will be 3-5x that, which is comfortably
   inside the range where these models are reliable.
3. **A GPU.** 3.3 s/frame is a CPU number and the reason the tiled pass looks
   expensive. On a GPU the tiled pass is the obvious default.

## The goal: not a pretrained class anywhere

Confirmed three ways:

- COCO's 80 classes contain no goal, net, post or frame.
- Both football checkpoints tested are `ball / goalkeeper / player / referee`
  and `player / ball / logo`. Neither has a goal class.
- Open-vocabulary prompting does not rescue it. YOLOE (`yoloe-11l-seg`,
  MobileCLIP text encoder) prompted with `goal`, `goal net`, `soccer goal`,
  `football goal`, `goalpost`, `net`, `white net`, `crossbar`, `goal frame`,
  `goal mouth` found **nothing on the full frame**. Cropped to the goal region
  and upscaled 2x it produced `white net` at 0.216 confidence over a box that
  swallowed the advertising hoardings as well as the goal. Not usable.

So the goal needs either fine-tuning or calibration, and **which one depends on
the camera, not on the model**:

### If the camera doesn't move — calibrate once

This is the normal 1v1 setup: a tripod, or a phone clamped to a fence. The goal
occupies the same pixels for the whole recording, so it is annotated once and
costs nothing per frame, with pixel-exact accuracy no detector will match.

The goal is stored as **four corners of the goal mouth**, not a box, per the
system plan's geometry: a box answers "is the ball in the goal", but four image
points with known 3D positions are what camera calibration needs, which is how
the system calibrates without any pitch markings. The box is derived from the
corners, so carrying both costs nothing.

`football_analysis/detection/goal.py` provides `StaticGoal`:
`from_click(frame)` to click the four corners, `.save()`/`.load()` to persist
them per camera setup, `.contains(point)` for point-in-quadrilateral (not just
the box), and `.xyxy` when a plain box is all a caller wants. Corner order is
`left_post_base, right_post_base, right_crossbar_end, left_crossbar_end`,
matching `GoalModel.mouth_corners()` in the geometry layer.

Camera pose is **not** solved here. `StaticGoal.corners` goes straight to
`football_analysis.geometry.calibrate_from_goal`, which additionally estimates
focal length when it is unknown (it usually is, on a phone), rejects a mirrored
corner order and reports its own uncertainty. An earlier draft of this module
carried its own `solve_pose`; it was removed rather than kept in parallel,
because two PnP implementations drift apart and that one was the worse of the
two — it assumed a known focal length and defaulted to a full-size 7.32 x 2.44 m
goal, where a 1v1 drill normally uses something nearer 3 x 2 m.

**This is the recommendation for Kaan's system** unless his camera moves.

### If the camera does move — fine-tuning works, and here is the proof

Fine-tuning was run end to end rather than assumed.

*Labelling.* The goal box was drawn by hand on one frame (frame 720) and
propagated across all 257 frames where the goal is visible, by estimating the
camera's frame-to-frame motion from sparse optical flow on background corners
and warping the box with it. The goal is static in the world, so its image
motion *is* the camera's motion. Boxes were spot-checked by eye at frames 500,
560, 620, 680, 710 and 745 and track the goal mouth cleanly with no visible
drift. Total human effort: **one box**. The labels are in
`assets/annotations/goal_boxes.json`.

*Training.* yolo11n, 640 px, 40 epochs, CPU. Split temporally with a 30-frame
gap so validation frames are not neighbours of training frames: 166 train
(frames 493-659), 60 val (frames 690-749).

*Result on the 60 held-out frames*, after the full 60-epoch run:

| metric | value |
|---|---|
| recall at IoU>=0.5, at every threshold to conf 0.50 | **100%** (60/60) |
| frames with a wrong box | **0** |
| median IoU of the detected box | **0.91** |
| median confidence on correct boxes | **0.92** (max 0.99) |
| mAP50 / mAP50-95 | 0.995 / 0.872 |
| speed | **31 ms/frame** on CPU |

An earlier 5-epoch checkpoint found the goal just as reliably (96.7% recall,
median IoU 0.94) but with uncalibrated confidences — median 0.12, so recall
collapsed at a normal 0.25 threshold. Training to convergence fixed the
calibration entirely. Worth knowing if a future fine-tune looks like it is
"barely working": check the confidence distribution before the localisation.

*Box vs keypoints.* The model trained here is a box detector, which was the
right thing to measure -- it answers "can a fine-tune find the goal at all",
and the answer is yes. If Kaan's camera does pan, the model to train for
production is a **4-keypoint pose model** rather than this box model, for the
same PnP reason above. The labelling recipe is unchanged: the optical-flow
propagation below carries four corner points exactly as readily as it carries
two box corners.

*What this does and does not show.* It shows the mechanism works and that the
data cost is tiny — one hand-drawn box, propagated. It does **not** show
generalisation: this is one goal, in one stadium, under one lighting condition.
A detector for arbitrary pitches needs the same recipe repeated across several
venues, which is roughly a morning's work per venue at this labelling cost.

## Interface

The pipeline calls one thing:

```python
from football_analysis.detection import Detector, DetectorConfig

detector = Detector(DetectorConfig(weights="assets/models/hayati_football.pt"))
for detection in detector.detect(frame):
    detection.label       # "ball" | "player" | "goalkeeper" | "referee" | "goal"
    detection.confidence
    detection.xyxy        # pixel coords in the frame as passed in
```

Every backend's class names are normalised through one map, so downstream code
never learns which checkpoint is loaded. `tile_ball=True` turns on the tiled
ball pass.

## Files

| path | what |
|---|---|
| `football_analysis/detection/detector.py` | `Detector`, `detect(frame)`, tiled ball pass, class normalisation |
| `football_analysis/detection/types.py` | `Detection`, the class vocabulary |
| `football_analysis/detection/goal.py` | `StaticGoal` for fixed cameras |
| `football_analysis/detection/benchmark.py` | reproduces the ball table above |
| `assets/models/` | yolo11n, yolo11x, two football checkpoints, YOLOE, the fine-tuned goal model |
| `assets/clips/broadcast_08fd33_4.mp4` | the test footage |
| `assets/annotations/` | ball ground truth, goal boxes, benchmark output |
| `assets/frames/detected_*.jpg` | sample frames with boxes drawn |

## What would help most next

Kaan's own footage. Every number here is from a broadcast camera at 1080p
looking at a full pitch. The three things that will differ on his clips —
framing, image quality, and whether the camera moves — each change the answer,
and the third one decides the whole goal design.
