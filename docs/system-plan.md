# 1v1 Football Analysis — System Plan

**Status:** proposed design, v1 · **Date:** 2026-09-22 · **Owner thread:** System plan and approach

**Published page:** https://claude.ai/artifact/MxzQbHFWTN5nTgg1wUh8Pq

The system takes a video of a 1v1 football drill (one attacker, one defender, one goal) shot
from a single fixed side-angled camera, and emits a timestamped record of **tackles, tricks,
passes, shots and goals**, having detected the ball, the players and the goal.

This document is the design the other threads build against. Every decision below is stated
with the reason and the evidence behind it.

---

## 0. The honest baseline

The closest published benchmark to what we are building is **SoccerNet Ball Action Spotting**:
12 classes including Pass, Shot, Goal and *Player Successful Tackle*, evaluated at **mAP@1**
(a 1-second tolerance). The 2024 challenge winner, T-DEED, scored **73.39 mAP@1** against a
**56.15** baseline — with 7 fully annotated games of training data and a heavy learned video
model behind it. ([SoccerNet BAS task](https://www.soccer-net.org/tasks/ball-action-spotting),
[SoccerNet 2024 results](https://arxiv.org/pdf/2409.10587))

So: the state of the art on broadcast football gets about three quarters of ball events right
at one-second precision. Any plan that promises "perfect" is lying.

But our problem is *not* broadcast football, and the differences run almost entirely in our
favour:

| Broadcast football | Our 1v1 clips |
| --- | --- |
| 22 players, constant occlusion | 2 players |
| Panning, zooming, cutting camera | one fixed camera, no cuts |
| Ball often 8–15 px | ball plausibly 20–40 px (camera is much closer) |
| Goal frequently off-screen | goal in frame, static, the whole time |
| Identity is a hard ReID problem | exactly two identities, permanently |

A fixed camera and two players is a qualitatively easier problem, and the plan below spends
that advantage deliberately: **geometry and physics do the work wherever the event is defined
by where the ball went, and a small learned model is used only where the event is defined by
how a body moved.**

---

## 1. Detection

### 1.1 Model: YOLO11 as the workhorse, YOLO26 A/B'd on the ball

**Decision.** Build on **Ultralytics YOLO11m** for players and the goal. Run a controlled
comparison of **YOLO26s/m** for the *ball model specifically* and adopt it there if it wins.
Keep the architecture behind one config key so a swap is a one-line change.

**Reason.** YOLO11 is the version the entire football-CV ecosystem is built on — the Roboflow
reference models, the public datasets, the published tutorials — so we can reproduce known-good
results instead of debugging our own novelty. YOLO26 (released January 2026) is the newer
family and its headline changes include **STAL**, which explicitly targets *small-object label
coverage*, plus DFL removal and NMS-free inference
([Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)). Small-object coverage
is precisely our bottleneck, so it deserves a measured trial on the ball — and nowhere else,
because everywhere else the ecosystem advantage outweighs a point of mAP.

**Licensing note.** Ultralytics is AGPL-3.0. Fine for a personal or open project; if this is
ever shipped closed-source, a commercial licence is required. Worth knowing now, not later.

### 1.2 Classes, and what pretrained weights actually give us

Three classes: **`player`**, **`ball`**, **`goal`**. No referee or goalkeeper class — a 1v1
drill has neither, and an unused class only costs us capacity and confusion.

What COCO pretrained weights cover:

| Class | In COCO? | Usable zero-shot? | Verdict |
| --- | --- | --- | --- |
| `player` | yes — `person` (class 0) | yes, for a day-one smoke test | **fine-tune anyway**: COCO persons are upright and photographic; ours are sliding, lunging, motion-blurred, and shot from a low side angle |
| `ball` | yes — `sports ball` (class 32) | partly — 56.5% recall unaided | **Don't reason about it, benchmark it.** COCO is usable and beats one football checkpoint; another football checkpoint beats COCO. See the measured table below. |
| `goal` | **no class covers it** | no | A goal frame + net is not a COCO concept. **Fine-tuning required, from scratch.** |

So: fine-tuning is required, and the ball and goal are the reason.

**Measured — and this section has now been revised twice by benchmarking, which is itself the
finding.** The detection thread tested candidates against 25 hand-labelled frames of real
side-angle footage with the ball at ~18 px (`docs/detection-report.md`):

| Configuration | Ball recall | Precision | ms/frame |
| --- | --- | --- | --- |
| `yolo11x` COCO @1280 | 56.5% | 100% | 1554 |
| `yolo11x` COCO + tiled ball | 73.9% | 94.4% | 4523 |
| `yolo11x` COCO + tiled + grass filter | 73.9% | 89.5% | 4449 |
| hayati football checkpoint @1280 | 34.8% | 80.0% | 207 |
| **`forzasys_soccer.pt` @1280, conf 0.35** | **82.6%** | **86.4%** | **533** |
| union: forzasys + tiled COCO + grass | 82.6% | 50.0% | 3784 |

**Use `forzasys_soccer.pt` at `imgsz=1280`, ball confidence 0.35, for the ball.** It beats the
tiled COCO pass by 9 points of recall at roughly an eighth of the cost.

The path here matters more than the winner. This plan first assumed COCO's `sports ball` was
unusable; a benchmark showed COCO beating a football checkpoint by 22 points and the plan flipped
to "initialise from COCO"; then a *second* football checkpoint, downloaded but left unrun because
the first had scored badly, turned out to beat everything. **The lesson is not "COCO wins" or
"football weights win" — it is that ball-detector performance is not predictable from the training
corpus, so benchmark every candidate you have on your own frames before reasoning about it.** One
unrun checkpoint cost two revisions of this document.

Note that forzasys wins on *recall* while giving up precision to the hybrid's 100%. That is the
right trade and it follows the principle in mitigation 6 below rather than contradicting it: a
trajectory filter rejects a physically impossible detection easily, and cannot invent a ball that
was never detected.

Three further findings, each narrower than it first appeared:

- **Tiling is a resolution fix, and only helps models that need it.** It takes COCO from 56.5% to
  73.9%, and does *nothing* for forzasys (82.6% either way) while costing it precision — a model
  already trained on small footballs doesn't need the extra pixels. A control that cropped and
  upsampled 2× made things *worse* (56.5% → 52.2%), because interpolation adds no detail. So tile
  only where a benchmark shows it helps, and never upsample.
- **Pitch masking is conditional, not general.** Rejecting ball candidates that aren't on grass
  took the tiled COCO pass from 60.7% to 89.5% precision at identical recall — but does nothing for
  forzasys, whose false positives are a whitish smear *on* the grass near the touchline. It fixes
  one kind of false positive, not all of them.
- **Ensembling is a dead end.** Unioning forzasys with the grass-filtered tiled COCO pass holds
  recall at 82.6% and halves precision to 50%, because both models miss *the same* frames — the
  ball occluded in a tackle, or motion-smeared. Those frames are the trajectory filter's job, not a
  second detector's.

**18 px is the edge of what any detector resolves**, and that is broadcast framing. In Kaan's own
1v1 framing the ball will be 3–5× that, comfortably inside the reliable range — the plan's central
bet, now with a measurement behind it rather than an estimate.

**Starting data — and a hard constraint on how to get it.** This environment's network allowlist
covers `raw.githubusercontent.com`, GitHub release assets and PyPI. **Roboflow, SoccerNet, Kaggle,
HuggingFace and YouTube are all unreachable** — they fail at CONNECT with a proxy 403, verified
from two threads.

The GitHub **API** is a different case and the distinction matters: `api.github.com` is reachable
but **gated per session, not blocked**. `/zen` and `/rate_limit` return *"This GitHub API path is
not available in agent sessions"*, and a repo path returns *"GitHub access to … is not enabled for
this session. Use add_repo to request access."* So the API is **not** a way around the dataset
problem — you cannot browse arbitrary public repos looking for datasets or committed weights.

What works: `raw.githubusercontent.com` with a path you already know, GitHub release asset URLs,
or `add_repo` for a repository you specifically need attached. The bootstrap route is therefore
checkpoints already fine-tuned on those datasets and committed as files in public GitHub repos,
fetched by known path. That is how the working player/ball model was obtained. Worth stating
plainly, because it costs about an hour to rediscover.

Do not label from zero. Start from
[Roboflow `football-players-detection-3zvbc`](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
(classes ball / player / referee / goalkeeper, 372 images, reported mAP@50 83.0%, precision
72.1%, recall 88.0%) — noting that an 83% aggregate is carried by the player class and the ball
class will sit far below it. Also pull the ball and pitch datasets used by
[roboflow/sports](https://github.com/roboflow/sports), which derive from the DFL Bundesliga Data
Shootout Kaggle competition. Use these to bootstrap, then fine-tune on frames from Kaan's own
camera — the fixed viewpoint means a few hundred frames from the real rig will beat thousands of
broadcast frames.

### 1.3 The ball is the hard part — six things we do about it

The futsal literature names the three failure modes directly: player–ball **occlusion** (worse in
small-court football than on a full pitch), **rapid unpredictable motion**, and **appearance
change** from motion blur ([Ball and Player Detection in Futsal Videos Using YOLOv8](https://ceur-ws.org/Vol-3885/paper28.pdf)).
The mitigations, in the order we apply them (the last two added from measurement):

1. **A separate ball model.** Not a fourth class on the player model — its own weights, its own
   image size, its own confidence threshold, its own augmentation recipe. This is what the
   Roboflow reference pipeline does: `football-ball-detection.pt` is a distinct checkpoint from
   `football-player-detection.pt`, players run at `imgsz=1280` on the full frame while the ball
   runs on slices ([roboflow/sports `examples/soccer/main.py`](https://github.com/roboflow/sports/blob/main/examples/soccer/main.py)).

2. **Sliced inference — where a benchmark shows it helps.** `supervision.InferenceSlicer` with
   640×640 tiles and NMS at 0.1, as the reference pipeline does, makes the ball occupy a far larger
   fraction of the network input at roughly 4–6× the cost. It is worth that cost for a model whose
   recall is resolution-limited (COCO: 56.5% → 73.9%) and worth nothing for one that isn't
   (forzasys: unchanged, and precision drops). Measure before paying for it.

3. **ROI-tracked crops for the fast pass.** Once the ball has been located, crop a window around
   the Kalman-predicted position and run full resolution there only. There is exactly one ball,
   so this is dramatically cheaper than full tiling and nearly as accurate. Fall back to a full
   sliced sweep whenever the track is lost.

4. **Train for blur.** High `imgsz` (1280), mosaic augmentation off or very low (the Roboflow
   calibration work found mosaic actively *worsened* results on this kind of task —
   [Roboflow: Camera Calibration in Sports with Keypoints](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)),
   and heavy motion-blur / downscale augmentation so the model sees the ball as it actually
   appears mid-flight.

5. **Exploit the fixed camera.** The background is static, so frame differencing or a running
   background model proposes candidate moving blobs almost free. Use these as a *prior* to
   raise ball-region confidence and to recover frames the detector misses outright. This signal
   simply does not exist for broadcast footage, and it is one of the best things our rig gives us.

6. **Invert the precision/recall split — but name the checkpoint.** Run the ball detector for
   recall and get precision back from the trajectory filter in §2.2 rather than from the detector:
   a physically impossible detection is trivially rejected by a motion model, while a ball that was
   never detected is gone forever. The threshold is **not** checkpoint-independent, though. On the
   COCO model, conf 0.25 already gives 73.9% recall at 94.4% precision. On the football checkpoint,
   dropping to conf 0.10 buys the same 73.9% recall but collapses precision to 51.5%. So pair the
   threshold with the weights, and don't carry a number across a model swap.

7. **Mask to the pitch — for off-pitch false positives only.** Rejecting ball candidates that
   aren't on grass took the tiled COCO pass from 60.7% to 89.5% precision at identical recall, which
   makes it the cheapest accuracy available *when the false positives are off the pitch*. It does
   nothing for forzasys, whose false positives sit on the grass. Conditional, not general.

---

## 2. Tracking

### 2.1 Players: ByteTrack plus a hard two-identity lock

**Decision.** **ByteTrack** for frame-to-frame association, followed by a **global two-identity
lock**: cluster player crops into exactly two appearance clusters for the whole clip and snap
every track to one of them each frame.

**Reason.** Ultralytics' own guidance is that ByteTrack suits "static or near-static cameras
where detector cost dominates", while BoT-SORT's extra machinery — camera motion compensation
and appearance ReID — earns its cost on *moving* cameras and in crowds
([Ultralytics tracking docs](https://docs.ultralytics.com/modes/track/)). Our camera is fixed
and our crowd is two people, so BoT-SORT's advantages are largely moot.

More importantly, 1v1 lets us do something broadcast systems cannot: we know a priori that the
number of identities is **exactly two, for the entire clip**. That converts identity from a
frame-to-frame association problem into a two-way classification problem with a global
constraint. Cluster crop embeddings (SigLIP features → UMAP → 2-means is the approach
roboflow/sports uses for team assignment; for two differently-coloured bibs an HSV histogram is
sufficient and far cheaper) and assign every detection to cluster A or B. **ID switches should
be zero**, and we should measure that rather than assume it.

**Fallback.** If the two players wear similar colours, appearance clustering degrades. Then fall
back to BoT-SORT with `with_reid` enabled, and surface a warning — because every downstream
event depends on knowing who is who, and a silent identity swap corrupts the whole record.

### 2.2 The ball: a physics-gated Kalman filter, smoothed both ways

The reference implementation's `BallTracker` keeps a 20-frame buffer and picks the detection
closest to the recent centroid ([`sports/common/ball.py`](https://github.com/roboflow/sports/blob/main/sports/common/ball.py)).
That is a decent outlier rejector and a poor motion model — it cannot predict, so it cannot
survive an occlusion.

**Decision.** A **constant-acceleration Kalman filter in image space with gravity as a known
bias on the vertical axis**, association gated by Mahalanobis distance, plus an offline
**forward–backward (RTS) smoothing pass**.

**Reason.** The side angle is what makes this work. In a side view the image's vertical axis is
close to world vertical, so a ball in flight traces a near-parabola *in pixel coordinates* —
gravity becomes a constant, known term in the state model rather than an unobservable. That is a
strong prior for free, and it is specifically a property of the side-angled camera the project
already specifies.

**Surviving occlusion.** When no detection passes the gate, coast on the prediction for up to
~0.4 s (12 frames at 30 fps), marking every coasted frame with an `interpolated` flag so no
downstream rule mistakes a guess for an observation. Because we analyse recorded clips rather
than a live feed, we can also **use the future**: the backward smoothing pass reconstructs the
trajectory through an occlusion from the frames on *both* sides of it, which is far more reliable
than forward-only prediction. Offline is an advantage and we should spend it.

**Occlusions are signal, not just noise.** A ball that disappears at a player's feet and
re-emerges with a changed velocity vector is not a tracking failure — that is very often exactly
where a tackle or a trick happened. The tracker records every gap with its duration, its location
and the nearest player, and the event layer reads those gaps as evidence.

---

## 3. Geometry: what the side angle actually costs us

**The question:** is a homography to pitch coordinates worth it for 1v1, and how do we get one
when there may be no pitch line markings?

**Decision.** **No full pitch homography. Yes to a lightweight ground-plane and scale
calibration, obtained from the goal.**

**Why not a pitch homography.** The standard approach detects ~32 characteristic pitch landmarks
with a pose model and solves `cv2.findHomography` against their known positions
([Roboflow](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)). Every one of
those landmarks is a full-pitch feature — centre circle, penalty box, corner arcs. A 1v1 drill
area has none of them, and may have no markings at all. The method is not merely inconvenient
here; its inputs do not exist.

And we do not need it. Every event we care about is **relational**: is the ball at this player's
feet, did it move toward the goal, did it enter the net, did these two bodies converge. Relations
survive in image space.

**Why we still need *some* metric grounding.** Two things break in raw pixels. Ball speed — which
separates a pass from a shot — is not comparable between near and far, and neither is player
separation, which gates contact events. Under a side angle, a metre near the camera is many more
pixels than a metre far from it.

**The goal is the ruler.** A goal has known, fixed dimensions (typically 3 × 2 m for a 1v1 or
futsal goal; 7.32 × 2.44 m for a full-size one). So we train the goal not as a bounding box but
as a **4-keypoint object** — the four corners of the goal mouth — and solve PnP against those
four known 3D points to recover the full camera pose. Four known coplanar points and an assumed
or calibrated focal length is all PnP needs, and it requires **no pitch markings whatsoever**.
The goal is already in frame, already static, and already something we have to detect.

**Validated.** The detection thread trained a goal *box* detector purely to settle whether a
fine-tune can find a goal at all. Fully trained: **100% recall on held-out frames at every
threshold up to conf 0.5, zero wrong boxes, median confidence 0.92, median IoU 0.91, 31 ms/frame** —
from labels propagated off a single hand-drawn box by optical flow. (An earlier 96.7% figure was a
5-epoch checkpoint with uncalibrated confidence.) That propagation carries four
corner points as readily as two, so the labelling cost for the keypoint model we actually want is
the same. `StaticGoal` now stores four goal-mouth corners in fixed order, with `.solve_pose(K)`
doing PnP against a 7.32 × 2.44 m goal (overridable for a smaller training goal), `.contains()` as
point-in-quadrilateral, and `.xyxy` derived.

Two fallbacks, in order:

- **Player height as a continuous scale.** Ask once for the two players' heights and use each
  detection's foot-point and head-point to estimate the ground plane and the local scale. Works
  everywhere in frame, self-corrects each frame, and degrades gracefully.
- **A one-time manual rectangle.** Place four cones in a measured rectangle before filming (or
  click four points once on the first frame). Ten seconds of setup for the most accurate answer
  available. For someone filming their own drills this is entirely reasonable, and where accuracy
  matters most it should be the recommended default.

The output of this module is not a top-down pitch map. It is a **ground plane, a scale function,
and a goal polygon in 3D** — the three things the event rules actually ask for.

---

## 4. Events: what actually separates one from another

This is the core of the system, and it needs a clear architectural split.

**Decision.** A **hybrid**, split by what defines the event:

- **Tier 1 — geometry and physics on tracks:** possession, **pass**, **shot**, **goal**.
  These are defined by where the ball went relative to known geometry. A rule-based state machine
  is interpretable, needs zero training labels, and can be debugged from a single clip.
- **Tier 2 — a small learned temporal model over pose features:** **tackle**, **trick**.
  These are defined by *how a body moved*, which geometry alone cannot express.

**Why not learn everything.** Because the best learned system in the field scores 73.39 mAP@1
with seven annotated games behind it, and we have none. **Why not rule everything.** Because a
lunging tackle and a shoulder-to-shoulder duel can produce near-identical ball geometry, and the
difference lives entirely in the legs.

**Why pose features rather than raw video for Tier 2.** A raw-video classifier needs tens of
thousands of labelled frames. Two players' pose keypoints plus the ball over a ~1.5 s window is a
low-dimensional, highly structured feature vector, and a small 1D-CNN or GRU over it can learn
from a few hundred labelled examples. Pose-based dribble analysis is an established line of work
([Fine-Grained Visual Dribbling Style Analysis, CVPR-W 2019](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Li_Fine-Grained_Visual_Dribbling_Style_Analysis_for_Soccer_Videos_With_Augmented_CVPRW_2019_paper.pdf);
[From Pixels to Play: Dribble and Tackle Detection in Football, UiO 2025](https://home.simula.no/~paalh/students/2025-UiO-EirikEggset.pdf)).

### 4.0 Possession — the substrate everything else sits on

Player *p* possesses the ball at frame *t* when the ball centre lies within a scale-corrected
distance of *p*'s foot region (the bottom ~25% of the box), sustained over *k* consecutive
frames. This yields a single signal, `possessor(t) ∈ {A, B, none}`, and almost every rule below
is a statement about how that signal changes.

### 4.1 Pass vs shot

Both are "the ball leaves a player at speed". Three discriminators, in order of weight:

1. **Terminal state — decisive, and available only because we are offline.** Follow the ball
   forward. Ends at another player's feet → pass. Ends in the net, off the frame, or rebounding
   off the goal → shot.
2. **Direction relative to the goal.** The angle between the ball's release velocity and the
   vector from the release point to the goal centre. Necessary but *not* sufficient — an attacker
   can legitimately play the ball goalward without shooting.
3. **Release speed and elevation.** Shots are faster and more often lofted. A tiebreaker, never
   the primary test.

**A scope problem worth raising now.** In a strict 1v1 — one attacker, one defender — *there is
no teammate*, so a pass in the ordinary sense barely exists. Three readings are possible: a pass
to a feeder/server who is also in shot, a rebound off a wall, or a deliberate knock past the
defender to re-collect. These are genuinely different events.

**Default taken so the work isn't blocked:** support an optional third `feeder` role, so real
passes are detected whenever a third player is present; and in a genuine two-player clip classify
a deliberate forward knock beyond the defender as **`push_past`** rather than mislabelling it a
pass. Kaan can redirect this once he says which his clips contain — see §7.

### 4.2 Tackle vs shoulder bump

Both are two bodies converging. The difference is **what happens to the ball, and what the
defender's legs do**.

| | Tackle | Shoulder bump / duel |
| --- | --- | --- |
| Possession across contact | **changes or is destroyed** (A→B, or A→none) | **retained** by the same player |
| Ball velocity | **deflected** — clear direction/speed discontinuity | continuous |
| Defender pose signature | **leg extension**: ankle accelerates toward the ball, ankle–hip distance extends, ankle drops low or goes wide — a lunge | **torso-to-torso**: shoulders converge, ankles do *not* extend toward the ball |
| Point of nearest approach | defender's **ankle** to ball | the two **torsos** to each other |

The primary rule is therefore: *contact + loss or change of ball control + ball velocity
deflection = tackle; contact + retained control = shoulder duel.*

Pose adds what the possession rule alone would miss — the **failed tackle**, where the defender
lunges and the attacker survives it. That is a real event a player wants logged, and it is
invisible to any rule that only watches possession.

Feature vector over a ~1 s window around the contact: minimum inter-player distance (ground-plane
corrected), possessor before/after, ball speed and heading change, defender ankle velocity
projected onto the ball direction, minimum defender-ankle-to-ball distance, and body-orientation
change. Labels: `tackle_won`, `tackle_failed`, `shoulder_duel`, `no_contact`.

### 4.3 Trick vs ordinary dribble

The hardest of the five, and we should say so plainly rather than over-promise.

An ordinary dribble is *repeated small touches, ball staying ahead of the player, roughly constant
travel direction, regular touch cadence*. A trick breaks one of those in a specific way:

- **Feet move a lot, the ball barely moves.** This is the single most general signature. A
  stepover is the feet circling a nearly stationary ball. The feature is **foot-keypoint path
  length per unit of ball displacement** over the window; a trick has a high ratio, a dribble a
  low one. This one feature catches stepovers, roulettes and most feints without naming any of
  them.
- **Sharp ball direction reversal preceded by a body feint** — the drag-back and the elastico.
  Feature: ball heading change magnitude, plus a mismatch between body orientation and actual
  travel direction in the frames just before.
- **The nutmeg is purely geometric, and we should implement it as an explicit rule.** The ball
  track passes between the defender's two ankle keypoints while the defender's body is between
  the attacker and the ball. That is unambiguous, needs no classifier, and it is the trick people
  most want logged.

**Honest sequencing.** A general named-trick taxonomy (stepover / nutmeg / drag-back / roulette /
elastico) needs roughly 50–100 labelled instances *per class*. So ship in two steps: first a
binary **`notable_skill_move`** detector driven by the foot-motion-per-ball-displacement ratio and
the reversal feature, plus the explicit nutmeg rule; then add named classes as labels accumulate.
A flagged "something happened at 4:17" the user can name themselves is more useful than a
confident wrong label.

### 4.4 Goal — confirmed against the net, not guessed

**Say this plainly:** professional goal-line technology uses 7–14 synchronised high-speed cameras
(Hawk-Eye and equivalents) to adjudicate whether a ball wholly crossed a line. **A single side
camera cannot do that**, and no amount of modelling will change it. What a single fixed side
camera *can* do reliably for a 1v1 goal is confirm that the ball went **into the net**, which for
this project's purposes is the question that actually matters.

Four conditions, all required:

1. **Entry.** The ball track crosses *into* the goal-mouth polygon (from §3's 4-keypoint goal
   detection) through its front face.
2. **Absorption.** The ball's speed drops sharply, or its direction reverses, within the goal
   depth — the net stopping the ball is a distinctive kinematic event.
3. **No re-emergence.** The ball does not come back out of the front face within ~0.5 s. This is
   what separates a goal from a shot that strikes the post or crossbar and rebounds into play.
4. **Net motion.** The net visibly ripples when it takes a ball. With a *fixed* camera, a
   frame-difference energy spike inside the net region is a cheap, robust confirmatory signal —
   and one that almost nobody exploits, because it is only available when the camera doesn't move.
   Ours doesn't.

Every goal is emitted with a confidence score and the conditions that fired. **Anything ambiguous
— a post hit, a ball crossing near the line, a goal during a ball-track gap — is flagged for human
review rather than silently decided.** For a record a player will actually trust, a flagged
maybe is worth more than a confident wrong.

---

## 5. Modules, and the contract between them

Package `football_analysis`, under `/mnt/project-files/football-analysis`:

```
football_analysis/
  state.py     FrameState — THE INTERFACE. Implemented, with JSONL caching.
  pipeline.py  stage orchestration
  adapters.py  detector seam, so the pipeline takes no hard model dependency
  interfaces.py / types.py / config/   shared protocols, types, config schema

  io/          video reader/writer, frame indexing, frame->timestamp mapping
  detection/   player, ball and goal detectors; sliced + ROI ball inference
  track/       player tracker + two-identity lock; ball Kalman + smoother
  geometry/    goal-corner PnP, ground plane, scale function, image<->world
  pose/        pose estimation on player crops                      [not built yet]

  events.py    EventType, Event, EventTimeline — the vocabulary
  ball_events/ Tier 1: possession state machine, pass / shot / goal rules
  body_events/ Tier 2: tackle / trick classifier over pose features [not built yet]

  eval/        metrics, label format, ground-truth comparison       [not built yet]
  viz/         annotated overlay video, event timeline
  cli/         run the pipeline -> timestamps JSON + overlay video
```


**The one architectural decision that matters most: the world-state record.** *(Built — see `state.py`.)*

Everything upstream of events writes a single per-frame record — frame index, timestamp, player
boxes with stable ids and pose keypoints, ball position with confidence and an `interpolated`
flag, the goal quad, and the current possessor. Every event module reads *only* that record,
never the video.

This is what lets several threads work at once. Detection and tracking threads produce the
record; event threads consume it. The schema is the interface, it should be frozen early, and
event logic can then be developed and re-run against a cached record in seconds instead of
re-running inference every time.

**Status: done.** `state.py` carries `FrameState` with `PlayerState`, `BallState` (including the
`interpolated` and `frames_since_seen` flags), `GoalState` and the possessor. `EventDetector.update()`
takes a `FrameState` and is no longer handed the frame at all, so event rules *structurally cannot*
reach the video. Caching runs through `--state-cache state.jsonl` / `--replay state.jsonl`, as JSON
Lines behind a schema-version header, so an interrupted run still leaves a readable file and a
stale-schema cache is refused rather than misread. Measured on 100 frames: 26 s of inference versus
0.003 s to replay.

## 6. Build order

The sequence matters, and one step is deliberately earlier than instinct suggests.

1. **Video IO and the world-state schema.** Freeze the interface first; everything else depends
   on it.
2. **Detection** — player, ball and goal models fine-tuned and evaluated per class.
3. **The labelling tool and the evaluation harness.** *Before any event logic.* You cannot tune a
   threshold you cannot measure, and every rule in §4 has thresholds. Building eval third rather
   than last is the difference between tuning and guessing.
4. **Ball trajectory filter and the two-identity lock** — the quality ceiling for everything
   downstream.
5. **Geometry and calibration.**
6. **Possession state machine, then the pass / shot / goal rules** (Tier 1). First end-to-end
   timestamps come out here.
7. **Pose, then the tackle / trick classifier** (Tier 2).
8. **CLI and overlay video** — the overlay is also the fastest debugging tool we will have.

## 7. How we know it's right

**Detection metrics.** mAP@50 and mAP@50-95 per class. Report the **ball separately and
prominently** — it is the bottleneck, and an aggregate mAP dominated by the easy `player` class
hides exactly the number we need to watch. Also report **ball presence rate**: the percentage of
*in-play* frames with a valid ball position, before and after interpolation.

**Tracking metrics.** ID switches (target: **zero**; with two players, anything else is a bug),
and ball track continuity — number and duration of gaps.

**Event metrics.** Precision / recall / F1 per event class at a temporal tolerance. Use
**mAP@1s**, SoccerNet's convention, so our numbers are directly comparable to published work, and
additionally report a tighter **±0.5 s** for goals and shots where precision is cheap.

**The test clip.** 3–5 minutes of Kaan's own footage, exhaustively labelled with frame-accurate
event timestamps, and held out completely from every training and tuning step. This is the single
most valuable input Kaan can supply, and the project's accuracy claims are worth exactly as much
as this clip's coverage.

**What good looks like — v1 targets:**

| Signal | Target | Why this number |
| --- | --- | --- |
| Ball present, in-play frames | ≥ 95% after interpolation | everything downstream degrades below this; per-frame detector recall is the *floor*, not the ceiling — 82.6% per frame on a smoothly moving ball leaves most of the gap recoverable by the tracker |
| Player ID switches | 0 | two known identities; anything else is a defect |
| **Goals** | ≥ 0.95 P / ≥ 0.95 R | few, unambiguous, four independent confirming conditions |
| **Shots** | ≥ 0.85 F1 | clear kinematic signature, occasionally confusable with a hard pass |
| **Tackles** | ≥ 0.75 F1 | needs pose; comparable to the SoccerNet class |
| **Passes** | ≥ 0.80 F1 | conditional on §4.1 being resolved |
| **Tricks** (binary `notable_skill_move`) | ≥ 0.60 F1 | hardest class; named sub-classes deferred |

These are deliberately *above* SoccerNet's 73.39 mAP@1 for the ball-geometry events and *below*
it for the body-motion ones. That asymmetry is the plan's central bet: a fixed camera and two
players makes ball geometry much easier and makes reading a body no easier at all.

**Traceability.** Every emitted event carries its confidence and the evidence that triggered it —
which conditions fired, which frames, whether the ball track was interpolated through the event.
A wrong call should be traceable to a specific threshold or a specific bad frame, not to "the
model".

## 8. Stack

Confirmed as proposed by the other threads: **Python**, **Ultralytics YOLO**, **OpenCV**, package
name **`football_analysis`**. No changes requested.

Additions recommended: **`scipy`/`filterpy`** for the Kalman and RTS smoother, and **`numpy`**
throughout.

On **`supervision`** (Roboflow's), narrowed from a blanket recommendation: take
`InferenceSlicer` for sliced ball detection and the ByteTrack wrapper, which are where it earns
its keep. Do *not* take its annotators — the `viz/` layer is already built and tested, so they
would be a rewrite for no gain. The detector seam is an adapter (`adapters.py`) rather than a
hard dependency, which keeps this a per-module choice instead of a package-wide one.

## 9. Open questions for Kaan

Neither of these blocks the work — defaults are taken above — but both change the result:

1. **What is a "pass" in your clips?** Is there a third player feeding the ball, a rebound wall,
   or is it strictly one attacker and one defender? Default taken: support an optional feeder
   role, and otherwise log a deliberate knock past the defender as `push_past` (§4.1).
2. **Can you supply 3–5 minutes of representative footage, plus the goal's width and height?**
   The footage becomes the held-out test clip; the goal dimensions are what make the PnP
   calibration in §3 exact rather than approximate.

---

## Sources

- [SoccerNet — Ball Action Spotting task](https://www.soccer-net.org/tasks/ball-action-spotting)
- [SoccerNet 2024 Challenges Results (arXiv 2409.10587)](https://arxiv.org/pdf/2409.10587)
- [SoccerNet 2025 Challenges Results (arXiv 2508.19182)](https://arxiv.org/abs/2508.19182)
- [T-DEED (CVPR-W 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Xarles_T-DEED_Temporal-Discriminability_Enhancer_Encoder-Decoder_for_Precise_Event_Spotting_in_Sports_CVPRW_2024_paper.pdf)
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/)
- [roboflow/sports](https://github.com/roboflow/sports) · [soccer example](https://github.com/roboflow/sports/blob/main/examples/soccer/main.py) · [`common/ball.py`](https://github.com/roboflow/sports/blob/main/sports/common/ball.py)
- [Roboflow — Camera Calibration in Sports with Keypoints](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)
- [Roboflow Universe — football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- [Ball and Player Detection in Futsal Videos Using YOLOv8 (CEUR Vol-3885)](https://ceur-ws.org/Vol-3885/paper28.pdf)
- [Fine-Grained Visual Dribbling Style Analysis (CVPR-W 2019)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Li_Fine-Grained_Visual_Dribbling_Style_Analysis_for_Soccer_Videos_With_Augmented_CVPRW_2019_paper.pdf)
- [From Pixels to Play: Dribble and Tackle Detection in Football (UiO, 2025)](https://home.simula.no/~paalh/students/2025-UiO-EirikEggset.pdf)
- `docs/detection-report.md` — this project's own measured benchmark on real side-angle footage
