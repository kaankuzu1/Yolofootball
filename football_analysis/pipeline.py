"""The run: a clip goes in one end, an event timeline comes out the other.

:class:`AnalysisPipeline` owns the loop and nothing else.  It reads frames,
hands each to the detector, the tracker and the event logic in turn, collects
what comes back, optionally draws and encodes an annotated clip, and assembles
the :class:`~football_analysis.events.EventTimeline`.

Every stage is injected.  Pass real implementations and this is the system;
pass the placeholders from :mod:`football_analysis.stubs` and it still runs
end to end, which is what makes the plumbing testable before the models exist.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from football_analysis.config import Config, ConfigError
from football_analysis.events import (
    Event,
    EventTimeline,
    EventType,
    PlayerInfo,
    SourceInfo,
)
from football_analysis.interfaces import (
    Detector,
    EventDetector,
    FrameObserver,
    StateRefiner,
    Tracker,
)
from football_analysis.io import AnnotatedVideoWriter, VideoReader
from football_analysis.state import (
    FrameState,
    StateCacheWriter,
    read_state_cache,
    state_from_tracks,
    tracks_from_state,
)
from football_analysis.types import Detection, PLAYER, Track, VideoFrame
from football_analysis.viz import FrameAnnotator

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisPipeline",
    "PipelineResult",
    "analyze",
    "replay",
    "default_detector",
    "default_event_stage",
    "default_state_refiners",
    "find_weights",
    "default_tracker",
    "resolve_weights",
]

ProgressHook = Callable[["ProgressUpdate"], None]


@dataclass
class ProgressUpdate:
    """Handed to a progress hook once per processed frame."""

    frame_index: int
    timestamp_s: float
    frames_processed: int
    events_so_far: int
    elapsed_s: float
    phase: str = "analyse"
    """Which pass this is: ``analyse``, ``refine``, ``events`` or ``render``.

    A run has more than one pass over the clip, and a progress line that does
    not say which is running looks like it has restarted."""


@dataclass
class PipelineResult:
    """What a run produced."""

    timeline: EventTimeline
    json_path: Path | None = None
    video_path: Path | None = None
    state_cache_path: Path | None = None
    """Where the per-frame world-state record was written, when it was."""

    @property
    def events(self) -> list[Event]:
        return self.timeline.events


class AnalysisPipeline:
    """Runs one clip through the stages."""

    def __init__(
        self,
        config: Config,
        *,
        detector: Detector,
        tracker: Tracker,
        event_detector: EventDetector,
        annotator: FrameAnnotator | None = None,
        observers: Sequence[FrameObserver] = (),
        refiners: Sequence[StateRefiner] = (),
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.event_detector = event_detector
        self.annotator = annotator or FrameAnnotator(config.output)
        self.observers = list(observers)
        """Pixel measurements that annotate the record before the rules read it."""
        self.refiners = list(refiners)
        """Second-pass readings, taken once the tracker has settled the boxes."""

    # -- filtering --------------------------------------------------------

    def _enabled_types(self) -> set[EventType]:
        enabled: set[EventType] = set()
        for name in self.config.events.enabled_types:
            try:
                enabled.add(EventType(name))
            except ValueError:
                logger.warning("ignoring unknown event type in config: %r", name)
        return enabled

    def _accept(self, event: Event, enabled: set[EventType], seen: dict[tuple, float]) -> bool:
        """Drop events that are disabled, under-confident, or a repeat."""
        if event.type not in enabled:
            return False
        if float(event.confidence) < self.config.events.min_confidence:
            return False

        if event.id is not None:
            # The producing stage assigned its own id, which means it has
            # already decided these are separate events. Second-guessing that
            # loses real ones -- a shot and the rebound the same player puts
            # away half a second later are two shots, not one counted twice.
            # The min_gap_s rule exists for stages that fire every frame, and
            # those do not name their events.
            return True

        key = (event.type, event.player_id)
        last = seen.get(key)
        if last is not None and float(event.timestamp_s) - last < self.config.events.min_gap_s:
            return False
        seen[key] = float(event.timestamp_s)
        return True

    # -- the loop ---------------------------------------------------------

    def run(self, input_path: str | Path, *, progress: ProgressHook | None = None) -> PipelineResult:
        video_cfg = self.config.video
        output_cfg = self.config.output

        for stage in (self.detector, self.tracker, self.event_detector,
                      *self.observers, *self.refiners):
            reset = getattr(stage, "reset", None)
            if callable(reset):
                reset()
        self.annotator.reset()

        enabled = self._enabled_types()
        seen: dict[tuple, float] = {}
        collected: list[Event] = []
        states: list[FrameState] = []
        player_labels: dict[int, str] = {}
        started = time.monotonic()
        frames_processed = 0
        # What to draw on each frame, kept for the render pass below. Only
        # boxes and ids, so this stays small even on a long clip.
        render_data: list[tuple[int, list[Track], list[Detection]]] = []
        rendering = bool(output_cfg.video_path)

        reader = VideoReader(
            input_path,
            target_width=video_cfg.resize_width,
            target_height=video_cfg.resize_height,
            start_s=video_cfg.start_s,
            end_s=video_cfg.end_s,
            frame_stride=video_cfg.frame_stride,
            max_frames=video_cfg.max_frames,
        )

        try:
            for frame in reader:
                frames_processed += 1
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(frame, detections)

                # The record is the interface to the event logic, which never
                # sees the frame itself. Writing it to disk lets the rules be
                # re-run in seconds against this run (see `replay`).
                state = state_from_tracks(frame, tracks)

                # Observers run here, where the frame still exists, and leave
                # their readings on the record -- so the rules get them, and so
                # does the cache, which keeps a replay honest.
                for observer in self.observers:
                    observer.observe(frame, state)
                states.append(state)

                if rendering:
                    render_data.append((frame.index, tracks, detections))

                if progress is not None:
                    progress(
                        ProgressUpdate(
                            frame_index=frame.index,
                            timestamp_s=frame.timestamp_s,
                            frames_processed=frames_processed,
                            events_so_far=len(collected),
                            elapsed_s=time.monotonic() - started,
                        )
                    )

            source_info = reader.source_info()
            output_fps = reader.output_fps
        finally:
            reader.close()

        # A tracker whose best answer needs the whole clip -- a backward ball
        # smoother, an identity lock that clusters every crop at once -- refines
        # the records here. The rules read the refined ones, never the
        # provisional ones, because a provisional identity is the wrong player.
        states, tracking_summary, states_refined = self._refine(states)

        # With the boxes settled, the second-pass readings can be taken -- pose
        # keypoints being the one this exists for. They run before the cache is
        # written, so a ``--replay`` sees exactly what the rules saw.
        states = self._run_refiners(input_path, states, progress, started)

        # Only now is the record final, so only now is it worth caching or
        # handing to the rules.
        cache_path: Path | None = None
        if output_cfg.state_cache_path and states:
            with StateCacheWriter(
                output_cfg.state_cache_path, header_extra=tracking_summary
            ) as state_cache:
                for state in states:
                    state_cache.write(state)
            cache_path = state_cache.path

        for position, state in enumerate(states, start=1):
            for player in state.players:
                player_labels[player.track_id] = player.player_id
            collected.extend(
                e for e in self.event_detector.update(state)
                if self._accept(e, enabled, seen)
            )
            if progress is not None:
                progress(
                    ProgressUpdate(
                        frame_index=state.frame_index,
                        timestamp_s=state.timestamp_s,
                        frames_processed=position,
                        events_so_far=len(collected),
                        elapsed_s=time.monotonic() - started,
                        phase="events",
                    )
                )

        # Events the logic could only confirm once the clip ended.
        for event in self.event_detector.finalize():
            if self._accept(event, enabled, seen):
                collected.append(event)

        # Drawing happens only now, because ``finalize`` is where the event
        # logic emits anything that needed lookahead -- a pass that turned out
        # to be a shot, a shot that turned out to be a goal. Drawing during the
        # analysis pass would silently omit exactly the events that matter most.
        # Draw from the refined records, not the provisional tracks captured
        # during the loop. Otherwise the clip captions a player by a tracklet id
        # while the timeline calls them Player 1, and the two disagree about who
        # did what -- which is worse than either being wrong alone.
        if rendering and states_refined:
            detections_by_index = {i: d for i, _, d in render_data}
            render_data = [
                (state.frame_index,
                 tracks_from_state(state),
                 detections_by_index.get(state.frame_index, []))
                for state in states
            ]

        video_path: Path | None = None
        frames_rendered = 0
        if rendering and render_data:
            video_path, frames_rendered = self._render(
                input_path, render_data, collected, output_fps
            )

        elapsed = time.monotonic() - started
        timeline = EventTimeline(source=source_info, config=self.config.to_dict())
        timeline.extend(collected)
        timeline.sort()
        # One entry per person, not per track: a player whose track was lost
        # and re-acquired is still one player, and ``player_id`` is the key the
        # events refer to, so it has to be unique here.
        by_player: dict[str, PlayerInfo] = {}
        for track_id, label in sorted(player_labels.items()):
            existing = by_player.get(label)
            if existing is None:
                by_player[label] = PlayerInfo(
                    player_id=label, label=label, track_id=track_id,
                    attributes={"track_ids": [track_id]},
                )
            else:
                existing.attributes["track_ids"].append(track_id)
        timeline.players = list(by_player.values())
        timeline.stats = {
            "frames_read": reader.frames_read,
            "frames_processed": frames_processed,
            "inexact_timestamps": reader.inexact_timestamp_count,
            "states_refined": states_refined,
            "elapsed_s": round(elapsed, 3),
            "fps_processed": round(frames_processed / elapsed, 2) if elapsed > 0 else None,
            "stages": {
                "detector": _describe(self.detector),
                "tracker": _describe(self.tracker),
                "event_detector": _describe(self.event_detector),
                "observers": [_describe(o) for o in self.observers],
                "refiners": [_describe(r) for r in self.refiners],
            },
        }

        timeline.stats["frames_rendered"] = frames_rendered

        result = PipelineResult(timeline=timeline)
        if output_cfg.json_path:
            result.json_path = timeline.write_json(
                output_cfg.json_path, indent=output_cfg.json_indent
            )
        result.video_path = video_path
        if cache_path is not None:
            result.state_cache_path = cache_path
        return result

    def _refine(
        self, states: list[FrameState]
    ) -> tuple[list[FrameState], dict[str, Any] | None, bool]:
        """Let the tracker replace the provisional records with its best ones.

        A tracker that can only answer from the whole clip exposes
        ``finalize_states()``; one that works frame by frame does not, and its
        records pass through untouched.

        The care here is in the attributes. Observer readings -- whether the net
        rippled, say -- were written onto the provisional records during the
        loop, and the tracker's refined records know nothing about them. Simply
        taking the new list would drop every one of them, and the only visible
        symptom would be a goal rule quietly getting less sure of itself. So
        they are carried across by frame index, with the tracker's own
        attributes winning any collision, since it is the one that just
        recomputed them.
        """
        finalize_states = getattr(self.tracker, "finalize_states", None)
        if not callable(finalize_states):
            return states, None, False

        refined = list(finalize_states() or ())
        if not refined:
            logger.warning(
                "%s.finalize_states() returned nothing; keeping the provisional records",
                type(self.tracker).__name__,
            )
            return states, None, False

        observed = {
            state.frame_index: state.attributes
            for state in states
            if state.attributes
        }
        for state in refined:
            carried = observed.get(state.frame_index)
            if carried:
                state.attributes = {**carried, **state.attributes}

        if len(refined) != len(states):
            logger.warning(
                "tracker returned %d records for %d processed frames",
                len(refined), len(states),
            )

        summary = None
        summarise = getattr(self.tracker, "summary", None)
        if callable(summarise):
            try:
                summary = summarise()
            except Exception:  # pragma: no cover - a stage's own reporting bug
                logger.debug("tracker failed to summarise itself", exc_info=True)
        return refined, summary, True

    def _run_refiners(
        self,
        input_path: str | Path,
        states: list[FrameState],
        progress: ProgressHook | None,
        started: float,
    ) -> list[FrameState]:
        """Give each refiner the settled records and the clip to re-read.

        A refiner that raises is reported and skipped rather than taking the
        run down: pose weights that will not load should cost the tackles, not
        the goals. What it would have filled in is simply absent, and the rules
        that need it abstain.
        """
        if not states:
            return states
        for refiner in self.refiners:
            name = type(refiner).__name__
            if progress is not None:
                progress(
                    ProgressUpdate(
                        frame_index=states[-1].frame_index,
                        timestamp_s=states[-1].timestamp_s,
                        frames_processed=len(states),
                        events_so_far=0,
                        elapsed_s=time.monotonic() - started,
                        phase="refine",
                    )
                )
            logger.info("%s over %d records", name, len(states))
            try:
                result = refiner.refine(states, str(input_path))
            except Exception:
                logger.warning("%s failed; its readings are missing", name, exc_info=True)
                continue
            if result:
                states = list(result)
            else:
                # Returning nothing is a bug in the refiner, not an instruction
                # to throw the clip away.
                logger.warning("%s returned no records; keeping the ones it was given", name)
        return states

    def _render(
        self,
        input_path: str | Path,
        render_data: Sequence[tuple[int, list[Track], list[Detection]]],
        events: Sequence[Event],
        fps: float,
    ) -> tuple[Path | None, int]:
        """Second pass: decode the clip again and draw the finished timeline.

        Decoding twice is cheap next to running the models once, and it is the
        only way an event the logic could not call until the clip ended can
        appear on the frame where it happened.
        """
        video_cfg = self.config.video
        output_cfg = self.config.output
        by_index = {index: (tracks, dets) for index, tracks, dets in render_data}
        ordered = sorted(events, key=lambda e: float(e.timestamp_s))

        self.annotator.reset()
        logger.info("rendering %s", output_cfg.video_path)

        reader = VideoReader(
            input_path,
            target_width=video_cfg.resize_width,
            target_height=video_cfg.resize_height,
            start_s=video_cfg.start_s,
            end_s=video_cfg.end_s,
            frame_stride=video_cfg.frame_stride,
            max_frames=video_cfg.max_frames,
        )
        writer = AnnotatedVideoWriter(
            output_cfg.video_path, fps=fps, codec=output_cfg.video_codec
        )
        next_event = 0
        # An event is drawn on the frame nearest its time. Without the half-gap
        # tolerance an event timestamped a hair after the last frame -- which
        # rounding alone can produce -- would be dropped from the video.
        tolerance = 0.5 / fps if fps > 0 else 0.0
        try:
            for frame in reader:
                tracks, detections = by_index.get(frame.index, ([], []))

                # Every event that happened at or before this frame's time and
                # has not been shown yet fires here. Matching on time rather
                # than on frame_index catches events that only carry a
                # timestamp, which finalize-time events often do.
                due: list[Event] = []
                while (
                    next_event < len(ordered)
                    and float(ordered[next_event].timestamp_s)
                    <= frame.timestamp_s + tolerance
                ):
                    due.append(ordered[next_event])
                    next_event += 1

                writer.write(
                    self.annotator.annotate(
                        frame, tracks=tracks, detections=detections, events=due
                    )
                )
        finally:
            reader.close()
            writer.close()

        if not writer.frames_written:
            return None, 0
        if next_event < len(ordered):
            # An event timed past the last rendered frame cannot be drawn; say
            # so rather than letting it vanish from the video silently.
            logger.warning(
                "%d event(s) fall after the last rendered frame and are not drawn",
                len(ordered) - next_event,
            )
        return writer.path, writer.frames_written


def _stage_unavailable(exc: ImportError, stage: str, consequence: str) -> None:
    """Report a stage that could not be imported, loudly and by name.

    A stage missing because the checkout is partly built is expected. A stage
    missing because one dependency is absent is a silently degraded run, and
    the two look identical from here -- so both get a warning naming the module
    that failed and what the run loses without it. Debug-level was wrong: it
    cost a day of CLI runs on a placeholder detector once already.
    """
    logger.warning(
        "%s unavailable (%s); %s",
        stage,
        f"no module named {exc.name!r}" if getattr(exc, "name", None) else exc,
        consequence,
    )
    logger.debug("import of %s failed", stage, exc_info=True)


def _describe(stage: object) -> dict:
    describe = getattr(stage, "describe", None)
    if callable(describe):
        try:
            return describe()
        except Exception:  # pragma: no cover - a stage's own reporting bug
            logger.debug("stage %r failed to describe itself", stage, exc_info=True)
    return {"name": type(stage).__name__}


def analyze(
    input_path: str | Path,
    config: Config | None = None,
    *,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    event_detector: EventDetector | None = None,
    progress: ProgressHook | None = None,
) -> PipelineResult:
    """Run one clip, filling in placeholder stages for any not supplied.

    This is the function the CLI calls and the one to import from a notebook::

        from football_analysis import analyze
        result = analyze("match.mp4")
        for event in result.timeline.primary():
            print(event.clock, event.type.value, event.player_id)
    """
    config = (config or Config()).validate()
    if event_detector is None:
        event_detector, observers = default_event_stage(config)
        refiners = default_state_refiners(config)
    else:
        # A caller who brings their own event stage brings its inputs too: we
        # cannot know what readings their rules expect.
        observers = []
        refiners = []

    pipeline = AnalysisPipeline(
        config,
        detector=detector or default_detector(config),
        tracker=tracker or default_tracker(config),
        event_detector=event_detector,
        observers=observers,
        refiners=refiners,
    )
    return pipeline.run(input_path, progress=progress)


def resolve_weights(config: Config) -> Path | None:
    """Find the detector weights on disk, or ``None`` if they are not there.

    A bare name like ``forzasys_soccer.pt`` is looked for as given and then
    under each of ``detection.weights_search_paths``, so the models that ship
    with the project can be named without a path.
    """
    model_path = (config.detection.model_path or "").strip()
    if not model_path:
        return None

    return find_weights(model_path, config.detection.weights_search_paths)


def _working_dir() -> Path:
    """Where a relative search path starts. Injectable, and never raises.

    ``Path.cwd()`` raises if the working directory has been deleted under the
    process, which should cost a lookup at most, not the run.
    """
    try:
        return Path.cwd()
    except OSError:  # pragma: no cover - only when the cwd has vanished
        return Path(".")


def _checkout_root() -> Path | None:
    """The directory holding the installed package, or ``None``.

    For a checkout or an editable install this is the project root, the one
    with ``assets/`` beside ``football_analysis/``. For a wheel in
    site-packages it is site-packages, where nothing will be found -- which is
    correct, since those weights were never installed.
    """
    try:
        import football_analysis

        return Path(football_analysis.__file__).resolve().parent.parent
    except Exception:  # pragma: no cover - only if the package is unimportable
        return None


def find_weights(name: str, search_paths: Sequence[str]) -> Path | None:
    """Locate a checkpoint by name, as given or under ``search_paths``.

    The search paths are relative by default (``assets/models``), which means
    the working directory decides whether the weights exist. That is the wrong
    answer: someone running ``football-analyse ~/clips/match.mp4`` from their
    clips folder should get the same detector as someone running it from the
    checkout, not a silent downgrade to the placeholder. So a relative search
    path is tried against the working directory first, then against the
    project the package was installed from.

    Weights are never downloaded implicitly. A stage whose weights are missing
    says so and stands down, because a silent download is a surprise and a
    silent fallback is worse -- see the CLI running a placeholder detector for
    a day while reporting success.
    """
    candidate = Path(name)
    if candidate.exists():
        return candidate
    if candidate.is_absolute() or len(candidate.parts) > 1:
        # A caller who spelled out a path meant that path, not a name to hunt for.
        return None

    absolute = [Path(d) for d in search_paths if Path(d).is_absolute()]
    relative = [Path(d) for d in search_paths if not Path(d).is_absolute()]

    # An absolute search path means itself, wherever the run started.
    for base in absolute:
        found = base / candidate.name
        if found.exists():
            return found

    for root in (_working_dir(), _checkout_root()):
        if root is None:
            continue
        for base in relative:
            found = root / base / candidate.name
            if found.exists():
                return found
    return None


def _resolve_device(requested: str) -> str:
    """Turn ``auto`` into something the backend understands."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - depends on the torch build
        logger.debug("could not probe for a GPU", exc_info=True)
    return "cpu"


def default_detector(config: Config) -> Detector:
    """The detector a run gets when the caller does not supply one.

    The real model is used whenever its weights are actually on disk. When they
    are not, the run falls back to the placeholder -- but says so loudly,
    because a run that quietly analyses motion blobs instead of players looks
    like it worked and is wrong about everything.

    Weights are never downloaded implicitly: a run that silently pulls a model
    off the internet is a surprise, and one that hangs trying to is worse.
    """
    from football_analysis.stubs import StubDetector

    weights = resolve_weights(config)
    if weights is None:
        logger.warning(
            "detector weights %r not found (looked in %s); "
            "falling back to the placeholder detector, which finds motion, not players. "
            "Pass --model with a path to real weights.",
            config.detection.model_path,
            ", ".join(config.detection.weights_search_paths) or "the working directory",
        )
        return StubDetector(config)

    try:
        from football_analysis.adapters import adapt_detector
        from football_analysis.detection import Detector as YoloDetector
        from football_analysis.detection import DetectorConfig
    except ImportError:
        logger.warning(
            "detection backend unavailable (install requirements-detect.txt); "
            "falling back to the placeholder detector",
        )
        return StubDetector(config)

    detection = config.detection
    settings: dict[str, Any] = {
        "weights": str(weights),
        "conf": detection.confidence_threshold,
        "ball_conf": detection.threshold_for("ball"),
        "iou": detection.iou_threshold,
        "device": _resolve_device(detection.device),
    }
    if detection.half_precision:
        settings["half"] = True
    if detection.imgsz:
        settings["imgsz"] = detection.imgsz
    settings.update(detection.options)

    try:
        raw = YoloDetector(DetectorConfig(**settings))
    except TypeError as exc:
        raise ConfigError(f"invalid detection setting: {exc}") from exc

    logger.info("detector: %s on %s", weights, settings["device"])
    # The backend speaks its own vocabulary: a goalkeeper is a player as far as
    # the event rules care, and a referee is not a participant in a 1v1.
    return adapt_detector(
        raw,
        label_map={"goalkeeper": "player"},
        keep=["player", "ball", "goal"],
    )


def default_tracker(config: Config) -> Tracker:
    """The tracking layer a run gets when the caller does not supply one.

    The real tracker is preferred; the simple IoU tracker is the fallback, so a
    partly-built checkout still runs. The ``track`` and ``geometry`` sections of
    the config are handed to the layer's own dataclasses, which is where an
    unknown key is caught -- this module deliberately does not know what those
    keys are.
    """
    from football_analysis.stubs import GreedyIouTracker

    try:
        from football_analysis.geometry.config import GeometryConfig
        from football_analysis.track import WorldStateTracker
        from football_analysis.track.config import TrackLayerConfig
    except ImportError as exc:
        _stage_unavailable(
            exc,
            "the tracking layer",
            "falling back to simple IoU tracking, so identities, the ball's "
            "Kalman filter and the camera geometry are all lost",
        )
        return GreedyIouTracker(config)

    layer = TrackLayerConfig.from_config(config)
    try:
        if config.track:
            layer = _apply(layer, config.track, "track")
        geometry = GeometryConfig(**config.geometry) if config.geometry else GeometryConfig()
    except TypeError as exc:
        raise ConfigError(f"invalid geometry setting: {exc}") from exc

    return WorldStateTracker(config, layer=layer, geometry=geometry)


def _apply(target: Any, settings: dict[str, Any], section: str) -> Any:
    """Set dotted keys on a nested config dataclass, e.g. ``ball.max_gap``."""
    for dotted, value in settings.items():
        node = target
        parts = str(dotted).split(".")
        for part in parts[:-1]:
            if not hasattr(node, part):
                raise ConfigError(f"unknown key in {section}: {dotted!r}")
            node = getattr(node, part)
        if not hasattr(node, parts[-1]):
            raise ConfigError(f"unknown key in {section}: {dotted!r}")
        setattr(node, parts[-1], value)
    return target


def default_event_stage(
    config: Config,
) -> tuple[EventDetector, list[FrameObserver]]:
    """The event logic a run gets when the caller does not supply one.

    The real rule engines are preferred; the placeholder is the fallback, so a
    partly-built checkout still runs rather than failing to import. Each engine
    brings whatever pixel observers its rules need, because only it knows what
    it measures.
    """
    from football_analysis.adapters import CompositeEventDetector, MeasurementObserver
    from football_analysis.stubs import StubEventDetector

    detectors: list[Any] = []
    observers: list[FrameObserver] = []

    try:
        from football_analysis.ball_events import (
            BallEventConfig,
            BallEventDetector,
            NetMotionMeter,
        )

        # ``from_config`` maps the keys the rules share with the ``events``
        # section; the ``ball_events`` section reaches the rest. Layering rather
        # than replacing means a run that sets only ``pass_mode`` still gets the
        # shared keys, and an unknown key is still rejected by ``from_dict``.
        rules = BallEventConfig.from_config(config)
        if config.ball_events:
            merged = {**rules.to_dict(), **config.ball_events}
            try:
                rules = BallEventConfig.from_dict(merged)
            except ValueError as exc:
                raise ConfigError(f"invalid ball_events setting: {exc}") from exc
        detectors.append(BallEventDetector(config, rules=rules))
        # The goal rule needs to know whether the net rippled, and only pixels
        # can say. The observer seam is how that reading reaches the record.
        observers.append(MeasurementObserver(NetMotionMeter(), "net_motion"))
    except ImportError as exc:
        _stage_unavailable(
            exc, "the ball event rules", "no passes, shots or goals will be reported"
        )

    try:
        from football_analysis.body_events import BodyEventConfig, BodyEventDetector

        body_config = _apply(BodyEventConfig(), config.body_events, "body_events")
        detectors.append(BodyEventDetector(config, body_config=body_config))
    except ImportError as exc:
        _stage_unavailable(
            exc, "the tackle and trick stage", "no tackles or tricks will be reported"
        )

    if not detectors:
        logger.info("no rule engine available; using the placeholder event stage")
        return StubEventDetector(config), []
    if len(detectors) == 1:
        return detectors[0], observers
    return CompositeEventDetector(*detectors), observers


def default_state_refiners(config: Config) -> list[StateRefiner]:
    """The second-pass readings a run gets when the caller does not supply any.

    Pose is the only one so far, and it is what tackle and trick detection
    reads. It runs here rather than inside the loop for two reasons. The boxes
    are settled, so a skeleton is never attributed to a provisional identity
    that the tracker is about to merge into someone else; and on a crowded clip
    it runs on the two players that survived rather than on every blob the
    detector proposed, which is most of the cost.

    If the pose weights are not on disk the step is skipped with a warning
    rather than downloaded. The rules that need keypoints then find none and
    abstain, which is the honest outcome.
    """
    from football_analysis.adapters import PoseRefiner

    # Nothing but tackles and tricks reads keypoints, so a run that has turned
    # both off should not pay for pose. This is the common case for a quick
    # goals-only pass over a long clip.
    wanted = {EventType.TACKLE, EventType.TRICK}
    if not wanted & {
        EventType(name) for name in config.events.enabled_types
        if name in EventType._value2member_map_
    }:
        logger.info("neither tackles nor tricks are enabled; skipping pose")
        return []

    try:
        from football_analysis.pose import PoseConfig, PoseEstimator
    except ImportError as exc:
        _stage_unavailable(
            exc, "the pose stage", "tackles and tricks have no keypoints to read"
        )
        return []

    pose_config = _apply(PoseConfig(), config.pose, "pose")
    weights = find_weights(pose_config.weights, config.detection.weights_search_paths)
    if weights is None:
        logger.warning(
            "pose weights %r not found (looked in %s); skipping pose, so tackles "
            "and tricks will not be detected",
            pose_config.weights,
            ", ".join(config.detection.weights_search_paths) or "nowhere",
        )
        return []
    pose_config.weights = str(weights)
    if "device" not in config.pose:
        # Follow the detector unless the config says otherwise.
        pose_config.device = _resolve_device(config.detection.device)

    try:
        estimator = PoseEstimator(pose_config)
    except Exception:
        logger.warning("could not load the pose model; skipping pose", exc_info=True)
        return []

    return [
        PoseRefiner(
            estimator,
            target_width=config.video.resize_width,
            target_height=config.video.resize_height,
        )
    ]


def replay(
    state_cache_path: str | Path,
    event_detector: EventDetector,
    config: Config | None = None,
) -> EventTimeline:
    """Re-run event logic over a cached run, without touching the video.

    This is the payoff of the world-state record: tuning a tackle threshold
    means re-reading a JSON Lines file, not re-running inference over the clip.

    The returned timeline carries no ``source`` resolution or fps, because the
    cache holds per-frame state rather than clip metadata; what it does carry
    is the events, which is what is being iterated on.
    """
    config = (config or Config()).validate()
    enabled: set[EventType] = set()
    for name in config.events.enabled_types:
        try:
            enabled.add(EventType(name))
        except ValueError:
            logger.warning("ignoring unknown event type in config: %r", name)

    seen: dict[tuple, float] = {}
    collected: list[Event] = []
    player_labels: dict[int, str] = {}
    records = 0
    last_timestamp = 0.0

    reset = getattr(event_detector, "reset", None)
    if callable(reset):
        reset()

    def accept(event: Event) -> bool:
        if event.type not in enabled:
            return False
        if float(event.confidence) < config.events.min_confidence:
            return False
        if event.id is not None:
            return True  # producer-named events are already de-duplicated
        key = (event.type, event.player_id)
        last = seen.get(key)
        if last is not None and float(event.timestamp_s) - last < config.events.min_gap_s:
            return False
        seen[key] = float(event.timestamp_s)
        return True

    started = time.monotonic()
    for state in read_state_cache(state_cache_path):
        records += 1
        last_timestamp = state.timestamp_s
        for player in state.players:
            player_labels[player.track_id] = player.player_id
        collected.extend(e for e in event_detector.update(state) if accept(e))

    for event in event_detector.finalize():
        if accept(event):
            collected.append(event)

    source = SourceInfo(
        path=str(state_cache_path),
        width=0,
        height=0,
        fps=0.0,
        frame_count=records,
        duration_s=round(last_timestamp, 3) if records else None,
    )
    timeline = EventTimeline(source=source, config=config.to_dict())
    timeline.extend(collected)
    timeline.sort()

    by_player: dict[str, PlayerInfo] = {}
    for track_id, label in sorted(player_labels.items()):
        if label not in by_player:
            by_player[label] = PlayerInfo(
                player_id=label, label=label, track_id=track_id,
                attributes={"track_ids": [track_id]},
            )
        else:
            by_player[label].attributes["track_ids"].append(track_id)
    timeline.players = list(by_player.values())

    timeline.stats = {
        "replayed_from_cache": str(state_cache_path),
        "frames_processed": records,
        "elapsed_s": round(time.monotonic() - started, 3),
        "stages": {"event_detector": _describe(event_detector)},
    }
    return timeline
