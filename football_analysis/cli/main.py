"""``football-analyse`` -- run a clip, get a JSON timeline.

    football-analyse match.mp4 -o events.json --render annotated.mp4

Flags beat the config file, which beats the defaults.  The resolved config is
written into the output document, so a result always says how it was made.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from football_analysis import __version__
from football_analysis.config import (
    Config,
    ConfigError,
    EventConfig,
    load_config,
    merge_overrides,
)
from football_analysis.events import EventType
from football_analysis.io.video_reader import VideoSourceError

logger = logging.getLogger("football_analysis")

__all__ = ["main", "build_parser"]

_EXIT_OK = 0
_EXIT_USAGE = 2
_EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-analyse",
        description=(
            "Analyse a 1v1 football clip and write a timestamped event timeline."
        ),
        epilog=(
            "example:\n"
            "  football-analyse match.mp4 -o events.json --render annotated.mp4\n"
            "  football-analyse match.mp4 --start 30 --end 90 --resize-width 960\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="path to the clip to analyse")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument(
        "-c", "--config", metavar="PATH",
        help="YAML or JSON config file layered over the defaults",
    )
    parser.add_argument(
        "-o", "--output", metavar="PATH",
        help="where to write the event timeline (default: stdout)",
    )
    parser.add_argument(
        "--render", metavar="PATH",
        help="also write an annotated clip to this path",
    )
    parser.add_argument(
        "--state-cache", metavar="PATH",
        help="write the per-frame world-state record here, for --replay",
    )
    parser.add_argument(
        "--replay", metavar="PATH",
        help="re-run event logic over a cached record instead of a clip",
    )

    segment = parser.add_argument_group("segment")
    segment.add_argument("--start", type=float, metavar="SECONDS",
                         help="skip to this point in the clip")
    segment.add_argument("--end", type=float, metavar="SECONDS",
                         help="stop at this point in the clip")
    segment.add_argument("--max-frames", type=int, metavar="N",
                         help="stop after N processed frames")

    frames = parser.add_argument_group("frames")
    frames.add_argument("--resize-width", type=int, metavar="PX",
                        help="resize frames to this width before analysis")
    frames.add_argument("--no-resize", action="store_true",
                        help="analyse at the clip's own resolution")
    frames.add_argument("--stride", type=int, metavar="N",
                        help="process every Nth frame")

    goal = parser.add_argument_group(
        "goal geometry (annotate once per camera setup)"
    )
    goal.add_argument(
        "--goal-corners", metavar="X1,Y1,...,X4,Y4",
        help="the four goal-mouth corners: left post base, right post base, "
             "right crossbar end, left crossbar end (left/right as seen in the "
             "image). Read them off the clip at its own resolution",
    )
    goal.add_argument(
        "--goal-corners-space", choices=("source", "processed"), default="source",
        help="what resolution the corners were read at (default: source, the "
             "clip's own; they are rescaled for you if --resize-width applies)",
    )
    goal.add_argument(
        "--goal-size", nargs=2, type=float, metavar=("WIDTH_M", "HEIGHT_M"),
        help="goal mouth in metres (default 3 2, a small-sided goal; "
             "a full-size goal is 7.32 2.44)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model", metavar="PATH", help="detector weights to load")
    model.add_argument("--device", metavar="DEV", help="auto, cpu, cuda, cuda:0 or mps")
    model.add_argument("--confidence", type=float, metavar="X",
                       help="global detection confidence floor")
    model.add_argument("--min-event-confidence", type=float, metavar="X",
                       help="drop events below this confidence")
    model.add_argument(
        "--events", metavar="LIST",
        help="comma-separated event types to report (default: %s)"
             % ",".join(EventConfig().enabled_types),
    )

    render = parser.add_argument_group("rendering")
    render.add_argument("--codec", metavar="FOURCC", help="force an output codec")
    render.add_argument("--no-boxes", action="store_true", help="do not draw boxes")
    render.add_argument("--no-trail", action="store_true", help="do not draw the ball trail")

    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved config as YAML and exit")
    parser.add_argument("--quiet", "-q", action="store_true", help="errors only")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    parser.add_argument("--no-progress", action="store_true",
                        help="do not print progress while running")
    return parser


def _parse_goal_corners(text: str) -> list[list[float]]:
    """Turn ``"x1,y1,x2,y2,x3,y3,x4,y4"`` into four points."""
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise ConfigError(f"--goal-corners takes numbers: {exc}") from exc
    if len(numbers) != 8:
        raise ConfigError(
            f"--goal-corners needs 8 numbers (4 corners), got {len(numbers)}"
        )
    return [[numbers[i], numbers[i + 1]] for i in range(0, 8, 2)]


def _scale_goal_corners(
    corners: list[list[float]], input_path: str | None, config: Config
) -> list[list[float]]:
    """Convert corners read at the clip's own size into processed-frame space.

    People annotate the goal by looking at their footage, which is the source
    resolution; the pipeline works on resized frames. Silently mixing the two
    puts the goal in the wrong place and every goal call with it, so the
    conversion happens here rather than in the user's head.
    """
    if input_path is None or config.video.resize_width is None:
        return corners

    from football_analysis.io.video_reader import probe

    info = probe(input_path)
    if not info.width:
        return corners

    scale_x = config.video.resize_width / info.width
    if config.video.resize_height is not None:
        scale_y = config.video.resize_height / info.height
    else:
        scale_y = scale_x
    if abs(scale_x - 1.0) < 1e-9 and abs(scale_y - 1.0) < 1e-9:
        return corners

    logger.info(
        "rescaling goal corners from %dx%d to the processed frame",
        info.width, info.height,
    )
    return [[x * scale_x, y * scale_y] for x, y in corners]


def _overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Map flags onto dotted config paths.  ``None`` entries are ignored."""
    overrides: dict[str, object] = {
        "video.start_s": args.start,
        "video.end_s": args.end,
        "video.max_frames": args.max_frames,
        "video.frame_stride": args.stride,
        "video.resize_width": args.resize_width,
        "detection.model_path": args.model,
        "detection.device": args.device,
        "detection.confidence_threshold": args.confidence,
        "events.min_confidence": args.min_event_confidence,
        "output.json_path": args.output,
        "output.video_path": args.render,
        "output.video_codec": args.codec,
        "output.state_cache_path": args.state_cache,
    }
    if args.no_resize:
        # An explicit --resize-width alongside --no-resize is a contradiction;
        # the parser catches that before we get here.
        overrides["video.resize_width"] = None
        overrides["video.resize_height"] = None
    if args.no_boxes:
        overrides["output.draw_boxes"] = False
    if args.no_trail:
        overrides["output.draw_ball_trail"] = False
    if args.events:
        overrides["events.enabled_types"] = [
            part.strip() for part in args.events.split(",") if part.strip()
        ]
    return overrides


def _resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    if args.no_resize:
        # merge_overrides skips None, so clearing a size needs its own pass.
        config.video.resize_width = None
        config.video.resize_height = None
    config = merge_overrides(config, _overrides_from_args(args))

    if args.goal_corners:
        corners = _parse_goal_corners(args.goal_corners)
        if args.goal_corners_space == "source":
            corners = _scale_goal_corners(corners, args.input, config)
        config.geometry["goal_corners_px"] = corners
    if args.goal_size:
        config.geometry["goal_width_m"] = args.goal_size[0]
        config.geometry["goal_height_m"] = args.goal_size[1]

    if args.input:
        config.input_path = args.input
    config.log_level = "DEBUG" if args.verbose else ("ERROR" if args.quiet else "INFO")
    return config


def _validate_event_types(config: Config) -> None:
    valid = {t.value for t in EventType}
    unknown = [name for name in config.events.enabled_types if name not in valid]
    if unknown:
        raise ConfigError(
            f"unknown event type(s): {', '.join(unknown)}. "
            f"Valid types: {', '.join(sorted(valid))}"
        )


@contextlib.contextmanager
def _clean_stdout(config: Config):
    """Keep stdout free of anything but the timeline.

    With no ``-o``, the document goes to stdout, so ``football-analyse clip.mp4
    | jq`` has to work. Model backends print banners and deprecation notices
    there without asking, and one stray line makes the output unparseable. So
    while the run is happening, anything written to stdout is sent to stderr
    instead, where it is still visible and harmless.
    """
    if config.output.json_path:
        yield  # the document is going to a file; stdout is already free
        return
    with contextlib.redirect_stdout(sys.stderr):
        yield


class _ProgressPrinter:
    """One rewriting line on stderr, so piping stdout to a file still works."""

    def __init__(self, enabled: bool, every_s: float = 0.5) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self.every_s = every_s
        self._last = 0.0

    def __call__(self, update) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last < self.every_s:
            return
        self._last = now
        rate = update.frames_processed / update.elapsed_s if update.elapsed_s else 0.0
        sys.stderr.write(
            f"\r  {update.phase:8s} {update.timestamp_s:7.2f}s  "
            f"{update.frames_processed} frames  "
            f"{update.events_so_far} events  "
            f"{rate:5.1f} fps "
        )
        sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write("\r" + " " * 70 + "\r")
            sys.stderr.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else (
            logging.ERROR if args.quiet else logging.INFO
        ),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.no_resize and args.resize_width is not None:
        parser.error("--no-resize and --resize-width cannot both be given")
    if args.replay and args.input:
        parser.error("--replay reads a cached record; do not also give a clip")
    if args.replay and args.render:
        parser.error("--replay has no video to draw on, so --render cannot work")

    try:
        config = _resolve_config(args)
        _validate_event_types(config)
    except ConfigError as exc:
        parser.error(str(exc))
        return _EXIT_USAGE  # unreachable; parser.error exits

    if args.print_config:
        try:
            print(config.to_yaml())
        except ImportError:
            print(config.to_json())
        return _EXIT_OK

    if args.replay:
        from football_analysis.pipeline import default_event_stage, replay

        event_detector, _ = default_event_stage(config)
        try:
            with _clean_stdout(config):
                timeline = replay(args.replay, event_detector, config)
        except (OSError, ValueError) as exc:
            logger.error("%s", exc)
            return _EXIT_FAILED
        return _emit(timeline, config, logger)

    if not args.input:
        parser.error("an input clip is required (or use --replay or --print-config)")

    # Imported here so --help and --print-config stay fast and dependency-free.
    from football_analysis.pipeline import analyze

    progress = _ProgressPrinter(enabled=not args.no_progress and not args.quiet)
    try:
        with _clean_stdout(config):
            result = analyze(args.input, config, progress=progress)
    except VideoSourceError as exc:
        logger.error("%s", exc)
        return _EXIT_FAILED
    except KeyboardInterrupt:
        logger.error("interrupted")
        return _EXIT_FAILED
    finally:
        progress.finish()

    if result.video_path is not None:
        logger.info("annotated clip -> %s", result.video_path)
    if result.state_cache_path is not None:
        logger.info("state record -> %s (replay with --replay)", result.state_cache_path)
    return _emit(result.timeline, config, logger)


def _emit(timeline, config: Config, log: logging.Logger) -> int:
    """Write the timeline where the config says, or print it."""
    if not config.output.json_path:
        print(timeline.to_json(indent=config.output.json_indent))
        return _EXIT_OK

    path = timeline.write_json(config.output.json_path, indent=config.output.json_indent)
    counts = timeline.counts()
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no events"
    log.info("%d frames, %s -> %s",
             timeline.stats.get("frames_processed", 0), summary, path)
    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
