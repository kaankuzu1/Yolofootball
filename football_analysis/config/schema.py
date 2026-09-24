"""The configuration dataclasses, their defaults, and the loader."""

from __future__ import annotations

import json
from dataclasses import (
    MISSING as _MISSING,
    asdict,
    dataclass,
    field,
    fields,
    is_dataclass,
)
from pathlib import Path
from typing import Any

__all__ = [
    "Config",
    "VideoConfig",
    "DetectionConfig",
    "TrackingConfig",
    "EventConfig",
    "OutputConfig",
    "ConfigError",
    "load_config",
    "merge_overrides",
    "default_config",
    "DEFAULT_CLASS_MAP",
]


class ConfigError(ValueError):
    """A configuration file was unreadable, or held a value that cannot work."""


# What the detector's class ids mean to the rest of the system.  A detection
# thread swapping in its own weights overrides this in the config file rather
# than editing code.
DEFAULT_CLASS_MAP: dict[int, str] = {
    0: "player",
    1: "ball",
    2: "goal",
}


@dataclass
class VideoConfig:
    """How the clip is read."""

    resize_width: int | None = 1280
    """Long-edge width fed to the model.  ``None`` keeps source resolution."""

    resize_height: int | None = None
    """Set alongside ``resize_width`` to force an aspect ratio."""

    frame_stride: int = 1
    """Process every Nth frame.  Raise it to trade accuracy for speed."""

    start_s: float = 0.0
    end_s: float | None = None
    max_frames: int | None = None

    def validate(self) -> None:
        if self.frame_stride < 1:
            raise ConfigError(f"video.frame_stride must be >= 1, got {self.frame_stride}")
        if self.start_s < 0:
            raise ConfigError(f"video.start_s must be >= 0, got {self.start_s}")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ConfigError(
                f"video.end_s ({self.end_s}) must be greater than "
                f"video.start_s ({self.start_s})"
            )
        for name in ("resize_width", "resize_height"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(f"video.{name} must be positive, got {value}")


@dataclass
class DetectionConfig:
    """Which model finds the ball, the players and the goal."""

    model_path: str = "forzasys_soccer.pt"
    """Ultralytics weights, resolved through ``weights_search_paths``.

    The default is the checkpoint the detection benchmark picked: 82.6% ball
    recall at 86.4% precision on the broadcast clip, better than every
    alternative on recall, precision and speed at once (``docs/detection-report.md``).
    Weights are never downloaded, so a checkout without them falls back to the
    placeholder detector and says so loudly."""

    device: str = "auto"
    """``auto``, ``cpu``, ``cuda``, ``cuda:0``, ``mps``."""

    confidence_threshold: float = 0.25
    """Global floor.  ``class_confidence`` overrides it per class."""

    class_confidence: dict[str, float] = field(
        default_factory=lambda: {"ball": 0.35, "player": 0.35, "goal": 0.30}
    )
    """Per-class floors. 0.35 for the ball is the benchmark's operating point:
    below it precision falls away faster than recall rises (0.25 gives the same
    82.6% recall at 82.6% precision instead of 86.4%). A model that finds the
    ball less confidently than this one wants a lower bar."""

    iou_threshold: float = 0.45
    """NMS overlap threshold."""

    class_map: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_MAP))
    """Deprecated and not read by the detector, which normalises each model's
    own ``names`` onto this vocabulary instead -- so swapping checkpoints needs
    no config change. Kept so existing config files still load."""
    max_detections: int = 30
    half_precision: bool = False
    imgsz: int | None = 1280
    """Model input size.  ``None`` lets the model use its own default.

    1280 rather than the model's 640: at 640 the ball on wide footage is a
    handful of pixels and the recall collapses. It is a resolution effect, not
    a model one (``docs/detection-report.md``)."""

    weights_search_paths: list[str] = field(
        default_factory=lambda: ["assets/models"]
    )
    """Where a bare ``model_path`` name is looked for, after the path as given."""

    options: dict[str, Any] = field(default_factory=dict)
    """Extra settings for the detector backend, e.g. ``tile_ball: true``.

    Held as a plain mapping for the same reason the ``track`` and ``geometry``
    sections are: this module should not have to know every knob the detector
    grows. Unknown keys are caught where the backend is built."""

    def threshold_for(self, class_name: str) -> float:
        return float(self.class_confidence.get(class_name, self.confidence_threshold))

    def validate(self) -> None:
        if not isinstance(self.options, dict):
            raise ConfigError(
                f"detection.options must be a mapping, got {type(self.options).__name__}"
            )
        for name in ("confidence_threshold", "iou_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"detection.{name} must be in [0, 1], got {value}")
        for cls, value in self.class_confidence.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigError(
                    f"detection.class_confidence[{cls}] must be in [0, 1], got {value}"
                )


@dataclass
class TrackingConfig:
    """How single-frame detections become identities that persist."""

    max_age_frames: int = 30
    """Keep a lost track alive this long before dropping it."""

    min_hits: int = 3
    """Frames a track must be seen before it is reported as confirmed."""

    iou_threshold: float = 0.3
    """Association threshold between a prediction and a new detection."""

    max_players: int = 2
    """A 1v1: exactly two player identities are expected."""

    ball_max_age_frames: int = 12
    """The ball disappears behind legs constantly, but not for long."""

    player_labels: list[str] = field(default_factory=lambda: ["Player 1", "Player 2"])

    def validate(self) -> None:
        if self.max_age_frames < 1:
            raise ConfigError("tracking.max_age_frames must be >= 1")
        if self.min_hits < 1:
            raise ConfigError("tracking.min_hits must be >= 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ConfigError("tracking.iou_threshold must be in [0, 1]")
        if self.max_players < 1:
            raise ConfigError("tracking.max_players must be >= 1")


@dataclass
class EventConfig:
    """Thresholds the event logic works to.

    These are the knobs a person tunes after watching the annotated output
    disagree with them, so they live in the config rather than in constants.
    """

    min_confidence: float = 0.40
    """Events below this are dropped before the timeline is written."""

    possession_distance_px: float = 80.0
    """Ball within this of a player's feet counts as that player having it."""

    possession_min_frames: int = 3
    """Frames of proximity before possession is called, to reject a graze."""

    pass_min_distance_px: float = 120.0
    """A ball transfer shorter than this is a touch, not a pass."""

    pass_max_duration_s: float = 4.0
    """Ball in flight longer than this is not a pass between two players."""

    tackle_max_distance_px: float = 90.0
    """How close two players must be for a possession change to be a tackle."""

    tackle_window_s: float = 1.0
    """Possession must change within this long of the players converging."""

    shot_min_speed_px_s: float = 400.0
    """Ball leaving a player slower than this is a pass, not a shot."""

    shot_goal_cone_deg: float = 35.0
    """Ball direction must be within this of the goal to count as a shot."""

    goal_confirm_frames: int = 3
    """Frames the ball must stay inside the net box to confirm a goal."""

    trick_min_touches: int = 3
    """Touches inside the window before a sequence is considered a trick."""

    trick_window_s: float = 1.5
    """How long that burst of touches may span."""

    min_gap_s: float = 0.5
    """Two events of the same type closer than this are de-duplicated."""

    enabled_types: list[str] = field(
        default_factory=lambda: [
            "tackle", "trick", "pass", "shot", "goal", "push_past",
        ]
    )
    """Which event types reach the timeline.

    The five headline types, plus ``push_past``: a strict 1v1 has no teammate,
    so a deliberate knock beyond the defender to re-collect is a real event
    that would otherwise be mislabelled a pass.  The remaining supporting
    types (``ball_touch``, ``possession_change``, ``out_of_play``) are opt-in,
    being intermediate signal rather than something to report."""

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError("events.min_confidence must be in [0, 1]")
        for name in (
            "possession_distance_px", "pass_min_distance_px",
            "tackle_max_distance_px", "shot_min_speed_px_s",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"events.{name} must be positive")
        if not self.enabled_types:
            raise ConfigError("events.enabled_types must not be empty")


@dataclass
class OutputConfig:
    """What the run leaves behind."""

    json_path: str | None = None
    """Where the event timeline goes.  ``None`` prints it to stdout."""

    video_path: str | None = None
    """Where the annotated clip goes.  ``None`` skips rendering."""

    video_codec: str | None = None
    """Force a FourCC.  ``None`` picks one that works for the extension."""

    state_cache_path: str | None = None
    """Write the per-frame world-state record here, as JSON Lines.  Event logic
    can then be re-run against it in seconds instead of re-running inference."""

    draw_boxes: bool = True
    draw_labels: bool = True
    draw_tracks: bool = True
    draw_ball_trail: bool = True
    trail_length: int = 24
    """Ball positions kept in the trail, in processed frames."""

    draw_events: bool = True
    event_banner_seconds: float = 2.0
    """How long an event caption stays on screen after it fires."""

    draw_clock: bool = True
    json_indent: int | None = 2

    def validate(self) -> None:
        if self.trail_length < 0:
            raise ConfigError("output.trail_length must be >= 0")
        if self.event_banner_seconds < 0:
            raise ConfigError("output.event_banner_seconds must be >= 0")


@dataclass
class Config:
    """The whole resolved configuration for one run."""

    input_path: str | None = None
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    events: EventConfig = field(default_factory=EventConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    track: dict[str, Any] = field(default_factory=dict)
    """Settings for the tracking layer, passed to ``TrackLayerConfig``.

    Held as a plain mapping rather than a nested dataclass so this module stays
    free of imports from the layers it configures -- a partly-built checkout
    still loads its config. The keys are validated when the layer is built,
    where the real dataclass can reject an unknown one."""

    geometry: dict[str, Any] = field(default_factory=dict)
    """Settings for the geometry layer, passed to ``GeometryConfig``.

    ``goal_corners_px`` is the important one: the four goal-mouth corners,
    annotated once per camera setup, in the order left post base, right post
    base, right crossbar end, left crossbar end."""

    ball_events: dict[str, Any] = field(default_factory=dict)
    """Settings for the ball-event rules, layered over what ``events`` already says.

    The `events` section holds the keys the rules share with everything else.
    This one reaches the rest of ``BallEventConfig`` -- in particular
    ``pass_mode``, ``feeders`` and ``teams``, which decide what counts as a pass
    when there is no teammate to pass to."""

    pose: dict[str, Any] = field(default_factory=dict)
    """Settings for the pose stage, passed to ``PoseConfig``.

    ``weights`` names the pose checkpoint; a bare name is looked for under
    ``detection.weights_search_paths``, and pose is skipped with a warning
    rather than downloaded if it is not there."""

    body_events: dict[str, Any] = field(default_factory=dict)
    """Settings for the tackle and trick stage, passed to ``BodyEventConfig``.

    Dotted keys reach the nested sections, e.g. ``windows.contact_ball_h``."""

    log_level: str = "INFO"

    def validate(self) -> "Config":
        for name in ("track", "geometry", "pose", "body_events", "ball_events"):
            section = getattr(self, name)
            if not isinstance(section, dict):
                raise ConfigError(
                    f"{name} must be a mapping, got {type(section).__name__}"
                )
        self.video.validate()
        self.detection.validate()
        self.tracking.validate()
        self.events.validate()
        self.output.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        import yaml  # imported lazily so JSON-only users need no PyYAML

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Config":
        return _build(cls, payload or {})


# -- building ------------------------------------------------------------


def _build(target: type, payload: dict[str, Any]) -> Any:
    """Construct a nested dataclass from a plain dict, rejecting stray keys."""
    if not isinstance(payload, dict):
        raise ConfigError(
            f"expected a mapping for {target.__name__}, got {type(payload).__name__}"
        )

    known = {f.name: f for f in fields(target)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {target.__name__}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        field_type = known[name].type
        # Nested config sections are dataclasses with a default_factory.
        factory = known[name].default_factory  # type: ignore[misc]
        if factory is not None and factory is not _MISSING and isinstance(value, dict):
            default_instance = factory()
            if is_dataclass(default_instance) and not isinstance(default_instance, dict):
                kwargs[name] = _build(type(default_instance), value)
                continue
        if name == "class_map" and isinstance(value, dict):
            # YAML may give string keys for what are model class ids.
            kwargs[name] = {int(k): str(v) for k, v in value.items()}
            continue
        kwargs[name] = value
    return target(**kwargs)



def default_config() -> Config:
    """A fresh config with every default in place."""
    return Config()


def load_config(path: str | Path | None) -> Config:
    """Read a YAML or JSON config file, layering it over the defaults.

    ``None`` returns the defaults, so callers do not need to branch.
    """
    if path is None:
        return default_config()

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"no such config file: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            import yaml

            payload = yaml.safe_load(text)
        elif suffix == ".json":
            payload = json.loads(text)
        else:
            raise ConfigError(
                f"unsupported config format {suffix!r}; use .yaml, .yml or .json"
            )
    except ConfigError:
        raise
    except Exception as exc:  # pragma: no cover - depends on the broken file
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc

    if payload is None:
        return default_config()
    return Config.from_dict(payload).validate()


def merge_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """Apply dotted-path overrides in place, e.g. ``{"video.start_s": 12.0}``.

    This is how command-line flags beat the config file.  ``None`` values are
    ignored so an unset flag never clobbers a configured value.
    """
    for dotted, value in overrides.items():
        if value is None:
            continue
        target: Any = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ConfigError(f"unknown config section in override: {dotted!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ConfigError(f"unknown config key in override: {dotted!r}")
        setattr(target, leaf, value)
    return config.validate()
