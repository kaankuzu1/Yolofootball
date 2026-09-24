"""Configuration for an analysis run.

One :class:`Config` object describes everything a run does: which segment of
which clip, how big the frames are, which model detects what, how the tracker
links detections, the thresholds the event logic uses, and what gets written
out.  It is built from defaults, optionally overlaid with a YAML or JSON file,
then optionally overlaid with command-line overrides -- in that order, so the
most specific thing wins.

The resolved config is copied into the output timeline, so any result can be
traced back to the settings that produced it.
"""

from football_analysis.config.schema import (
    Config,
    VideoConfig,
    DetectionConfig,
    TrackingConfig,
    EventConfig,
    OutputConfig,
    ConfigError,
    load_config,
    merge_overrides,
    default_config,
    DEFAULT_CLASS_MAP,
)

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
