"""Drawing detections, tracks and events onto frames."""

from football_analysis.viz.overlay import FrameAnnotator
from football_analysis.viz.palette import color_for_class, color_for_track

__all__ = ["FrameAnnotator", "color_for_class", "color_for_track"]
