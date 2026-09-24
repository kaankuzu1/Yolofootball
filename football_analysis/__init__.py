"""Football 1v1 analysis: a clip in, a timestamped event timeline out.

    from football_analysis import analyze
    result = analyze("clip.mp4")
    print(result.timeline.to_json())

The package is built in stages, each replaceable:

``io``          read frames with trustworthy timestamps; write annotated clips
``config``      one object describing a whole run
``interfaces``  the contracts a detector, tracker and event detector satisfy
``detection``   the YOLO stage (built separately)
``events``      the output schema every stage writes into
``pipeline``    the loop that joins them
``viz``         boxes, trails and captions drawn on frames
``stubs``       placeholder stages so a run works before the real ones land
"""

from football_analysis.config import Config, load_config
from football_analysis.events import Event, EventTimeline, EventType, SCHEMA_VERSION
from football_analysis.pipeline import AnalysisPipeline, PipelineResult, analyze
from football_analysis.types import BBox, Detection, Track, VideoFrame

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "analyze",
    "AnalysisPipeline",
    "PipelineResult",
    "Config",
    "load_config",
    "Event",
    "EventType",
    "EventTimeline",
    "BBox",
    "Detection",
    "Track",
    "VideoFrame",
]
