"""Reading clips in and writing annotated clips out."""

from football_analysis.io.video_reader import VideoReader, VideoSourceError
from football_analysis.io.video_writer import AnnotatedVideoWriter, VideoWriteError

__all__ = [
    "VideoReader",
    "VideoSourceError",
    "AnnotatedVideoWriter",
    "VideoWriteError",
]
