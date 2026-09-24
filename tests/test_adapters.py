"""The seam between a foreign detector and this package's contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from football_analysis.adapters import FrameDetectorAdapter, adapt_detector
from football_analysis.interfaces import Detector
from football_analysis.types import BBox, Detection, VideoFrame


@dataclass(frozen=True)
class ForeignDetection:
    """The shape the detection module returns: label, confidence, xyxy."""

    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class ForeignDetector:
    """Takes raw pixels, not a VideoFrame -- the mismatch this adapter exists for."""

    def __init__(self, records):
        self.records = records
        self.seen = []

    def detect(self, frame: np.ndarray):
        self.seen.append(frame.shape)
        return self.records


def _frame(width=320, height=180):
    return VideoFrame(
        index=0,
        timestamp_s=0.0,
        image=np.zeros((height, width, 3), np.uint8),
        source_size=(width, height),
    )


def test_adapter_satisfies_the_detector_protocol():
    adapter = FrameDetectorAdapter(ForeignDetector([]))
    assert isinstance(adapter, Detector)


def test_records_are_converted_and_pixels_are_passed_through():
    inner = ForeignDetector([
        ForeignDetection("player", 0.91, (10, 20, 50, 140)),
        ForeignDetection("ball", 0.33, (100, 120, 112, 132)),
    ])
    out = FrameDetectorAdapter(inner).detect(_frame())

    assert inner.seen == [(180, 320, 3)], "the detector gets raw pixels"
    assert [d.class_name for d in out] == ["player", "ball"]
    assert isinstance(out[0].bbox, BBox)
    assert out[0].bbox.as_tuple() == (10, 20, 50, 140)
    assert out[1].confidence == pytest.approx(0.33)


def test_boxes_are_clamped_to_the_frame():
    inner = ForeignDetector([ForeignDetection("player", 0.9, (-20, -5, 400, 300))])
    box = FrameDetectorAdapter(inner).detect(_frame())[0].bbox
    assert box.as_tuple() == (0.0, 0.0, 320.0, 180.0)


def test_inverted_corners_are_repaired_not_raised_on():
    inner = ForeignDetector([ForeignDetection("ball", 0.5, (80, 90, 40, 30))])
    box = FrameDetectorAdapter(inner).detect(_frame())[0].bbox
    assert box.as_tuple() == (40.0, 30.0, 80.0, 90.0)


@pytest.mark.parametrize(
    "xyxy",
    [
        (50, 50, 50, 50),                  # zero area
        (50, 50, 50.4, 90),                # sub-pixel width
        (float("nan"), 0, 10, 10),         # not a number
        None,                              # missing entirely
    ],
)
def test_unusable_boxes_are_dropped_and_counted(xyxy):
    # A zero-area box downstream becomes a division by zero, so it must not pass.
    inner = ForeignDetector([ForeignDetection("ball", 0.5, xyxy)])
    adapter = FrameDetectorAdapter(inner)

    assert adapter.detect(_frame()) == []
    assert adapter.dropped == 1


def test_labels_can_be_remapped_and_filtered():
    inner = ForeignDetector([
        ForeignDetection("goalkeeper", 0.8, (10, 10, 40, 100)),
        ForeignDetection("referee", 0.7, (60, 10, 90, 100)),
        ForeignDetection("ball", 0.4, (150, 150, 162, 162)),
    ])
    adapter = FrameDetectorAdapter(
        inner, label_map={"goalkeeper": "player"}, keep=["player", "ball"]
    )
    out = adapter.detect(_frame())

    assert [d.class_name for d in out] == ["player", "ball"]  # the referee is gone


def test_dict_records_work_too():
    inner = ForeignDetector([
        {"label": "ball", "confidence": 0.6, "xyxy": (10, 10, 30, 30)}
    ])
    out = FrameDetectorAdapter(inner).detect(_frame())
    assert out[0].class_name == "ball"


def test_describe_reports_the_wrapped_detector():
    class Configured(ForeignDetector):
        config = type("C", (), {"weights": "yolo11n.pt", "imgsz": 1280})()

    payload = FrameDetectorAdapter(Configured([])).describe()
    assert payload["name"] == "Configured"
    assert payload["weights"] == "yolo11n.pt"


def test_a_native_detector_is_left_alone():
    class Native:
        def detect(self, frame: VideoFrame) -> list[Detection]:
            return []

    native = Native()
    assert adapt_detector(native) is native


def test_a_foreign_detector_gets_wrapped():
    wrapped = adapt_detector(ForeignDetector([]))
    assert isinstance(wrapped, FrameDetectorAdapter)
    assert isinstance(wrapped, Detector)


def test_something_that_is_not_a_detector_is_rejected():
    with pytest.raises(TypeError, match="detect"):
        adapt_detector(object())
