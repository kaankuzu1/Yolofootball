"""Geometry and frame bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.types import BBox, Point, VideoFrame


def test_bbox_geometry():
    box = BBox(10, 20, 30, 60)
    assert box.width == 20
    assert box.height == 40
    assert box.area == 800
    assert box.center.as_tuple() == (20.0, 40.0)
    assert box.bottom_center.as_tuple() == (20.0, 60.0)


def test_bbox_rejects_inverted_corners():
    with pytest.raises(ValueError):
        BBox(30, 0, 10, 10)
    with pytest.raises(ValueError):
        BBox(0, 30, 10, 10)


def test_bbox_iou():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0
    # Half-overlap: intersection 50, union 150.
    assert a.iou(BBox(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_bbox_from_xywh_and_scaling():
    box = BBox.from_xywh(10, 10, 20, 40)
    assert box.as_tuple() == (10, 10, 30, 50)
    assert box.scaled(0.5).as_tuple() == (5, 5, 15, 25)


def test_point_distance():
    assert Point(0, 0).distance_to(Point(3, 4)) == pytest.approx(5.0)


def test_frame_maps_boxes_back_to_source_pixels():
    frame = VideoFrame(
        index=7,
        timestamp_s=0.28,
        image=np.zeros((180, 320, 3), np.uint8),
        source_size=(1280, 720),
        scale=0.25,
    )
    assert frame.size == (320, 180)
    assert frame.to_source_bbox(BBox(10, 10, 20, 20)).as_tuple() == (40, 40, 80, 80)
