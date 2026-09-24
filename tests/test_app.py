"""The desktop app's logic that does not need a camera or a screen."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from football_analysis.app.cameras import (  # noqa: E402
    OPENCV,
    CameraInfo,
    choose_format,
    order_cameras,
    pick_camera,
)
from football_analysis.app.export import write_csv  # noqa: E402
from football_analysis.app.recorder import PacedRecorder  # noqa: E402
from football_analysis.app.storage import (  # noqa: E402
    GoalSetup,
    goal_path,
    load_goal,
    new_recording_path,
    processed_corners,
    save_goal,
)
from football_analysis.app.theme import event_name  # noqa: E402
from football_analysis.app.workers import (  # noqa: E402
    AnalysisRequest,
    LiveBox,
    build_config,
    mark_possessor,
)
from football_analysis.events import Event, EventType  # noqa: E402


# -- cameras -------------------------------------------------------------------

def test_built_in_camera_comes_first():
    usb = CameraInfo("usb", "Logitech BRIO")
    iphone = CameraInfo("iphone", "Kaan's iPhone Camera", position="back")
    facetime = CameraInfo("ft", "FaceTime HD Camera", is_default=True, position="front")
    assert [c.id for c in order_cameras([usb, iphone, facetime])] == ["ft", "iphone", "usb"]


def test_front_camera_beats_others_when_nothing_is_default():
    usb = CameraInfo("usb", "A USB camera")
    front = CameraInfo("front", "Z built-in", position="front")
    assert order_cameras([usb, front])[0].id == "front"


def test_remembered_camera_is_picked_while_connected():
    cams = [CameraInfo("ft", "FaceTime", is_default=True), CameraInfo("usb", "USB")]
    assert pick_camera(cams, "usb").id == "usb"
    assert pick_camera(cams, "unplugged").id == "ft"
    assert pick_camera([], "usb") is None


def test_label_names_the_default_front_camera():
    cam = CameraInfo("ft", "FaceTime HD Camera", is_default=True, position="front")
    assert cam.label == "FaceTime HD Camera (varsayılan, ön kamera)"
    assert CameraInfo("x", "USB", backend=OPENCV).label == "USB"


def test_format_choice_prefers_frame_rate_then_size():
    formats = [
        (1920, 1080, 1.0, 15.0),   # right size, too slow
        (1280, 720, 1.0, 30.0),
        (640, 480, 1.0, 30.0),
        (1920, 1080, 1.0, 30.0),
    ]
    assert choose_format(formats, (1920, 1080)) == 3
    assert choose_format(formats, (1280, 720)) == 1
    assert choose_format([(1920, 1080, 1.0, 15.0)], (1280, 720)) == 0
    assert choose_format([], (1280, 720)) is None


# -- recording -----------------------------------------------------------------

def _frame(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def test_recorder_places_frames_by_arrival_time(tmp_path):
    rec = PacedRecorder(tmp_path / "a.avi", fps=10.0)
    rec.write(_frame(0), 100.00)   # frame 0
    rec.write(_frame(1), 100.10)   # frame 1
    rec.write(_frame(2), 100.13)   # early: still frame 1's slot, dropped
    rec.write(_frame(3), 100.45)   # late: fills 2, 3, 4
    assert rec.frames_written == 5
    assert rec.frames_dropped == 1
    assert rec.frames_repeated == 2
    assert rec.duration_s == pytest.approx(0.5)
    assert rec.close() == tmp_path / "a.avi"


def test_recorder_caps_a_stalled_camera(tmp_path):
    rec = PacedRecorder(tmp_path / "b.avi", fps=10.0, max_fill_s=1.0)
    rec.write(_frame(0), 0.0)
    rec.write(_frame(1), 30.0)     # 30 s stall: at most ~1 s of frozen picture
    assert rec.frames_written <= 12
    rec.close()


def test_recorder_with_no_frames_reports_nothing(tmp_path):
    assert PacedRecorder(tmp_path / "c.avi", fps=30.0).close() is None


# -- goal and files ------------------------------------------------------------

def test_goal_round_trips_beside_the_clip(tmp_path):
    clip = tmp_path / "2026-09-24 18.05.12.mp4"
    goal = GoalSetup([[10, 200], [110, 200], [110, 150], [10, 150]], (640, 360), (5.0, 2.0))
    save_goal(clip, goal)
    assert goal_path(clip).name == "2026-09-24 18.05.12.goal.json"
    loaded = load_goal(clip)
    assert loaded.corners == goal.corners
    assert loaded.frame_size == (640, 360)
    assert loaded.size_m == (5.0, 2.0)


def test_goal_file_from_the_corner_script_is_read_at_clip_size(tmp_path):
    clip = tmp_path / "match.mp4"
    goal_path(clip).write_text(json.dumps({"corners": [[1, 2], [3, 4], [5, 6], [7, 8]]}))
    assert load_goal(clip) is None                 # no size known: refuse to guess
    assert load_goal(clip, (1920, 1080)).frame_size == (1920, 1080)


def test_goal_scales_between_resolutions():
    goal = GoalSetup([[640, 360], [1280, 720], [0, 0], [100, 50]], (1280, 720))
    assert goal.scaled_to((1920, 1080))[0] == [960.0, 540.0]
    assert processed_corners([[1920, 1080]], (1920, 1080), 1280) == [[1280.0, 720.0]]
    assert processed_corners([[5, 5]], (640, 360), None) == [[5, 5]]


def test_new_recording_never_overwrites(tmp_path):
    from datetime import datetime

    now = datetime(2026, 9, 24, 18, 5, 12)
    first = new_recording_path(tmp_path, now)
    first.write_bytes(b"x")
    second = new_recording_path(tmp_path, now)
    assert first.name == "2026-09-24 18.05.12.mp4"
    assert second.name == "2026-09-24 18.05.12 (2).mp4"


# -- analysis and export -------------------------------------------------------

def test_analysis_config_carries_goal_outputs_and_stride(tmp_path):
    request = AnalysisRequest(
        clip=tmp_path / "c.mp4", events_json=tmp_path / "c.events.json",
        annotated=None, states=tmp_path / "c.states.jsonl",
        goal_corners_px=[[1, 2], [3, 4], [5, 6], [7, 8]], goal_size_m=(7.32, 2.44),
        frame_stride=2,
    )
    config = build_config(request)
    assert config.output.json_path == str(tmp_path / "c.events.json")
    assert config.output.video_path is None
    assert config.video.frame_stride == 2
    assert config.geometry["goal_corners_px"][3] == [7.0, 8.0]
    assert config.geometry["goal_width_m"] == 7.32


def test_turkish_event_names():
    assert event_name("goal") == "Gol"
    assert event_name("trick", {"trick_name": "nutmeg"}) == "Çalım (tünel)"
    assert event_name("shot", {"on_target": True}) == "Şut (isabetli)"
    assert event_name("something_new") == "Something new"


def test_csv_opens_in_turkish_excel(tmp_path):
    events = [
        Event(EventType.GOAL, 12.5, 0.9, player_id="Player 1"),
        Event(EventType.TACKLE, 3.0, 0.7, player_id="Player 2", secondary_player_id="Player 1",
              detail={"needs_review": True, "review_reason": "ball hidden"}),
    ]
    path = write_csv(tmp_path / "e.csv", events)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")        # BOM so Excel reads UTF-8
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig"), delimiter=";"))
    assert [r["olay"] for r in rows] == ["Top kapma", "Gol"]   # time order
    assert rows[0]["kontrol_et"] == "evet"
    assert rows[1]["zaman"] == "00:12.500"


# -- live possession mark ------------------------------------------------------

def test_player_with_ball_at_feet_is_marked():
    near = LiveBox("player", (100, 100, 140, 200), 0.9, 1)
    far = LiveBox("player", (400, 100, 440, 200), 0.9, 2)
    ball = LiveBox("ball", (135, 190, 145, 200), 0.8)
    assert mark_possessor([near, far, ball]) is near
    assert near.has_ball and not far.has_ball


def test_loose_ball_marks_nobody():
    player = LiveBox("player", (100, 100, 140, 200), 0.9, 1)
    ball = LiveBox("ball", (300, 190, 310, 200), 0.8)
    assert mark_possessor([player, ball]) is None
    assert mark_possessor([player]) is None
