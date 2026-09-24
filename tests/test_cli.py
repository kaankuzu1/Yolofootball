"""The command line: flag parsing, layering, exit codes and output."""

from __future__ import annotations

import json

import pytest

from football_analysis.cli.main import build_parser, main
from football_analysis.cli.main import _resolve_config


def _args(*argv):
    return build_parser().parse_args(list(argv))


def test_flags_land_on_the_config():
    config = _resolve_config(
        _args("clip.mp4", "--start", "12", "--end", "30", "--stride", "2",
              "--resize-width", "960", "--confidence", "0.4")
    )
    assert config.input_path == "clip.mp4"
    assert config.video.start_s == pytest.approx(12.0)
    assert config.video.end_s == pytest.approx(30.0)
    assert config.video.frame_stride == 2
    assert config.video.resize_width == 960
    assert config.detection.confidence_threshold == pytest.approx(0.4)


def test_flags_beat_the_config_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("video:\n  start_s: 5.0\n  frame_stride: 4\n", encoding="utf-8")
    config = _resolve_config(_args("clip.mp4", "-c", str(path), "--start", "20"))

    assert config.video.start_s == pytest.approx(20.0)   # the flag wins
    assert config.video.frame_stride == 4                # the file still applies


def test_no_resize_clears_the_configured_size(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("video:\n  resize_width: 1280\n", encoding="utf-8")
    config = _resolve_config(_args("clip.mp4", "-c", str(path), "--no-resize"))
    assert config.video.resize_width is None


def test_event_list_is_parsed():
    config = _resolve_config(_args("clip.mp4", "--events", "goal, shot ,tackle"))
    assert config.events.enabled_types == ["goal", "shot", "tackle"]


def test_output_paths_reach_the_config():
    config = _resolve_config(_args("clip.mp4", "-o", "e.json", "--render", "a.mp4"))
    assert config.output.json_path == "e.json"
    assert config.output.video_path == "a.mp4"


def test_print_config_exits_cleanly_without_a_clip(capsys):
    assert main(["--print-config"]) == 0
    assert "resize_width" in capsys.readouterr().out


def test_an_unknown_event_type_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["clip.mp4", "--events", "backflip"])
    assert excinfo.value.code == 2
    assert "unknown event type" in capsys.readouterr().err


def test_contradictory_resize_flags_are_rejected(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["clip.mp4", "--no-resize", "--resize-width", "640"])
    assert excinfo.value.code == 2


def test_missing_input_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_a_missing_clip_fails_without_a_traceback(tmp_path):
    assert main([str(tmp_path / "absent.mp4"), "--quiet"]) == 1


def test_full_run_writes_the_timeline(sample_clip, tmp_path):
    path, _ = sample_clip
    out = tmp_path / "events.json"
    code = main([
        str(path), "-o", str(out), "--resize-width", "320",
        "--events", "ball_touch,possession_change",
        "--min-event-confidence", "0.1", "--no-progress", "--quiet",
    ])

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["stats"]["frames_processed"] > 0
    assert payload["source"]["processed_width"] == 320


def test_full_run_renders_a_clip(sample_clip, tmp_path):
    path, _ = sample_clip
    rendered = tmp_path / "annotated.mp4"
    code = main([
        str(path), "-o", str(tmp_path / "e.json"), "--render", str(rendered),
        "--resize-width", "320", "--end", "2", "--no-progress", "--quiet",
    ])

    assert code == 0
    assert rendered.exists() and rendered.stat().st_size > 0


def test_timeline_goes_to_stdout_when_no_output_path(sample_clip, capsys):
    path, _ = sample_clip
    code = main([str(path), "--max-frames", "5", "--no-progress", "--quiet"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"


def test_state_cache_flag_reaches_the_config():
    config = _resolve_config(_args("clip.mp4", "--state-cache", "state.jsonl"))
    assert config.output.state_cache_path == "state.jsonl"


def test_replay_with_a_clip_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["clip.mp4", "--replay", "state.jsonl"])
    assert excinfo.value.code == 2


def test_replay_cannot_render_since_there_is_no_video():
    with pytest.raises(SystemExit) as excinfo:
        main(["--replay", "state.jsonl", "--render", "out.mp4"])
    assert excinfo.value.code == 2


def test_a_missing_cache_fails_without_a_traceback(tmp_path):
    assert main(["--replay", str(tmp_path / "absent.jsonl"), "--quiet"]) == 1


def test_run_then_replay_from_the_cli(sample_clip, tmp_path):
    path, _ = sample_clip
    cache = tmp_path / "state.jsonl"
    live = tmp_path / "live.json"
    again = tmp_path / "again.json"
    common = ["--events", "ball_touch,possession_change",
              "--min-event-confidence", "0.3", "--no-progress", "--quiet"]

    assert main([str(path), "-o", str(live), "--resize-width", "320",
                 "--state-cache", str(cache)] + common) == 0
    assert cache.exists()

    # The payoff: the same events, from the record, with no clip involved.
    assert main(["--replay", str(cache), "-o", str(again)] + common) == 0
    live_events = json.loads(live.read_text(encoding="utf-8"))["events"]
    again_events = json.loads(again.read_text(encoding="utf-8"))["events"]
    assert [e["type"] for e in again_events] == [e["type"] for e in live_events]
    assert [e["clock"] for e in again_events] == [e["clock"] for e in live_events]


def test_goal_corners_are_parsed_into_four_points():
    config = _resolve_config(
        _args("--goal-corners", "100,300, 400,300, 400,120, 100,120",
              "--goal-corners-space", "processed")
    )
    assert config.geometry["goal_corners_px"] == [
        [100.0, 300.0], [400.0, 300.0], [400.0, 120.0], [100.0, 120.0]
    ]


def test_goal_corners_are_rescaled_from_the_clips_own_resolution(sample_clip):
    # People read corners off their footage, which is the source size; the
    # pipeline works on resized frames. Mixing the two misplaces the goal.
    path, truth = sample_clip
    source_w = truth.size[0]
    config = _resolve_config(
        _args(str(path), "--resize-width", "320",
              "--goal-corners", f"{source_w},100, {source_w},200, 0,200, 0,100")
    )
    scale = 320 / source_w
    corners = config.geometry["goal_corners_px"]
    assert corners[0][0] == pytest.approx(source_w * scale)
    assert corners[0][1] == pytest.approx(100 * scale)


def test_processed_space_corners_are_left_alone(sample_clip):
    path, _ = sample_clip
    config = _resolve_config(
        _args(str(path), "--resize-width", "320", "--goal-corners-space", "processed",
              "--goal-corners", "10,20,30,40,50,60,70,80")
    )
    assert config.geometry["goal_corners_px"][0] == [10.0, 20.0]


@pytest.mark.parametrize("text", ["1,2,3,4", "1,2,3,4,5,6,7,8,9", "a,b,c,d,e,f,g,h"])
def test_bad_goal_corners_are_a_usage_error(text, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["clip.mp4", "--goal-corners", text])
    assert excinfo.value.code == 2


def test_goal_size_reaches_the_config():
    config = _resolve_config(_args("clip.mp4", "--goal-size", "7.32", "2.44"))
    assert config.geometry["goal_width_m"] == pytest.approx(7.32)
    assert config.geometry["goal_height_m"] == pytest.approx(2.44)


def test_goal_settings_default_to_absent():
    # An empty section means "the layer's own defaults", not "override with none".
    assert _resolve_config(_args("clip.mp4")).geometry == {}
