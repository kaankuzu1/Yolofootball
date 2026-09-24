"""Config defaults, file loading, layering and validation."""

from __future__ import annotations

import json

import pytest

from football_analysis.config import (
    Config,
    ConfigError,
    default_config,
    load_config,
    merge_overrides,
)


def test_defaults_are_valid_and_sensible():
    cfg = default_config().validate()
    assert cfg.video.frame_stride == 1
    assert cfg.tracking.max_players == 2  # it is a 1v1
    # "The ball is small, so it needs a lower bar than a player" was the
    # assumption here until the detection benchmark measured it: on the default
    # checkpoint, dropping the ball below 0.35 costs precision faster than it
    # buys recall. Every class floor still has to be a usable probability, and
    # an unknown class still falls back to the global one.
    for name in ("ball", "player", "goal"):
        assert 0.0 < cfg.detection.threshold_for(name) < 1.0
    assert cfg.detection.threshold_for("unheard_of") == cfg.detection.confidence_threshold


def test_load_yaml_layers_over_defaults(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "video:\n  start_s: 12.5\n  frame_stride: 2\n"
        "events:\n  min_confidence: 0.6\n",
        encoding="utf-8",
    )
    cfg = load_config(path)

    assert cfg.video.start_s == pytest.approx(12.5)
    assert cfg.video.frame_stride == 2
    assert cfg.events.min_confidence == pytest.approx(0.6)
    # Untouched sections keep their defaults.
    assert cfg.detection.model_path == default_config().detection.model_path


def test_load_json_config(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"detection": {"device": "cpu"}}), encoding="utf-8")
    assert load_config(path).detection.device == "cpu"


def test_load_config_none_gives_defaults():
    assert load_config(None).video.frame_stride == 1


def test_class_map_string_keys_become_ints(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "detection:\n  class_map:\n    0: player\n    1: ball\n    2: goal\n",
        encoding="utf-8",
    )
    assert load_config(path).detection.class_map == {0: "player", 1: "ball", 2: "goal"}


def test_unknown_keys_are_rejected_rather_than_ignored():
    # A silently ignored typo in a config file is a bug that takes an hour to find.
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"video": {"stride": 2}})


def test_missing_and_unsupported_config_files(tmp_path):
    with pytest.raises(ConfigError, match="no such config"):
        load_config(tmp_path / "absent.yaml")
    bad = tmp_path / "run.ini"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported config format"):
        load_config(bad)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"video": {"frame_stride": 0}}, "frame_stride"),
        ({"video": {"start_s": 10.0, "end_s": 5.0}}, "end_s"),
        ({"detection": {"confidence_threshold": 1.4}}, "confidence_threshold"),
        ({"events": {"enabled_types": []}}, "enabled_types"),
    ],
)
def test_validation_rejects_impossible_values(payload, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_dict(payload).validate()


def test_overrides_beat_the_file_and_skip_unset_flags():
    cfg = Config.from_dict({"video": {"start_s": 5.0}})
    merge_overrides(cfg, {"video.start_s": 9.0, "video.end_s": None, "output.json_path": "a.json"})

    assert cfg.video.start_s == pytest.approx(9.0)
    assert cfg.video.end_s is None  # a None override must not clobber
    assert cfg.output.json_path == "a.json"


def test_overrides_reject_unknown_paths():
    with pytest.raises(ConfigError, match="unknown config"):
        merge_overrides(Config(), {"video.nonexistent": 1})
    with pytest.raises(ConfigError, match="unknown config section"):
        merge_overrides(Config(), {"nosuchsection.key": 1})


def test_config_serialises_for_the_output_document():
    payload = default_config().to_dict()
    assert payload["tracking"]["max_players"] == 2
    assert "yaml" not in payload  # to_dict is plain data, not a rendering
    assert "video" in json.loads(default_config().to_json())


def test_push_past_is_reported_by_default():
    # A 1v1 has no teammate, so a knock past the defender to re-collect is its
    # own event; leaving it out of the defaults would silently lose it.
    enabled = default_config().events.enabled_types
    assert "push_past" in enabled
    assert {"tackle", "trick", "pass", "shot", "goal"} <= set(enabled)


def test_intermediate_signal_stays_opt_in():
    # ball_touch and friends are how the headline events are derived, not
    # something to report; a default run should not be full of them.
    enabled = set(default_config().events.enabled_types)
    assert not enabled & {"ball_touch", "possession_change", "out_of_play"}


def test_the_defaults_are_the_benchmark_s_operating_point():
    # These three are not arbitrary: docs/detection-report.md measured them at
    # 82.6% ball recall and 86.4% precision, and a plain `football-analyse
    # clip.mp4` should get that rather than stock COCO weights at 640.
    detection = default_config().detection
    assert detection.model_path == "forzasys_soccer.pt"
    assert detection.imgsz == 1280
    assert detection.threshold_for("ball") == 0.35
    # And the weights are findable by that bare name in a checkout.
    assert "assets/models" in detection.weights_search_paths


def test_ball_event_rules_are_reachable_from_a_config_file(tmp_path):
    # `pass_mode`, `feeders` and `teams` decide what counts as a pass when there
    # is no teammate to pass to, and were Python-only until this section existed.
    from football_analysis.pipeline import default_event_stage

    path = tmp_path / "run.yaml"
    path.write_text(
        "ball_events:\n"
        "  pass_mode: teams\n"
        "  feeders: [Feeder]\n"
        "  teams: {Feeder: home, Player 1: home, Player 2: away}\n"
    )
    cfg = load_config(path)
    assert cfg.ball_events["pass_mode"] == "teams"

    stage, _ = default_event_stage(cfg)
    described = stage.describe()
    stages = described.get("stages", [described])
    assert any(s.get("pass_mode") == "teams" for s in stages)


def test_the_ball_events_section_layers_over_the_shared_keys():
    # Setting one key must not throw away what `events` already said.
    from football_analysis.ball_events import BallEventConfig

    cfg = Config()
    cfg.events.possession_min_frames = 7
    shared = BallEventConfig.from_config(cfg)
    assert shared.possession_min_frames == 7

    cfg.ball_events = {"pass_mode": "any"}
    merged = BallEventConfig.from_dict({**shared.to_dict(), **cfg.ball_events})
    assert merged.pass_mode == "any"
    assert merged.possession_min_frames == 7


def test_a_misspelled_ball_events_key_is_rejected():
    from football_analysis.pipeline import default_event_stage

    cfg = Config()
    cfg.ball_events = {"pss_mode": "any"}
    with pytest.raises(ConfigError):
        default_event_stage(cfg.validate())


def test_a_feeder_needs_a_third_identity_from_the_tracking_section():
    # The config comment in configs/default.yaml promises this works; if the
    # tracker stops honouring max_players the promise becomes a lie.
    from football_analysis.track.config import TrackLayerConfig

    cfg = Config()
    cfg.tracking.max_players = 3
    cfg.tracking.player_labels = ["Player 1", "Player 2", "Feeder"]
    layer = TrackLayerConfig.from_config(cfg.validate())
    assert layer.players.num_identities == 3
    assert "Feeder" in layer.players.labels
