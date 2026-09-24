"""The CLI has to actually load the model it was told to load.

This regressed once: --model set the config, nothing read it, and every run
quietly analysed motion blobs while reporting success. A run that looks like it
worked and is wrong about everything is the worst failure mode there is, so
these tests pin both halves -- the model gets built, and a fallback is loud.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from football_analysis.cli.main import main
from football_analysis.config import Config, ConfigError
from football_analysis.pipeline import default_detector, resolve_weights

WEIGHTS = "assets/models/forzasys_soccer.pt"

# Weights are not in git; a fresh checkout skips the tests that load them.
needs_weights = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / WEIGHTS).is_file(),
    reason=f"{WEIGHTS} is not on disk (see 'Model weights' in the README)",
)


def _config(model_path, **detection):
    config = Config()
    config.detection.model_path = model_path
    for key, value in detection.items():
        setattr(config.detection, key, value)
    return config.validate()


def test_weights_are_found_as_given(tmp_path):
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"not really a model")
    assert resolve_weights(_config(str(weights))) == weights


def test_a_bare_name_is_found_in_the_search_paths(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "custom.pt").write_bytes(b"not really a model")
    config = _config("custom.pt", weights_search_paths=[str(models)])

    assert resolve_weights(config) == models / "custom.pt"


def test_missing_weights_resolve_to_nothing(tmp_path):
    assert resolve_weights(_config(str(tmp_path / "absent.pt"))) is None
    assert resolve_weights(_config("")) is None


def test_an_explicit_path_is_not_searched_for_elsewhere(tmp_path):
    # A path the user spelled out is either right or wrong; quietly finding a
    # different file somewhere else would be worse than failing.
    models = tmp_path / "models"
    models.mkdir()
    (models / "custom.pt").write_bytes(b"x")
    config = _config("elsewhere/custom.pt", weights_search_paths=[str(models)])

    assert resolve_weights(config) is None


def test_missing_weights_fall_back_loudly(tmp_path, caplog):
    config = _config(str(tmp_path / "absent.pt"))
    with caplog.at_level("WARNING"):
        detector = default_detector(config)

    assert type(detector).__name__ == "StubDetector"
    assert "falling back to the placeholder" in caplog.text
    assert "absent.pt" in caplog.text


@needs_weights
def test_the_real_detector_is_built_when_the_weights_exist():
    detector = default_detector(_config(WEIGHTS, imgsz=640))
    payload = detector.describe()

    assert payload["name"] == "Detector"
    assert payload["weights"] == WEIGHTS
    assert payload["imgsz"] == 640


@needs_weights
def test_detection_settings_reach_the_backend():
    config = _config(WEIGHTS, confidence_threshold=0.4, iou_threshold=0.6)
    config.detection.class_confidence = {"ball": 0.05}
    raw = default_detector(config).detector

    assert raw.config.conf == pytest.approx(0.4)
    assert raw.config.iou == pytest.approx(0.6)
    # The ball is small and blurs, so its own threshold has to survive.
    assert raw.config.ball_conf == pytest.approx(0.05)


@needs_weights
def test_backend_specific_options_pass_through():
    config = _config(WEIGHTS)
    config.detection.options = {"tile_ball": True, "tile_imgsz": 512}
    raw = default_detector(config).detector

    assert raw.config.tile_ball is True
    assert raw.config.tile_imgsz == 512


@needs_weights
def test_an_unknown_detector_option_is_rejected_rather_than_ignored():
    config = _config(WEIGHTS)
    config.detection.options = {"no_such_knob": 1}
    with pytest.raises(ConfigError, match="detection setting"):
        default_detector(config)


@needs_weights
def test_goalkeepers_count_as_players_and_referees_do_not():
    adapter = default_detector(_config(WEIGHTS))
    assert adapter.label_map == {"goalkeeper": "player"}
    assert adapter.keep == {"player", "ball", "goal"}


@needs_weights
def test_the_cli_uses_the_model_it_was_given(sample_clip, tmp_path):
    # The regression itself: --model must reach the run, not just the config.
    path, _ = sample_clip
    out = tmp_path / "events.json"
    code = main([
        str(path), "-o", str(out), "--model", WEIGHTS, "--max-frames", "2",
        "--resize-width", "320", "--no-progress", "--quiet",
    ])

    assert code == 0
    stages = json.loads(out.read_text(encoding="utf-8"))["stats"]["stages"]
    assert stages["detector"]["name"] == "Detector"
    assert stages["detector"].get("placeholder") is None
    assert stages["detector"]["weights"] == WEIGHTS


def test_stdout_stays_parseable_while_a_backend_chatters(sample_clip, capsys):
    # Model backends print banners and deprecation notices to stdout. One stray
    # line and `football-analyse clip.mp4 | jq` stops working.
    path, _ = sample_clip
    code = main([
        str(path), "--model", WEIGHTS, "--max-frames", "2",
        "--resize-width", "320", "--no-progress", "--quiet",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"


# -- a stage that cannot be imported says so out loud --------------------------


def _hide_module(monkeypatch, prefix):
    """Make `import <prefix>...` raise, the way a missing dependency does."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == prefix or name.startswith(prefix + "."):
            raise ModuleNotFoundError(f"No module named 'scipy'", name="scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod in [m for m in list(sys.modules) if m == prefix or m.startswith(prefix + ".")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)


def test_a_missing_tracking_dependency_warns_by_name(monkeypatch, caplog):
    # This was a logger.debug: one absent dependency quietly downgraded every
    # run to IoU tracking, with no identities, no ball filter and no geometry.
    from football_analysis.config import Config
    from football_analysis.pipeline import default_tracker

    _hide_module(monkeypatch, "football_analysis.track")
    with caplog.at_level("WARNING"):
        tracker = default_tracker(Config().validate())

    assert type(tracker).__name__ == "GreedyIouTracker"
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("tracking layer" in m for m in warnings), warnings
    # Name the module, or the reader has no idea what to install.
    assert any("scipy" in m for m in warnings), warnings
    # And say what the run loses, not just that something is missing.
    assert any("identities" in m for m in warnings), warnings


def test_a_missing_rule_engine_warns_by_name(monkeypatch, caplog):
    from football_analysis.config import Config
    from football_analysis.pipeline import default_event_stage

    _hide_module(monkeypatch, "football_analysis.ball_events")
    with caplog.at_level("WARNING"):
        default_event_stage(Config().validate())

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("ball event rules" in m and "scipy" in m for m in warnings), warnings
