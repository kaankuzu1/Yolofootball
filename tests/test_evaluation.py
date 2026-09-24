"""The evaluation harness: label format, matching, metrics, targets and the CLI."""

from __future__ import annotations

import json

import pytest

from football_analysis.evaluation import Label, LabelSet, draft_labels, load_labels, match, score, score_many
from football_analysis.evaluation.__main__ import main
from football_analysis.evaluation.labels import FORMAT, VERSION


def _labels(events, types=None, **extra) -> LabelSet:
    payload = {"format": FORMAT, "version": VERSION, "fps": 25.0, "events": events, **extra}
    if types is not None:
        payload["labelled_types"] = types
    return LabelSet.from_dict(payload)


def _ev(event_type, t, confidence=0.9, player=None):
    e = {"type": event_type, "timestamp_s": t, "confidence": confidence}
    if player:
        e["player_id"] = player
    return e


def test_label_times_parse_every_way():
    ls = _labels([
        {"type": "goal", "timestamp_s": 12.5},
        {"type": "goal", "clock": "01:02.500"},
        {"type": "goal", "frame_index": 250},
        {"type": "goal", "window_s": [3.0, 4.0]},
    ])
    assert [(lab.start_s, lab.end_s) for lab in ls.labels] == [(12.5, 12.5), (62.5, 62.5), (10.0, 10.0), (3.0, 4.0)]
    assert ls.labelled_types == ("goal",)


@pytest.mark.parametrize("bad", [
    {"type": "goal"},                                          # no time
    {"type": "goal", "timestamp_s": 1, "clock": "00:01"},      # two times
    {"type": "header", "timestamp_s": 1},                      # unknown type
    {"type": "goal", "window_s": [5, 4]},                      # backwards window
])
def test_bad_labels_are_refused(bad):
    with pytest.raises(ValueError):
        _labels([bad])


def test_labels_of_an_unlisted_type_are_refused():
    with pytest.raises(ValueError):
        _labels([{"type": "shot", "timestamp_s": 1}], types=["goal"])


def test_matching_is_one_to_one_and_maximal():
    labels = [Label("pass", 1.0, 1.0), Label("pass", 2.0, 2.0)]
    # Greedy-by-nearest would give 1.9 to the label at 2.0 and strand 2.8.
    assert sorted(match([1.9, 2.8], labels, 1.0)) == [(0, 0), (1, 1)]
    # Two events can't share one label.
    assert len(match([1.0, 1.1], labels[:1], 1.0)) == 1


def test_tolerance_is_inclusive_and_windows_count_as_zero_error():
    assert match([2.0], [Label("pass", 1.0, 1.0)], 1.0) == [(0, 0)]
    assert match([2.01], [Label("pass", 1.0, 1.0)], 1.0) == []
    assert Label("pass", 3.0, 5.0).distance_s(4.2) == 0.0
    assert match([5.9], [Label("pass", 3.0, 5.0)], 1.0) == [(0, 0)]


def test_right_missed_invented_and_metrics():
    labels = _labels([{"type": "shot", "timestamp_s": t} for t in (10.0, 20.0, 30.0, 40.0)])
    events = [_ev("shot", 10.3), _ev("shot", 19.5), _ev("shot", 30.2), _ev("shot", 55.0)]
    s = score(events, labels).types["shot"]
    assert (s.right, len(s.missed), len(s.invented)) == (3, 1, 1)
    assert s.precision == pytest.approx(0.75) and s.recall == pytest.approx(0.75)
    assert s.f1 == pytest.approx(0.75)
    assert s.target_status() == "not met"
    assert s.missed[0].start_s == 40.0 and s.invented[0]["timestamp_s"] == 55.0
    assert s.mean_abs_error_s == pytest.approx((0.3 + 0.5 + 0.2) / 3)
    # Tight tolerance: the shot 0.5 s off still counts, nothing beyond it would.
    assert s.tight is not None and s.tight.right == 3


def test_goal_target_needs_both_precision_and_recall():
    labels = _labels([{"type": "goal", "timestamp_s": float(t)} for t in range(0, 200, 10)])
    perfect = [_ev("goal", float(t)) for t in range(0, 200, 10)]
    assert score(perfect, labels).types["goal"].target_status() == "met"
    one_extra = perfect + [_ev("goal", 205.0)]
    s = score(one_extra, labels).types["goal"]
    assert s.precision == pytest.approx(20 / 21) and s.target_status() == "met"
    assert score(perfect[:18], labels).types["goal"].target_status() == "not met"


def test_ambiguous_labels_neither_reward_nor_punish():
    labels = _labels([
        {"type": "trick", "timestamp_s": 5.0},
        {"type": "trick", "timestamp_s": 9.0, "ambiguous": True},
        {"type": "trick", "timestamp_s": 14.0, "ambiguous": True},
    ])
    s = score([_ev("trick", 5.1), _ev("trick", 9.2)], labels).types["trick"]
    assert (s.labelled, s.right, len(s.missed), len(s.invented), s.ignored) == (1, 1, 0, 0, 1)
    assert s.precision == 1.0 and s.recall == 1.0


def test_only_labelled_types_and_span_are_scored():
    labels = _labels([{"type": "goal", "timestamp_s": 5.0}], types=["goal", "shot"], span_s=[0.0, 30.0])
    events = [_ev("goal", 5.0), _ev("tackle", 7.0), _ev("tackle", 8.0), _ev("shot", 45.0)]
    report = score(events, labels)
    assert set(report.types) == {"goal", "shot"}
    assert report.unscored == {"tackle": 2}
    assert report.types["shot"].predicted == 0  # 45 s is outside the labelled span


def test_nothing_found_scores_zero_not_blank():
    s = score([], _labels([{"type": "pass", "timestamp_s": 1.0}])).types["pass"]
    assert s.recall == 0.0 and s.f1 == 0.0 and s.ap == 0.0 and s.precision is None


def test_average_precision_rewards_confident_right_answers():
    labels = _labels([{"type": "pass", "timestamp_s": t} for t in (1.0, 5.0)])
    good_first = [_ev("pass", 1.0, 0.9), _ev("pass", 5.0, 0.8), _ev("pass", 9.0, 0.1)]
    bad_first = [_ev("pass", 1.0, 0.5), _ev("pass", 5.0, 0.4), _ev("pass", 9.0, 0.9)]
    assert score(good_first, labels).types["pass"].ap == pytest.approx(1.0)
    assert score(bad_first, labels).types["pass"].ap == pytest.approx(2 / 3)


def test_type_confusions_and_player_agreement_are_reported():
    labels = _labels([
        {"type": "pass", "timestamp_s": 3.0, "player_id": "Player 1"},
        {"type": "tackle", "timestamp_s": 8.0},
    ])
    events = [_ev("pass", 3.1, player="Player 2"), _ev("pass", 8.2)]
    report = score(events, labels)
    assert report.confusions == [
        {"predicted_type": "pass", "predicted_s": 8.2, "label_type": "tackle", "label_s": 8.0}
    ]
    assert (report.types["pass"].player_checked, report.types["pass"].player_same) == (1, 0)
    md = report.to_markdown()
    assert "invented pass at 8.20 s" in md and "missed tackle at 8.00 s" in md


def test_clips_pool_without_matching_across_them():
    a = _labels([{"type": "shot", "timestamp_s": 10.0}], clip="a.mp4")
    b = _labels([{"type": "shot", "timestamp_s": 50.0}], clip="b.mp4")
    report = score_many([([_ev("shot", 10.0)], a), ([_ev("shot", 10.0)], b)])
    s = report.types["shot"]
    assert (s.labelled, s.right, len(s.missed), len(s.invented)) == (2, 1, 1, 1)
    assert report.clips == ["a.mp4", "b.mp4"]


def test_draft_round_trips_and_marks_labels_unchecked(tmp_path):
    events = [_ev("shot", 4.0, player="Player 1"), _ev("possession_change", 5.0), _ev("goal", 4.4)]
    drafted = draft_labels(events, clip="clip.mp4", fps=25.0)
    path = drafted.write_json(tmp_path / "clip.labels.json")
    back = load_labels(path)
    assert [(lab.type, lab.start_s) for lab in back.labels] == [("shot", 4.0), ("goal", 4.4)]
    assert all(lab.note.startswith("unchecked") for lab in back.labels)
    assert score(events, back).types["shot"].f1 == 1.0


def test_cli_scores_and_writes_json(tmp_path, capsys):
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps({"events": [_ev("goal", 12.0), _ev("goal", 30.0)]}))
    labels_path = _labels([{"type": "goal", "clock": "00:12.300"}], clip="c.mp4").write_json(tmp_path / "c.labels.json")
    out = tmp_path / "score.json"
    assert main(["score", str(events_path), str(labels_path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| goal | 1 | 2 | 1 | 0 | 1 |" in printed
    report = json.loads(out.read_text())
    assert report["types"]["goal"]["precision"] == 0.5
    assert report["types"]["goal"]["target_status"] == "not met"
    assert main(["check", str(labels_path)]) == 0
