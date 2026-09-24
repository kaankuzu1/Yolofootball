"""Command line for the tackle and trick stage.

    python -m football_analysis.body_events train
    python -m football_analysis.body_events evaluate
    python -m football_analysis.body_events propose CLIP --states STATES.jsonl
    python -m football_analysis.body_events label WINDOW_ID LABEL [--sub-label NAME]
    python -m football_analysis.body_events detect --states STATES.jsonl

The loop for adding real footage: run the pipeline on the clip with
``--state-cache`` to get STATES.jsonl, ``propose`` to cut candidate windows
and review sheets, watch them and ``label`` each (or fill the ``label``
column of ``assets/body_events/labels.csv``), then ``train`` again.  Real
windows are weighted above simulated ones, and ``evaluate`` reports on the
real ones separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "assets" / "body_events"


def _synthetic_samples(per_scenario: int, seed: int, workers: int):
    from football_analysis.body_events.dataset import synthetic_samples
    from football_analysis.body_events.synthetic import generate, seeds_for

    pairs = seeds_for(per_scenario, seed)
    if workers <= 1:
        return [s for a in pairs for s in synthetic_samples(generate(*a))]
    from multiprocessing import get_context

    # Generation only: no OpenMP has run yet in this process, so forking is safe.
    with get_context("fork").Pool(workers) as pool:
        return [s for r in pool.map(_gen_job, pairs, chunksize=8) for s in r]


def _gen_job(pair):
    from football_analysis.body_events.dataset import synthetic_samples
    from football_analysis.body_events.synthetic import generate

    return synthetic_samples(generate(*pair))


def cmd_train(args: argparse.Namespace) -> int:
    from football_analysis.body_events.dataset import WindowDataset
    from football_analysis.body_events.model import BodyEventModel, default_model_path

    t0 = time.time()
    samples = _synthetic_samples(args.per_scenario, args.seed, args.workers)
    real = WindowDataset(args.dataset).samples()
    print(f"{len(samples)} simulated windows, {len(real)} real labelled windows "
          f"({time.time() - t0:.0f}s)", file=sys.stderr)
    model = BodyEventModel.train(samples + real, real_weight=args.real_weight, seed=args.seed,
                                 meta={"synthetic_per_scenario": args.per_scenario,
                                       "synthetic_seed": args.seed,
                                       "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out = Path(args.out) if args.out else default_model_path()
    model.save(out)
    print(f"saved {out}", file=sys.stderr)
    print(json.dumps(model.meta["trained_on"], indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from football_analysis.body_events.dataset import WindowDataset
    from football_analysis.body_events.detector import BodyEventConfig
    from football_analysis.body_events.evaluate import evaluate_synthetic
    from football_analysis.body_events.model import BodyEventModel, default_model_path
    from football_analysis.body_events.synthetic import HARD_SENSOR, generate, seeds_for

    model = BodyEventModel.load(args.model or default_model_path())
    cfg = BodyEventConfig(trick_threshold=args.trick_threshold, tackle_threshold=args.tackle_threshold)
    report: dict = {"model": model.meta, "config": {"trick_threshold": cfg.trick_threshold,
                                                    "tackle_threshold": cfg.tackle_threshold}}
    # Held-out simulated clips: seeds disjoint from any training seed.
    for name, sensor in (("synthetic_heldout", None), ("synthetic_hard_sensor", HARD_SENSOR)):
        seqs = [generate(n, s, sensor=sensor) for n, s in seeds_for(args.per_scenario, args.seed)]
        rep, _ = evaluate_synthetic(model, seqs, cfg)
        report[name] = rep
    real = WindowDataset(args.dataset).samples()
    if real:
        report["real_windows"] = _window_report(model, real)
    text = json.dumps(report, indent=2, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def _window_report(model, samples) -> dict:
    """Window-level results on real labelled windows, per head.  Only
    meaningful once there are a few dozen; the counts are reported so a
    reader can see how little a number rests on."""
    out = {}
    for kind, head in (("contact", model.contact), ("skill", model.skill)):
        rows = [s for s in samples if s.kind == kind]
        if not rows:
            continue
        p = head.predict_proba([s.features for s in rows])
        pred = [head.classes[i] for i in np.argmax(p, axis=1)]
        truth = [s.label for s in rows]
        out[kind] = {
            "windows": len(rows),
            "accuracy": round(float(np.mean([a == b for a, b in zip(pred, truth)])), 3),
            "confusion": {t: {c: sum(1 for a, b in zip(truth, pred) if a == t and b == c)
                              for c in head.classes} for t in sorted(set(truth))},
        }
    return out


def cmd_propose(args: argparse.Namespace) -> int:
    """Cut candidate windows from a clip's records for someone to label."""
    from football_analysis.body_events.dataset import WindowDataset
    from football_analysis.body_events.features import WindowConfig, contact_windows, skill_windows
    from football_analysis.body_events.review import render_review
    from football_analysis.body_events.series import ClipSeries
    from football_analysis.state import read_state_cache

    states = list(read_state_cache(args.states))
    clip = ClipSeries.from_states(states)
    ds = WindowDataset(args.dataset)
    stem = Path(args.clip).stem
    existing = {r["window_id"] for r in ds.rows()}
    cfg = WindowConfig()
    proposals = []
    for w in contact_windows(clip, cfg):
        proposals.append(("contact", w.attacker_id, w.defender_id, w.start, w.end, w.anchor))
    # Skill windows overlap heavily; one per 0.6 s of each spell is plenty to label.
    last_start: dict[str, float] = {}
    for w in skill_windows(clip, cfg):
        t = float(clip.t[w.start])
        if t - last_start.get(w.player_id, -9.0) >= args.skill_every:
            proposals.append(("skill", w.player_id, w.opponent_id or "", w.start, w.end, None))
            last_start[w.player_id] = t
    print(f"{len(proposals)} candidate windows", file=sys.stderr)
    added = 0
    for kind, pid, other, a, b, anchor in proposals:
        t0, t1 = float(clip.t[a]), float(clip.t[b])
        wid = f"{stem}_{kind[0]}_{t0:07.2f}_{pid}".replace(" ", "")
        if wid in existing:
            continue
        keep = [s for s in states if t0 - 0.5 <= s.timestamp_s <= t1 + 0.5]
        row = {"kind": kind, "label": "", "source_clip": str(args.clip), "start_s": round(t0, 3),
               "end_s": round(t1, 3), "anchor_s": round(float(clip.t[anchor]), 3) if anchor is not None else "",
               "player_id": pid, "other_player_id": other, "origin": "real"}
        ds.add_window(wid, keep, row)
        if not args.no_review:
            render_review(args.clip, keep, [pid] + ([other] if other else []), t0, t1,
                          ds.root / "review" / wid, title=f"{wid}  ({kind})",
                          anchor_s=float(clip.t[anchor]) if anchor is not None else None)
        added += 1
    print(f"added {added} windows to {ds.labels_path}; review sheets in {ds.root / 'review'}",
          file=sys.stderr)
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    from football_analysis.body_events.dataset import CONTACT_LABELS, SKILL_LABELS, WindowDataset

    ds = WindowDataset(args.dataset)
    rows = ds.rows()
    for r in rows:
        if r["window_id"] == args.window_id:
            valid = (CONTACT_LABELS if r["kind"] == "contact" else SKILL_LABELS) + ("skip",)
            if args.label not in valid:
                print(f"{args.label!r} is not one of {valid}", file=sys.stderr)
                return 2
            r["label"] = args.label
            if args.sub_label is not None:
                r["sub_label"] = args.sub_label
            if args.notes is not None:
                r["notes"] = args.notes
            r["labeller"] = args.labeller
            ds.write_rows(rows)
            return 0
    print(f"no window {args.window_id!r}", file=sys.stderr)
    return 1


def cmd_detect(args: argparse.Namespace) -> int:
    from football_analysis.body_events.detector import BodyEventConfig, detect_body_events
    from football_analysis.body_events.model import BodyEventModel, default_model_path
    from football_analysis.state import read_state_cache

    model = BodyEventModel.load(args.model or default_model_path())
    cfg = BodyEventConfig(trick_threshold=args.trick_threshold, tackle_threshold=args.tackle_threshold)
    events = detect_body_events(list(read_state_cache(args.states)), model, cfg)
    events.sort(key=lambda e: e.timestamp_s)
    print(json.dumps([e.to_dict() for e in events], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m football_analysis.body_events",
                                description="Tackle and trick detection over pose.")
    p.add_argument("--dataset", default=str(DATASET), help="labelled-window store")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on simulated clips plus any labelled real windows")
    t.add_argument("--per-scenario", type=int, default=200)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--real-weight", type=float, default=5.0)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", help="model path (default: assets/body_events/model/body_events.joblib)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="event-level scores on held-out clips and real windows")
    e.add_argument("--model")
    e.add_argument("--per-scenario", type=int, default=40)
    e.add_argument("--seed", type=int, default=7, help="must differ from the training seed")
    e.add_argument("--trick-threshold", type=float, default=0.85)
    e.add_argument("--tackle-threshold", type=float, default=0.5)
    e.add_argument("--out", help="also write the report here")
    e.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("propose", help="cut candidate windows from a clip for labelling")
    pr.add_argument("clip")
    pr.add_argument("--states", required=True, help="state cache of the clip (--state-cache)")
    pr.add_argument("--skill-every", type=float, default=0.6)
    pr.add_argument("--no-review", action="store_true", help="skip the review sheets")
    pr.set_defaults(func=cmd_propose)

    lb = sub.add_parser("label", help="set the label of one window")
    lb.add_argument("window_id")
    lb.add_argument("label")
    lb.add_argument("--sub-label")
    lb.add_argument("--notes")
    lb.add_argument("--labeller", default="")
    lb.set_defaults(func=cmd_label)

    d = sub.add_parser("detect", help="run the stage over a state cache and print the events")
    d.add_argument("--states", required=True)
    d.add_argument("--model")
    d.add_argument("--trick-threshold", type=float, default=0.85)
    d.add_argument("--tackle-threshold", type=float, default=0.5)
    d.set_defaults(func=cmd_detect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
