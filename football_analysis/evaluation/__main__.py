"""``python -m football_analysis.evaluation``: score a run, or draft labels to correct.

    # Score a run's events against a label file (prints a table, optional JSON / Markdown)
    python -m football_analysis.evaluation score out/events.json \\
        assets/annotations/events/my_1v1.labels.json --json out/score.json

    # Several clips pooled: pass --pair EVENTS LABELS once per clip
    python -m football_analysis.evaluation score --pair a.json a.labels.json --pair b.json b.labels.json

    # Seed a label file from a run, then correct it against the video
    python -m football_analysis.evaluation draft out/events.json -o assets/annotations/events/my_1v1.labels.json

    # Check a label file parses and says what it covers
    python -m football_analysis.evaluation check assets/annotations/events/my_1v1.labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_analysis.evaluation.labels import draft_labels, load_labels
from football_analysis.evaluation.score import DEFAULT_TOLERANCE_S, load_events, score_many


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m football_analysis.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("score", help="score a run's events against hand labels")
    sc.add_argument("events", nargs="?", help="events.json from a run")
    sc.add_argument("labels", nargs="?", help="label file for the same clip")
    sc.add_argument("--pair", nargs=2, action="append", default=[], metavar=("EVENTS", "LABELS"),
                    help="one more clip to pool into the score (repeatable)")
    sc.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S,
                    help="seconds either side a match may be off by (default 1.0, SoccerNet's convention)")
    sc.add_argument("--json", type=Path, help="write the full report as JSON")
    sc.add_argument("--markdown", type=Path, help="write the table as Markdown")

    dr = sub.add_parser("draft", help="seed a label file from a run, to be corrected by hand")
    dr.add_argument("events", help="events.json from a run")
    dr.add_argument("-o", "--output", type=Path, required=True)
    dr.add_argument("--types", nargs="+", help="event types to include (default: the five headline types)")

    ck = sub.add_parser("check", help="validate a label file")
    ck.add_argument("labels", nargs="+")

    args = parser.parse_args(argv)

    if args.command == "score":
        pairs = list(args.pair)
        if args.events or args.labels:
            if not (args.events and args.labels):
                parser.error("score needs both EVENTS and LABELS")
            pairs.insert(0, [args.events, args.labels])
        if not pairs:
            parser.error("score needs EVENTS LABELS or at least one --pair")
        report = score_many([(load_events(ev), load_labels(lab)) for ev, lab in pairs], args.tolerance)
        text = report.to_markdown()
        print(text, end="")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(text, encoding="utf-8")
        return 0

    if args.command == "draft":
        payload = json.loads(Path(args.events).read_text(encoding="utf-8"))
        source = payload.get("source", {}) if isinstance(payload, dict) else {}
        drafted = draft_labels(load_events(args.events), clip=source.get("path"),
                               fps=source.get("fps"), types=args.types)
        drafted.write_json(args.output)
        print(f"wrote {len(drafted.labels)} unchecked labels to {args.output}")
        return 0

    for path in args.labels:
        ls = load_labels(path)
        counts = {t: sum(1 for lab in ls.labels if lab.type == t and not lab.ambiguous) for t in ls.labelled_types}
        span = f"{ls.span_s[0]:g}–{ls.span_s[1]:g} s" if ls.span_s else "whole clip"
        print(f"{path}: {span}, " + ", ".join(f"{n} {t}" for t, n in counts.items())
              + f", {sum(lab.ambiguous for lab in ls.labels)} ambiguous")
    return 0


if __name__ == "__main__":
    sys.exit(main())
