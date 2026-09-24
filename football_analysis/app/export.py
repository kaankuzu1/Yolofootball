"""The timeline as a spreadsheet: one row per event, readable in Numbers or Excel."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from football_analysis.app.theme import event_name
from football_analysis.events import Event

COLUMNS = (
    "zaman", "saniye", "olay", "type", "oyuncu", "diger_oyuncu",
    "guven", "kontrol_et", "neden", "ayrinti",
)


def event_rows(events: Iterable[Event]) -> list[dict[str, str]]:
    rows = []
    for event in sorted(events, key=lambda e: float(e.timestamp_s)):
        detail = dict(event.detail or {})
        rows.append({
            "zaman": event.clock,
            "saniye": f"{float(event.timestamp_s):.3f}",
            "olay": event_name(event.type.value, detail),
            "type": event.type.value,
            "oyuncu": event.player_id or "",
            "diger_oyuncu": event.secondary_player_id or "",
            "guven": f"{float(event.confidence):.2f}",
            "kontrol_et": "evet" if detail.get("needs_review") else "",
            "neden": str(detail.get("review_reason") or ""),
            "ayrinti": json.dumps(detail, ensure_ascii=False, default=str) if detail else "",
        })
    return rows


def write_csv(path: str | Path, events: Iterable[Event]) -> Path:
    """Write the events as CSV.

    Semicolon-separated with a UTF-8 byte-order mark: that is what Excel on a
    Turkish-locale Mac opens straight into columns with ş and ğ intact, since
    the comma is the decimal separator there.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(event_rows(events))
    return path
