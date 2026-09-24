"""Colours, names and the style sheet. Nothing here touches a camera or a model.

The window is dark on purpose: the video is the content, and a light frame
around it pulls the eye away from the pitch. Each event type has one colour
used everywhere it appears -- the timeline strip, the list, the counters -- so
a goal is the same green in all three.
"""

from __future__ import annotations

from football_analysis.events import EventType

# Turkish names for what the timeline reports. The JSON keeps the English
# ``type`` values; these are for people.
EVENT_NAMES: dict[str, str] = {
    EventType.GOAL.value: "Gol",
    EventType.SHOT.value: "Şut",
    EventType.PASS.value: "Pas",
    EventType.TACKLE.value: "Top kapma",
    EventType.TRICK.value: "Çalım",
    EventType.PUSH_PAST.value: "Topu önüne atma",
    EventType.POSSESSION_CHANGE.value: "Top el değiştirdi",
    EventType.BALL_TOUCH.value: "Topa dokunma",
    EventType.OUT_OF_PLAY.value: "Oyun dışı",
}

EVENT_COLORS: dict[str, str] = {
    EventType.GOAL.value: "#3ddc84",
    EventType.SHOT.value: "#ff9f43",
    EventType.PASS.value: "#54a0ff",
    EventType.TACKLE.value: "#ff6b6b",
    EventType.TRICK.value: "#c56cf0",
    EventType.PUSH_PAST.value: "#48dbfb",
    EventType.POSSESSION_CHANGE.value: "#8395a7",
    EventType.BALL_TOUCH.value: "#576574",
    EventType.OUT_OF_PLAY.value: "#576574",
}

# The five the system was asked for, in the order the counters show them.
HEADLINE_TYPES: tuple[str, ...] = ("goal", "shot", "pass", "tackle", "trick")

TRICK_NAMES: dict[str, str] = {
    "nutmeg": "tünel",
    "skill_move": "hareket",
    "stepover": "makas",
    "dragback": "topu geri çekme",
    "feint": "vücut çalımı",
}

ACCENT = "#3ddc84"
DANGER = "#ff4d4f"
BG = "#101418"
PANEL = "#171c22"
PANEL_2 = "#1e252d"
BORDER = "#2a323c"
TEXT = "#e8edf2"
MUTED = "#8a96a3"


def event_name(event_type: str, detail: dict | None = None) -> str:
    """What to call an event in the list, e.g. ``Çalım (makas)``."""
    name = EVENT_NAMES.get(event_type, event_type.replace("_", " ").capitalize())
    trick = (detail or {}).get("trick_name")
    if event_type == EventType.TRICK.value and trick:
        return f"{name} ({TRICK_NAMES.get(trick, str(trick).replace('_', ' '))})"
    if event_type == EventType.SHOT.value and (detail or {}).get("on_target"):
        return f"{name} (isabetli)"
    return name


def event_color(event_type: str) -> str:
    return EVENT_COLORS.get(event_type, MUTED)


STYLE_SHEET = f"""
* {{
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#sidebar, QWidget#eventsPanel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QLabel#appTitle {{ font-size: 17px; font-weight: 700; }}
QLabel#sectionTitle {{
    color: {MUTED};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding-top: 6px;
}}
QLabel#muted {{ color: {MUTED}; }}
QLabel#statusPill {{
    background: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 3px 10px;
    color: {MUTED};
}}
QLabel#recClock {{
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 15px;
    font-weight: 600;
}}
QComboBox, QDoubleSpinBox, QSpinBox {{
    background: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 8px;
    min-height: 20px;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QAbstractSpinBox {{ padding-right: 22px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    width: 18px; border: none; background: transparent;
}}
QComboBox QAbstractItemView {{
    background: {PANEL_2};
    border: 1px solid {BORDER};
    selection-background-color: #26603f;
}}
QPushButton {{
    background: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 12px;
}}
QPushButton:hover {{ border-color: #3c4855; background: #232b34; }}
QPushButton:pressed {{ background: #151a20; }}
QPushButton:disabled {{ color: #56616d; border-color: #222a32; }}
QPushButton#primary {{
    background: {ACCENT};
    color: #062312;
    border: none;
    font-weight: 700;
}}
QPushButton#primary:hover {{ background: #5be49a; }}
QPushButton#primary:disabled {{ background: #1f4a31; color: #4b7a5e; }}
QPushButton#record {{
    background: {DANGER};
    color: white;
    border: none;
    border-radius: 22px;
    font-weight: 700;
    font-size: 14px;
    padding: 10px 22px;
}}
QPushButton#record:hover {{ background: #ff6b6d; }}
QPushButton#record[recording="true"] {{ background: #ffffff; color: {DANGER}; }}
QPushButton#record:disabled {{ background: #5a2a2b; color: #a07071; }}
QPushButton#chip {{
    border-radius: 12px;
    padding: 5px 10px;
    font-size: 12px;
    text-align: left;
}}
QPushButton#chip:checked {{ background: #26303a; border-color: #4b5968; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 2px; }}
QCheckBox {{ spacing: 8px; }}
QProgressBar {{
    background: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 10px;
    text-align: center;
    font-size: 11px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    border-radius: 8px;
    padding: 2px;
    margin: 1px 0;
}}
QListWidget::item:selected {{ background: #22303b; }}
QListWidget::item:hover {{ background: #1b232b; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {PANEL_2}; border: 1px solid {BORDER}; padding: 4px; }}
QStatusBar {{ color: {MUTED}; }}
"""
