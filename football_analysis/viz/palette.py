"""Colours for the overlay.

Picked to stay legible on grass, which is the one background this system is
guaranteed to be drawn over: nothing here is green, and every colour is light
enough to read against both a bright pitch and a shadowed one.  Values are BGR
because that is what OpenCV draws in.
"""

from __future__ import annotations

__all__ = ["color_for_class", "color_for_track", "CLASS_COLORS", "TRACK_COLORS",
           "TEXT_COLOR", "BANNER_COLOR"]

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (255, 176, 0),    # amber
    "ball": (60, 60, 255),      # red
    "goal": (255, 255, 255),    # white
}
_UNKNOWN_CLASS_COLOR = (200, 200, 200)

# Cycled per track id so two players never share a colour.
TRACK_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 176, 0),    # amber
    (200, 80, 255),   # magenta
    (255, 220, 120),  # pale blue
    (120, 200, 255),  # peach
    (230, 120, 60),   # slate
)

TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
BANNER_COLOR: tuple[int, int, int] = (40, 40, 40)


def color_for_class(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, _UNKNOWN_CLASS_COLOR)


def color_for_track(track_id: int, class_name: str | None = None) -> tuple[int, int, int]:
    """Per-identity colour, except the ball, which is always the ball colour."""
    if class_name == "ball":
        return CLASS_COLORS["ball"]
    if class_name == "goal":
        return CLASS_COLORS["goal"]
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]
