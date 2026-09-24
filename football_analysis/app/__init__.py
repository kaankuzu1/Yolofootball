"""A desktop app around the analysis: pick a camera, record a 1v1, get the timeline.

    python -m football_analysis.app

Built on Qt (PySide6) so the same window runs on macOS, Windows and Linux. The
camera list comes from the operating system, so a built-in FaceTime camera, a
USB webcam and an iPhone used as a Continuity Camera all appear by name, and
the choice is remembered between launches.

The app never reimplements the analysis. Live, it runs the detector on the
newest camera frame so boxes follow the players; the events themselves come
from the same :func:`football_analysis.analyze` run the CLI uses, over the
clip that was just recorded, because the event rules need the whole clip
(they decide in ``finalize``) and would be guessing on a live feed.

Qt is an optional dependency (``requirements-app.txt``); nothing else in the
package imports this module.
"""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from football_analysis.app.main import main as _main

    return _main(argv)
