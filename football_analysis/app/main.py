"""Start the app: ``python -m football_analysis.app`` (or ``football-app``)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="football-app", description="1v1 football analysis app")
    parser.add_argument("clip", nargs="?", help="open this video instead of the camera")
    parser.add_argument("-v", "--verbose", action="store_true", help="log more to the terminal")
    return parser.parse_args(argv)


def _request_camera_permission(app, window) -> None:
    """Ask macOS for the camera, once, before opening it.

    Without asking, AVFoundation hands back black frames and no error. Run from
    Terminal, the prompt and the grant belong to Terminal (or iTerm, or VS
    Code) rather than to Python, which is why the denied message names both.
    """
    try:
        from PySide6.QtCore import QCameraPermission, Qt
    except ImportError:  # Qt older than 6.5: no permission API, just try
        window.set_camera_permission(True)
        return

    permission = QCameraPermission()
    status = app.checkPermission(permission)
    if status == Qt.PermissionStatus.Granted:
        window.set_camera_permission(True)
    elif status == Qt.PermissionStatus.Denied:
        window.set_camera_permission(False)
    else:
        window.set_camera_permission(None)
        app.requestPermission(
            permission, window,
            lambda result: window.set_camera_permission(
                result.status() == Qt.PermissionStatus.Granted
            ),
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Same two environment fixes the CLI needs: scipy and torch fighting over
    # MKL on Intel builds, and torch's MPS path falling back to the CPU for
    # the odd operator it lacks rather than raising.
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "The app needs PySide6. Install it with:\n\n"
            "    pip install -r requirements-app.txt\n",
            file=sys.stderr,
        )
        return 2

    from PySide6.QtGui import QIcon

    from football_analysis.app.theme import STYLE_SHEET
    from football_analysis.app.window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("1v1 Analiz")
    app.setOrganizationName("FootballAnalysis")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    icon = Path(__file__).with_name("icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    window = MainWindow()
    window.show()
    if args.clip:
        window.open_clip(Path(args.clip).expanduser())
    _request_camera_permission(app, window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
