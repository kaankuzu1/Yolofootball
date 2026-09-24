"""The main window: camera on the left, video in the middle, events on the right.

Two modes share the one video area:

``live``    the chosen camera, with boxes from the live detector and a record
            button. Stopping a recording switches to review and, by default,
            starts the analysis.
``review``  a recorded or opened clip with a player, the event strip under it
            and the event list beside it. Clicking an event plays the moment.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from football_analysis.app import theme
from football_analysis.app.cameras import (
    RESOLUTIONS,
    CameraInfo,
    FrameSource,
    list_cameras,
    open_source,
    pick_camera,
)
from football_analysis.app.export import write_csv
from football_analysis.app.player import ClipPlayer
from football_analysis.app.recorder import PacedRecorder
from football_analysis.app.storage import (
    GoalSetup,
    annotated_path,
    csv_path,
    default_output_dir,
    events_path,
    load_goal,
    new_recording_path,
    processed_corners,
    save_goal,
    states_path,
)
from football_analysis.app.widgets import EventRow, TimelineStrip, VideoView
from football_analysis.app.workers import (
    AnalysisRequest,
    AnalysisWorker,
    LiveDetector,
    describe_device,
)
from football_analysis.events import EventTimeline

logger = logging.getLogger(__name__)

GOAL_PRESETS = [
    ("Futsal / küçük kale · 3 × 2 m", (3.0, 2.0)),
    ("Mini kale · 1.8 × 1.2 m", (1.8, 1.2)),
    ("Halı saha · 5 × 2 m", (5.0, 2.0)),
    ("Tam boy · 7.32 × 2.44 m", (7.32, 2.44)),
    ("Özel", None),
]

VIDEO_FILTER = "Videolar (*.mp4 *.mov *.m4v *.avi *.mkv);;Tüm dosyalar (*)"
RECORD_FPS = 30.0
MIN_RECORDING_S = 1.0


def _section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("sectionTitle")
    return label


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or QSettings("FootballAnalysis", "1v1 Analiz")
        self.setWindowTitle("1v1 Analiz")
        self.resize(1440, 860)
        self.setMinimumSize(1100, 680)

        self.mode = "live"
        self.cameras: list[CameraInfo] = []
        self.camera: CameraInfo | None = None
        self.source: FrameSource | None = None
        self.live: LiveDetector | None = None
        self.recorder: PacedRecorder | None = None
        self.analysis: AnalysisWorker | None = None
        self.clip: Path | None = None
        self.timeline: EventTimeline | None = None
        self.warnings: list[str] = []
        self.review_goal: GoalSetup | None = None
        self.camera_frame_size: tuple[int, int] | None = None
        self._frozen = False
        self._type_filter: str | None = None
        self._permission_ok = False
        """The camera is opened only once macOS has said yes (see main.py)."""

        self.player = ClipPlayer(self)
        self.player.frame.connect(self._on_player_frame)
        self.player.state_changed.connect(self._on_play_state)

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self._watch_devices()
        self._restore_settings()
        self._set_mode("live")

    # ======================================================================
    # layout
    # ======================================================================

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)
        outer.addWidget(self._build_sidebar())
        outer.addWidget(self._build_center(), 1)
        outer.addWidget(self._build_events_panel())

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(284)
        box = QVBoxLayout(side)
        box.setContentsMargins(16, 16, 16, 16)
        box.setSpacing(8)

        title = QLabel("⚽  1v1 Analiz")
        title.setObjectName("appTitle")
        box.addWidget(title)
        self.device_label = _muted()
        box.addWidget(self.device_label)
        box.addSpacing(6)

        # -- source ---------------------------------------------------------
        box.addWidget(_section("Kamera"))
        row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(12)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_chosen)
        row.addWidget(self.camera_combo, 1)
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setToolTip("Kameraları yeniden tara")
        self.refresh_btn.setFixedWidth(36)
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        row.addWidget(self.refresh_btn)
        box.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(_muted("Çözünürlük"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(list(RESOLUTIONS))
        self.resolution_combo.currentTextChanged.connect(self._on_resolution_changed)
        row.addWidget(self.resolution_combo, 1)
        box.addLayout(row)
        self.camera_hint = _muted(
            "USB kamera ya da iPhone (Süreklilik Kamerası) takınca listede kendiliğinden görünür."
        )
        self.camera_hint.setStyleSheet("font-size: 11px;")
        box.addWidget(self.camera_hint)

        self.open_btn = QPushButton("Video dosyası aç…")
        self.open_btn.clicked.connect(self.open_file_dialog)
        box.addWidget(self.open_btn)

        # -- goal -----------------------------------------------------------
        box.addSpacing(8)
        box.addWidget(_section("Kale"))
        self.goal_status = _muted()
        box.addWidget(self.goal_status)
        row = QHBoxLayout()
        self.mark_btn = QPushButton("Kaleyi işaretle")
        self.mark_btn.setToolTip("Kalenin dört köşesine sırayla tıklayın (kısayol: G)")
        self.mark_btn.clicked.connect(self.toggle_marking)
        row.addWidget(self.mark_btn, 1)
        self.clear_goal_btn = QPushButton("Sil")
        self.clear_goal_btn.setFixedWidth(52)
        self.clear_goal_btn.clicked.connect(self.clear_goal)
        row.addWidget(self.clear_goal_btn)
        box.addLayout(row)

        self.goal_preset = QComboBox()
        for name, _size in GOAL_PRESETS:
            self.goal_preset.addItem(name)
        self.goal_preset.currentIndexChanged.connect(self._on_goal_preset)
        box.addWidget(self.goal_preset)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self.goal_w = QDoubleSpinBox()
        self.goal_h = QDoubleSpinBox()
        for spin, value in ((self.goal_w, 3.0), (self.goal_h, 2.0)):
            spin.setRange(0.5, 10.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setSuffix(" m")
            spin.setValue(value)
            spin.valueChanged.connect(self._on_goal_size_changed)
        grid.addWidget(_muted("Genişlik"), 0, 0)
        grid.addWidget(self.goal_w, 0, 1)
        grid.addWidget(_muted("Yükseklik"), 1, 0)
        grid.addWidget(self.goal_h, 1, 1)
        box.addLayout(grid)

        # -- analysis -------------------------------------------------------
        box.addSpacing(8)
        box.addWidget(_section("Analiz"))
        self.live_check = QCheckBox("Canlı tespit (kutular)")
        self.live_check.setToolTip("Kamerada oyuncuları, topu ve kaleyi anlık gösterir")
        self.live_check.toggled.connect(self._on_live_toggled)
        box.addWidget(self.live_check)
        self.auto_check = QCheckBox("Kayıt bitince analiz et")
        box.addWidget(self.auto_check)
        self.render_check = QCheckBox("Çizimli video üret")
        self.render_check.setToolTip("Kutuların ve olay yazılarının çizildiği bir kopya kaydeder")
        box.addWidget(self.render_check)
        row = QHBoxLayout()
        row.addWidget(_muted("Hız"))
        self.stride_combo = QComboBox()
        self.stride_combo.addItem("Her kare (en doğru)", 1)
        self.stride_combo.addItem("2 karede bir (2× hızlı)", 2)
        self.stride_combo.addItem("3 karede bir (3× hızlı)", 3)
        row.addWidget(self.stride_combo, 1)
        box.addLayout(row)

        box.addStretch(1)
        self.folder_btn = QPushButton("Kayıt klasörü…")
        self.folder_btn.setObjectName("link")
        self.folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.folder_btn.clicked.connect(self.choose_output_dir)
        box.addWidget(self.folder_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.folder_label = _muted()
        self.folder_label.setStyleSheet("font-size: 11px;")
        box.addWidget(self.folder_label)
        return side

    def _build_center(self) -> QWidget:
        center = QWidget()
        box = QVBoxLayout(center)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)

        header = QHBoxLayout()
        self.mode_label = QLabel()
        self.mode_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        header.addWidget(self.mode_label)
        self.status_pill = QLabel()
        self.status_pill.setObjectName("statusPill")
        header.addWidget(self.status_pill)
        header.addStretch(1)
        self.back_live_btn = QPushButton("← Kameraya dön")
        self.back_live_btn.clicked.connect(lambda: self._set_mode("live"))
        header.addWidget(self.back_live_btn)
        box.addLayout(header)

        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background: #07090b; border: 1px solid {theme.BORDER};"
                            " border-radius: 12px; }")
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(1, 1, 1, 1)
        self.view = VideoView()
        self.view.corner_clicked.connect(self._on_corner_clicked)
        inner.addWidget(self.view)
        box.addWidget(frame, 1)

        # Player controls (review).
        self.player_bar = QWidget()
        pbar = QHBoxLayout(self.player_bar)
        pbar.setContentsMargins(0, 0, 0, 0)
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(44)
        self.play_btn.clicked.connect(self.player.toggle)
        pbar.addWidget(self.play_btn)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._on_slider)
        pbar.addWidget(self.slider, 1)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("recClock")
        pbar.addWidget(self.time_label)
        self.overlay_check = QCheckBox("Çizimli")
        self.overlay_check.setToolTip("Analizin çizdiği kutularla oynat")
        self.overlay_check.toggled.connect(self._on_overlay_toggled)
        pbar.addWidget(self.overlay_check)
        box.addWidget(self.player_bar)

        self.strip = TimelineStrip()
        self.strip.seek.connect(self._seek_to)
        box.addWidget(self.strip)

        # Record / analyse bar.
        actions = QHBoxLayout()
        actions.setSpacing(12)
        self.record_btn = QPushButton("●  Kaydı başlat")
        self.record_btn.setObjectName("record")
        self.record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_btn.clicked.connect(self.toggle_recording)
        actions.addWidget(self.record_btn)
        self.analyse_btn = QPushButton("Analiz et")
        self.analyse_btn.setObjectName("primary")
        self.analyse_btn.setMinimumWidth(130)
        self.analyse_btn.clicked.connect(self.start_analysis)
        actions.addWidget(self.analyse_btn)

        progress_box = QVBoxLayout()
        progress_box.setSpacing(3)
        self.progress_label = _muted()
        progress_box.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        progress_box.addWidget(self.progress)
        actions.addLayout(progress_box, 1)
        self.record_hint = _muted("R ya da boşluk: kaydı başlat / bitir. Kamera sabit dursun.")
        actions.addWidget(self.record_hint, 1)
        self.cancel_btn = QPushButton("Durdur")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        actions.addWidget(self.cancel_btn)
        box.addLayout(actions)
        return center

    def _build_events_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("eventsPanel")
        panel.setFixedWidth(340)
        box = QVBoxLayout(panel)
        box.setContentsMargins(14, 16, 14, 14)
        box.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("Olaylar")
        title.setObjectName("appTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.warn_btn = QPushButton()
        self.warn_btn.setObjectName("link")
        self.warn_btn.setStyleSheet("color: #ffd166;")
        self.warn_btn.clicked.connect(self._show_warnings)
        head.addWidget(self.warn_btn)
        box.addLayout(head)

        # Counters that double as filters.
        chips = QGridLayout()
        chips.setSpacing(6)
        self.chip_group = QButtonGroup(self)
        self.chip_group.setExclusive(False)
        self.chips: dict[str, QPushButton] = {}
        for i, kind in enumerate(theme.HEADLINE_TYPES):
            chip = QPushButton()
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setIcon(_dot_icon(theme.event_color(kind)))
            chip.clicked.connect(lambda _=False, k=kind: self._on_chip(k))
            self.chips[kind] = chip
            chips.addWidget(chip, i // 3, i % 3)
        box.addLayout(chips)
        self.support_check = QCheckBox("Destek olaylarını da göster")
        self.support_check.setToolTip("Top el değiştirmesi, topu önüne atma, topa dokunma")
        self.support_check.toggled.connect(self._refresh_event_list)
        box.addWidget(self.support_check)

        self.event_list = QListWidget()
        self.event_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.event_list.itemClicked.connect(self._on_event_clicked)
        self.event_list.itemActivated.connect(self._on_event_clicked)
        box.addWidget(self.event_list, 1)
        self.empty_label = _muted()
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.empty_label)

        row = QHBoxLayout()
        self.csv_btn = QPushButton("CSV")
        self.csv_btn.setToolTip("Olayları Excel/Numbers için dışa aktar")
        self.csv_btn.clicked.connect(self.export_csv)
        row.addWidget(self.csv_btn)
        self.json_btn = QPushButton("JSON")
        self.json_btn.setToolTip("Tam zaman çizelgesini kaydet")
        self.json_btn.clicked.connect(self.export_json)
        row.addWidget(self.json_btn)
        self.reveal_btn = QPushButton("Klasörü aç")
        self.reveal_btn.clicked.connect(self.reveal_clip)
        row.addWidget(self.reveal_btn)
        box.addLayout(row)
        return panel

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Dosya")
        action = QAction("Video aç…", self)
        action.setShortcut(QKeySequence.StandardKey.Open)
        action.triggered.connect(self.open_file_dialog)
        menu.addAction(action)
        action = QAction("Kayıt klasörünü göster", self)
        action.triggered.connect(lambda: self._reveal(self.output_dir()))
        menu.addAction(action)
        help_menu = self.menuBar().addMenu("Yardım")
        action = QAction("Nasıl kullanılır", self)
        action.triggered.connect(self._show_help)
        help_menu.addAction(action)

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._on_space)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, self._on_record_key)
        QShortcut(QKeySequence(Qt.Key.Key_G), self, self.toggle_marking)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._cancel_marking)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))

    # ======================================================================
    # settings
    # ======================================================================

    def _restore_settings(self) -> None:
        s = self.settings
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.setCurrentText(str(s.value("resolution", "720p")))
        self.resolution_combo.blockSignals(False)
        self.live_check.blockSignals(True)
        self.live_check.setChecked(s.value("live_detect", True, type=bool))
        self.live_check.blockSignals(False)
        self.auto_check.setChecked(s.value("auto_analyse", True, type=bool))
        self.render_check.setChecked(s.value("render", True, type=bool))
        stride = int(s.value("stride", 1))
        self.stride_combo.setCurrentIndex(max(0, self.stride_combo.findData(stride)))
        self._set_goal_size(self._default_goal_size())
        self._update_folder_label()
        self.device_label.setText(f"Hesaplama: {describe_device()}")

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("resolution", self.resolution_combo.currentText())
        s.setValue("live_detect", self.live_check.isChecked())
        s.setValue("auto_analyse", self.auto_check.isChecked())
        s.setValue("render", self.render_check.isChecked())
        s.setValue("stride", self.stride_combo.currentData())
        s.setValue("goal_size", f"{self.goal_w.value()},{self.goal_h.value()}")
        if self.camera is not None:
            s.setValue("camera_id", self.camera.id)

    def output_dir(self) -> Path:
        stored = self.settings.value("output_dir", "")
        return Path(stored) if stored else default_output_dir()

    def _update_folder_label(self) -> None:
        self.folder_label.setText(str(self.output_dir()).replace(str(Path.home()), "~"))

    def choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Kayıtlar nereye kaydedilsin?",
                                                  str(self.output_dir()))
        if folder:
            self.settings.setValue("output_dir", folder)
            self._update_folder_label()

    def _default_goal_size(self) -> tuple[float, float]:
        raw = str(self.settings.value("goal_size", "3.0,2.0"))
        try:
            w, h = (float(v) for v in raw.split(","))
            return (w, h)
        except ValueError:
            return (3.0, 2.0)

    # ======================================================================
    # cameras
    # ======================================================================

    def _watch_devices(self) -> None:
        try:
            from PySide6.QtMultimedia import QMediaDevices

            self._media_devices = QMediaDevices(self)
            self._media_devices.videoInputsChanged.connect(self.refresh_cameras)
        except Exception:  # pragma: no cover - Qt Multimedia missing
            self._media_devices = None

    def refresh_cameras(self) -> None:
        if self.recorder is not None:
            return  # never swap cameras under a recording
        remembered = self.camera.id if self.camera else self.settings.value("camera_id", "")
        self.cameras = list_cameras()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for camera in self.cameras:
            self.camera_combo.addItem(camera.name, camera.id)
            tags = camera.label[len(camera.name):].strip(" ()")
            if tags:
                self.camera_combo.setItemData(self.camera_combo.count() - 1, tags,
                                              Qt.ItemDataRole.ToolTipRole)
        chosen = pick_camera(self.cameras, remembered)
        if chosen is not None:
            self.camera_combo.setCurrentIndex(self.cameras.index(chosen))
        self.camera_combo.blockSignals(False)
        if not self.cameras:
            self.camera_combo.addItem("Kamera bulunamadı")
            self.camera_combo.setEnabled(False)
            self._stop_camera()
            self.camera = None
            self.view.set_frame(None)
            self.view.placeholder = "Kamera bulunamadı. Bir kamera bağlayıp ↻ ile yeniden tarayın."
            self.view.update()
            return
        self.camera_combo.setEnabled(True)
        if chosen is not None and (self.camera is None or chosen.id != self.camera.id
                                   or self.source is None):
            self._switch_camera(chosen)

    def _on_camera_chosen(self, index: int) -> None:
        if 0 <= index < len(self.cameras):
            self._switch_camera(self.cameras[index])

    def _on_resolution_changed(self, _text: str) -> None:
        self._save_settings()
        if self.camera is not None and self.mode == "live":
            self._switch_camera(self.camera)

    def _switch_camera(self, camera: CameraInfo) -> None:
        self.camera = camera
        self._save_settings()
        self._update_goal_ui()
        if self.mode == "live" and self._permission_ok:
            self._start_camera()

    def _start_camera(self) -> None:
        self._stop_camera()
        if self.camera is None:
            return
        self.camera_frame_size = None
        self.view.clear_overlay()
        self.view.set_frame(None)
        self.view.placeholder = f"{self.camera.name} açılıyor…"
        resolution = RESOLUTIONS.get(self.resolution_combo.currentText(), RESOLUTIONS["720p"])
        self.source = open_source(self.camera, resolution, self)
        self.source.frame.connect(self._on_camera_frame)
        self.source.error.connect(self._on_camera_error)
        self.source.started.connect(lambda mode: self._set_status(
            f"{self.camera.name}" + (f" · {mode}" if mode else "")))
        self.source.start()
        self._ensure_live_detector()
        self._update_controls()

    def _stop_camera(self) -> None:
        if self.source is not None:
            try:
                self.source.frame.disconnect(self._on_camera_frame)
            except (RuntimeError, TypeError):
                pass
            self.source.stop()
            self.source.deleteLater()
            self.source = None

    def _on_camera_error(self, message: str) -> None:
        self.view.set_frame(None)
        self.view.placeholder = f"Kamera açılamadı: {message}"
        self.view.update()
        self._set_status("Kamera hatası")

    def _on_camera_frame(self, image, t: float) -> None:
        if self.mode != "live":
            return
        size = (image.shape[1], image.shape[0])
        if size != self.camera_frame_size:
            self.camera_frame_size = size
            self._update_goal_ui()
        if self.recorder is not None:
            self.recorder.write(image, t)
            self.view.recording_s = self.recorder.duration_s
        if not self._frozen:
            self.view.set_frame(image)
        if self.live is not None and self.live_check.isChecked() and not self._frozen:
            self.live.submit(image, t)

    def set_camera_permission(self, granted: bool | None) -> None:
        """``None`` while the macOS prompt is still on screen."""
        self._permission_ok = bool(granted)
        if granted is None:
            self.view.placeholder = "Kamera izni bekleniyor…"
            self.view.update()
        elif not granted:
            self._stop_camera()
            if self.mode != "live":
                return
            self.view.set_frame(None)
            self.view.placeholder = (
                "Kamera izni yok. Sistem Ayarları › Gizlilik ve Güvenlik › Kamera'dan\n"
                "bu uygulamaya (ya da Terminal'e) izin verip yeniden açın.\n"
                "Bu arada kayıtlı bir videoyu açabilirsiniz."
            )
            self.view.update()
        elif self.mode == "live":
            self.refresh_cameras()

    # ======================================================================
    # live detection
    # ======================================================================

    def _ensure_live_detector(self) -> None:
        if not self.live_check.isChecked() or self.live is not None:
            return
        self.live = LiveDetector(parent=self)
        self.live.result.connect(self._on_live_result)
        self.live.status.connect(self._on_live_status)
        self.live.failed.connect(self._on_live_failed)
        self.live.start()

    def _stop_live_detector(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live.deleteLater()
            self.live = None
        self.view.clear_overlay()
        self.view.badge = ""

    def _on_live_toggled(self, on: bool) -> None:
        self._save_settings()
        if on and self.mode == "live":
            self._ensure_live_detector()
        elif not on:
            self._stop_live_detector()

    def _on_live_result(self, result) -> None:
        if self.mode != "live" or not self.live_check.isChecked() or self._frozen:
            return
        self.view.set_boxes(result.boxes)
        self.view.badge = f"Canlı tespit · {result.fps:.0f} fps"

    def _on_live_status(self, text: str) -> None:
        self.view.badge = text
        self.view.update()

    def _on_live_failed(self, message: str) -> None:
        self.view.badge = ""
        self.view.clear_overlay()
        self.live_check.blockSignals(True)
        self.live_check.setChecked(False)
        self.live_check.blockSignals(False)
        if self.live is not None:
            self.live.deleteLater()
            self.live = None
        self._set_status("Canlı tespit kapalı")
        QMessageBox.warning(self, "Canlı tespit", message)

    # ======================================================================
    # recording
    # ======================================================================

    def _on_record_key(self) -> None:
        if self.mode == "live":
            self.toggle_recording()

    def toggle_recording(self) -> None:
        if self.recorder is None:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        if self.source is None or self.camera_frame_size is None:
            QMessageBox.information(self, "Kayıt", "Önce kameradan görüntü gelmesi gerekiyor.")
            return
        folder = self.output_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.recorder = PacedRecorder(new_recording_path(folder), fps=RECORD_FPS)
        except Exception as exc:
            QMessageBox.critical(self, "Kayıt", f"Kayıt başlatılamadı: {exc}")
            self.recorder = None
            return
        self.view.recording_s = 0.0
        self.record_btn.setText("■  Kaydı bitir")
        self.record_btn.setProperty("recording", True)
        self._repolish(self.record_btn)
        self._update_controls()

    def stop_recording(self) -> None:
        recorder, self.recorder = self.recorder, None
        self.view.recording_s = None
        self.record_btn.setText("●  Kaydı başlat")
        self.record_btn.setProperty("recording", False)
        self._repolish(self.record_btn)
        if recorder is None:
            return
        duration = recorder.duration_s
        clip = recorder.close()
        self._update_controls()
        if clip is None or duration < MIN_RECORDING_S:
            if clip is not None:
                clip.unlink(missing_ok=True)
            self._set_status("Kayıt çok kısaydı, silindi")
            return
        goal = self._camera_goal()
        if goal is not None:
            save_goal(clip, goal)
        self._set_status(f"Kaydedildi: {clip.name} ({duration:.0f} sn)")
        self.open_clip(clip)
        if self.auto_check.isChecked():
            QTimer.singleShot(200, self.start_analysis)

    # ======================================================================
    # review
    # ======================================================================

    def open_file_dialog(self) -> None:
        if self.recorder is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Video aç", str(self.output_dir()),
                                              VIDEO_FILTER)
        if path:
            self.open_clip(Path(path))

    def open_clip(self, clip: Path) -> None:
        clip = Path(clip)
        if clip.name.endswith(".annotated.mp4"):
            original = clip.with_name(clip.name.replace(".annotated.mp4", ".mp4"))
            if original.exists():
                clip = original
        if not self.player.open(clip):
            QMessageBox.critical(self, "Video aç", f"{clip.name} açılamadı.")
            return
        self.clip = clip
        self.review_goal = load_goal(clip, self.player.size)
        self.timeline = None
        self.warnings = []
        events_file = events_path(clip)
        if events_file.exists():
            try:
                self.timeline = EventTimeline.read_json(events_file)
            except Exception:
                logger.warning("could not read %s", events_file, exc_info=True)
        self._set_mode("review")
        self.overlay_check.blockSignals(True)
        self.overlay_check.setChecked(annotated_path(clip).exists() and self.timeline is not None)
        self.overlay_check.blockSignals(False)
        self._load_player_source()
        self._show_timeline()

    def _load_player_source(self, keep_position: bool = False) -> None:
        if self.clip is None:
            return
        position = self.player.position_s if keep_position else 0.0
        annotated = annotated_path(self.clip)
        use_annotated = self.overlay_check.isChecked() and annotated.exists()
        self.player.open(annotated if use_annotated else self.clip)
        if position:
            self.player.seek(position)
        self._update_goal_ui()

    def _on_overlay_toggled(self, _on: bool) -> None:
        self._load_player_source(keep_position=True)

    def _on_player_frame(self, image, t: float) -> None:
        if self.mode != "review" or self._frozen:
            return
        self.view.set_frame(image)
        duration = self.player.duration_s
        if duration > 0 and not self.slider.isSliderDown():
            self.slider.setValue(int(1000 * t / duration))
        self.time_label.setText(f"{_clock(t)} / {_clock(duration)}")
        self.strip.set_position(t)
        self._highlight_event_near(t)

    def _on_play_state(self, playing: bool) -> None:
        self.play_btn.setText("❚❚" if playing else "▶")

    def _on_slider(self, value: int) -> None:
        self.player.seek(self.player.duration_s * value / 1000.0)

    def _seek_to(self, t: float) -> None:
        if self.mode == "review":
            self.player.seek(t)

    def _step(self, frames: int) -> None:
        if self.mode == "review" and not self.view.marking:
            self.player.step(frames)

    def _on_space(self) -> None:
        if self.mode == "review":
            self.player.toggle()
        else:
            self.toggle_recording()

    # ======================================================================
    # goal
    # ======================================================================

    def _camera_goal_key(self) -> str | None:
        return f"goal/{self.camera.id}" if self.camera else None

    def _camera_goal(self) -> GoalSetup | None:
        key = self._camera_goal_key()
        raw = self.settings.value(key, "") if key else ""
        if not raw:
            return None
        try:
            return GoalSetup.from_dict(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            return None

    def _current_goal(self) -> GoalSetup | None:
        return self.review_goal if self.mode == "review" else self._camera_goal()

    def _store_goal(self, goal: GoalSetup | None) -> None:
        if self.mode == "review":
            self.review_goal = goal
            if self.clip is not None:
                if goal is None:
                    from football_analysis.app.storage import goal_path

                    goal_path(self.clip).unlink(missing_ok=True)
                else:
                    save_goal(self.clip, goal)
        else:
            key = self._camera_goal_key()
            if key is None:
                return
            if goal is None:
                self.settings.remove(key)
            else:
                self.settings.setValue(key, json.dumps(goal.to_dict()))
        self._update_goal_ui()

    def _displayed_size(self) -> tuple[int, int] | None:
        return self.view.frame_size() or (
            self.player.size if self.mode == "review" else self.camera_frame_size
        )

    def _update_goal_ui(self) -> None:
        goal = self._current_goal()
        size = self._displayed_size()
        self.view.set_goal(goal.scaled_to(size) if goal and size else None)
        if goal is None:
            where = "bu video" if self.mode == "review" else "bu kamera"
            self.goal_status.setText(
                f"{where.capitalize()} için işaretlenmedi. Gol ve isabetli şut için gerekli; "
                "kamerayı sabitledikten sonra bir kez işaretleyin."
            )
            self.clear_goal_btn.setEnabled(False)
        else:
            w, h = goal.size_m
            self.goal_status.setText(f"✓ İşaretli · {w:g} × {h:g} m")
            self.goal_status.setStyleSheet(f"color: {theme.ACCENT};")
            self.clear_goal_btn.setEnabled(True)
            self._set_goal_size(goal.size_m, persist=False)
        if goal is None:
            self.goal_status.setStyleSheet("")

    def toggle_marking(self) -> None:
        if self.view.marking:
            self._cancel_marking()
            return
        if self.view.frame_size() is None:
            QMessageBox.information(self, "Kale", "İşaretlemek için önce görüntü gerekiyor.")
            return
        if self.mode == "review":
            self.player.pause()
        self._frozen = True
        self.view.clear_overlay()
        self.view.start_marking()
        self.mark_btn.setText("İptal")

    def _cancel_marking(self) -> None:
        if self.view.marking:
            self.view.stop_marking()
        self._frozen = False
        self.mark_btn.setText("Kaleyi işaretle")

    def _on_corner_clicked(self, x: float, y: float) -> None:
        corners = self.view.add_marking_point(x, y)
        if corners is None:
            return
        size = self.view.frame_size()
        self._frozen = False
        self.mark_btn.setText("Kaleyi yeniden işaretle")
        goal = GoalSetup(corners=corners, frame_size=size,
                         size_m=(self.goal_w.value(), self.goal_h.value()))
        if self.mode == "review" and self.overlay_check.isChecked() and self.clip is not None:
            # Clicked on the annotated copy, which may be smaller than the clip.
            goal = GoalSetup(corners=goal.scaled_to(self.player.size) if size != self.player.size
                             else corners, frame_size=self.player.size, size_m=goal.size_m)
        self._store_goal(goal)
        self._set_status("Kale kaydedildi")

    def clear_goal(self) -> None:
        self._store_goal(None)
        self.mark_btn.setText("Kaleyi işaretle")

    def _set_goal_size(self, size: tuple[float, float], persist: bool = True) -> None:
        for spin, value in ((self.goal_w, size[0]), (self.goal_h, size[1])):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        preset = next((i for i, (_n, s) in enumerate(GOAL_PRESETS)
                       if s and abs(s[0] - size[0]) < 1e-6 and abs(s[1] - size[1]) < 1e-6),
                      len(GOAL_PRESETS) - 1)
        self.goal_preset.blockSignals(True)
        self.goal_preset.setCurrentIndex(preset)
        self.goal_preset.blockSignals(False)
        if persist:
            self._save_settings()

    def _on_goal_preset(self, index: int) -> None:
        size = GOAL_PRESETS[index][1]
        if size is not None:
            self._set_goal_size(size)
            self._on_goal_size_changed()

    def _on_goal_size_changed(self, *_args) -> None:
        size = (self.goal_w.value(), self.goal_h.value())
        self._set_goal_size(size)
        goal = self._current_goal()
        if goal is not None and tuple(goal.size_m) != size:
            goal.size_m = size
            self._store_goal(goal)

    # ======================================================================
    # analysis
    # ======================================================================

    def start_analysis(self) -> None:
        if self.analysis is not None or self.clip is None:
            return
        goal = self.review_goal
        if goal is None:
            answer = QMessageBox.question(
                self, "Kale işaretlenmedi",
                "Bu videoda kale işaretli değil. Paslar, top kapmalar ve çalımlar yine bulunur "
                "ama gol ve isabetli şut bulunamaz.\n\nYine de analiz edilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        from football_analysis.config import load_config
        from football_analysis.io.video_reader import probe

        config = load_config(None)
        corners = None
        if goal is not None:
            info = probe(self.clip)
            source_size = (info.width, info.height)
            corners = processed_corners(goal.scaled_to(source_size), source_size,
                                        config.video.resize_width, config.video.resize_height)
        request = AnalysisRequest(
            clip=self.clip,
            events_json=events_path(self.clip),
            annotated=annotated_path(self.clip) if self.render_check.isChecked() else None,
            states=states_path(self.clip),
            goal_corners_px=corners,
            goal_size_m=tuple(goal.size_m) if goal else (self.goal_w.value(), self.goal_h.value()),
            frame_stride=int(self.stride_combo.currentData() or 1),
            config=config,
        )
        self._save_settings()
        if self.live is not None:
            self.live.pause(True)
        # Reading the clip while it is also being played is fine, but the
        # annotated copy is about to be rewritten: play the original.
        if self.overlay_check.isChecked():
            self.overlay_check.setChecked(False)
        self.analysis = AnalysisWorker(request, self)
        self.analysis.progress.connect(self._on_analysis_progress)
        self.analysis.finished_ok.connect(self._on_analysis_done)
        self.analysis.failed.connect(self._on_analysis_failed)
        self.analysis.finished.connect(self._on_analysis_thread_done)
        self.progress.setValue(0)
        self.progress_label.setText("Başlıyor…")
        self.analysis.start()
        self._update_controls()

    def cancel_analysis(self) -> None:
        if self.analysis is not None:
            self.analysis.cancel()
            self.progress_label.setText("Durduruluyor…")

    def _on_analysis_progress(self, text: str, fraction: float) -> None:
        self.progress_label.setText(text)
        if fraction >= 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
        else:
            self.progress.setRange(0, 0)

    def _on_analysis_done(self, result, warnings: list) -> None:
        self.timeline = result.timeline
        self.warnings = list(warnings)
        n = len(self.timeline.primary())
        self.progress_label.setText(f"Bitti · {n} olay" + (
            f" · {len(self.timeline.needs_review())} tanesi kontrol istiyor"
            if self.timeline.needs_review() else ""))
        self.progress.setValue(1000)
        if self.mode == "review":
            if result.video_path is not None:
                self.overlay_check.setChecked(True)
            self._show_timeline()

    def _on_analysis_failed(self, message: str) -> None:
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        if not message:
            self.progress_label.setText("Durduruldu")
            return
        self.progress_label.setText("Analiz başarısız")
        QMessageBox.critical(self, "Analiz", message)

    def _on_analysis_thread_done(self) -> None:
        if self.analysis is not None:
            self.analysis.deleteLater()
        self.analysis = None
        if self.live is not None:
            self.live.pause(False)
        self._update_controls()

    # ======================================================================
    # events
    # ======================================================================

    def _visible_events(self) -> list:
        if self.timeline is None:
            return []
        events = [e for e in self.timeline.events
                  if e.type.is_primary or self.support_check.isChecked()]
        if self._type_filter:
            events = [e for e in events if e.type.value == self._type_filter]
        return events

    def _show_timeline(self) -> None:
        counts = self.timeline.counts() if self.timeline else {}
        for kind, chip in self.chips.items():
            chip.setText(f"{theme.EVENT_NAMES[kind]}  {counts.get(kind, 0)}")
            chip.setChecked(self._type_filter == kind)
        self.warn_btn.setText(f"⚠ {len(self.warnings)} uyarı" if self.warnings else "")
        self.warn_btn.setVisible(bool(self.warnings))
        self._refresh_event_list()

    def _refresh_event_list(self, *_args) -> None:
        self.event_list.clear()
        events = self._visible_events()
        for event in events:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, float(event.timestamp_s))
            row = EventRow(event)
            item.setSizeHint(row.sizeHint())
            self.event_list.addItem(item)
            self.event_list.setItemWidget(item, row)
        duration = self.player.duration_s if self.mode == "review" else 0.0
        strip_events = [e for e in (self.timeline.events if self.timeline else [])
                        if e.type.is_primary or self.support_check.isChecked()]
        self.strip.set_events(strip_events, duration)

        if self.timeline is None:
            self.empty_label.setText(
                "Henüz analiz yok.\nKayıt yapın ya da bir video açıp “Analiz et”e basın."
                if self.mode == "review" else
                "Kaydı başlatın. Kayıt bitince olaylar burada zaman sırasıyla listelenir."
            )
        elif not events:
            self.empty_label.setText("Bu filtreyle olay yok.")
        else:
            self.empty_label.setText("")
        self.empty_label.setVisible(bool(self.empty_label.text()))
        has = self.timeline is not None
        self.csv_btn.setEnabled(has)
        self.json_btn.setEnabled(has)

    def _on_chip(self, kind: str) -> None:
        self._type_filter = None if self._type_filter == kind else kind
        self._show_timeline()

    def _on_event_clicked(self, item: QListWidgetItem) -> None:
        t = float(item.data(Qt.ItemDataRole.UserRole))
        if self.mode != "review":
            return
        # Start a moment early so the build-up is visible, then play through it.
        self.player.seek(max(0.0, t - 1.5))
        self.player.play()

    def _highlight_event_near(self, t: float) -> None:
        for i in range(self.event_list.count()):
            item = self.event_list.item(i)
            if abs(float(item.data(Qt.ItemDataRole.UserRole)) - t) < 0.25:
                if self.event_list.currentRow() != i:
                    self.event_list.setCurrentRow(i)
                    self.event_list.scrollToItem(item)
                return

    def export_csv(self) -> None:
        if self.timeline is None:
            return
        suggested = csv_path(self.clip) if self.clip else self.output_dir() / "olaylar.csv"
        path, _ = QFileDialog.getSaveFileName(self, "CSV olarak kaydet", str(suggested),
                                              "CSV (*.csv)")
        if path:
            write_csv(path, self.timeline.events)
            self._set_status(f"Kaydedildi: {Path(path).name}")

    def export_json(self) -> None:
        if self.timeline is None:
            return
        suggested = events_path(self.clip) if self.clip else self.output_dir() / "olaylar.json"
        path, _ = QFileDialog.getSaveFileName(self, "JSON olarak kaydet", str(suggested),
                                              "JSON (*.json)")
        if not path:
            return
        source = events_path(self.clip) if self.clip else None
        if source is not None and source.exists() and Path(path) != source:
            shutil.copyfile(source, path)
        else:
            self.timeline.write_json(path)
        self._set_status(f"Kaydedildi: {Path(path).name}")

    def reveal_clip(self) -> None:
        self._reveal(self.clip.parent if self.clip else self.output_dir())

    def _reveal(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _show_warnings(self) -> None:
        QMessageBox.information(self, "Analizin uyarıları", "\n\n".join(self.warnings))

    def _show_help(self) -> None:
        QMessageBox.information(
            self, "Nasıl kullanılır",
            "1. Soldan kamerayı seçin. Kamerayı tripoda ya da sabit bir yere koyun; "
            "sistem kameranın oynamadığını varsayar.\n\n"
            "2. “Kaleyi işaretle”ye basıp kalenin dört köşesine sırayla tıklayın: sol direğin "
            "dibi, sağ direğin dibi, üst direğin sağ ucu, üst direğin sol ucu. Kamera yerinden "
            "oynamadıkça bir kez yeter.\n\n"
            "3. “Kaydı başlat” (R ya da boşluk). Bitince analiz kendiliğinden başlar.\n\n"
            "4. Sağdaki listede bir olaya tıklayınca video o ana gider. Alttaki şeritte de "
            "olaylar renkli çizgilerle görünür.\n\n"
            "Kısayollar: Boşluk oynat/durdur ya da kayıt · ←/→ kare kare · G kale · Esc iptal.",
        )

    # ======================================================================
    # modes and chrome
    # ======================================================================

    def _set_mode(self, mode: str) -> None:
        if mode == "live" and self.recorder is not None:
            return
        self._cancel_marking()
        self.mode = mode
        live = mode == "live"
        self.view.clear_overlay()
        self.view.show_boxes = live
        if live:
            self.player.close()
            self.clip = None
            self.review_goal = None
            self.timeline = None
            self.warnings = []
            self.mode_label.setText("Canlı kamera")
            if self._permission_ok:
                if self.cameras and self.camera is not None:
                    self._start_camera()
                else:
                    self.refresh_cameras()
        else:
            self._stop_camera()
            self.view.badge = ""
            self.mode_label.setText(self.clip.name if self.clip else "Video")
            self._set_status("İnceleme")
        self.player_bar.setVisible(not live)
        self.strip.setVisible(not live)
        self.back_live_btn.setVisible(not live)
        self._update_goal_ui()
        self._show_timeline()
        self._update_controls()

    def _update_controls(self) -> None:
        live = self.mode == "live"
        recording = self.recorder is not None
        analysing = self.analysis is not None
        self.record_btn.setVisible(live)
        self.record_hint.setVisible(live)
        self.record_btn.setEnabled(self.source is not None)
        self.analyse_btn.setVisible(not live)
        self.analyse_btn.setEnabled(not analysing and self.clip is not None)
        self.analyse_btn.setText("Yeniden analiz et" if self.timeline is not None else "Analiz et")
        self.cancel_btn.setVisible(analysing)
        self.progress.setVisible(not live and (analysing or bool(self.progress_label.text())))
        self.progress_label.setVisible(not live)
        self.camera_combo.setEnabled(not recording and bool(self.cameras))
        self.resolution_combo.setEnabled(not recording)
        self.refresh_btn.setEnabled(not recording)
        self.open_btn.setEnabled(not recording)
        self.back_live_btn.setEnabled(not recording)
        self.mark_btn.setEnabled(not recording)
        if live and not analysing:
            self.progress_label.setText("")

    def _set_status(self, text: str) -> None:
        self.status_pill.setText(text)
        self.status_pill.setVisible(bool(text))

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def closeEvent(self, event) -> None:
        if self.recorder is not None:
            self.stop_recording_silently()
        if self.analysis is not None:
            answer = QMessageBox.question(
                self, "Analiz sürüyor", "Analiz yarıda kalacak. Çıkılsın mı?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.analysis.cancel()
            self.analysis.wait(10000)
        self._save_settings()
        self._stop_live_detector()
        self._stop_camera()
        self.player.close()
        super().closeEvent(event)

    def stop_recording_silently(self) -> None:
        recorder, self.recorder = self.recorder, None
        if recorder is not None:
            clip = recorder.close()
            goal = self._camera_goal()
            if clip is not None and goal is not None:
                save_goal(clip, goal)


def _dot_icon(color: str, size: int = 10) -> QIcon:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(0, 0, size * 2, size * 2)
    painter.end()
    return QIcon(pixmap)


def _clock(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    return f"{m:02d}:{s:02d}"
