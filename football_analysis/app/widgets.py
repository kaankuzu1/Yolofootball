"""The custom-drawn parts of the window: the video, the event strip, an event row."""

from __future__ import annotations

from collections import deque

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from football_analysis.app import theme
from football_analysis.detection.goal import CORNER_ORDER

CORNER_PROMPTS = {
    "left_post_base": "Sol direğin dibi",
    "right_post_base": "Sağ direğin dibi",
    "right_crossbar_end": "Üst direğin sağ ucu",
    "left_crossbar_end": "Üst direğin sol ucu",
}

PLAYER_COLORS = ["#54a0ff", "#ff9f43", "#c56cf0", "#48dbfb", "#feca57", "#ff6b6b"]


def bgr_to_qimage(image: np.ndarray) -> QImage:
    """Wrap a BGR array as a QImage. The array must outlive the image."""
    height, width = image.shape[:2]
    image = np.ascontiguousarray(image)
    return QImage(image.data, width, height, image.strides[0], QImage.Format.Format_BGR888)


class VideoView(QWidget):
    """Shows frames, letterboxed, with live boxes and the goal drawn on top.

    Everything drawn here is in the frame's own pixel coordinates and mapped
    to the widget on paint, so it stays put whatever size the window is.
    """

    corner_clicked = Signal(float, float)
    """In goal-marking mode: a click, in frame pixels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)
        self._array: np.ndarray | None = None
        self._image: QImage | None = None
        self._boxes: list = []
        self._trail: deque = deque(maxlen=24)
        self._goal: list[list[float]] | None = None
        self._marking: list[list[float]] | None = None
        self._hover: QPointF | None = None
        self.placeholder = "Kamera bekleniyor…"
        self.recording_s: float | None = None
        self.badge = ""
        self.show_boxes = True

    # -- content ---------------------------------------------------------------

    def set_frame(self, image: np.ndarray | None) -> None:
        self._array = image
        self._image = bgr_to_qimage(image) if image is not None else None
        self.update()

    def frame_size(self) -> tuple[int, int] | None:
        if self._array is None:
            return None
        return (self._array.shape[1], self._array.shape[0])

    def current_frame(self) -> np.ndarray | None:
        return self._array

    def set_boxes(self, boxes: list) -> None:
        self._boxes = list(boxes)
        ball = next((b for b in self._boxes if b.kind == "ball"), None)
        if ball is not None:
            x1, y1, x2, y2 = ball.xyxy
            self._trail.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        self.update()

    def clear_overlay(self) -> None:
        self._boxes = []
        self._trail.clear()
        self.update()

    def set_goal(self, corners: list[list[float]] | None) -> None:
        self._goal = corners
        self.update()

    def start_marking(self) -> None:
        self._marking = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def stop_marking(self) -> None:
        self._marking = None
        self.unsetCursor()
        self.update()

    @property
    def marking(self) -> bool:
        return self._marking is not None

    def add_marking_point(self, x: float, y: float) -> list[list[float]] | None:
        """Record a clicked corner; returns all four once the last is in."""
        if self._marking is None:
            return None
        self._marking.append([x, y])
        self.update()
        if len(self._marking) == 4:
            done = self._marking
            self.stop_marking()
            return done
        return None

    # -- geometry --------------------------------------------------------------

    def _target_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def _to_widget(self, x: float, y: float) -> QPointF:
        rect = self._target_rect()
        iw = self._image.width() if self._image else 1
        ih = self._image.height() if self._image else 1
        return QPointF(rect.x() + x * rect.width() / iw, rect.y() + y * rect.height() / ih)

    def _to_frame(self, point: QPointF) -> tuple[float, float] | None:
        if self._image is None:
            return None
        rect = self._target_rect()
        if not rect.contains(point):
            return None
        x = (point.x() - rect.x()) * self._image.width() / rect.width()
        y = (point.y() - rect.y()) * self._image.height() / rect.height()
        return (x, y)

    # -- input -----------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._marking is not None and event.button() == Qt.MouseButton.LeftButton:
            hit = self._to_frame(event.position())
            if hit is not None:
                self.corner_clicked.emit(*hit)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._marking is not None:
            self._hover = event.position()
            self.update()
        super().mouseMoveEvent(event)

    # -- drawing ---------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#07090b"))

        if self._image is None:
            painter.setPen(QColor(theme.MUTED))
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            painter.end()
            return

        painter.drawImage(self._target_rect(), self._image)
        if self._goal:
            self._draw_goal(painter, self._goal, QColor(theme.ACCENT))
        if self.show_boxes:
            self._draw_trail(painter)
            for box in self._boxes:
                self._draw_box(painter, box)
        if self._marking is not None:
            self._draw_marking(painter)
        self._draw_hud(painter)
        painter.end()

    def _draw_goal(self, painter: QPainter, corners, color: QColor) -> None:
        poly = QPolygonF([self._to_widget(x, y) for x, y in corners])
        fill = QColor(color)
        fill.setAlpha(40)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
        painter.drawPolygon(poly)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_box(self, painter: QPainter, box) -> None:
        x1, y1, x2, y2 = box.xyxy
        top_left = self._to_widget(x1, y1)
        bottom_right = self._to_widget(x2, y2)
        rect = QRectF(top_left, bottom_right)
        if box.kind == "ball":
            center = rect.center()
            radius = max(6.0, max(rect.width(), rect.height()) / 2.0 + 3.0)
            glow = QColor("#ffffff")
            glow.setAlpha(60)
            painter.setBrush(glow)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(center, radius, radius)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            return
        if box.kind == "goal":
            if self._goal:  # the marked goal is the truth; skip the detector's box
                return
            color = QColor(theme.ACCENT)
            label = "Kale"
        else:
            color = QColor(PLAYER_COLORS[((box.track_id or 1) - 1) % len(PLAYER_COLORS)])
            label = f"Oyuncu {box.track_id}" if box.track_id is not None else "Oyuncu"
        if getattr(box, "has_ball", False):
            # The player on the ball gets a glow and a filled tag, so who is
            # playing it can be read at a glance from across the room.
            glow = QColor(color)
            glow.setAlpha(55)
            painter.setBrush(glow)
            painter.setPen(QPen(color, 4))
            painter.drawRoundedRect(rect.adjusted(-3, -3, 3, 3), 6, 6)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            self._draw_tag(painter, rect.topLeft() + QPointF(-3, -3), f"⚽ {label} · topta", color)
            return
        painter.setPen(QPen(color, 2))
        painter.drawRoundedRect(rect, 4, 4)
        self._draw_tag(painter, rect.topLeft(), label, color)

    def _draw_tag(self, painter: QPainter, anchor: QPointF, text: str, color: QColor) -> None:
        font = QFont(painter.font())
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(text) + 10
        h = metrics.height() + 4
        tag = QRectF(anchor.x(), anchor.y() - h - 2, w, h)
        if tag.y() < 0:
            tag.moveTop(anchor.y() + 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(tag, 4, 4)
        painter.setPen(QColor("#0b0f13"))
        painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, text)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_trail(self, painter: QPainter) -> None:
        points = list(self._trail)
        for i in range(1, len(points)):
            alpha = int(200 * i / len(points))
            color = QColor("#ffffff")
            color.setAlpha(alpha)
            painter.setPen(QPen(color, 2))
            painter.drawLine(self._to_widget(*points[i - 1]), self._to_widget(*points[i]))

    def _draw_marking(self, painter: QPainter) -> None:
        points = self._marking or []
        color = QColor("#ff7be5")
        if points:
            path = QPainterPath(self._to_widget(*points[0]))
            for p in points[1:]:
                path.lineTo(self._to_widget(*p))
            if self._hover is not None and len(points) < 4:
                path.lineTo(self._hover)
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.drawPath(path)
        for i, p in enumerate(points):
            c = self._to_widget(*p)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(color)
            painter.drawEllipse(c, 6, 6)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            self._draw_tag(painter, c + QPointF(8, 0), str(i + 1), color)

        step = len(points)
        if step < 4:
            prompt = f"{step + 1}/4 · {CORNER_PROMPTS[CORNER_ORDER[step]]} noktasına tıklayın"
            self._draw_banner(painter, prompt + "   (Esc: iptal)", QColor("#ff7be5"))

    def _draw_banner(self, painter: QPainter, text: str, color: QColor) -> None:
        font = QFont(painter.font())
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(text) + 28
        h = metrics.height() + 14
        rect = QRectF((self.width() - w) / 2.0, self.height() - h - 16, w, h)
        painter.setPen(Qt.PenStyle.NoPen)
        bg = QColor("#0b0f13")
        bg.setAlpha(215)
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, h / 2.0, h / 2.0)
        painter.setPen(color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_hud(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        if self.recording_s is not None:
            m, s = divmod(int(self.recording_s), 60)
            text = f"KAYIT  {m:02d}:{s:02d}"
            w = metrics.horizontalAdvance(text) + 36
            rect = QRectF(14, 14, w, metrics.height() + 12)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
            painter.setBrush(QColor(theme.DANGER))
            painter.drawEllipse(QPointF(rect.x() + 14, rect.center().y()), 5, 5)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(rect.adjusted(24, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, text)
        if self.badge:
            w = metrics.horizontalAdvance(self.badge) + 20
            rect = QRectF(self.width() - w - 14, 14, w, metrics.height() + 12)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
            painter.setPen(QColor("#d6dde4"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.badge)
        painter.setBrush(Qt.BrushStyle.NoBrush)


class TimelineStrip(QWidget):
    """The clip as a bar, each event a coloured mark; click to jump there."""

    seek = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(46)
        self.setMouseTracking(True)
        self.duration_s = 0.0
        self.position_s = 0.0
        self._events: list = []
        self._hover_x: float | None = None

    def set_events(self, events: list, duration_s: float) -> None:
        self._events = list(events)
        self.duration_s = max(0.0, float(duration_s))
        self.update()

    def set_position(self, t: float) -> None:
        self.position_s = t
        self.update()

    def _bar(self) -> QRectF:
        return QRectF(10, 18, self.width() - 20, 12)

    def _x_for(self, t: float) -> float:
        bar = self._bar()
        if self.duration_s <= 0:
            return bar.left()
        return bar.left() + bar.width() * min(1.0, max(0.0, t / self.duration_s))

    def _t_for(self, x: float) -> float:
        bar = self._bar()
        return max(0.0, min(1.0, (x - bar.left()) / max(1.0, bar.width()))) * self.duration_s

    def _event_at(self, x: float):
        best, best_d = None, 6.0
        for event in self._events:
            d = abs(self._x_for(float(event.timestamp_s)) - x)
            if d < best_d:
                best, best_d = event, d
        return best

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.duration_s > 0:
            hit = self._event_at(event.position().x())
            self.seek.emit(float(hit.timestamp_s) if hit else self._t_for(event.position().x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._hover_x = event.position().x()
        hit = self._event_at(self._hover_x)
        if hit is not None:
            who = f" · {hit.player_id}" if hit.player_id else ""
            self.setToolTip(f"{hit.clock}  {theme.event_name(hit.type.value, hit.detail)}{who}")
        else:
            self.setToolTip("")
        self.update()

    def leaveEvent(self, _event) -> None:
        self._hover_x = None
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = self._bar()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.PANEL_2))
        painter.drawRoundedRect(bar, 6, 6)
        if self.duration_s > 0:
            played = QRectF(bar.left(), bar.top(), self._x_for(self.position_s) - bar.left(),
                            bar.height())
            painter.setBrush(QColor("#243240"))
            painter.drawRoundedRect(played, 6, 6)
            # Supporting events first, so headline marks sit on top of them.
            ordered = sorted(self._events, key=lambda e: e.type.is_primary)
            for event in ordered:
                x = self._x_for(float(event.timestamp_s))
                color = QColor(theme.event_color(event.type.value))
                if event.type.is_primary:
                    painter.setBrush(color)
                    painter.drawRoundedRect(QRectF(x - 2, bar.top() - 7, 4, bar.height() + 14), 2, 2)
                else:
                    color.setAlpha(150)
                    painter.setBrush(color)
                    painter.drawRect(QRectF(x - 0.75, bar.top() + 2, 1.5, bar.height() - 4))
            x = self._x_for(self.position_s)
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPointF(x, bar.center().y()), 6, 6)
        if self._hover_x is not None and self.duration_s > 0:
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
            painter.drawLine(QPointF(self._hover_x, 4), QPointF(self._hover_x, self.height() - 4))
        painter.end()

    def sizeHint(self) -> QSize:
        return QSize(600, 46)


class Dot(QWidget):
    def __init__(self, color: str, size: int = 10, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(self.rect())
        painter.end()


class EventRow(QWidget):
    """One line of the event list: colour, time, what, who, how sure."""

    def __init__(self, event, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        detail = event.detail or {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)
        layout.addWidget(Dot(theme.event_color(event.type.value)), 0, Qt.AlignmentFlag.AlignVCenter)

        clock = QLabel(event.clock[:-2] if len(event.clock) > 8 else event.clock)
        clock.setStyleSheet("font-family: 'SF Mono', Menlo, monospace; color: #a9b4bf;")
        clock.setFixedWidth(64)
        layout.addWidget(clock)

        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel(theme.event_name(event.type.value, detail))
        weight = 600 if event.type.is_primary else 400
        title.setStyleSheet(f"font-weight: {weight};")
        text.addWidget(title)
        who = event.player_id or ""
        if event.secondary_player_id:
            who = f"{who} → {event.secondary_player_id}" if who else event.secondary_player_id
        if who:
            sub = QLabel(who)
            sub.setObjectName("muted")
            sub.setStyleSheet("font-size: 11px;")
            text.addWidget(sub)
        layout.addLayout(text, 1)

        if detail.get("needs_review"):
            flag = QLabel("kontrol et")
            flag.setToolTip(str(detail.get("review_reason") or "Kural emin olamadı"))
            flag.setStyleSheet(
                "background:#4a3a12; color:#ffd166; border-radius:8px; padding:1px 7px;"
                "font-size:11px;"
            )
            layout.addWidget(flag)

        conf = QLabel(f"%{int(round(float(event.confidence) * 100))}")
        conf.setToolTip("Güven")
        conf.setObjectName("muted")
        conf.setFixedWidth(38)
        conf.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(conf)
