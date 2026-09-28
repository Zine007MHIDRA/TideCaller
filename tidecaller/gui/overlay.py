"""F1 area editor: a transparent always-on-top window with draggable/resizable colored boxes."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from tidecaller.config.schema import Regions
from tidecaller.core.regions import Rect, Region

COLORS = {
    "fish_bar": QColor(60, 140, 255),
    "reel_progress": QColor(80, 220, 120),
    "shake": QColor(255, 80, 200),
    "cast_meter": QColor(255, 210, 60),
    "hotbar": QColor(0, 230, 230),
}
HANDLE = 12


class AreaOverlay(QWidget):
    def __init__(self, regions: Regions, client: Rect, on_close: Callable[[], None]) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(client.left, client.top, client.width, client.height)
        self.regions, self.client, self.on_close = regions, client, on_close
        self.local = Rect(0, 0, client.width, client.height)
        self._drag: tuple[str, str, QPoint, QRect] | None = None

    def _qrect(self, name: str) -> QRect:
        r = getattr(self.regions, name).to_rect(self.local)
        return QRect(r.left, r.top, r.width, r.height)

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 60))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        for name, color in COLORS.items():
            r = self._qrect(name)
            fill = QColor(color)
            fill.setAlpha(45)
            p.fillRect(r, fill)
            p.setPen(QPen(color, 2))
            p.drawRect(r)
            p.fillRect(QRect(r.right() - HANDLE, r.bottom() - HANDLE, HANDLE, HANDLE), color)
            p.drawText(r.left() + 4, r.top() - 4, name.replace("_", " "))
        p.setPen(QColor(255, 255, 255))
        p.drawText(12, 22, "Drag boxes to move, corner to resize. F1 / Esc to save and close.")

    def mousePressEvent(self, e) -> None:
        pos = e.position().toPoint()
        for name in reversed(list(COLORS)):  # topmost first
            r = self._qrect(name)
            corner = QRect(r.right() - HANDLE, r.bottom() - HANDLE, HANDLE + 2, HANDLE + 2)
            if corner.contains(pos):
                self._drag = (name, "resize", pos, r)
                return
            if r.contains(pos):
                self._drag = (name, "move", pos, r)
                return

    def mouseMoveEvent(self, e) -> None:
        if not self._drag:
            return
        name, mode, start, r0 = self._drag
        d = e.position().toPoint() - start
        r = QRect(r0)
        if mode == "move":
            r.translate(d)
        else:
            r.setWidth(max(10, r0.width() + d.x()))
            r.setHeight(max(6, r0.height() + d.y()))
        setattr(self.regions, name, Region.from_rect(Rect(r.x(), r.y(), r.width(), r.height()), self.local))
        self.update()

    def mouseReleaseEvent(self, _) -> None:
        self._drag = None

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_F1):
            self.close()

    def closeEvent(self, e) -> None:
        self.on_close()
        super().closeEvent(e)
