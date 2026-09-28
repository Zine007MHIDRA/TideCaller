"""'Calibrate reel' dialog: capture (or load) a few seconds of reeling, click the bar and the fish, apply."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from tidecaller.config.schema import RodProfile
from tidecaller.core.regions import Rect, Region
from tidecaller.modules.reel.calibrate import CalibrationResult, Click, learn_profile, preview

VIEW_W = 820
V_ZOOM = 4  # the strip is only ~25 px tall; stretch it so it can be clicked


def frames_from_video(path: str, region: Region, start_s: float, end_s: float, fps: float = 12) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r = region.to_rect(Rect(0, 0, w, h))
    out, t = [], start_s
    while t <= end_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[r.top:r.bottom, r.left:r.right].copy())
        t += 1 / fps
    cap.release()
    return out


class ClickableImage(QLabel):
    def __init__(self, on_click: Callable[[int, int], None]) -> None:
        super().__init__()
        self.on_click = on_click
        self.src_size = (1, 1)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, e) -> None:
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return
        p = e.position()
        sx, sy = self.src_size[0] / pm.width(), self.src_size[1] / pm.height()
        x, y = int(p.x() * sx), int(p.y() * sy)
        if 0 <= x < self.src_size[0] and 0 <= y < self.src_size[1]:
            self.on_click(x, y)


class CalibrateDialog(QDialog):
    def __init__(self, rod: RodProfile, region: Region, grab_frame: Callable[[], np.ndarray | None] | None,
                 hide_windows: Callable[[bool], None] = lambda hidden: None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Calibrate reel - {rod.name}")
        self.rod, self.region, self.grab_frame, self.hide_windows = rod, region, grab_frame, hide_windows
        self.frames: list[np.ndarray] = []
        self.bar_click: Click | None = None
        self.result: CalibrationResult | None = None

        v = QVBoxLayout(self)
        v.addWidget(QLabel("1. Get a few seconds of an active reel (bar and fish on screen)."))
        row = QHBoxLayout()
        self.capture_btn = QPushButton("Capture from game (3 s countdown, 4 s capture)")
        self.capture_btn.clicked.connect(self._start_capture)
        self.capture_btn.setEnabled(grab_frame is not None)
        load_btn = QPushButton("Load recording...")
        load_btn.clicked.connect(self._load_video)
        self.start_s, self.end_s = QDoubleSpinBox(), QDoubleSpinBox()
        for sb, val in ((self.start_s, 0.0), (self.end_s, 5.0)):
            sb.setRange(0, 36000)
            sb.setDecimals(1)
            sb.setSuffix(" s")
            sb.setValue(val)
        row.addWidget(self.capture_btn)
        row.addWidget(load_btn)
        row.addWidget(QLabel("from"))
        row.addWidget(self.start_s)
        row.addWidget(QLabel("to"))
        row.addWidget(self.end_s)
        v.addLayout(row)

        self.step = QLabel("")
        self.step.setStyleSheet("font-weight:600")
        v.addWidget(self.step)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._show_frame)
        v.addWidget(self.slider)
        self.view = ClickableImage(self._clicked)
        self.view.setMinimumSize(VIEW_W, 60)
        v.addWidget(self.view)
        self.no_fish_btn = QPushButton("No fish visible - skip")
        self.no_fish_btn.clicked.connect(lambda: self._learn(None))
        v.addWidget(self.no_fish_btn)

        v.addWidget(QLabel("Result (blue: catch bar, red: fish, orange: fish assumed at bar center)"))
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        self.result_view = QLabel()
        v.addWidget(self.result_view)
        buttons = QHBoxLayout()
        self.redo_btn = QPushButton("Redo clicks")
        self.redo_btn.clicked.connect(self._reset_clicks)
        self.apply_btn = QPushButton("Apply to rod")
        self.apply_btn.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        for b in (self.redo_btn, self.apply_btn, cancel):
            buttons.addWidget(b)
        v.addLayout(buttons)
        self._reset_clicks()

    # ------------------------------------------------------------ getting frames
    CAPTURE_SAMPLES = 40  # 4 s at 10 fps

    def _start_capture(self) -> None:
        self.hide_windows(True)  # keep our windows off the fish bar while capturing
        self._countdown = 3
        self._captured: list[np.ndarray] = []
        self._grabs = 0
        QTimer.singleShot(1000, self._tick_countdown)

    def _tick_countdown(self) -> None:
        self._countdown -= 1
        if self._countdown > 0:
            QTimer.singleShot(1000, self._tick_countdown)
            return
        self._timer = QTimer(self, interval=100, timeout=self._grab_one)
        self._timer.start()

    def _grab_one(self) -> None:
        try:
            f = self.grab_frame() if self.grab_frame else None
        except Exception:
            f = None
        if f is not None:
            self._captured.append(f)
        self._grabs += 1
        if self._grabs >= self.CAPTURE_SAMPLES:
            self._timer.stop()
            self.hide_windows(False)
            self._set_frames(self._captured)

    def _load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Recording", "", "Video (*.mp4 *.mkv *.mov *.avi)")
        if not path:
            return
        try:
            frames = frames_from_video(path, self.region, self.start_s.value(), self.end_s.value())
        except Exception as e:
            QMessageBox.warning(self, "Calibrate", str(e))
            return
        self._set_frames(frames)

    def _set_frames(self, frames: list[np.ndarray]) -> None:
        if not frames:
            QMessageBox.warning(self, "Calibrate", "No frames captured. Is Roblox open and the reel on screen?")
            return
        self.frames = frames
        self.slider.setRange(0, len(frames) - 1)
        self.slider.setValue(len(frames) // 2)
        self._reset_clicks()
        self._show_frame(self.slider.value())

    # ------------------------------------------------------------ clicking
    def _reset_clicks(self) -> None:
        self.bar_click, self.result = None, None
        self.apply_btn.setEnabled(False)
        self.no_fish_btn.setEnabled(False)
        self.summary.setText("")
        self.result_view.clear()
        self.step.setText("2. Pick a frame where the bar and fish are clear, then click the CATCH BAR."
                          if self.frames else "")

    def _show_frame(self, i: int) -> None:
        if not self.frames:
            return
        from tidecaller.gui.app import to_pixmap

        f = self.frames[i]
        big = cv2.resize(f, (f.shape[1], f.shape[0] * V_ZOOM), interpolation=cv2.INTER_NEAREST)
        if self.bar_click and self.bar_click.frame == i:
            cv2.circle(big, (self.bar_click.x, self.bar_click.y * V_ZOOM), 6, (255, 80, 0), 2)
        self.view.src_size = (f.shape[1], f.shape[0])
        self.view.setPixmap(to_pixmap(big, VIEW_W))

    def _clicked(self, x: int, y: int) -> None:
        if not self.frames or self.result is not None:
            return
        i = self.slider.value()
        if self.bar_click is None:
            self.bar_click = Click(i, x, y)
            self.step.setText("3. Now click the FISH marker (or press 'No fish visible').")
            self.no_fish_btn.setEnabled(True)
            self._show_frame(i)
        else:
            self._learn(Click(i, x, y))

    def _learn(self, fish: Click | None) -> None:
        if self.bar_click is None:
            return
        from tidecaller.gui.app import to_pixmap

        self.result = learn_profile(self.frames, self.bar_click, fish, self.rod)
        self.summary.setText(self.result.summary())
        self.result_view.setPixmap(to_pixmap(preview(self.frames, self.result.results, n=6), VIEW_W))
        self.step.setText("4. Check the preview, then Apply (or Redo clicks).")
        self.apply_btn.setEnabled(self.result.bar_rate > 0)
        self.no_fish_btn.setEnabled(False)

    def learned_profile(self) -> RodProfile | None:
        return None if self.result is None else self.result.profile
