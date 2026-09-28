"""Main window: Hydra-style tabs, start/stop, live stats and a debug preview of what the tracker sees."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tidecaller.config import rod_presets
from tidecaller.config.schema import Config
from tidecaller.core.state_machine import Macro, Phase, Stats
from tidecaller.gui.forms import build_form
from tidecaller.modules.reel import TrackResult


class Bridge(QObject):
    """Marshals worker-thread callbacks onto the Qt thread."""
    phase = Signal(str)
    debug = Signal(object)
    finished = Signal()
    hotkey = Signal(str)


def to_pixmap(img: np.ndarray, max_w: int = 560) -> QPixmap:
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy())


def annotate(img: np.ndarray, r: TrackResult | None, hold: bool) -> np.ndarray:
    out = cv2.resize(img, (img.shape[1], max(img.shape[0], 40)), interpolation=cv2.INTER_NEAREST)
    h = out.shape[0]
    if r is not None:
        cv2.rectangle(out, (int(r.bar_left), 0), (int(r.bar_right), h - 1), (255, 200, 0), 2)
        cv2.line(out, (int(r.fish_x), 0), (int(r.fish_x), h - 1), (0, 0, 255), 2)
        cv2.line(out, (int(r.bar_center), h // 2), (int(r.fish_x), h // 2), (0, 255, 255), 1)
    cv2.circle(out, (8, 8), 5, (0, 220, 0) if hold else (80, 80, 80), -1)
    return out


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, cfg_path: Path, macro_factory: Callable[[], Macro],
                 overlay_factory: Callable[[Callable[[], None]], QWidget | None] | None = None,
                 notifier_factory=None) -> None:
        super().__init__()
        self.cfg, self.cfg_path = cfg, cfg_path
        self.macro_factory, self.overlay_factory, self.notifier_factory = macro_factory, overlay_factory, notifier_factory
        self.macro: Macro | None = None
        self.thread: threading.Thread | None = None
        self.bridge = Bridge()
        self.bridge.phase.connect(self._on_phase)
        self.bridge.debug.connect(self._on_debug)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.hotkey.connect(self._on_hotkey)
        self._overlay = None
        self._last_debug = 0.0

        self.setWindowTitle("Tidecaller")
        self.setMinimumWidth(620)
        if cfg.main.always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        root = QWidget()
        v = QVBoxLayout(root)
        self.tabs = QTabWidget()
        v.addWidget(self.tabs)
        self._build_tabs()

        bar = QHBoxLayout()
        self.status = QLabel("Idle")
        self.status.setStyleSheet("font-weight:600")
        self.start_btn = QPushButton(f"Start ({cfg.main.hotkeys.start.upper()})")
        self.stop_btn = QPushButton(f"Stop ({cfg.main.hotkeys.stop.upper()})")
        self.area_btn = QPushButton(f"Areas ({cfg.main.hotkeys.area_editor.upper()})")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.area_btn.clicked.connect(self.toggle_overlay)
        self.stop_btn.setEnabled(False)
        for w in (self.status, self.area_btn, self.start_btn, self.stop_btn):
            bar.addWidget(w, 1 if w is self.status else 0)
        v.addLayout(bar)
        self.setCentralWidget(root)

        self._stats_timer = QTimer(self, interval=500, timeout=self._refresh_stats)
        self._stats_timer.start()

    # ------------------------------------------------------------ tabs
    def _save(self) -> None:
        self.cfg.save(self.cfg_path)

    def _scroll(self, w: QWidget) -> QScrollArea:
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        return s

    def _build_tabs(self) -> None:
        c = self.cfg
        self.tabs.addTab(self._scroll(self._main_tab()), "Main")
        self.tabs.addTab(self._scroll(build_form(c.cast, self._save)), "Cast")
        self.tabs.addTab(self._scroll(build_form(c.shake, self._save)), "Shake")
        self.tabs.addTab(self._scroll(self._fish_tab()), "Fish")
        self.tabs.addTab(self._scroll(build_form(c.totem, self._save)), "Totem")
        self.tabs.addTab(self._scroll(self._discord_tab()), "Discord")
        self.tabs.addTab(self._stats_tab(), "Stats")

    def _main_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(build_form(self.cfg.main, self._save))
        v.addWidget(QLabel("<b>Hotkeys</b> (restart to apply)"))
        v.addWidget(build_form(self.cfg.main.hotkeys, self._save))
        v.addStretch()
        return w

    def _fish_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(build_form(self.cfg.fish, self._save, skip=("rod",)))
        row = QHBoxLayout()
        self.rod_box = QComboBox()
        self.rod_box.setEditable(True)  # type to search 260+ rods
        self.rod_box.addItems(list(self.cfg.rods) + [n for n in rod_presets.names() if n not in self.cfg.rods])
        self.rod_box.setCurrentText(self.cfg.fish.rod)
        self.rod_box.currentTextChanged.connect(self._select_rod)
        new_btn = QPushButton("New profile")
        new_btn.clicked.connect(self._new_rod)
        row.addWidget(QLabel("Rod profile"))
        row.addWidget(self.rod_box, 1)
        row.addWidget(new_btn)
        v.addLayout(row)
        self.rod_form_holder = QVBoxLayout()
        v.addLayout(self.rod_form_holder)
        self._render_rod_form()
        v.addStretch()
        return w

    def _render_rod_form(self) -> None:
        while self.rod_form_holder.count():
            self.rod_form_holder.takeAt(0).widget().deleteLater()
        name = self.cfg.fish.rod
        profile = self.cfg.rod()
        stats = rod_presets.load_stats().get(name)
        if stats is not None:
            origin = "custom tuning" if name in self.cfg.rods else "preset from wiki stats"
            self.rod_form_holder.addWidget(QLabel(
                f"Control {stats.control:+.2f} · Resilience {stats.resilience_pct:g}% · "
                f"Lure {stats.lure_pct}% · Luck {stats.luck_pct}% — <i>{origin}</i>"))

        def on_change() -> None:
            # the first edit to a preset turns it into a saved custom profile
            self.cfg.rods.setdefault(profile.name, profile)
            self._save()

        self.rod_form_holder.addWidget(build_form(profile, on_change, skip=("name",)))

    def _select_rod(self, name: str) -> None:
        self.cfg.fish.rod = name
        stats = rod_presets.load_stats().get(name)
        if stats is not None and name not in self.cfg.rods:
            self.cfg.fish.track = stats.recommended_track
        self._save()
        self._render_rod_form()

    def _new_rod(self) -> None:
        n = f"Rod {len(self.cfg.rods) + 1}"
        self.cfg.rods[n] = self.cfg.rod().model_copy(update={"name": n})
        self.rod_box.addItem(n)
        self.rod_box.setCurrentText(n)

    def _discord_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(build_form(self.cfg.discord, self._save))
        test = QPushButton("Test webhook")
        test.clicked.connect(self._test_webhook)
        v.addWidget(test)
        v.addWidget(QLabel("Discord: channel settings -> Integrations -> Webhooks -> New webhook -> Copy URL."))
        v.addStretch()
        return w

    def _test_webhook(self) -> None:
        if self.notifier_factory is None:
            return
        ok = False
        try:
            ok = self.notifier_factory().test()
        except Exception as e:  # network errors etc.
            QMessageBox.warning(self, "Webhook", str(e))
            return
        QMessageBox.information(self, "Webhook", "Sent! Check your channel." if ok else "Failed - check the URL.")

    def _stats_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        form = QFormLayout()
        self.stat_labels = {k: QLabel("-") for k in ("Runtime", "Catches", "Lost", "Per hour", "Recoveries",
                                                      "Last reel")}
        for k, lab in self.stat_labels.items():
            form.addRow(k, lab)
        v.addLayout(form)
        v.addWidget(QLabel("<b>Tracker view</b> - blue: catch bar, red: fish, dot: mouse held"))
        self.preview = QLabel()
        self.preview.setMinimumHeight(60)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111")
        v.addWidget(self.preview)
        v.addStretch()
        return w

    # ------------------------------------------------------------ run control
    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        try:
            self.macro = self.macro_factory()
        except Exception as e:
            QMessageBox.critical(self, "Tidecaller", f"Could not start: {e}")
            return
        self.macro.on_phase = lambda p: self.bridge.phase.emit(p.value)
        self.macro.on_debug = self._debug_hook
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _run(self) -> None:
        try:
            self.macro.run()
        except Exception as e:  # surface crashes instead of dying silently
            self.bridge.phase.emit(f"error: {e}")
        finally:
            self.bridge.finished.emit()

    def _debug_hook(self, img: np.ndarray, r: TrackResult | None, hold: bool) -> None:
        now = time.perf_counter()
        if now - self._last_debug > 1 / 20:  # cap the preview at 20 fps
            self._last_debug = now
            self.bridge.debug.emit(annotate(img, r, hold))

    def stop(self) -> None:
        if self.macro:
            self.macro.stop()

    def toggle_overlay(self) -> None:
        if self._overlay is not None and self._overlay.isVisible():
            self._overlay.close()
            return
        if self.overlay_factory is None:
            return
        self._overlay = self.overlay_factory(self._save)
        if self._overlay is None:
            QMessageBox.warning(self, "Tidecaller", "Roblox window not found - open Fisch first.")
            return
        self._overlay.show()
        self._overlay.activateWindow()

    # ------------------------------------------------------------ slots
    def _on_phase(self, p: str) -> None:
        self.status.setText(p.capitalize())

    def _on_debug(self, img: np.ndarray) -> None:
        self.preview.setPixmap(to_pixmap(img))

    def _on_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if not self.status.text().lower().startswith("error"):
            self.status.setText(Phase.IDLE.value.capitalize())

    def _on_hotkey(self, name: str) -> None:
        {"start": self.start, "stop": self.stop, "area": self.toggle_overlay}[name]()

    def _refresh_stats(self) -> None:
        if not self.macro:
            return
        s: Stats = self.macro.stats
        now = time.perf_counter()
        el = int(now - s.started)
        vals = {
            "Runtime": f"{el // 3600:d}:{el // 60 % 60:02d}:{el % 60:02d}",
            "Catches": str(s.catches),
            "Lost": str(s.lost),
            "Per hour": f"{s.per_hour(now):.0f}",
            "Recoveries": str(s.recoveries),
            "Last reel": f"{s.last_reel_s:.1f}s",
        }
        for k, v in vals.items():
            self.stat_labels[k].setText(v)

    def closeEvent(self, e) -> None:
        self.stop()
        self._save()
        super().closeEvent(e)

