"""Entry point.

    python -m tidecaller                 # live: plays Fisch in the Roblox window
    python -m tidecaller --sim           # demo: plays the built-in simulator (no Roblox needed)
    python -m tidecaller --replay a.mp4  # dry run: reads a recording, input is only logged
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel

from tidecaller.config.schema import Config
from tidecaller.core.capture import ReplaySource, StaticSource, make_source
from tidecaller.core.regions import Rect
from tidecaller.core.state_machine import Macro
from tidecaller.core.window import find_roblox_client_rect, roblox_is_focused
from tidecaller.gui.app import MainWindow, to_pixmap
from tidecaller.notify.discord import DiscordNotifier

CONFIG_PATH = Path.home() / ".tidecaller" / "config.json"


def register_hotkeys(win: MainWindow, cfg: Config) -> None:
    try:
        import keyboard
    except ImportError:
        return
    hk = cfg.main.hotkeys
    for key, name in ((hk.start, "start"), (hk.stop, "stop"), (hk.area_editor, "area")):
        keyboard.add_hotkey(key, lambda n=name: win.bridge.hotkey.emit(n))


def screenshot_png(cfg: Config) -> bytes | None:
    import cv2

    rect = find_roblox_client_rect()
    if rect is None:
        return None
    img = make_source("mss").grab(rect)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tidecaller")
    ap.add_argument("--sim", action="store_true", help="play the built-in Fisch simulator")
    ap.add_argument("--replay", type=Path, help="run against a screen recording (input is not sent)")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    app = QApplication(sys.argv)
    notifier_factory = lambda: DiscordNotifier(cfg.discord, lambda: screenshot_png(cfg))  # noqa: E731

    if args.sim:
        from tidecaller.sim.fisch_sim import FischSim, SimInput

        sim = FischSim(regions=cfg.regions)
        sim_view = QLabel()
        sim_view.setWindowTitle("Fisch simulator")
        refresh = QTimer(interval=33, timeout=lambda: sim_view.setPixmap(to_pixmap(sim.render(), 960)))
        refresh.start()
        sim_view.show()

        def factory() -> Macro:
            cfg.main.pause_when_unfocused = False
            return Macro(cfg, StaticSource(sim.render), SimInput(sim), client_rect=lambda: sim.client,
                         notifier=notifier_factory())

        overlay_factory = None
    elif args.replay:
        from tidecaller.core.input import RecordingInput

        def factory() -> Macro:
            src = ReplaySource(args.replay)
            w, h = src.size
            inp = RecordingInput()
            m = Macro(cfg, src, inp, client_rect=lambda: Rect(0, 0, w, h))

            def watch() -> None:  # stop when the recording ends
                while m.running and not src.finished:
                    threading.Event().wait(0.2)
                m.stop()

            threading.Thread(target=watch, daemon=True).start()
            return m

        overlay_factory = None
    else:
        from tidecaller.core.input import DirectInput
        from tidecaller.gui.overlay import AreaOverlay

        def factory() -> Macro:
            return Macro(cfg, make_source(cfg.main.capture.value), DirectInput(), find_roblox_client_rect,
                         is_focused=roblox_is_focused, notifier=notifier_factory())

        def overlay_factory(on_close):
            client = find_roblox_client_rect()
            return None if client is None else AreaOverlay(cfg.regions, client, on_close)

    win = MainWindow(cfg, args.config, factory, overlay_factory, notifier_factory)
    win.show()
    register_hotkeys(win, cfg)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
