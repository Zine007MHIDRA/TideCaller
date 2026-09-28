"""Record real fish-bar crops from Fisch while you (or the macro) reel, with pre-labels to correct.

Frames where the line tracker finds the bar are saved with its guess as a YOLO label; open the folder in a
labelling tool (e.g. Label Studio, CVAT) to fix mistakes, then merge with the sim dataset.

    python ml/capture_frames.py --out datasets/fishbar_live --fps 8 --minutes 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from tidecaller.config.schema import Config
from tidecaller.core.capture import RateLimiter, make_source
from tidecaller.core.window import find_roblox_client_rect
from tidecaller.modules.reel.trackers import LineTracker

CONFIG_PATH = Path.home() / ".tidecaller" / "config.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/fishbar_live"))
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    tracker = LineTracker(cfg.rod())
    src = make_source(cfg.main.capture.value)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(parents=True, exist_ok=True)
    limiter, end, n = RateLimiter(a.fps), time.time() + a.minutes * 60, 0
    print("Recording - start fishing in Roblox. Ctrl+C to stop.")
    try:
        while time.time() < end:
            limiter.wait()
            client = find_roblox_client_rect()
            if client is None:
                continue
            img = src.grab(cfg.regions.fish_bar.to_rect(client))
            if img is None:
                continue
            r = tracker.track(img)
            if r is None:
                continue  # reel UI not visible
            w = img.shape[1]
            name = f"live_{int(time.time() * 1000)}"
            cv2.imwrite(str(a.out / "images" / f"{name}.png"), img)
            (a.out / "labels" / f"{name}.txt").write_text(
                f"0 {r.fish_x / w:.6f} 0.5 {11 / w:.6f} 1.0\n"
                f"1 {r.bar_center / w:.6f} 0.5 {(r.bar_right - r.bar_left) / w:.6f} 1.0\n")
            n += 1
    except KeyboardInterrupt:
        pass
    print(f"saved {n} frames to {a.out}")


if __name__ == "__main__":
    main()
