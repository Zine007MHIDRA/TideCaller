"""A tiny stand-in for Fisch's fishing UI, driven by the same Input interface the macro uses.

It is NOT a pixel-perfect copy of the game; it reproduces the mechanics the macro has to solve:
  * cast meter that fills while the mouse is held,
  * white round shake buttons that must be clicked N times,
  * a reel bar pushed right while holding / falling left when released, chasing a wandering fish,
  * a progress bar that fills while the fish is inside the bar (catch at 100%, lose at 0%).
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from tidecaller.config.schema import Regions
from tidecaller.core.input import Input
from tidecaller.core.regions import Rect

W, H = 1280, 720


@dataclass
class SimParams:
    meter_fill_s: float = 1.1
    bite_delay_s: float = 0.6
    shakes_needed: int = 3
    bar_width_frac: float = 0.28
    bar_accel: float = 2.6        # bar-widths/s^2 while holding (in fractions of track)
    bar_gravity: float = 2.4
    fish_speed: float = 0.35      # track fractions per second
    fish_jumpiness: float = 1.2   # target changes per second
    progress_gain: float = 0.45   # per second while inside
    progress_loss: float = 0.30
    seed: int | None = 7


@dataclass
class FischSim:
    regions: Regions = field(default_factory=Regions)
    p: SimParams = field(default_factory=SimParams)
    clock: callable = time.perf_counter

    def __post_init__(self) -> None:
        self.rng = random.Random(self.p.seed)
        self.client = Rect(0, 0, W, H)
        self.lock = threading.Lock()
        self.mouse_down = False
        self.cursor = (W // 2, H // 2)
        self.equipped_slot: int | None = None
        self.state = "idle"
        self.t_state = self.clock()
        self.fill = 0.0
        self.shakes_left = 0
        self.button: tuple[int, int, int] | None = None
        self.bar_x = 0.3      # left edge, fraction of track
        self.bar_v = 0.0
        self.fish_x = 0.5
        self.fish_target = 0.5
        self.progress = 0.3
        self.caught = 0
        self.lost = 0
        self.last_cast_fill: float | None = None
        self._t_last = self.clock()

    # ------------------------------------------------------------ input hooks
    def press(self) -> None:
        with self.lock:
            self.mouse_down = True
            if self.state == "idle" and self.equipped_slot == 1:
                self._goto("casting")
                self.fill = 0.0
            elif self.state == "shake" and self.button is not None:
                bx, by, r = self.button
                if (self.cursor[0] - bx) ** 2 + (self.cursor[1] - by) ** 2 <= r * r:
                    self.shakes_left -= 1
                    self._spawn_button() if self.shakes_left > 0 else self._start_reel()

    def release(self) -> None:
        with self.lock:
            self.mouse_down = False
            if self.state == "casting":
                self.last_cast_fill = self.fill
                self._goto("waiting")

    def key(self, key: str) -> None:
        with self.lock:
            if key.isdigit():
                slot = int(key)
                self.equipped_slot = None if self.equipped_slot == slot else slot
                if self.equipped_slot is None:
                    self._goto("idle")

    # ------------------------------------------------------------ simulation
    def _goto(self, s: str) -> None:
        self.state, self.t_state = s, self.clock()

    def _spawn_button(self) -> None:
        rr = self.regions.shake.to_rect(self.client)
        r = 34
        self.button = (self.rng.randint(rr.left + r, rr.right - r), self.rng.randint(rr.top + r, rr.bottom - r), r)

    def _start_reel(self) -> None:
        self.button = None
        self.bar_x, self.bar_v = 0.5 - self.p.bar_width_frac / 2, 0.0
        self.fish_x = self.fish_target = self.rng.uniform(0.2, 0.8)
        self.progress = 0.3
        self._goto("reel")

    def tick(self) -> None:
        with self.lock:
            now = self.clock()
            dt = min(0.05, now - self._t_last)
            self._t_last = now
            if self.state == "casting":
                self.fill = min(1.0, self.fill + dt / self.p.meter_fill_s)
            elif self.state == "waiting" and now - self.t_state > self.p.bite_delay_s:
                self.shakes_left = self.p.shakes_needed
                self._spawn_button()
                self._goto("shake")
            elif self.state == "reel":
                self._tick_reel(dt)
            elif self.state in ("caught", "escaped") and now - self.t_state > 0.6:
                self._goto("idle")

    def _tick_reel(self, dt: float) -> None:
        p, bw = self.p, self.p.bar_width_frac
        self.bar_v += (p.bar_accel if self.mouse_down else -p.bar_gravity) * dt
        self.bar_x += self.bar_v * dt
        if self.bar_x <= 0 or self.bar_x >= 1 - bw:  # walls kill velocity
            self.bar_x = min(max(self.bar_x, 0.0), 1 - bw)
            self.bar_v = 0.0
        if self.rng.random() < p.fish_jumpiness * dt:
            self.fish_target = self.rng.uniform(0.05, 0.95)
        step = p.fish_speed * dt
        self.fish_x += max(-step, min(step, self.fish_target - self.fish_x))
        inside = self.bar_x <= self.fish_x <= self.bar_x + bw
        self.progress += (p.progress_gain if inside else -p.progress_loss) * dt
        if self.progress >= 1:
            self.caught += 1
            self._goto("caught")
        elif self.progress <= 0:
            self.lost += 1
            self._goto("escaped")

    # ------------------------------------------------------------ rendering
    def render(self) -> np.ndarray:
        self.tick()
        with self.lock:
            img = np.full((H, W, 3), (120, 90, 40), np.uint8)  # "ocean"
            img[: H // 3] = (200, 160, 110)                     # "sky"
            self._draw_hotbar(img)
            if self.state == "casting":
                self._draw_meter(img)
            if self.state == "shake" and self.button:
                bx, by, r = self.button
                cv2.circle(img, (bx, by), r, (255, 255, 255), -1)
                cv2.circle(img, (bx, by), r, (60, 60, 60), 2)
            if self.state == "reel":
                self._draw_reel(img)
            return img

    def _draw_hotbar(self, img: np.ndarray) -> None:
        r = self.regions.hotbar.to_rect(self.client)
        sw = r.width / 9
        for i in range(9):
            x0 = int(r.left + i * sw)
            cv2.rectangle(img, (x0 + 2, r.top + 2), (int(x0 + sw) - 2, r.bottom - 2), (40, 40, 40), -1)
            if self.equipped_slot == i + 1:
                cv2.rectangle(img, (x0, r.top), (int(x0 + sw) - 1, r.bottom - 1), (255, 255, 255), 3)

    def _draw_meter(self, img: np.ndarray) -> None:
        r = self.regions.cast_meter.to_rect(self.client)
        cv2.rectangle(img, (r.left, r.top), (r.right - 1, r.bottom - 1), (30, 30, 30), -1)
        top = int(r.bottom - self.fill * r.height)
        hue_bgr = (40, 220, 60) if self.fill < 0.9 else (40, 200, 240)
        cv2.rectangle(img, (r.left, top), (r.right - 1, r.bottom - 1), hue_bgr, -1)

    def _draw_reel(self, img: np.ndarray) -> None:
        r = self.regions.fish_bar.to_rect(self.client)
        cv2.rectangle(img, (r.left, r.top), (r.right - 1, r.bottom - 1), (70, 60, 50), -1)
        bl = int(r.left + self.bar_x * r.width)
        br = int(bl + self.p.bar_width_frac * r.width)
        cv2.rectangle(img, (bl, r.top + 2), (br, r.bottom - 3), (245, 245, 245), -1)
        fx = int(r.left + self.fish_x * r.width)
        cv2.rectangle(img, (fx - 5, r.top + 1), (fx + 5, r.bottom - 2), (25, 25, 25), -1)
        pr = self.regions.reel_progress.to_rect(self.client)
        cv2.rectangle(img, (pr.left, pr.top), (pr.right - 1, pr.bottom - 1), (50, 50, 50), -1)
        cv2.rectangle(img, (pr.left, pr.top), (int(pr.left + self.progress * pr.width), pr.bottom - 1),
                      (90, 230, 90), -1)


class SimInput(Input):
    """Routes the macro's input straight into the simulator."""

    def __init__(self, sim: FischSim) -> None:
        self.sim = sim

    def mouse_down(self) -> None:
        if not self.sim.mouse_down:
            self.sim.press()

    def mouse_up(self) -> None:
        if self.sim.mouse_down:
            self.sim.release()

    def move(self, x: int, y: int) -> None:
        self.sim.cursor = (x, y)

    def key_down(self, key: str) -> None:
        self.sim.key(key)

    def key_up(self, key: str) -> None:
        pass
