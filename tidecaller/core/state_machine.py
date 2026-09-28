"""The macro loop: EQUIP -> CAST -> SHAKE -> REEL -> CATCH -> CONSUMABLES -> EQUIP ...

Every dependency (frames, input, clock, window lookup, notifier) is injected, so the exact same loop drives the
real game, a recorded replay, or the built-in simulator.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import cv2
import numpy as np

from tidecaller.config.schema import Config, ShakeStyle
from tidecaller.core.capture import FrameSource, RateLimiter
from tidecaller.core.input import Input, jitter
from tidecaller.core.regions import Rect, Region
from tidecaller.modules.cast import FullTimeEstimator, ReleaseDecider, meter_is_full
from tidecaller.modules.reel import ReelController, TrackResult, make_tracker
from tidecaller.modules.rod import rod_held
from tidecaller.modules.shake import find_circles, find_pixel
from tidecaller.modules.totem import ConsumableScheduler


class Phase(str, Enum):
    IDLE = "idle"
    EQUIP = "equip"
    CAST = "cast"
    SHAKE = "shake"
    REEL = "reel"
    CATCH = "catch"
    CONSUMABLES = "consumables"
    RECOVER = "recover"


class Stopped(Exception):
    pass


@dataclass
class Stats:
    started: float = field(default_factory=time.perf_counter)
    cycles: int = 0
    catches: int = 0
    lost: int = 0
    recoveries: int = 0
    last_reel_s: float = 0.0
    phase_s: dict[str, float] = field(default_factory=dict)

    def per_hour(self, now: float) -> float:
        hours = max(1e-9, (now - self.started) / 3600)
        return self.catches / hours


def read_progress(img: np.ndarray) -> float | None:
    """Horizontal fill of the reel progress bar (bright pixels from the left)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row = gray[gray.shape[0] // 2]
    filled = row > 170
    if not filled.any():
        return None
    return float(filled.mean())


class Macro:
    def __init__(
        self,
        cfg: Config,
        source: FrameSource,
        inp: Input,
        client_rect: Callable[[], Rect | None],
        is_focused: Callable[[], bool] = lambda: True,
        clock: Callable[[], float] = time.perf_counter,
        notifier=None,
    ) -> None:
        self.cfg, self.src, self.inp = cfg, source, inp
        self.client_rect, self.is_focused, self.clock = client_rect, is_focused, clock
        self.notifier = notifier
        self.phase = Phase.IDLE
        self.stats = Stats(started=clock())
        self.full_time = FullTimeEstimator()
        self.consumables = ConsumableScheduler(cfg.totem, clock())
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._client: Rect | None = None
        self._last_progress: float | None = None
        # hooks for the GUI
        self.on_phase: Callable[[Phase], None] = lambda p: None
        self.on_debug: Callable[[np.ndarray, TrackResult | None, bool], None] | None = None
        self.on_catch: Callable[[Stats, bool], None] = lambda s, ok: None

    # ---------------------------------------------------------------- control
    def stop(self) -> None:
        self._stop.set()

    def pause(self, value: bool = True) -> None:
        (self._paused.set if value else self._paused.clear)()

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    def run(self, max_cycles: int | None = None) -> Stats:
        handlers = {
            Phase.EQUIP: self._equip,
            Phase.CAST: self._cast,
            Phase.SHAKE: self._shake,
            Phase.REEL: self._reel,
            Phase.CATCH: self._catch,
            Phase.CONSUMABLES: self._consumables,
            Phase.RECOVER: self._recover,
        }
        nxt = Phase.CONSUMABLES
        try:
            while not self._stop.is_set():
                if max_cycles is not None and self.stats.cycles >= max_cycles:
                    break
                self._set_phase(nxt)
                t0 = self.clock()
                nxt = handlers[nxt]()
                self.stats.phase_s[self.phase.value] = self.clock() - t0
        except Stopped:
            pass
        finally:
            self.inp.release_all()
            self._set_phase(Phase.IDLE)
        return self.stats

    # ---------------------------------------------------------------- helpers
    def _set_phase(self, p: Phase) -> None:
        self.phase = p
        self.on_phase(p)

    def _checkpoint(self) -> None:
        """Called inside every loop: honours stop/pause and focus loss."""
        if self._stop.is_set():
            raise Stopped
        while self._paused.is_set() or (self.cfg.main.pause_when_unfocused and not self.is_focused()):
            self.inp.release_all()
            if self._stop.wait(0.1):
                raise Stopped

    def _sleep(self, seconds: float) -> None:
        end = self.clock() + seconds
        while (left := end - self.clock()) > 0:
            self._checkpoint()
            time.sleep(min(left, 0.05))

    def _rect(self, region: Region) -> Rect:
        client = self.client_rect() or self._client
        if client is None:
            raise RuntimeError("Roblox window not found")
        self._client = client
        return region.to_rect(client)

    def _grab(self, region: Region) -> tuple[np.ndarray | None, Rect]:
        rect = self._rect(region)
        return self.src.grab(rect), rect

    def _tracker(self):
        return make_tracker(self.cfg.fish.track.value, self.cfg.rod(), self.cfg.fish.yolo_model)

    def _notify(self, kind: str, **kw) -> None:
        if self.notifier is not None:
            try:
                self.notifier.send(kind, stats=self.stats, **kw)
            except Exception:
                pass  # notifications must never break the loop

    # ---------------------------------------------------------------- phases
    def _rod_held(self) -> bool | None:
        img, _ = self._grab(self.cfg.regions.rod_info)
        return None if img is None else rod_held(img)

    def _wait_rod_ready(self, timeout_s: float) -> bool:
        """Wait for the rod's Power bar: it only comes back once catch popups/animations are done."""
        end = self.clock() + timeout_s
        while self.clock() < end:
            if self._rod_held():
                return True
            self._sleep(0.05)
        return False

    def _equip(self) -> Phase:
        """Press the rod slot only when we can SEE the rod isn't held. Pressing it while held unequips the rod,
        so this never presses twice in a row: if the check still fails afterwards we cast anyway and let the
        cast/shake timeouts decide."""
        m = self.cfg.main
        # give the UI a moment first: right after a catch the hotbar can still be hidden
        if m.auto_select_rod and not self._wait_rod_ready(1.5):
            self.inp.tap(str(m.rod_slot))
            self._wait_rod_ready(1.5)
        return Phase.CAST

    def _cast(self) -> Phase:
        c = self.cfg.cast
        decider = ReleaseDecider(c, self.clock(), latency_s=c.input_latency_ms / 1000,
                                 time_to_full_s=self.full_time.value)
        limiter = RateLimiter(self.cfg.fish.scan_fps)
        # center the cursor in the game area so the click lands on the world, not UI
        client = self._rect(Region(0, 0, 1, 1))
        self.inp.move(client.left + client.width // 2, client.top + client.height // 2)
        self.inp.mouse_down()
        while True:
            self._checkpoint()
            img, _ = self._grab(self.cfg.regions.cast_meter)
            full = img is not None and meter_is_full(img)
            now = self.clock()
            if full:
                self.full_time.update(now - decider.t0)
            if decider.should_release(now, full):
                break
            limiter.wait()
        self.inp.mouse_up()
        self._sleep(c.post_cast_delay_ms / 1000)
        return Phase.SHAKE

    def skips_shake(self) -> bool:
        """High-lure rods (e.g. Masterline) and instant baits go straight from cast to reel."""
        s = self.cfg.shake
        if s.style == ShakeStyle.NONE:
            return True
        if s.skip_shake_lure_pct <= 0:
            return False
        from tidecaller.config.rod_presets import load_stats

        stats = load_stats().get(self.cfg.fish.rod)
        return bool(stats and stats.lure_pct is not None and stats.lure_pct >= s.skip_shake_lure_pct)

    def _shake(self) -> Phase:
        s = self.cfg.shake
        tracker = self._tracker()
        start = self.clock()
        deadline = start + s.timeout_s
        skip = self.skips_shake()
        nav_on = False
        seen = 0  # consecutive frames with a bar: one stray frame (glow, particles) must not start the reel
        while self.clock() < deadline:
            self._checkpoint()
            bar, _ = self._grab(self.cfg.regions.fish_bar)
            seen = seen + 1 if bar is not None and tracker.track(bar) is not None else 0
            if seen >= 3:
                if nav_on:
                    self.inp.tap(s.navigation_key)
                return Phase.REEL
            if seen:
                self._sleep(0.01)
                continue
            if skip or self.clock() - start < s.instant_grace_s:
                self._sleep(0.03)  # only watch for the reel; never click
                continue
            if s.style == ShakeStyle.NAVIGATION:
                if not nav_on:
                    self.inp.tap(s.navigation_key)
                    nav_on = True
                self.inp.tap("enter")
            else:
                img, rect = self._grab(self.cfg.regions.shake)
                if img is not None:
                    if s.style == ShakeStyle.CIRCLE:
                        hits = find_circles(img, s)
                        pt = hits[0] if hits else None
                    else:
                        pt = find_pixel(img, s)
                    if pt is not None:
                        self.inp.click(rect.left + pt[0], rect.top + pt[1])
            self._sleep(jitter(s.click_interval_ms))
        return Phase.RECOVER

    def _reel(self) -> Phase:
        f = self.cfg.fish
        tracker = self._tracker()
        ctl = ReelController(self.cfg.rod())
        limiter = RateLimiter(f.scan_fps)
        t_start = self.clock()
        last_seen = t_start
        self._last_progress = None
        while True:
            self._checkpoint()
            now = self.clock()
            if now - t_start > f.reel_timeout_s:
                self.inp.mouse_up()
                return Phase.RECOVER
            img, _ = self._grab(self.cfg.regions.fish_bar)
            r = tracker.track(img) if img is not None else None
            if r is None:
                if now - last_seen > 0.35:  # UI gone -> reel finished
                    break
                limiter.wait()
                continue
            last_seen = now
            hold = ctl.step(r, now)
            (self.inp.mouse_down if hold else self.inp.mouse_up)()
            prog, _ = self._grab(self.cfg.regions.reel_progress)
            if prog is not None:
                p = read_progress(prog)
                if p is not None:
                    self._last_progress = p
            if self.on_debug is not None:
                self.on_debug(img, r, hold)
            limiter.wait()
        self.inp.mouse_up()
        self.stats.last_reel_s = self.clock() - t_start
        return Phase.CATCH

    def _catch(self) -> Phase:
        # Progress was climbing when the UI vanished -> caught; otherwise the fish escaped.
        ok = self._last_progress is None or self._last_progress >= 0.5
        self.stats.cycles += 1
        if ok:
            self.stats.catches += 1
        else:
            self.stats.lost += 1
        if self.cfg.fish.bag_spam:
            for _ in range(2):  # open+close the backpack to dismiss the catch popup
                self.inp.tap(self.cfg.fish.bag_key)
                self._sleep(jitter(120))
        self._wait_rod_ready(3.0)  # casting during the catch popup is ignored by the game
        self.on_catch(self.stats, ok)
        if self.cfg.discord.on_catch:
            self._notify("catch", ok=ok)
        return Phase.CONSUMABLES

    def _consumables(self) -> Phase:
        t, cyc = self.clock(), self.stats.cycles
        used = False
        if self.consumables.totem_due(t, cyc):
            self._use_slot(self.cfg.totem.slot)
            self.consumables.mark_totem(t, cyc)
            used = True
        if self.consumables.potion_due(t):
            self._use_slot(self.cfg.totem.potion_slot)
            self.consumables.mark_potion(t)
            used = True
        if used and not self.cfg.main.auto_select_rod:
            self.inp.tap(str(self.cfg.main.rod_slot))
        return Phase.EQUIP

    def _use_slot(self, slot: int) -> None:
        self.inp.tap(str(slot))
        self._sleep(jitter(350))
        self.inp.click()
        self._sleep(jitter(1500))

    def _recover(self) -> Phase:
        self.stats.recoveries += 1
        self.inp.release_all()
        # no blind key presses here: EQUIP re-checks the rod, and the next cast's click reels in any stuck line
        self._sleep(jitter(800))
        if self.cfg.discord.on_error:
            self._notify("stuck", phase=self.phase.value)
        return Phase.EQUIP
