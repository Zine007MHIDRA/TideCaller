"""Reel controller: turns tracker output into mouse hold/release.

Holding pushes the catch bar right, releasing lets it drift left. A PD term gives a desired duty cycle
(0 = always released, 1 = always held) and a sigma-delta modulator turns that into per-frame hold decisions,
so the effective force is smooth even though the mouse is binary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tidecaller.config.schema import RodProfile
from tidecaller.modules.reel.trackers import TrackResult


@dataclass
class ReelController:
    rod: RodProfile
    _prev_err: float | None = None
    _prev_t: float | None = None
    _acc: float = 0.0
    last_duty: float = field(default=0.5, init=False)

    def reset(self) -> None:
        self._prev_err = self._prev_t = None
        self._acc = 0.0

    def duty(self, r: TrackResult, t: float) -> float:
        err = r.error
        if abs(err) < self.rod.deadzone_px:
            err = 0.0
        derr = 0.0
        if self._prev_err is not None and self._prev_t is not None and t > self._prev_t:
            derr = (err - self._prev_err) / (t - self._prev_t)
        self._prev_err, self._prev_t = err, t
        u = self.rod.kp * err + self.rod.kd * derr
        self.last_duty = min(1.0, max(0.0, 0.5 + u))
        return self.last_duty

    def step(self, r: TrackResult, t: float) -> bool:
        """True -> hold the mouse this frame."""
        self._acc += self.duty(r, t)
        if self._acc >= 1.0:
            self._acc -= 1.0
            return True
        return False
