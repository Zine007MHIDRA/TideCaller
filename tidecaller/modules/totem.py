"""Totem and potion scheduling. Pure bookkeeping; the state machine performs the actual key presses."""
from __future__ import annotations

from dataclasses import dataclass

from tidecaller.config.schema import TotemSettings, TotemTrigger


@dataclass
class ConsumableScheduler:
    cfg: TotemSettings
    start_t: float
    _last_totem_t: float | None = None
    _last_totem_cycle: int = 0
    _last_potion_t: float | None = None

    def totem_due(self, t: float, cycles: int) -> bool:
        if not self.cfg.enabled:
            return False
        if self._last_totem_t is None:
            return True
        if self.cfg.trigger == TotemTrigger.CYCLES:
            return cycles - self._last_totem_cycle >= self.cfg.every_cycles
        return t - self._last_totem_t >= self.cfg.every_minutes * 60

    def potion_due(self, t: float) -> bool:
        if not self.cfg.auto_potion:
            return False
        return self._last_potion_t is None or t - self._last_potion_t >= self.cfg.potion_every_minutes * 60

    def mark_totem(self, t: float, cycles: int) -> None:
        self._last_totem_t, self._last_totem_cycle = t, cycles

    def mark_potion(self, t: float) -> None:
        self._last_potion_t = t
