"""Keyboard/mouse output. Every module goes through `Input` so it can be swapped for a recorder in tests."""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def jitter(ms: float, pct: float = 0.15) -> float:
    """Seconds, randomised +-pct so timings are not machine-identical."""
    return max(0.0, ms * (1 + random.uniform(-pct, pct))) / 1000.0


class Input(ABC):
    @abstractmethod
    def mouse_down(self) -> None: ...
    @abstractmethod
    def mouse_up(self) -> None: ...
    @abstractmethod
    def move(self, x: int, y: int) -> None: ...
    @abstractmethod
    def key_down(self, key: str) -> None: ...
    @abstractmethod
    def key_up(self, key: str) -> None: ...

    def click(self, x: int | None = None, y: int | None = None, hold_ms: float = 30) -> None:
        if x is not None and y is not None:
            self.move(x, y)
        self.mouse_down()
        time.sleep(jitter(hold_ms))
        self.mouse_up()

    def tap(self, key: str, hold_ms: float = 30) -> None:
        self.key_down(key)
        time.sleep(jitter(hold_ms))
        self.key_up(key)

    def release_all(self) -> None:
        self.mouse_up()


class DirectInput(Input):
    """Scan-code input via pydirectinput; Roblox reads these reliably where VK-based SendInput can fail."""

    def __init__(self) -> None:
        import pydirectinput

        pydirectinput.PAUSE = 0
        pydirectinput.FAILSAFE = False
        self._pdi = pydirectinput
        self._mouse_is_down = False

    def mouse_down(self) -> None:
        if not self._mouse_is_down:
            self._pdi.mouseDown()
            self._mouse_is_down = True

    def mouse_up(self) -> None:
        if self._mouse_is_down:
            self._pdi.mouseUp()
            self._mouse_is_down = False

    def move(self, x: int, y: int) -> None:
        self._pdi.moveTo(x, y)
        # Roblox only registers hover after a relative nudge
        self._pdi.moveRel(1, 0, relative=True)
        self._pdi.moveRel(-1, 0, relative=True)

    def key_down(self, key: str) -> None:
        self._pdi.keyDown(key)

    def key_up(self, key: str) -> None:
        self._pdi.keyUp(key)


@dataclass
class RecordingInput(Input):
    """No-op backend that logs every event. Used by replay mode and tests."""
    events: list[tuple[float, str, object]] = field(default_factory=list)
    mouse_is_down: bool = False

    def _log(self, kind: str, arg: object = None) -> None:
        self.events.append((time.perf_counter(), kind, arg))

    def mouse_down(self) -> None:
        if not self.mouse_is_down:
            self.mouse_is_down = True
            self._log("mouse_down")

    def mouse_up(self) -> None:
        if self.mouse_is_down:
            self.mouse_is_down = False
            self._log("mouse_up")

    def move(self, x: int, y: int) -> None:
        self._log("move", (x, y))

    def key_down(self, key: str) -> None:
        self._log("key_down", key)

    def key_up(self, key: str) -> None:
        self._log("key_up", key)

    def kinds(self) -> list[str]:
        return [k for _, k, _ in self.events]
