"""Frame sources. Everything downstream consumes BGR numpy arrays via `FrameSource.grab`."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from tidecaller.core.regions import Rect


class FrameSource(ABC):
    @abstractmethod
    def grab(self, rect: Rect) -> np.ndarray | None:
        """Return a BGR image of `rect` (screen coords), or None if no frame is available."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class MssSource(FrameSource):
    """GDI capture. Slower, but works while OBS / Discord are capturing the screen."""

    def __init__(self) -> None:
        import mss

        self._sct = mss.mss()

    def grab(self, rect: Rect) -> np.ndarray | None:
        shot = self._sct.grab({"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height})
        return np.asarray(shot)[:, :, :3]  # BGRA -> BGR

    def close(self) -> None:
        self._sct.close()


class DxcamSource(FrameSource):
    """Desktop Duplication API capture. High FPS; returns None when the screen hasn't changed."""

    def __init__(self) -> None:
        import dxcam

        self._cam = dxcam.create(output_color="BGR")
        self._last: dict[tuple, np.ndarray] = {}

    def grab(self, rect: Rect) -> np.ndarray | None:
        key = rect.as_ltrb()
        frame = self._cam.grab(region=key)
        if frame is None:  # unchanged since last grab -> reuse
            return self._last.get(key)
        self._last[key] = frame
        return frame

    def close(self) -> None:
        del self._cam


class ReplaySource(FrameSource):
    """Plays back a recorded full-screen video so the whole pipeline runs without Roblox.

    `rect` is interpreted in the video's own pixel space.
    """

    def __init__(self, path: Path, fps: float | None = None, loop: bool = False) -> None:
        import cv2

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(path)
        self._fps = fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._loop = loop
        self._frame: np.ndarray | None = None
        self._t0 = time.perf_counter()
        self._idx = -1
        self.finished = False

    @property
    def size(self) -> tuple[int, int]:
        import cv2

        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _advance_to(self, idx: int) -> None:
        import cv2

        while self._idx < idx:
            ok, frame = self._cap.read()
            if not ok:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.finished = True
                return
            self._frame, self._idx = frame, self._idx + 1

    def grab(self, rect: Rect) -> np.ndarray | None:
        self._advance_to(int((time.perf_counter() - self._t0) * self._fps))
        if self._frame is None:
            return None
        return self._frame[rect.top:rect.bottom, rect.left:rect.right].copy()

    def close(self) -> None:
        self._cap.release()


class StaticSource(FrameSource):
    """A fixed full-screen image (or a callable producing one). Used by tests and the simulator."""

    def __init__(self, frame_fn) -> None:
        self._fn = frame_fn if callable(frame_fn) else (lambda: frame_fn)

    def grab(self, rect: Rect) -> np.ndarray | None:
        return self._fn()[rect.top:rect.bottom, rect.left:rect.right].copy()


def make_source(backend: str) -> FrameSource:
    if backend == "dxcam":
        try:
            return DxcamSource()
        except Exception:  # no D3D11 output (RDP, some laptops) -> degrade gracefully
            return MssSource()
    return MssSource()


class RateLimiter:
    """Sleeps just enough to hold a target loop frequency."""

    def __init__(self, hz: float) -> None:
        self.period = 1.0 / hz
        self._next = time.perf_counter()

    def wait(self) -> None:
        self._next += self.period
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.perf_counter()
