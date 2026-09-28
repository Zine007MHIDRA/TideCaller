"""Casting: hold the mouse while the power meter fills, release according to a strategy."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from tidecaller.config.schema import CastSettings, CastStyle, ReleaseStyle


def read_meter(img: np.ndarray) -> float | None:
    """Fill fraction (0..1) of the vertical cast meter, measured from the bottom. None if no meter is visible.

    The meter fill is a saturated, bright color (green -> yellow -> red depending on power); the empty part is a
    dark translucent track. We look at the central columns and find the topmost filled row.
    """
    h, w = img.shape[:2]
    cols = img[:, max(0, w // 2 - 2): w // 2 + 3]
    hsv = cv2.cvtColor(cols, cv2.COLOR_BGR2HSV)
    filled = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 120)).mean(axis=1) > 0.5
    if filled.sum() < 2:
        return None
    return float(h - np.flatnonzero(filled)[0]) / h


@dataclass
class ReleaseDecider:
    """Stateful per-cast decision: feed it (t, fill) samples, it says when to let go."""
    cfg: CastSettings
    t0: float
    latency_s: float = 0.035
    _hist: deque = field(default_factory=lambda: deque(maxlen=6))

    @property
    def style(self) -> ReleaseStyle:
        # perfect casts always need the predictive release, regardless of the chosen release style
        return ReleaseStyle.PREDICTIVE if self.cfg.style == CastStyle.PERFECT else self.cfg.release

    def should_release(self, t: float, fill: float | None) -> bool:
        elapsed = t - self.t0
        style = self.style
        if style == ReleaseStyle.TIMED:
            return elapsed * 1000 >= self.cfg.timed_hold_ms
        if fill is not None:
            self._hist.append((t, fill))
        if style == ReleaseStyle.PREDICTIVE:
            if len(self._hist) >= 3:
                (ta, fa), (tb, fb) = self._hist[0], self._hist[-1]
                v = (fb - fa) / (tb - ta) if tb > ta else 0.0
                # release so that the meter reaches the threshold when the input actually lands
                if fill is not None and v > 0 and fill + v * self.latency_s >= self.cfg.fill_threshold:
                    return True
        elif fill is not None and fill >= self.cfg.fill_threshold:
            return True
        if style == ReleaseStyle.THRESHOLD:
            return False
        return elapsed * 1000 >= self.cfg.timeout_ms  # idiot-proof / predictive fallback


class LatencyEstimator:
    """Learns the real input latency from perfect-cast outcomes (EMA)."""

    def __init__(self, seed_ms: float, alpha: float = 0.2) -> None:
        self.ms, self.alpha = seed_ms, alpha

    def update(self, overshoot: float, velocity: float) -> None:
        """overshoot = final fill - target (negative if short); velocity in fill/s."""
        if velocity <= 0:
            return
        observed = self.ms + overshoot / velocity * 1000
        self.ms = max(0.0, (1 - self.alpha) * self.ms + self.alpha * observed)
