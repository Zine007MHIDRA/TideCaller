"""Casting: hold the mouse while the power meter fills, release according to a strategy.

The real meter (measured on gameplay footage) is a thin, dark-outlined vertical bar right of the character that
fills WHITE from the bottom and turns entirely GREEN at full power, which is the "PERFECT!" window (~200 ms).
A tall solid green column is unmistakable even with bright auras around the player, so release keys off that
instead of trying to measure the white fill against busy backgrounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tidecaller.config.schema import CastSettings, CastStyle, ReleaseStyle

GREEN_MIN_FRAC = 0.18  # of the search box height; the full meter is ~0.6 of the default box


def _longest_run(col: np.ndarray, max_gap: int = 3) -> int:
    best = run = gap = 0
    for v in col:
        if v:
            run += gap + 1 if run else 1
            gap = 0
            best = max(best, run)
        elif run:
            gap += 1
            if gap > max_gap:
                run = gap = 0
    return best


def meter_is_full(img: np.ndarray) -> bool:
    """True while the cast meter is a solid green column (full power / perfect window)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (h > 35) & (h < 90) & (s > 80) & (v > 110)
    counts = green.sum(axis=0)
    if counts.max() < GREEN_MIN_FRAC * img.shape[0]:
        return False
    for x in np.argsort(counts)[::-1][:5]:
        if _longest_run(green[:, x]) >= GREEN_MIN_FRAC * img.shape[0]:
            return True
    return False


@dataclass
class ReleaseDecider:
    """Stateful per-cast decision: feed it (t, full) samples, it says when to let go."""
    cfg: CastSettings
    t0: float
    latency_s: float = 0.035
    time_to_full_s: float | None = None  # learned by FullTimeEstimator, used by the predictive style

    @property
    def style(self) -> ReleaseStyle:
        # perfect casts need a timing-aware release regardless of the chosen style
        if self.cfg.style == CastStyle.PERFECT and self.cfg.release == ReleaseStyle.TIMED:
            return ReleaseStyle.IDIOT_PROOF
        return self.cfg.release

    def should_release(self, t: float, full: bool) -> bool:
        elapsed = t - self.t0
        style = self.style
        if style == ReleaseStyle.TIMED:
            return elapsed * 1000 >= self.cfg.timed_hold_ms
        if full:
            return True
        if style == ReleaseStyle.PREDICTIVE and self.time_to_full_s is not None:
            if elapsed >= self.time_to_full_s - self.latency_s:
                return True
        cap = self.cfg.timeout_ms * (3 if style == ReleaseStyle.THRESHOLD else 1)
        return elapsed * 1000 >= cap


class FullTimeEstimator:
    """Learns how long the meter takes to fill (EMA over casts where green was actually seen)."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.value: float | None = None
        self.alpha = alpha

    def update(self, seconds: float) -> None:
        self.value = seconds if self.value is None else (1 - self.alpha) * self.value + self.alpha * seconds
