"""Locate the fish marker and the player's catch bar inside the fish-bar region.

All trackers return x-positions in region-local pixels, so the controller is backend-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from tidecaller.config.schema import RodProfile


@dataclass(frozen=True)
class TrackResult:
    fish_x: float
    bar_left: float
    bar_right: float
    width: int

    @property
    def bar_center(self) -> float:
        return (self.bar_left + self.bar_right) / 2

    @property
    def error(self) -> float:
        """Positive -> fish is right of the bar -> hold."""
        return self.fish_x - self.bar_center


def _span(mask_1d: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    """Extent of the largest cluster of True values, bridging gaps <= max_gap (the fish icon cuts the bar)."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > max_gap)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of each contiguous True run."""
    padded = np.r_[False, mask_1d, False].astype(np.int8)
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _fish_center(mask_1d: np.ndarray, max_width: int) -> float | None:
    """The fish is a narrow marker; wide dark runs are the empty track, so ignore them."""
    best = None
    for s, e in _runs(mask_1d):
        width = e - s + 1
        if 2 <= width <= max_width and (best is None or width > best[1] - best[0] + 1):
            best = (s, e)
    return None if best is None else (best[0] + best[1]) / 2


class Tracker(ABC):
    min_bar_px = 12

    def __init__(self, rod: RodProfile) -> None:
        self.rod = rod

    @abstractmethod
    def track(self, img: np.ndarray) -> TrackResult | None:
        """`img` is the BGR crop of the fish-bar region. None when the reel UI isn't visible."""


class LineTracker(Tracker):
    """Scans a few rows through the middle of the bar. Cheapest option, ~0.1 ms per frame."""

    def track(self, img: np.ndarray) -> TrackResult | None:
        h, w = img.shape[:2]
        rows = img[max(0, h // 2 - 1): h // 2 + 2]
        gray = cv2.cvtColor(rows, cv2.COLOR_BGR2GRAY).mean(axis=0)
        return self._from_masks(gray >= self.rod.line_bar_min, gray <= self.rod.line_fish_max, w)

    def _from_masks(self, bar: np.ndarray, fish: np.ndarray, w: int) -> TrackResult | None:
        fish_x = _fish_center(fish, max_width=max(6, int(w * 0.12)))
        span = _span(bar, max_gap=max(8, w // 25))
        if fish_x is None or span is None or span[1] - span[0] < self.min_bar_px:
            return None
        return TrackResult(fish_x, span[0], span[1], w)


class ColorTracker(LineTracker):
    """HSV masks projected onto the x axis. More robust to rods that tint the bar."""

    def track(self, img: np.ndarray) -> TrackResult | None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        r = self.rod
        bar = cv2.inRange(hsv, np.array(r.bar_hsv_lo), np.array(r.bar_hsv_hi))
        fish = cv2.inRange(hsv, np.array(r.fish_hsv_lo), np.array(r.fish_hsv_hi))
        h = img.shape[0]
        # a column counts if a meaningful share of its pixels match -> ignores thin borders/text
        return self._from_masks((bar > 0).sum(axis=0) > h * 0.4, (fish > 0).sum(axis=0) > h * 0.3, img.shape[1])


def make_tracker(style: str, rod: RodProfile, yolo_model: str | None = None) -> Tracker:
    if style == "color":
        return ColorTracker(rod)
    if style == "yolo":
        from tidecaller.modules.reel.yolo import YoloTracker

        return YoloTracker(rod, yolo_model or "models/fishbar.onnx")
    return LineTracker(rod)
