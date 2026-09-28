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
    fish_visible: bool = True

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


def _fish_center(mask_1d: np.ndarray, max_width: int, weight: np.ndarray | None = None) -> float | None:
    """The fish is a narrow marker; wide runs are the empty track or a skin decoration, so ignore them."""
    best, best_score = None, -1.0
    for s, e in _runs(mask_1d):
        width = e - s + 1
        if not 2 <= width <= max_width:
            continue
        score = float(weight[s:e + 1].sum()) if weight is not None else width
        if score > best_score:
            best, best_score = (s, e), score
    return None if best is None else (best[0] + best[1]) / 2


def in_hsv_range(hsv: np.ndarray, lo, hi) -> np.ndarray:
    """cv2.inRange, but a hue range with lo > hi wraps around red (e.g. 170..10)."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sv = (s >= lo[1]) & (s <= hi[1]) & (v >= lo[2]) & (v <= hi[2])
    hue = (h >= lo[0]) & (h <= hi[0]) if lo[0] <= hi[0] else (h >= lo[0]) | (h <= hi[0])
    return sv & hue


class Tracker(ABC):
    min_bar_px = 12

    def __init__(self, rod: RodProfile) -> None:
        self.rod = rod

    @abstractmethod
    def track(self, img: np.ndarray) -> TrackResult | None:
        """`img` is the BGR crop of the fish-bar region. None when the reel UI isn't visible."""

    def _from_masks(self, bar: np.ndarray, fish: np.ndarray, w: int,
                    fish_weight: np.ndarray | None = None) -> TrackResult | None:
        r = self.rod
        span = _span(bar, max_gap=max(8, int(w * r.bar_gap_frac)))
        if span is None or span[1] - span[0] < max(self.min_bar_px, r.bar_min_frac * w):
            return None  # no catch bar -> reel UI not on screen
        if span[1] - span[0] >= 0.9 * w:
            return None  # "bar" fills the whole box: that's bright background (sand, sky), not a catch bar
        fish_x = _fish_center(fish, max_width=max(6, int(w * r.fish_max_width_frac)), weight=fish_weight)
        if fish_x is None:
            if not r.calibrated:
                return None  # uncalibrated: without a fish we can't tell a real reel from a bright background
            # Bar visible but fish hidden: it is behind the bar or a skin decoration, i.e. roughly centred.
            return TrackResult((span[0] + span[1]) / 2, span[0], span[1], w, fish_visible=False)
        return TrackResult(fish_x, span[0], span[1], w)


class LineTracker(Tracker):
    """Scans a few rows through the middle of the bar. Cheapest option, ~0.1 ms per frame."""

    def track(self, img: np.ndarray) -> TrackResult | None:
        h, w = img.shape[:2]
        rows = img[max(0, h // 2 - 1): h // 2 + 2]
        gray = cv2.cvtColor(rows, cv2.COLOR_BGR2GRAY).mean(axis=0)
        return self._from_masks(gray >= self.rod.line_bar_min, gray <= self.rod.line_fish_max, w)


def _relaxed(lo, hi, dh: int = 6, ds: int = 30, dv: int = 40):
    """A looser version of an HSV range, used to measure how far a matching blob really extends."""
    wraps = lo[0] > hi[0]
    h_lo, h_hi = (lo[0] - dh) % 180, (hi[0] + dh) % 180
    if not wraps and lo[0] - dh >= 0 and hi[0] + dh <= 180:
        h_lo, h_hi = lo[0] - dh, hi[0] + dh
    return (h_lo, max(0, lo[1] - ds), max(0, lo[2] - dv)), (h_hi, hi[1], hi[2])


class ColorTracker(Tracker):
    """HSV masks projected onto the x axis. Works with bar skins once calibrated."""

    def track(self, img: np.ndarray) -> TrackResult | None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        r = self.rod
        h, w = img.shape[:2]
        bar = in_hsv_range(hsv, r.bar_hsv_lo, r.bar_hsv_hi)
        fish = in_hsv_range(hsv, r.fish_hsv_lo, r.fish_hsv_hi)
        fish_cols = fish.sum(axis=0)
        fish_mask = fish_cols > h * r.fish_col_share
        if r.calibrated and fish_mask.any():
            # Hysteresis: a candidate is only the fish if it stays narrow under a looser color match. A patch of
            # similarly-colored track (skins with gradient tracks) grows wide under the relaxed mask and is dropped.
            loose = in_hsv_range(hsv, *_relaxed(r.fish_hsv_lo, r.fish_hsv_hi)).sum(axis=0) > h * r.fish_col_share
            max_w = max(6, int(w * r.fish_max_width_frac))
            for s, e in _runs(fish_mask):
                ls, le = s, e
                while ls > 0 and loose[ls - 1]:
                    ls -= 1
                while le < w - 1 and loose[le + 1]:
                    le += 1
                if le - ls + 1 > 1.5 * max_w:
                    fish_mask[s:e + 1] = False
        # a column counts if a meaningful share of its pixels match -> ignores thin borders/text
        return self._from_masks(bar.sum(axis=0) > h * r.bar_col_share, fish_mask, w, fish_weight=fish_cols)


def make_tracker(style: str, rod: RodProfile, yolo_model: str | None = None) -> Tracker:
    if style == "color":
        return ColorTracker(rod)
    if style == "yolo":
        from tidecaller.modules.reel.yolo import YoloTracker

        return YoloTracker(rod, yolo_model or "models/fishbar.onnx")
    return LineTracker(rod)
