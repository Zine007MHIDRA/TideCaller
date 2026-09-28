"""Is the rod in hand?

While a rod is held, Fisch shows a "Current Bait ... Power" block above the hotbar. Its power bar is a thin
(~9 px at 1080p), long horizontal band: purple fill followed by a navy remainder. Measured on real gameplay the band
spans 0.40 of the strip width when held, 0.00 when not, ~0.14 during the reel (hotbar hidden).

To avoid mistaking open water or sky for the bar, the band must also be *thin* (big blue areas fill every row of
the strip) and contain some purple fill (water is plain blue).
"""
from __future__ import annotations

import cv2
import numpy as np

HELD_MIN_RUN = 0.3
MAX_BAND_FRAC = 0.4   # of the strip height
MIN_PURPLE_FRAC = 0.1  # of the run


def _row_runs(mask: np.ndarray) -> np.ndarray:
    """Longest True run per row, in pixels."""
    out = np.zeros(mask.shape[0], int)
    for y, row in enumerate(mask):
        d = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if starts.size:
            out[y] = int((ends - starts).max())
    return out


def power_bar_run(band: np.ndarray) -> float:
    """Width of the power bar as a fraction of the strip, or 0 if nothing bar-shaped is there."""
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    mask = (hue >= 95) & (hue <= 145) & (sat > 110)
    purple = (hue >= 118) & (hue <= 145) & (sat > 110)
    h, w = mask.shape
    runs = _row_runs(mask)
    long_rows = runs >= HELD_MIN_RUN * w
    if long_rows.sum() == 0 or long_rows.sum() > MAX_BAND_FRAC * h:
        return 0.0  # nothing, or a big blue area rather than a thin bar
    rows = np.flatnonzero(long_rows)
    if purple[rows].sum() < MIN_PURPLE_FRAC * runs[rows].sum():
        return 0.0
    return float(runs.max()) / w


def rod_held(band: np.ndarray) -> bool:
    return power_bar_run(band) >= HELD_MIN_RUN
