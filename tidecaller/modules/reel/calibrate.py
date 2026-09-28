"""Reel calibration: learn a bar skin's colors from two clicks.

Cosmetic bar skins (pastel tracks, white bars with decorations, colored fish markers) defeat fixed thresholds.
The user captures a few seconds of reeling, clicks once on the catch bar and once on the fish, and we learn:

  * bar color   - the clicked bar's pixels, grown along the row until the color changes sharply; achromatic bars
                  (white/grey) are matched on saturation+value only, since their hue drifts with gradients
  * bar shape   - its width (-> minimum width) and the holes decorations punch into it (-> gap bridging)
  * fish color  - a tight hue band around the click, plus the marker's width (-> maximum fish width)

The learned profile is then scored on every captured frame so the user sees whether it actually works.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from tidecaller.config.schema import RodProfile
from tidecaller.modules.reel.trackers import ColorTracker, TrackResult, _runs, in_hsv_range

ACHROMATIC_S = 60


@dataclass
class Click:
    frame: int
    x: int
    y: int


@dataclass
class CalibrationResult:
    profile: RodProfile
    bar_rate: float                 # frames where the bar was found
    fish_rate: float                # frames (with a bar) where the fish was visible
    results: list[TrackResult | None] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.bar_rate >= 0.8

    def summary(self) -> str:
        lines = [f"Bar found in {self.bar_rate:.0%} of frames, fish visible in {self.fish_rate:.0%} of those."]
        lines += self.warnings
        return "\n".join(lines)


def _hue_band(hues: np.ndarray, pad: int) -> tuple[int, int]:
    """Tight hue range covering the samples, wrapping around red when that is narrower."""
    h = hues.astype(int)
    lo, hi = np.percentile(h, [5, 95])
    shifted = (h + 90) % 180
    slo, shi = np.percentile(shifted, [5, 95])
    if shi - slo < hi - lo:  # samples straddle red (0/180)
        lo, hi = (slo - 90) % 180, (shi - 90) % 180
    return int((lo - pad) % 180), int((hi + pad) % 180)


def _grow_row(hsv_row: np.ndarray, x: int, tol_s: int = 30, tol_v: int = 18,
              max_hole_frac: float = 0.12) -> tuple[int, int, np.ndarray]:
    """The clicked bar's extent along its row, plus the mask of pixels that look like it.

    Pixels "look like the bar" when saturation/value are close to the click (bars are hue gradients). Runs of those
    are merged outward from the clicked run across holes up to `max_hole_frac` of the width, which is what a skin
    decoration pinned to the bar (or the fish icon) punches into it.
    """
    w = len(hsv_row)
    s0, v0 = int(hsv_row[x, 1]), int(hsv_row[x, 2])
    ok = (np.abs(hsv_row[:, 1].astype(int) - s0) <= tol_s) & (np.abs(hsv_row[:, 2].astype(int) - v0) <= tol_v)
    runs = _runs(ok)
    here = next((i for i, (a, b) in enumerate(runs) if a <= x <= b), None)
    if here is None:
        return x, x, ok
    left, right = runs[here]
    max_hole, min_piece = max_hole_frac * w, 0.03 * w
    i = here - 1
    while i >= 0 and left - runs[i][1] - 1 <= max_hole and runs[i][1] - runs[i][0] + 1 >= min_piece:
        left, i = runs[i][0], i - 1
    i = here + 1
    while i < len(runs) and runs[i][0] - right - 1 <= max_hole and runs[i][1] - runs[i][0] + 1 >= min_piece:
        right, i = runs[i][1], i + 1
    return int(left), int(right), ok


def learn_profile(frames: list[np.ndarray], bar: Click, fish: Click | None, base: RodProfile) -> CalibrationResult:
    img = frames[bar.frame]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    warnings: list[str] = []

    # --- bar color from the clicked element (middle rows only, away from borders)
    left, right, like = _grow_row(hsv[bar.y], bar.x)
    band = hsv[max(0, h // 4): max(1, 3 * h // 4), left:right + 1]
    rows = band[:, like[left:right + 1]].reshape(-1, 3).astype(int)  # skip decoration pixels inside the bar
    if rows.size == 0:
        rows = band.reshape(-1, 3).astype(int)
    s_lo, s_hi = np.percentile(rows[:, 1], [2, 97])
    v_lo = np.percentile(rows[:, 2], 3)
    if np.median(rows[:, 1]) < ACHROMATIC_S:
        bar_lo, bar_hi = (0, 0, int(max(0, v_lo - 8))), (180, int(min(255, s_hi + 8)), 255)
    else:
        h_lo, h_hi = _hue_band(rows[:, 0], pad=6)
        bar_lo, bar_hi = (h_lo, int(max(0, s_lo - 25)), int(max(0, v_lo - 25))), (h_hi, 255, 255)

    # tighten until the empty track outside the clicked bar stops matching
    outside = np.ones(w, bool)
    outside[max(0, left - 10): right + 11] = False
    for _ in range(8):
        cols = in_hsv_range(hsv, bar_lo, bar_hi).sum(axis=0) > h * 0.4
        if cols[outside].mean() < 0.03:
            break
        bar_lo = (bar_lo[0], bar_lo[1], min(254, bar_lo[2] + 6))
        bar_hi = (bar_hi[0], max(1, bar_hi[1] - 4), bar_hi[2])
    else:
        warnings.append("The track still looks a lot like the bar; tracking may be unreliable.")

    profile = base.model_copy(update={"bar_hsv_lo": bar_lo, "bar_hsv_hi": bar_hi, "calibrated": True})

    # --- bar shape: widest run in the clicked frame, and the biggest hole inside it
    bar_cols = in_hsv_range(hsv, bar_lo, bar_hi).sum(axis=0) > h * 0.4
    inside = bar_cols[left:right + 1] if right > left else bar_cols
    holes = [e - s + 1 for s, e in _runs(~inside)]
    gap = max(holes, default=0)
    # decorations (e.g. an icon pinned to the bar) can sit just outside the grown span; look at the whole bar
    runs = _runs(bar_cols)
    if len(runs) >= 2:
        between = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
        near = [g for g, (a, b) in zip(between, zip(runs, runs[1:]))
                if min(a[1] - a[0], b[1] - b[0]) > 0.05 * w and g < 0.15 * w]
        gap = max([gap, *near])
    # margin: the decoration can overlap the fish marker, making the hole wider than in the clicked frame
    profile.bar_gap_frac = float(min(0.15, max(0.04, (gap * 1.6 + 10) / w)))

    # --- fish color and width
    if fish is not None:
        fimg = cv2.cvtColor(frames[fish.frame], cv2.COLOR_BGR2HSV)
        patch = fimg[max(0, fish.y - 2): fish.y + 3, max(0, fish.x - 2): fish.x + 3].reshape(-1, 3).astype(int)
        f_h, f_s, f_v = np.median(patch, axis=0)
        if f_s < ACHROMATIC_S:  # grey/black/white fish icon: match on value
            lo, hi = (0, 0, int(max(0, f_v - 40))), (180, int(f_s + 40), int(min(255, f_v + 40)))
        else:
            h_lo, h_hi = int(f_h - 10) % 180, int(f_h + 10) % 180
            lo, hi = (h_lo, int(max(40, f_s - 45)), int(max(0, f_v - 70))), (h_hi, 255, 255)
        cols = in_hsv_range(fimg, lo, hi)
        colmask = cols.sum(axis=0) > fimg.shape[0] * 0.3
        width = next((e - s + 1 for s, e in _runs(colmask) if s <= fish.x <= e), 8)
        profile = profile.model_copy(update={
            "fish_hsv_lo": lo, "fish_hsv_hi": hi,
            "fish_max_width_frac": float(min(0.12, max(4, width * 1.6) / w)),
            "fish_col_share": 0.3,
        })
    else:
        warnings.append("No fish clicked: the fish color was not learned (tracking assumes a centred fish).")

    # --- score it, then set the minimum bar width from what was actually seen. A stricter column share rejects
    # objects crossing the strip diagonally (the rod model swinging on cast); keep it only if recall holds up.
    profile.bar_col_share = 0.6
    results = [ColorTracker(profile).track(f) for f in frames]
    if sum(r is not None for r in results) < 0.9 * len(frames):
        profile.bar_col_share = 0.4
        results = [ColorTracker(profile).track(f) for f in frames]
    widths = [r.bar_right - r.bar_left for r in results if r is not None]
    if widths:
        profile.bar_min_frac = float(0.6 * np.median(widths) / w)
        results = [ColorTracker(profile).track(f) for f in frames]
    found = [r for r in results if r is not None]
    bar_rate = len(found) / max(1, len(frames))
    fish_rate = sum(r.fish_visible for r in found) / max(1, len(found))
    if bar_rate < 0.8:
        warnings.append("The bar was missed in many frames: make sure the capture covers an active reel "
                        "and the fish-bar box (F1) sits on the bar's inner rows.")
    return CalibrationResult(profile, bar_rate, fish_rate, results, warnings)


def preview(frames: list[np.ndarray], results: list[TrackResult | None], n: int = 8, scale: int = 1) -> np.ndarray:
    """Stack a few annotated frames: blue box = catch bar, red line = fish (dashed when assumed)."""
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    out = []
    for i in idx:
        f = frames[i].copy()
        r = results[i]
        if r is not None:
            cv2.rectangle(f, (int(r.bar_left), 0), (int(r.bar_right), f.shape[0] - 1), (255, 80, 0), 2)
            color = (0, 0, 255) if r.fish_visible else (0, 160, 255)
            cv2.line(f, (int(r.fish_x), 0), (int(r.fish_x), f.shape[0] - 1), color, 2)
        else:
            cv2.putText(f, "no bar", (4, f.shape[0] - 4), 0, 0.4, (0, 0, 255), 1)
        out.append(f)
        out.append(np.full((3, f.shape[1], 3), 30, np.uint8))
    img = np.vstack(out)
    return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
