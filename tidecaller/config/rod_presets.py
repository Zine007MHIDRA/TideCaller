"""Reel presets derived from each rod's in-game stats (see rods.json).

In Fisch, Control widens the player's catch bar and Resilience slows how fast progress drains while the fish is
outside it. So:
  * wider bar (high control)  -> gentler gains, bigger deadzone (no need to chase every pixel)
  * narrow bar (neg. control) -> stronger gains, tiny deadzone
  * low / negative resilience -> more damping (kd) so we don't overshoot and bleed progress
These are starting points; per-rod fine tuning is saved as a custom profile.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tidecaller.config.schema import RodProfile, TrackStyle

DATA = Path(__file__).with_name("rods.json")

# The video recommends the trained model for rods whose bar visuals break pixel rules.
YOLO_RODS = {"Tryhard Rod", "Maelstrom"}

BASE_KP, BASE_KD, BASE_DEADZONE = 0.012, 0.0025, 6


@dataclass(frozen=True)
class RodStats:
    name: str
    lure_pct: float | None
    luck_pct: float | None
    control: float
    resilience_pct: float

    @property
    def recommended_track(self) -> TrackStyle:
        return TrackStyle.YOLO if self.name in YOLO_RODS else TrackStyle.LINE


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@lru_cache(maxsize=1)
def load_stats() -> dict[str, RodStats]:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    return {
        r[0]: RodStats(r[0], r[1], r[2], float(r[3] or 0.0), float(r[4] or 0.0))
        for r in raw["rods"]
    }


def profile_from_stats(s: RodStats) -> RodProfile:
    control = _clamp(s.control, -0.5, 0.7)
    width = 1 + 1.5 * control  # relative catch-bar width vs. a 0-control rod
    resil = _clamp(s.resilience_pct, -100, 100) / 100
    return RodProfile(
        name=s.name,
        kp=round(BASE_KP / width, 5),
        kd=round(BASE_KD * (1 + 0.6 * max(0.0, -resil)) / width ** 0.5, 5),
        deadzone_px=int(round(_clamp(BASE_DEADZONE * width, 2, 14))),
    )


def preset(name: str) -> RodProfile | None:
    s = load_stats().get(name)
    return None if s is None else profile_from_stats(s)


def names() -> list[str]:
    return sorted(load_stats(), key=str.lower)
