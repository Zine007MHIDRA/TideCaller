"""Reel calibration against real frames of a cosmetic bar skin (pastel track, white bar with a pinned decoration,
teal fish capsule). Clicks match what a user would do in the Calibrate reel dialog."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from tidecaller.config.schema import RodProfile
from tidecaller.modules.reel.calibrate import Click, learn_profile, preview
from tidecaller.modules.reel.trackers import ColorTracker, LineTracker

FIX = Path(__file__).parent / "fixtures"
REEL = sorted(FIX.glob("reel_*.png"))
NO_REEL = sorted(FIX.glob("noreel_*.png"))
FISH_X = 402.5  # the fish capsule is frozen in place (bait effect) in these frames


@pytest.fixture(scope="module")
def frames():
    return [cv2.imread(str(p)) for p in REEL]


@pytest.fixture(scope="module")
def calibrated(frames):
    i = [p.name for p in REEL].index("reel_310.png")
    return learn_profile(frames, bar=Click(i, 100, 13), fish=Click(i, 402, 13), base=RodProfile())


def test_uncalibrated_default_tracker_cannot_read_this_skin(frames):
    assert all(LineTracker(RodProfile()).track(f) is None for f in frames)


def test_calibration_finds_bar_in_every_frame(calibrated):
    assert calibrated.ok and calibrated.bar_rate == 1.0
    assert calibrated.profile.calibrated


def test_bar_covers_most_of_the_real_catch_bar(calibrated):
    widths = [r.bar_right - r.bar_left for r in calibrated.results]
    assert np.median(widths) > 380  # the real bar is ~430 px of the 806 px strip


def test_fish_on_capsule_or_assumed_centred(calibrated):
    for r in calibrated.results:
        if r.fish_visible:
            assert abs(r.fish_x - FISH_X) <= 4
        # the fish sits inside the bar in all these frames; the capsule's outline can clip the bar's right end,
        # so allow a little slack past the detected edge
        assert r.bar_left - 40 <= r.fish_x <= r.bar_right + 40


def test_teal_track_is_not_mistaken_for_the_fish(calibrated):
    assert all(r.fish_x < 600 for r in calibrated.results)  # the right half of the track is teal too


@pytest.mark.parametrize("path", NO_REEL, ids=lambda p: p.name)
def test_no_reel_frames_have_no_bar(calibrated, path):
    assert ColorTracker(calibrated.profile).track(cv2.imread(str(path))) is None


def test_bar_hole_from_decoration_is_bridged(calibrated):
    assert calibrated.profile.bar_gap_frac > 0.05


def test_preview_renders(frames, calibrated):
    img = preview(frames, calibrated.results)
    assert img.shape[1] == frames[0].shape[1] and img.shape[0] > frames[0].shape[0]


def test_missing_fish_click_still_calibrates_bar(frames):
    res = learn_profile(frames, bar=Click(0, 250, 13), fish=None, base=RodProfile())
    assert res.bar_rate >= 0.8
    assert any("fish" in w for w in res.warnings)
