from pathlib import Path

import cv2
import numpy as np
import pytest

from tidecaller.config.schema import CastSettings, CastStyle, Config, ReleaseStyle, ShakeSettings
from tidecaller.core.regions import Rect, Region
from tidecaller.modules.cast import ReleaseDecider, meter_is_full
from tidecaller.modules.reel import ReelController, TrackResult, make_tracker
from tidecaller.modules.rod import power_bar_run, rod_held
from tidecaller.modules.shake import find_circles, find_pixel
from tidecaller.sim.fisch_sim import FischSim, SimInput

ROD = Config().rod()


def fish_bar(width=560, height=25, bar=(200, 350), fish=300):
    img = np.full((height, width, 3), (70, 60, 50), np.uint8)
    img[2:-3, bar[0]:bar[1]] = 245
    img[1:-2, fish - 5:fish + 6] = 25
    return img


@pytest.mark.parametrize("style", ["line", "color"])
@pytest.mark.parametrize("bar,fish", [((200, 350), 300), ((10, 160), 400), ((380, 540), 20), ((100, 250), 180)])
def test_trackers_locate_fish_and_bar(style, bar, fish):
    r = make_tracker(style, ROD).track(fish_bar(bar=bar, fish=fish))
    assert r is not None
    assert abs(r.fish_x - fish) <= 3
    assert abs(r.bar_left - bar[0]) <= 3 and abs(r.bar_right - (bar[1] - 1)) <= 3


@pytest.mark.parametrize("style", ["line", "color"])
def test_tracker_none_when_reel_hidden(style):
    assert make_tracker(style, ROD).track(np.full((25, 560, 3), (120, 90, 40), np.uint8)) is None


def test_controller_holds_when_fish_is_right_and_releases_when_left():
    ctl = ReelController(ROD)
    right = TrackResult(fish_x=500, bar_left=100, bar_right=200, width=560)
    holds = [ctl.step(right, t * 0.01) for t in range(20)]
    assert sum(holds) >= 18
    ctl.reset()
    left = TrackResult(fish_x=20, bar_left=300, bar_right=400, width=560)
    assert sum(ctl.step(left, t * 0.01) for t in range(20)) <= 2


def test_controller_centered_is_half_duty():
    ctl = ReelController(ROD)
    r = TrackResult(fish_x=150, bar_left=100, bar_right=200, width=560)
    holds = sum(ctl.step(r, t * 0.01) for t in range(100))
    assert 45 <= holds <= 55


FIX = Path(__file__).parent / "fixtures"


def fixture(name):
    img = cv2.imread(str(FIX / f"{name}.png"))
    assert img is not None, name
    return img


# --- real-game frames (1920x1040 recording) --------------------------------------------------------
def test_real_meter_full_detected():
    assert meter_is_full(fixture("meter_full"))


@pytest.mark.parametrize("name", ["meter_filling", "meter_none_aura", "meter_none_idle"])
def test_real_meter_not_full(name):
    assert not meter_is_full(fixture(name))


def test_real_rod_held():
    assert rod_held(fixture("rodinfo_held"))


@pytest.mark.parametrize("name", ["rodinfo_not_held", "rodinfo_reeling"])
def test_real_rod_not_held(name):
    assert not rod_held(fixture(name))
    assert power_bar_run(fixture(name)) < 0.2


# --- release strategies ----------------------------------------------------------------------------
def test_idiot_proof_releases_on_green_or_timeout():
    cfg = CastSettings(release=ReleaseStyle.IDIOT_PROOF, timeout_ms=1000)
    d = ReleaseDecider(cfg, t0=0.0)
    assert not d.should_release(0.5, False)
    assert d.should_release(0.6, True)
    assert ReleaseDecider(cfg, t0=0.0).should_release(1.05, False)  # meter never seen -> fallback


def test_threshold_waits_longer_than_timeout():
    cfg = CastSettings(release=ReleaseStyle.THRESHOLD, timeout_ms=1000)
    d = ReleaseDecider(cfg, t0=0.0)
    assert not d.should_release(1.5, False)
    assert d.should_release(1.5, True)


def test_predictive_uses_learned_time_to_full():
    cfg = CastSettings(style=CastStyle.PERFECT, release=ReleaseStyle.PREDICTIVE)
    d = ReleaseDecider(cfg, t0=0.0, latency_s=0.05, time_to_full_s=0.8)
    assert not d.should_release(0.70, False)
    assert d.should_release(0.76, False)


def test_shake_circle_found_and_square_rejected():
    img = np.full((400, 600, 3), (120, 90, 40), np.uint8)
    import cv2

    cv2.circle(img, (420, 150), 34, (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cfg = ShakeSettings(circularity=0.8)
    hits = find_circles(img, cfg)
    assert len(hits) == 1 and abs(hits[0][0] - 420) <= 2 and abs(hits[0][1] - 150) <= 2
    assert find_pixel(img, cfg) is not None


def test_rod_held_on_sim_power_bar():
    sim = FischSim()
    crop = lambda: (lambda f, r: f[r.top:r.bottom, r.left:r.right])(  # noqa: E731
        sim.render(), sim.regions.rod_info.to_rect(sim.client))
    assert not rod_held(crop())
    SimInput(sim).tap("1")
    assert rod_held(crop())


def test_region_roundtrip():
    client = Rect(100, 50, 1600, 900)
    rect = Region(0.25, 0.5, 0.1, 0.2).to_rect(client)
    back = Region.from_rect(rect, client)
    assert back.to_rect(client) == rect
