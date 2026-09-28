import numpy as np
import pytest

from tidecaller.config.schema import CastSettings, CastStyle, Config, ReleaseStyle, ShakeSettings
from tidecaller.core.regions import Rect, Region
from tidecaller.modules.cast import LatencyEstimator, ReleaseDecider, read_meter
from tidecaller.modules.reel import ReelController, TrackResult, make_tracker
from tidecaller.modules.rod import slot_is_selected
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


def meter(fill, h=200, w=30):
    img = np.full((h, w, 3), 30, np.uint8)
    img[int(h - fill * h):, :] = (40, 220, 60)
    return img


@pytest.mark.parametrize("fill", [0.1, 0.5, 0.95])
def test_read_meter(fill):
    assert read_meter(meter(fill)) == pytest.approx(fill, abs=0.02)


def test_read_meter_absent():
    assert read_meter(np.full((200, 30, 3), 30, np.uint8)) is None


def test_idiot_proof_release_threshold_and_timeout():
    cfg = CastSettings(release=ReleaseStyle.IDIOT_PROOF, fill_threshold=0.9, timeout_ms=1000)
    d = ReleaseDecider(cfg, t0=0.0)
    assert not d.should_release(0.2, 0.5)
    assert d.should_release(0.3, 0.92)
    d2 = ReleaseDecider(cfg, t0=0.0)
    assert d2.should_release(1.1, None)  # meter never seen -> fallback


def test_predictive_release_leads_by_latency():
    cfg = CastSettings(style=CastStyle.PERFECT, fill_threshold=0.95)
    d = ReleaseDecider(cfg, t0=0.0, latency_s=0.1)
    # meter fills at 1.0/s -> should release ~0.1 s before hitting 0.95
    fired = next(t for t in np.arange(0.0, 1.0, 0.01) if d.should_release(t, t))
    assert 0.83 <= fired <= 0.87


def test_latency_estimator_moves_toward_observed():
    est = LatencyEstimator(30)
    est.update(overshoot=0.02, velocity=1.0)  # released 20 ms too late -> real latency higher
    assert est.ms > 30


def test_shake_circle_found_and_square_rejected():
    img = np.full((400, 600, 3), (120, 90, 40), np.uint8)
    import cv2

    cv2.circle(img, (420, 150), 34, (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cfg = ShakeSettings(circularity=0.8)
    hits = find_circles(img, cfg)
    assert len(hits) == 1 and abs(hits[0][0] - 420) <= 2 and abs(hits[0][1] - 150) <= 2
    assert find_pixel(img, cfg) is not None


def test_hotbar_slot_detection_on_sim():
    sim = FischSim()
    SimInput(sim).tap("1")
    frame = sim.render()
    hb = sim.regions.hotbar.to_rect(sim.client)
    crop = frame[hb.top:hb.bottom, hb.left:hb.right]
    assert slot_is_selected(crop, 1)
    assert not slot_is_selected(crop, 2)


def test_region_roundtrip():
    client = Rect(100, 50, 1600, 900)
    rect = Region(0.25, 0.5, 0.1, 0.2).to_rect(client)
    back = Region.from_rect(rect, client)
    assert back.to_rect(client) == rect
