from tidecaller.config import rod_presets
from tidecaller.config.schema import Config, TrackStyle


def test_all_wiki_rods_load():
    stats = rod_presets.load_stats()
    assert len(stats) > 250
    assert "Tryhard Rod" in stats and "Masterline Rod" in stats


def test_narrow_bar_rods_get_stronger_gains_than_wide_bar_rods():
    tryhard = rod_presets.preset("Tryhard Rod")        # control -0.37
    anchor = rod_presets.preset("Anchor n' Chain")     # control +0.5
    flimsy = rod_presets.preset("Flimsy Rod")          # control 0
    assert tryhard.kp > flimsy.kp > anchor.kp
    assert tryhard.deadzone_px < flimsy.deadzone_px < anchor.deadzone_px


def test_hard_rods_recommend_yolo():
    s = rod_presets.load_stats()
    assert s["Tryhard Rod"].recommended_track == TrackStyle.YOLO
    assert s["Flimsy Rod"].recommended_track == TrackStyle.LINE


def test_config_prefers_custom_then_preset():
    cfg = Config()
    cfg.fish.rod = "Tryhard Rod"
    assert cfg.rod().name == "Tryhard Rod"
    cfg.rods["Tryhard Rod"] = cfg.rod().model_copy(update={"kp": 0.5})
    assert cfg.rod().kp == 0.5
    cfg.fish.rod = "Not A Real Rod"
    assert cfg.rod().name == "Default"
