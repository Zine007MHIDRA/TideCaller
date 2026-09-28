"""End-to-end: the real Macro loop plays the simulator, no Roblox involved."""
import threading

from tidecaller.config.schema import Config, ShakeStyle
from tidecaller.core.capture import StaticSource
from tidecaller.core.state_machine import Macro, Phase
from tidecaller.sim.fisch_sim import FischSim, SimInput


def run_sim(cfg: Config, cycles: int, timeout: float = 60.0):
    sim = FischSim(regions=cfg.regions)
    macro = Macro(cfg, StaticSource(sim.render), SimInput(sim), client_rect=lambda: sim.client)
    phases: list[Phase] = []
    macro.on_phase = phases.append
    th = threading.Thread(target=macro.run, kwargs={"max_cycles": cycles}, daemon=True)
    th.start()
    th.join(timeout)
    macro.stop()
    th.join(2)
    return sim, macro, phases


def test_full_loop_catches_fish():
    cfg = Config()
    cfg.discord.on_catch = False
    sim, macro, phases = run_sim(cfg, cycles=3)
    assert macro.stats.cycles == 3, phases
    assert sim.caught >= 2, (sim.caught, sim.lost)
    assert macro.stats.recoveries == 0
    for p in (Phase.EQUIP, Phase.CAST, Phase.SHAKE, Phase.REEL, Phase.CATCH):
        assert p in phases


def test_pixel_shake_and_color_tracker():
    cfg = Config()
    cfg.shake.style = ShakeStyle.PIXEL
    cfg.fish.track = "color"
    sim, macro, _ = run_sim(cfg, cycles=2)
    assert macro.stats.cycles == 2
    assert sim.caught >= 1


def test_stop_releases_mouse():
    cfg = Config()
    sim = FischSim(regions=cfg.regions)
    macro = Macro(cfg, StaticSource(sim.render), SimInput(sim), client_rect=lambda: sim.client)
    th = threading.Thread(target=macro.run, daemon=True)
    th.start()
    threading.Event().wait(1.5)
    macro.stop()
    th.join(3)
    assert not th.is_alive()
    assert not sim.mouse_down


# --- regressions from real gameplay (rod toggling, instant-bite rods) ------------------------------
class CountingSimInput(SimInput):
    def __init__(self, sim):
        super().__init__(sim)
        self.keys: list[str] = []
        self.clicks_at: list[tuple[int, int]] = []

    def key_down(self, key):
        self.keys.append(key)
        super().key_down(key)

    def move(self, x, y):
        self.clicks_at.append((x, y))
        super().move(x, y)


def run_counting(cfg: Config, cycles: int, sim: FischSim, timeout: float = 60.0):
    inp = CountingSimInput(sim)
    macro = Macro(cfg, StaticSource(sim.render), inp, client_rect=lambda: sim.client)
    th = threading.Thread(target=macro.run, kwargs={"max_cycles": cycles}, daemon=True)
    th.start()
    th.join(timeout)
    macro.stop()
    th.join(2)
    return macro, inp


def test_rod_already_held_is_never_unequipped():
    cfg = Config()
    cfg.fish.bag_spam = False
    sim = FischSim(regions=cfg.regions)
    sim.equipped_slot = 1
    macro, inp = run_counting(cfg, cycles=3, sim=sim)
    assert macro.stats.cycles == 3
    assert "1" not in inp.keys  # pressing the slot while holding the rod would put it away
    assert sim.equipped_slot == 1


def test_rod_not_held_is_equipped_once_and_keeps_fishing():
    cfg = Config()
    cfg.fish.bag_spam = False
    sim = FischSim(regions=cfg.regions)
    macro, inp = run_counting(cfg, cycles=3, sim=sim)
    assert macro.stats.cycles == 3
    assert inp.keys.count("1") == 1
    assert sim.caught >= 2


def test_high_lure_rod_skips_shake_clicks():
    cfg = Config()
    cfg.fish.rod = "Masterline Rod"  # lure 1000% -> instant bite, no shake
    cfg.fish.track = "line"
    cfg.fish.bag_spam = False
    sim = FischSim(regions=cfg.regions)
    sim.p.shakes_needed = 0
    sim.equipped_slot = 1
    macro = Macro(cfg, StaticSource(sim.render), SimInput(sim), client_rect=lambda: sim.client)
    assert macro.skips_shake()
    macro2, inp = run_counting(cfg, cycles=2, sim=sim)
    shake = cfg.regions.shake.to_rect(sim.client)
    center = (sim.client.width // 2, sim.client.height // 2)
    stray = [p for p in inp.clicks_at if p != center and shake.left <= p[0] <= shake.right
             and shake.top <= p[1] <= shake.bottom]
    assert macro2.stats.cycles == 2 and macro2.stats.recoveries == 0
    assert stray == []  # only the cast's own cursor centering, never a shake click
