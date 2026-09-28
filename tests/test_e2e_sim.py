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
