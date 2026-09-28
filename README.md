# Tidecaller

An external, computer-vision fishing macro for the Roblox game **Fisch**, inspired by the feature set of the
community "Hydra" macro. It only **reads pixels from the screen and sends keyboard/mouse input**. There's no memory
reading, no injection, and no executor. It's a real-time CV + control-loop project.

> ⚠️ Automating gameplay can break Roblox's Terms of Use and Fisch's rules. Use it on your own account and at
> your own risk.

## Features
| Tab | What it does |
|---|---|
| **Main** | DXcam (fast) or MSS (for screen-sharing and laptops) capture, auto-select rod (reads the rod's Power bar, never toggles a held rod away), pause when Roblox loses focus, rebindable hotkeys |
| **Cast** | Releases the moment the power meter turns green (perfect cast). Release styles: idiot-proof, threshold, timed, predictive (learned time-to-full) |
| **Shake** | Circle detection (HSV mask + contour circularity), fast pixel mode, UI-navigation mode, or none. Rods with lure ≥ 95% (e.g. Masterline) skip the shake automatically |
| **Fish** | Line / Color / YOLO trackers, PD reel controller with sigma-delta mouse modulation, bag spam, scan FPS |
| **Calibrate reel** | Two clicks teach the tracker a cosmetic bar skin: capture a reel (or load a recording), click the catch bar, click the fish |
| **Rod presets** | Reel presets for all 263 rods, generated from their Control and Resilience stats |
| **Totem** | Cycle- or timer-based totems, auto potions |
| **Discord** | Webhook with test button, catch pings, "stuck" alerts, and periodic summaries with a screenshot |
| **Stats** | Catches/hour, runtime, recoveries, plus a **live tracker view** showing what the bot sees |
| **F1 areas** | Transparent overlay to drag/resize every detection region; saved relative to the Roblox window |

## How it works
```
EQUIP ─▶ CAST ─▶ SHAKE ─▶ REEL ─▶ CATCH ─▶ CONSUMABLES ─┐
  ▲        (timeouts on every phase ─▶ RECOVER)          │
  └──────────────────────────────────────────────────────┘
```
* **Reel:** each frame, a tracker returns the fish x and the catch-bar extents. A PD controller turns the error into a
  duty cycle, and a sigma-delta modulator converts that duty into per-frame hold/release. The mouse is binary, but
  the force on the bar ends up smooth.
* **Rod presets:** Control widens the catch bar, so wide-bar rods get gentler gains and a bigger deadzone, while
  narrow-bar rods like the Tryhard Rod (−0.37) get strong gains and a tight deadzone. Low resilience adds damping.
  Stats come from the [Fisch wiki](https://fischipedia.org/wiki/Fishing_Rods).
* **Dependency injection everywhere:** the same `Macro` loop runs against the live game, a screen recording
  (`--replay`), or a built-in **Fisch simulator** (`--sim`).

## Run
```bash
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m tidecaller --sim            # demo, no Roblox needed
.venv\Scripts\python -m tidecaller                  # live
.venv\Scripts\python -m tidecaller --replay run.mp4 # dry run on a recording
```
Before the first live run: open Fisch, cast once, press **F1**, and check the boxes sit on the fish bar's inner rows,
the cast meter area, the shake area, the hotbar and the rod's Power bar (defaults are measured on a real 1080p window).

Using a reel **bar skin**? Fish tab → pick your rod → **Calibrate reel...** → capture while reeling (or load a
recording) → click the catch bar, then the fish → check the preview → Apply. On a pastel skin with a decoration
pinned to the bar, this found the bar in 106/106 frames of a real reel where the default tracker found 0.

## Tests
```bash
.venv\Scripts\python -m pytest -q
```
The unit tests cover trackers (±3 px), release strategies, shake detection, the controller and rod presets.
Crops from real gameplay (`tests/fixtures/`) pin the cast-meter, rod-held and reel-calibration detectors to the real
UI. The end-to-end tests run the real loop against the simulator, including regressions for rod toggling and
instant-bite rods.

## YOLO tracker (hard rods)
```bash
.venv\Scripts\python ml/make_sim_dataset.py --out datasets/fishbar --n 3000   # labelled sim data
.venv\Scripts\python ml/capture_frames.py --out datasets/fishbar_live          # real frames, pre-labelled
pip install ultralytics
.venv\Scripts\python ml/train_yolo.py --data datasets/fishbar/data.yaml         # -> models/fishbar.onnx
```

## Build
`python build.py` produces `dist/Tidecaller.exe`. CI (GitHub Actions, Windows) runs the tests and builds the exe on
every push, and attaches it to a GitHub release when you push a `v*` tag.

## Roadmap
- [ ] Train and ship a YOLO model on real Tryhard/Maelstrom frames
- [ ] Fixture frames captured from the live game for regression tests
- [ ] Demo GIF
