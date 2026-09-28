"""Generate a perfectly-labelled YOLO dataset of fish-bar crops from the simulator.

Good for bootstrapping the model and the pipeline. For real rods, mix in frames from capture_frames.py.

    python ml/make_sim_dataset.py --out datasets/fishbar --n 3000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from tidecaller.sim.fisch_sim import FischSim

FISH, BAR = 0, 1
FISH_HALF_W = 5  # matches FischSim._draw_reel


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Cheap domain randomisation so the model doesn't memorise the sim's flat colors."""
    out = img.astype(np.float32)
    out *= rng.uniform(0.75, 1.2)                       # brightness
    out += np.array([rng.uniform(-18, 18) for _ in range(3)], np.float32)  # tint
    out += np.random.normal(0, rng.uniform(0, 6), out.shape)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def sample(sim: FischSim, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    sim.state = "reel"
    sim.p.bar_width_frac = rng.uniform(0.12, 0.45)       # emulate different Control stats
    sim.bar_x = rng.uniform(0, 1 - sim.p.bar_width_frac)
    sim.fish_x = rng.uniform(0.02, 0.98)
    sim.progress = rng.uniform(0.05, 0.95)
    frame = np.full((720, 1280, 3), 0, np.uint8)
    sim._draw_reel(frame)
    r = sim.regions.fish_bar.to_rect(sim.client)
    crop = frame[r.top:r.bottom, r.left:r.right]
    w, h = crop.shape[1], crop.shape[0]
    labels = [
        (FISH, sim.fish_x, 0.5, (2 * FISH_HALF_W + 1) / w, 1.0),
        (BAR, sim.bar_x + sim.p.bar_width_frac / 2, 0.5, sim.p.bar_width_frac, 1.0),
    ]
    return augment(crop, rng), labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/fishbar"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    np.random.seed(a.seed)
    sim = FischSim()
    for i in range(a.n):
        split = "val" if rng.random() < a.val else "train"
        img, labels = sample(sim, rng)
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.out / "images" / split / f"sim_{i:05d}.png"), img)
        (a.out / "labels" / split / f"sim_{i:05d}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels) + "\n")
    (a.out / "data.yaml").write_text(
        f"path: {a.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: fish\n  1: bar\n")
    print(f"wrote {a.n} samples to {a.out}")


if __name__ == "__main__":
    main()
