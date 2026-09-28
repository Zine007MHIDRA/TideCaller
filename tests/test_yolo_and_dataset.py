import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from tidecaller.config.schema import Config  # noqa: E402
from tidecaller.sim.fisch_sim import FischSim  # noqa: E402

pytest.importorskip("onnxruntime")
from tidecaller.modules.reel.yolo import YoloTracker  # noqa: E402


class FakeSession:
    """Mimics a YOLOv8 ONNX output: (1, 4 + ncls, N) with boxes in letterboxed-input pixels."""

    def __init__(self, preds):
        self.preds = preds

    def run(self, _outs, _feeds):
        out = np.zeros((1, 6, len(self.preds)), np.float32)
        for i, (cls, cx, w, score) in enumerate(self.preds):
            out[0, :4, i] = (cx, 160, w, 10)
            out[0, 4 + cls, i] = score
        return [out]


def make_tracker(preds, width=560):
    t = YoloTracker.__new__(YoloTracker)  # skip model loading
    t.rod, t.imgsz, t.conf, t.inp = Config().rod(), 320, 0.35, "images"
    t.sess = FakeSession(preds)
    return t


def test_yolo_decode_scales_back_to_region_pixels():
    # imgsz 320 over a 560 px strip -> scale 320/560
    s = 320 / 560
    t = make_tracker([(0, 400 * s, 11 * s, 0.9), (1, 150 * s, 100 * s, 0.8), (1, 50 * s, 20 * s, 0.2)])
    r = t.track(np.zeros((25, 560, 3), np.uint8))
    assert r is not None
    assert r.fish_x == pytest.approx(400, abs=1)
    assert r.bar_left == pytest.approx(100, abs=1) and r.bar_right == pytest.approx(200, abs=1)


def test_yolo_none_when_class_missing():
    assert make_tracker([(0, 100, 5, 0.9)]).track(np.zeros((25, 560, 3), np.uint8)) is None


def test_sim_dataset_labels_match_render():
    import make_sim_dataset as mk

    rng = random.Random(1)
    np.random.seed(1)
    img, labels = mk.sample(FischSim(), rng)
    (_, fx, _, _, _), (_, bx, _, bw, _) = labels
    w = img.shape[1]
    # the fish column must be dark and the bar center bright in the (augmented) crop
    col = lambda x: img[:, int(x * w)].mean()  # noqa: E731
    assert col(fx) < 90
    if not (bx - bw / 2 <= fx <= bx + bw / 2) or abs(fx - bx) > 0.03:
        assert col(bx) > 150
