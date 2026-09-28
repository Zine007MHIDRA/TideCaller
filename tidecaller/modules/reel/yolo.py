"""YOLO tracker for rods whose bar visuals defeat pixel rules (Tryhard, Maelstrom, ...).

Expects a YOLOv8 ONNX export with two classes: 0 = fish, 1 = bar (see ml/train_yolo.py).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tidecaller.config.schema import RodProfile
from tidecaller.modules.reel.trackers import Tracker, TrackResult

FISH, BAR = 0, 1


class YoloTracker(Tracker):
    def __init__(self, rod: RodProfile, model_path: str, imgsz: int = 320, conf: float = 0.35) -> None:
        super().__init__(rod)
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path} (train one with ml/train_yolo.py)")
        providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.conf = imgsz, conf

    def track(self, img: np.ndarray) -> TrackResult | None:
        h, w = img.shape[:2]
        # letterbox the wide strip into a square without distorting x
        scale = self.imgsz / w
        resized = cv2.resize(img, (self.imgsz, max(1, round(h * scale))))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        pad = (self.imgsz - resized.shape[0]) // 2
        canvas[pad:pad + resized.shape[0]] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: blob})[0][0]  # (4 + ncls, N)
        boxes, scores = out[:4].T, out[4:].T
        best: dict[int, tuple[float, np.ndarray]] = {}
        for box, sc in zip(boxes, scores):
            cls = int(sc.argmax())
            if sc[cls] >= self.conf and sc[cls] > best.get(cls, (0.0, None))[0]:
                best[cls] = (float(sc[cls]), box)
        if FISH not in best or BAR not in best:
            return None
        fx = best[FISH][1][0] / scale
        bx, bw = best[BAR][1][0] / scale, best[BAR][1][2] / scale
        return TrackResult(float(fx), float(bx - bw / 2), float(bx + bw / 2), w)
