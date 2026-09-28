"""Train the fish/bar detector and export it to ONNX for the runtime tracker.

Training needs the heavy extras (torch via ultralytics), which the macro itself does not:

    pip install ultralytics
    python ml/make_sim_dataset.py --out datasets/fishbar
    python ml/train_yolo.py --data datasets/fishbar/data.yaml --epochs 60

The exported model lands at models/fishbar.onnx, the default `fish.yolo_model` path.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=320)  # must match YoloTracker(imgsz=...)
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--out", type=Path, default=Path("models/fishbar.onnx"))
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.base)
    # no mosaic/flip-heavy augmentation: the bar is a 1-D problem and horizontal flips are fine, vertical are not
    model.train(data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, flipud=0.0, fliplr=0.5, mosaic=0.0,
                degrees=0.0, scale=0.2, rect=False)
    onnx_path = Path(model.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, a.out)
    print(f"exported -> {a.out}")


if __name__ == "__main__":
    main()
