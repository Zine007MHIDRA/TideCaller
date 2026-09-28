"""Auto-select rod: an equipped hotbar slot gets a bright outline; if the rod slot lacks it, press the slot key."""
from __future__ import annotations

import cv2
import numpy as np


def slot_is_selected(hotbar: np.ndarray, slot: int, slots: int = 9, min_ratio: float = 0.12) -> bool:
    h, w = hotbar.shape[:2]
    sw = w / slots
    cell = hotbar[:, int((slot - 1) * sw): int(slot * sw)]
    border = np.concatenate([cell[:3].reshape(-1, 3), cell[-3:].reshape(-1, 3)])
    hsv = cv2.cvtColor(border[None], cv2.COLOR_BGR2HSV)[0]
    bright = (hsv[:, 2] > 200).mean()
    return bright >= min_ratio
