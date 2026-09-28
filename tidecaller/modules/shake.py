"""Shake phase: after the bite, Fisch spawns white round "shake" buttons that must be clicked."""
from __future__ import annotations

import math

import cv2
import numpy as np

from tidecaller.config.schema import ShakeSettings


def white_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 215), (180, 45, 255))


def find_circles(img: np.ndarray, cfg: ShakeSettings) -> list[tuple[int, int]]:
    """Centers (region-local) of round white blobs, largest first."""
    mask = cv2.morphologyEx(white_mask(img), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits = []
    for c in contours:
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if area < cfg.min_button_area or per == 0:
            continue
        if 4 * math.pi * area / (per * per) < cfg.circularity:
            continue
        m = cv2.moments(c)
        hits.append((area, int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return [(x, y) for _, x, y in sorted(hits, reverse=True)]


def find_pixel(img: np.ndarray, cfg: ShakeSettings) -> tuple[int, int] | None:
    """Cheapest option: first sufficiently large white region, no shape test."""
    mask = white_mask(img)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_button_area // 2:
            return int(centroids[i][0]), int(centroids[i][1])
    return None
