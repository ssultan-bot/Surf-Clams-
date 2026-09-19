"""Image loading and scaling helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_bgr(path: str | Path) -> np.ndarray:
    """Read an image as BGR. Raises if the file cannot be read."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def work_scale(shape: tuple[int, int], max_side: int) -> float:
    """Return the factor f (<=1) to multiply a full-res length by to reach the
    working-resolution length, i.e. work_len = full_len * f."""
    h, w = shape[:2]
    longest = max(h, w)
    return min(1.0, max_side / float(longest))


def downscale_gray(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale a gray image so its long side <= max_side.

    Returns (small_gray, f) where a point (x, y) in full-res maps to
    (x * f, y * f) in small_gray.
    """
    f = work_scale(gray.shape, max_side)
    if f >= 1.0:
        return gray, 1.0
    new_w = max(1, int(round(gray.shape[1] * f)))
    new_h = max(1, int(round(gray.shape[0] * f)))
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, f
