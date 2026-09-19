"""Locate the surf clam shell in the photo -> a binary mask.

The shell is a bright, elongated object on a darker background. The hard part
(documented at length in the old project's METHODS.md) is *which* bright blob
is the shell: it can merge with debris, a second fragment, an overexposed
background patch, or a scale-bar card.

We sidestep that entirely when the caller supplies seed points that lie on the
shell (in this project, the two endpoints the user clicks): we simply keep the
connected component(s) those seeds fall in. With no seeds we fall back to the
old heuristic (largest component whose fill ratio is below a threshold).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

import config


def _binarize(gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast boost -> Otsu threshold -> morphological cleanup."""
    g = cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL, config.MORPH_KERNEL))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def _fill(mask_bool: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask_bool).astype(np.uint8) * 255


def segment_shell(gray: np.ndarray, seeds: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Return a uint8 {0,255} mask of the shell.

    seeds: optional list of (x, y) points known to be on the shell (working
    scale). When given, the mask is the union of the components containing
    those seeds — this is the robust path and is what the 2 clicked endpoints
    feed in.
    """
    m = _binarize(gray)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return _fill(m > 0)

    if seeds:
        labels = set()
        h, w = lab.shape
        for x, y in seeds:
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            lb = int(lab[yi, xi])
            if lb == 0:  # seed landed on background -> snap to nearest foreground
                lb = _nearest_label(lab, xi, yi)
            if lb > 0:
                labels.add(lb)
        if labels:
            keep = np.isin(lab, list(labels))
            return _fill(keep)

    # No usable seeds: largest "shell-like" component (fill ratio < threshold),
    # else just the largest component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    boxes = stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]
    ratio = areas / np.maximum(boxes, 1)
    pool = np.where(ratio < config.FILL_RATIO_MAX)[0]
    if len(pool) == 0:
        pool = np.arange(len(areas))
    big = 1 + pool[int(np.argmax(areas[pool]))]
    return _fill(lab == big)


def _nearest_label(lab: np.ndarray, xi: int, yi: int, radius: int = 40) -> int:
    """Nearest non-zero label to (xi, yi) within a search window."""
    h, w = lab.shape
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    win = lab[y0:y1, x0:x1]
    ys, xs = np.where(win > 0)
    if len(ys) == 0:
        return 0
    d = (ys + y0 - yi) ** 2 + (xs + x0 - xi) ** 2
    return int(win[ys[int(np.argmin(d))], xs[int(np.argmin(d))]])
