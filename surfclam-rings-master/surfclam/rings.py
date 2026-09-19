"""Extract growth rings by reading brightness along the centerline.

Growth rings cross the shell roughly perpendicular to the centerline, appearing
as dark bands. We sample a perpendicular-averaged grayscale profile along the
line, smooth it, and take the dark valleys as rings (via scipy.find_peaks).
This reuses the proven 1-D approach from the earlier project but rides on the
automatic centerline instead of a hand-drawn line.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import config


def clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)


def profile_along(gray: np.ndarray, path_xy: np.ndarray, half_width: int = None, step: float = 1.0):
    """Perpendicular-averaged brightness sampled along the path at ~step px.

    Returns (s, x, y, nx, ny, prof):
      s        arc length of each sample (px)
      x, y     sample coordinates on the centerline
      nx, ny   unit normal at each sample (for drawing ring ticks)
      prof     mean gray over +/- half_width px along the normal
    """
    half_width = config.RING_HALF_WIDTH if half_width is None else half_width
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    d = np.r_[0.0, np.cumsum(seg)]
    length = float(d[-1])
    s = np.arange(0.0, length, step)
    x = np.interp(s, d, path_xy[:, 0])
    y = np.interp(s, d, path_xy[:, 1])

    # unit tangent -> unit normal
    tx = np.gradient(x)
    ty = np.gradient(y)
    tlen = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tlen, tx / tlen

    g = clahe(gray).astype(np.float64)
    h, w = g.shape
    offsets = np.arange(-half_width, half_width + 1)
    prof = np.zeros_like(s)
    for o in offsets:
        xs = np.clip(np.round(x + nx * o), 0, w - 1).astype(int)
        ys = np.clip(np.round(y + ny * o), 0, h - 1).astype(int)
        prof += g[ys, xs]
    prof /= len(offsets)
    return s, x, y, nx, ny, prof


def detect(prof: np.ndarray, gray_sigma: float = None, min_dist: int = None, prominence: float = None):
    """Return (valley_indices, smoothed_profile). Valleys = dark bands = rings."""
    gray_sigma = config.RING_GRAY_SIGMA if gray_sigma is None else gray_sigma
    min_dist = config.RING_MIN_DIST if min_dist is None else min_dist
    prominence = config.RING_PROMINENCE if prominence is None else prominence
    smooth = gaussian_filter1d(prof, sigma=gray_sigma)
    rng = float(smooth.max() - smooth.min()) or 1.0
    valleys, _ = find_peaks(-smooth, distance=min_dist, prominence=rng * prominence)
    return valleys, smooth
