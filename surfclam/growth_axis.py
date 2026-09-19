"""Trace the growth axis -- the line that cuts perpendicularly across the rings.

The medial axis of the shell mask is the wrong line: it follows the *outline* of
the shell, so it drifts wherever the shell is thicker on one side, and it knows
nothing about the rings. The rings themselves define the correct axis: growth
proceeds perpendicular to each ring, and ring spacing changes along the shell
(wide near the tail, narrow near the root), so the axis must be driven by ring
orientation, not by shell geometry.

Method: estimate local ring orientation with the structure tensor (cheap, no
need to segment individual rings), take its perpendicular as the local growth
direction, and integrate that direction field into a streamline from the root to
the tail. The clicked endpoints anchor the two ends.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, gaussian_filter1d
from scipy.interpolate import splprep, splev

import config


def orientation_field(gray: np.ndarray, sigma_grad: float = 2.0, sigma_tensor: float = 12.0,
                      mask: np.ndarray | None = None):
    """Local ring orientation via the structure tensor.

    Returns (ux, uy): a unit vector field along the *ring* direction (the
    direction in which brightness varies least, i.e. along a dark band). Its
    perpendicular is the growth direction.

    Pass `mask` when the region of interest has a strong outer boundary. The
    shell/background edge is a far stronger gradient than any growth ring, and
    the tensor's smoothing radius reaches it from well inside the shell, so
    without masking the field near the margin reports the *outline* direction
    rather than the ring direction. Gradients outside the (eroded) mask are
    dropped and the tensor is renormalised by the valid-pixel weight.
    """
    g = cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)
    f = g.astype(np.float32) / 255.0
    f = gaussian_filter(f, sigma_grad)
    gy, gx = np.gradient(f)

    if mask is not None:
        k = int(max(3, 2 * round(sigma_grad + 2) + 1))
        inner = cv2.erode((mask > 0).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
        gx = gx * inner
        gy = gy * inner
        wsum = gaussian_filter(inner, sigma_tensor) + 1e-6
    else:
        wsum = 1.0

    # structure tensor, smoothed over a neighbourhood
    Jxx = gaussian_filter(gx * gx, sigma_tensor) / wsum
    Jyy = gaussian_filter(gy * gy, sigma_tensor) / wsum
    Jxy = gaussian_filter(gx * gy, sigma_tensor) / wsum

    # principal direction: eigenvector of the SMALLER eigenvalue = along the band
    diff = Jxx - Jyy
    denom = np.hypot(diff, 2.0 * Jxy) + 1e-12
    cos2t = diff / denom
    sin2t = (2.0 * Jxy) / denom
    theta = 0.5 * np.arctan2(sin2t, cos2t)   # direction of MAX variation (across band)
    # rotate 90 deg -> along the band (the ring direction)
    ux, uy = -np.sin(theta), np.cos(theta)
    return ux.astype(np.float32), uy.astype(np.float32)


def _sample(field: np.ndarray, x: float, y: float) -> float:
    return float(map_coordinates(field, [[y], [x]], order=1, mode="nearest")[0])


def trace_growth_axis(gray: np.ndarray, mask: np.ndarray, start_xy, end_xy,
                      step: float = 2.0, max_steps: int = 20000,
                      sigma_tensor: float = 12.0, pull: float = 0.35) -> np.ndarray:
    """Streamline from start to end that stays perpendicular to the rings.

    At each step we move along the local growth direction (perpendicular to the
    ring), with a mild pull toward the target so the streamline actually lands on
    the clicked endpoint rather than drifting off. `pull` blends the two: 0 = pure
    ring-driven, 1 = straight line to the target.
    """
    ux, uy = orientation_field(gray, sigma_tensor=sigma_tensor)
    # growth direction = perpendicular to ring direction
    gx_dir, gy_dir = -uy, ux

    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    inside = mask > 0
    h, w = mask.shape

    pos = start.copy()
    pts = [pos.copy()]
    prev_dir = end - start
    prev_dir /= (np.linalg.norm(prev_dir) + 1e-9)

    for _ in range(max_steps):
        to_end = end - pos
        dist = float(np.linalg.norm(to_end))
        if dist < step * 1.5:
            break
        to_end_u = to_end / (dist + 1e-9)

        d = np.array([_sample(gx_dir, pos[0], pos[1]), _sample(gy_dir, pos[0], pos[1])])
        n = np.linalg.norm(d)
        if n < 1e-6:
            d = prev_dir.copy()
        else:
            d /= n
        # orientation is 180-deg ambiguous: keep heading consistent
        if np.dot(d, prev_dir) < 0:
            d = -d
        # blend ring-driven direction with a pull toward the endpoint
        v = (1.0 - pull) * d + pull * to_end_u
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            v = to_end_u
        else:
            v /= nv

        nxt = pos + v * step
        xi = int(np.clip(round(nxt[0]), 0, w - 1))
        yi = int(np.clip(round(nxt[1]), 0, h - 1))
        if not inside[yi, xi]:
            # stepped out of the shell: fall back toward the target this step
            nxt = pos + to_end_u * step
            xi = int(np.clip(round(nxt[0]), 0, w - 1))
            yi = int(np.clip(round(nxt[1]), 0, h - 1))
            if not inside[yi, xi]:
                break
        prev_dir = v
        pos = nxt
        pts.append(pos.copy())

    pts.append(end.copy())
    return np.asarray(pts, dtype=float)


def smooth_axis(xy: np.ndarray, n: int = None, tol: float = None) -> np.ndarray:
    """Arc-length resample + approximating spline (same treatment as the old path)."""
    n = config.N_OUTPUT_POINTS if n is None else n
    tol = config.SMOOTH_TOL_PX if tol is None else tol
    keep = np.r_[True, np.any(np.abs(np.diff(xy, axis=0)) > 1e-6, axis=1)]
    xy = xy[keep]
    if len(xy) < 4:
        return xy
    seg = np.hypot(*np.diff(xy, axis=0).T)
    d = np.r_[0.0, np.cumsum(seg)]
    length = float(d[-1])
    m = max(8, int(length))
    du = np.linspace(0.0, length, m)
    xr = np.interp(du, d, xy[:, 0])
    yr = np.interp(du, d, xy[:, 1])
    tck, _ = splprep([xr, yr], s=m * (tol ** 2), k=3)
    xs, ys = splev(np.linspace(0, 1, n), tck)
    return np.column_stack([xs, ys])


def trace(gray: np.ndarray, mask: np.ndarray, start_xy, end_xy,
          sigma_tensor: float = 12.0, pull: float = 0.35) -> np.ndarray:
    return smooth_axis(trace_growth_axis(gray, mask, start_xy, end_xy,
                                         sigma_tensor=sigma_tensor, pull=pull))
