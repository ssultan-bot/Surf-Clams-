"""Trace the shell's ventral (bottom) margin between two clicked points, and
offset it inward to get a measurement line that crosses the growth rings.

Why this beats the earlier attempts:
  * The medial axis follows the shell *outline*, so it drifts with shell shape
    and knows nothing about the rings.
  * Streamline integration along the ring-orientation field is unstable -- it
    exits the shell at the margin or spirals at the root, where the rings are
    concentric and the direction field is singular.
  * The ventral margin is a strong, unambiguous boundary, and the rings
    terminate on it, so a curve parallel to it crosses them.

The two clicked points split the shell outline into two arcs; we keep the lower
one. Constraining it to the two points means debris outside that span cannot
pull the trace off course.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

import config
from .growth_axis import smooth_axis


def shell_mask(gray: np.ndarray, seeds, close_k: int = None) -> np.ndarray:
    """Binary shell mask, keeping the component(s) the clicked points land in.

    A plain Otsu threshold is deliberate: raising it to drop the dim, ring-free
    gray band on the lower margin also erases the dimmer tail entirely, because
    the shell is not uniformly bright. The morphological close bridges the notch
    that debris bites out of the margin -- but only up to a point: too large a
    kernel merges the mask with the scale-bar card.
    """
    close_k = config.EDGE_CLOSE_K if close_k is None else close_k
    g = cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k | 1, close_k | 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = ndi.binary_fill_holes(m > 0).astype(np.uint8) * 255

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    labels = set()
    h, w = lab.shape
    for x, y in seeds:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        if lab[yi, xi] > 0:
            labels.add(int(lab[yi, xi]))
    if not labels:
        labels = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
    return np.isin(lab, list(labels)).astype(np.uint8) * 255


def bottom_arc(mask: np.ndarray, a_xy, b_xy) -> np.ndarray:
    """The lower of the two outline arcs delimited by the two points."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("empty shell mask")
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    ia = int(np.argmin(np.hypot(cnt[:, 0] - a_xy[0], cnt[:, 1] - a_xy[1])))
    ib = int(np.argmin(np.hypot(cnt[:, 0] - b_xy[0], cnt[:, 1] - b_xy[1])))
    if ia > ib:
        ia, ib = ib, ia
    arc1 = cnt[ia:ib + 1]
    arc2 = np.vstack([cnt[ib:], cnt[:ia + 1]])
    lower = arc1 if arc1[:, 1].mean() > arc2[:, 1].mean() else arc2
    # orient it so it starts at a_xy
    if np.hypot(*(lower[0] - np.asarray(a_xy))) > np.hypot(*(lower[-1] - np.asarray(a_xy))):
        lower = lower[::-1]
    return lower


def _inward_normals(path: np.ndarray, mask: np.ndarray, probe: float = 4.0) -> np.ndarray:
    """Unit normals at each path point, flipped to point into the shell."""
    tx = np.gradient(path[:, 0])
    ty = np.gradient(path[:, 1])
    tl = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tl, tx / tl
    h, w = mask.shape
    xs = np.clip(np.round(path[:, 0] + nx * probe), 0, w - 1).astype(int)
    ys = np.clip(np.round(path[:, 1] + ny * probe), 0, h - 1).astype(int)
    flip = mask[ys, xs] == 0          # that side is outside -> use the other
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    return np.column_stack([nx, ny])


def local_thickness(path: np.ndarray, normals: np.ndarray, mask: np.ndarray,
                    max_t: float = 600.0, step: float = 2.0) -> np.ndarray:
    """Shell thickness at each path point, measured along the inward normal."""
    h, w = mask.shape
    t = np.zeros(len(path))
    for i, (p, nvec) in enumerate(zip(path, normals)):
        d = 0.0
        while d < max_t:
            d += step
            x = int(np.clip(round(p[0] + nvec[0] * d), 0, w - 1))
            y = int(np.clip(round(p[1] + nvec[1] * d), 0, h - 1))
            if mask[y, x] == 0:
                break
        t[i] = d
    return t


def offset_inward(path: np.ndarray, mask: np.ndarray, frac: float,
                  smooth_thickness: float = 25.0) -> np.ndarray:
    """Shift the margin into the shell by `frac` of the local shell thickness.

    A fixed pixel offset is wrong here: the shell tapers strongly from root to
    tail, so the same offset would sit mid-shell at one end and outside it at
    the other. Offsetting along the inward normal (not straight up) keeps the
    line inside where the shell runs vertically, near the root.
    """
    if frac <= 0:
        return path
    normals = _inward_normals(path, mask)
    thick = local_thickness(path, normals, mask)
    if smooth_thickness:
        thick = ndi.gaussian_filter1d(thick, smooth_thickness, mode="nearest")
    return path + normals * (thick * frac)[:, None]


def offset_up(path: np.ndarray, dy: float) -> np.ndarray:
    """Translate the whole line straight up by `dy` pixels.

    Deliberately the plain thing: no normals, no per-point thickness. Offsetting
    along the local normal by a fraction of shell thickness was tried and failed
    badly -- at the curled root the normal ray runs *along* the shell instead of
    across it (47 rays hit the 600px measuring cap), the marker-ink holes stop
    rays early, and offsetting a concave stretch made the curve self-intersect.
    A rigid translation has none of those failure modes.
    """
    out = path.copy()
    out[:, 1] -= float(dy)
    return out


def trace(gray: np.ndarray, a_xy, b_xy, frac: float = 0.0,
          close_k: int = None, tol: float = 2.5, mask: np.ndarray = None):
    """Full pipeline: mask -> bottom arc -> inward offset -> smoothed line.

    Returns (line_xy, mask).
    """
    if mask is None:
        mask = shell_mask(gray, [a_xy, b_xy], close_k=close_k)
    arc = bottom_arc(mask, a_xy, b_xy)
    line = offset_inward(arc, mask, frac)
    return smooth_axis(line, tol=tol), mask
