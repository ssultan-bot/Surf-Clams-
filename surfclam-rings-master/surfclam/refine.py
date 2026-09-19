"""Improve a SAM mask: iterative refinement, then snap its boundary to real edges.

SAM gets the *topology* right -- it knows which blob is the shell and, given
negative points, which lumps of debris are not. What it is comparatively weak at
is placing the boundary to the pixel, because its mask is decoded from a
low-resolution embedding. Classical thresholding has the opposite profile: its
boundary sits exactly on the intensity step, but it cannot tell the shell from
a touching lump of debris.

So we use each for what it is good at: SAM decides the region, and the boundary
is then pulled onto the nearest strong image gradient.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

import config


def edge_strength(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gradient magnitude of the contrast-enhanced image, in [0, 1]."""
    g = cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)
    return m / (m.max() + 1e-9)


def boundary_score(gray: np.ndarray, mask: np.ndarray, grad: np.ndarray | None = None) -> float:
    """Mean gradient magnitude along the mask outline.

    A boundary that sits on the real intensity step scores high; one that floats
    across flat shell or flat background scores low. This is what lets us check
    that a refinement actually improved the mask rather than just moved it.
    """
    grad = edge_strength(gray) if grad is None else grad
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(float)
    return float(map_coordinates(grad, [pts[:, 1], pts[:, 0]], order=1, mode="nearest").mean())


def snap_to_edges(gray: np.ndarray, mask: np.ndarray, search: int = 14,
                  smooth: float = 12.0, min_gain: float = 3.0, probe: int = 8,
                  grad: np.ndarray | None = None) -> np.ndarray:
    """Pull the mask outline onto the true shell/background transition.

    The obvious objective -- move to the strongest nearby gradient -- is wrong
    here and measurably so: the growth rings inside the shell are themselves
    strong gradients, so the outline gets captured by a ring instead of the real
    margin. Scoring by gradient magnitude made the gradient metric rise 56-78%
    while the independent inside-vs-outside contrast *fell* 4-6%.

    So the score is the signed contrast across the candidate position -- bright
    inside, dark outside. A ring edge has bright material on both sides and
    therefore scores near zero, while the real margin scores high.

    A point only moves if it gains more than `min_gain` grey levels over staying
    put, and the displacement is smoothed around the contour so neighbours do
    not snap to different pixels and leave the outline ragged.
    """
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return m
    g = gray.astype(np.float32)
    out = np.zeros_like(m)
    h, w = m.shape
    for c in cnts:
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) < 16:
            cv2.fillPoly(out, [pts.astype(np.int32)], 1)
            continue
        tx = np.gradient(np.r_[pts[-1, 0], pts[:, 0], pts[0, 0]])[1:-1]
        ty = np.gradient(np.r_[pts[-1, 1], pts[:, 1], pts[0, 1]])[1:-1]
        tl = np.hypot(tx, ty) + 1e-9
        nx, ny = -ty / tl, tx / tl
        # orient the normal outward (contour winding is not guaranteed)
        probe_in = map_coordinates(m.astype(np.float32),
                                   [np.clip(pts[:, 1] + ny * 3, 0, h - 1),
                                    np.clip(pts[:, 0] + nx * 3, 0, w - 1)],
                                   order=0, mode="constant", cval=0)
        if probe_in.mean() > 0.5:                 # that side is inside -> flip
            nx, ny = -nx, -ny

        def sample(off):
            return map_coordinates(g, [np.clip(pts[:, 1] + ny * off, 0, h - 1),
                                       np.clip(pts[:, 0] + nx * off, 0, w - 1)],
                                   order=1, mode="nearest")

        offs = np.arange(-search, search + 1, dtype=float)
        score = np.empty((len(offs), len(pts)), dtype=np.float32)
        for i, o in enumerate(offs):
            score[i] = sample(o - probe) - sample(o + probe)   # inside minus outside
        k = int(np.argmin(np.abs(offs)))
        best = np.argmax(score, axis=0)
        gain = score[best, np.arange(len(pts))] - score[k]
        disp = np.where(gain > min_gain, offs[best], 0.0)
        if smooth:
            disp = gaussian_filter1d(disp, smooth, mode="wrap")
        moved = np.column_stack([pts[:, 0] + nx * disp, pts[:, 1] + ny * disp])
        cv2.fillPoly(out, [np.round(moved).astype(np.int32)], 1)
    return out


def refine_with_sam(predictor, coords, labels, logits_row, multimask: bool = True):
    """Second SAM pass, seeded with the first pass's own mask logits.

    This is SAM's documented refinement loop and it was simply missing: feeding
    the previous low-resolution mask back in alongside the same points lets the
    decoder sharpen a boundary it had to guess the first time.
    """
    return predictor.predict(point_coords=coords, point_labels=labels,
                             mask_input=logits_row[None, :, :],
                             multimask_output=multimask)
