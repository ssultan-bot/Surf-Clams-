"""Extract the growth lines themselves, in their original form.

Within the shell mask, growth rings are dark bands over a brighter shell body.
We flatten the shell's slow brightness variation and keep the locally-dark
pixels as ring bands -- returned as a clean binary line map, and (separately)
as the original grayscale pixels on a plain background, so the rings keep the
way they actually look rather than being reduced to a skeleton or a heatmap.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import sato, meijering, apply_hysteresis_threshold

import config

_RIDGE_FILTERS = {"sato": sato, "meijering": meijering}


def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)


def darkness(gray: np.ndarray, shell_mask: np.ndarray, bg_ksize: int = None) -> np.ndarray:
    """Grayscale 'how much darker than the local shell background' image.

    This flattens the shell body's slow brightness so only the ring bands stand
    out. Returned as uint8 (0 = shell body, bright = a dark ring band), which is
    a faithful grayscale view of the rings on their own.
    """
    bg_ksize = (config.RINGX_BLOCK * 2 + 1) if bg_ksize is None else bg_ksize
    bg_ksize |= 1
    g = _clahe(gray).astype(np.float32)
    bg = cv2.GaussianBlur(g, (bg_ksize, bg_ksize), 0)
    dark = np.clip(bg - g, 0, None)
    dark[shell_mask == 0] = 0
    m = float(dark.max()) or 1.0
    return (dark / m * 255).astype(np.uint8)


def band_mask(gray: np.ndarray, shell_mask: np.ndarray,
              block: int = None, C: int = None, abs_dark: int = None, min_area: int = None) -> np.ndarray:
    """Binary mask (uint8 0/255) of the dark growth-ring bands inside the shell."""
    block = config.RINGX_BLOCK if block is None else block
    C = config.RINGX_C if C is None else C
    abs_dark = config.RINGX_ABS_DARK if abs_dark is None else abs_dark
    min_area = config.RINGX_MIN_AREA if min_area is None else min_area

    g = _clahe(gray)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block | 1, C)
    bw[shell_mask == 0] = 0
    bw[g < abs_dark] = 0  # near-black is ink / holes / background, not a real ring

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def as_grayscale(gray: np.ndarray, mask: np.ndarray, background: int = 255) -> np.ndarray:
    """The ring bands in their original grayscale on a plain background."""
    out = np.full_like(gray, background)
    out[mask > 0] = gray[mask > 0]
    return out


# --- direction 1 improvement: ridge-filter extraction (more continuous) -------

def ridge_response(gray: np.ndarray, shell_mask: np.ndarray,
                   sigmas=None, erode: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Sato tubeness response for dark line structures, confined to the shell.

    Returns (response in [0,1], eroded shell mask). Unlike per-pixel
    thresholding, the ridge filter responds continuously along a line, so ring
    bands come out as unbroken curves.
    """
    sigmas = config.RIDGE_SIGMAS if sigmas is None else sigmas
    erode = config.RIDGE_ERODE if erode is None else erode
    g = _clahe(gray)
    m = shell_mask.copy()
    if erode:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode | 1, erode | 1))
        m = cv2.erode(m, k)
    resp = sato(g.astype(np.float64) / 255.0, sigmas=sigmas, black_ridges=True)
    resp[m == 0] = 0.0
    resp /= (resp.max() + 1e-9)
    return resp, m


def band_mask_ridge(gray: np.ndarray, shell_mask: np.ndarray,
                    thresh: float = None, min_area: int = None, close: int = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Binary ring-band mask from the ridge response. Returns (mask, response_u8)."""
    thresh = config.RIDGE_THRESH if thresh is None else thresh
    min_area = config.RIDGE_MIN_AREA if min_area is None else min_area
    close = config.RIDGE_CLOSE if close is None else close

    resp, _ = ridge_response(gray, shell_mask)
    g = _clahe(gray)
    bw = (resp > thresh).astype(np.uint8) * 255
    bw[g < config.RINGX_ABS_DARK] = 0  # ink / holes are not rings
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close | 1, close | 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out, (resp * 255).astype(np.uint8)


def _drop_small(bw: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def declutter(bw: np.ndarray, min_diag: float) -> np.ndarray:
    """Drop small connected components (short bounding-box diagonal).

    Detached capillary-like fragments are short; real ring curves span a long
    bounding box, so a diagonal threshold trims the former and keeps the latter.
    """
    if not min_diag:
        return bw
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = np.zeros_like(bw)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if np.hypot(w, h) >= min_diag:
            out[lab == i] = 255
    return out


def band_mask_hyst(gray: np.ndarray, shell_mask: np.ndarray, which: str = "sato",
                   sigmas=None, low: float = None, high: float = None,
                   erode: int = None, min_area: int = None, close: int = None,
                   declutter_diag: float = 0.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Ridge extraction with hysteresis thresholding (direction A / C).

    Multi-scale ridge filter (`which` = 'sato' or 'meijering') then grow lines
    from strong seeds (>high) down to a faint floor (>low), so faint ring
    continuations are kept only where they connect to a confident core -- more
    complete than a single threshold, without the speckle. Returns (mask, resp).
    """
    sigmas = config.RIDGE_SIGMAS_MULTI if sigmas is None else sigmas
    low = config.RIDGE_HYST_LOW if low is None else low
    high = config.RIDGE_HYST_HIGH if high is None else high
    erode = config.RIDGE_ERODE if erode is None else erode
    min_area = config.RIDGE_MIN_AREA if min_area is None else min_area
    close = config.RIDGE_CLOSE if close is None else close

    g = _clahe(gray)
    m = shell_mask.copy()
    if erode:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode | 1, erode | 1))
        m = cv2.erode(m, k)
    filt = _RIDGE_FILTERS[which]
    resp = filt(g.astype(np.float64) / 255.0, sigmas=sigmas, black_ridges=True)
    resp = np.nan_to_num(resp)
    resp[m == 0] = 0.0
    resp /= (resp.max() + 1e-9)

    bw = apply_hysteresis_threshold(resp, low, high).astype(np.uint8) * 255
    bw[g < config.RINGX_ABS_DARK] = 0
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close | 1, close | 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    bw = _drop_small(bw, min_area)
    bw = declutter(bw, declutter_diag)
    return bw, (resp * 255).astype(np.uint8)
