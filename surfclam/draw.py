"""Visualisation: overlay the mask and centerline on the photo."""

from __future__ import annotations

import cv2
import numpy as np


def overlay(bgr: np.ndarray, path_xy: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Draw the shell tint (optional), the centerline, and its two end markers.

    path_xy is in full-resolution image coordinates.
    """
    out = bgr.copy()
    if mask is not None:
        tint = np.zeros_like(out)
        tint[mask > 0] = (0, 180, 255)  # amber, BGR
        out = cv2.addWeighted(out, 1.0, tint, 0.25, 0)

    pts = np.round(path_xy).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=False, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    r = max(6, int(round(0.004 * max(out.shape[:2]))))
    cv2.circle(out, tuple(pts[0]), r, (0, 0, 255), -1, lineType=cv2.LINE_AA)   # start, red
    cv2.circle(out, tuple(pts[-1]), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)  # end, green
    return out


def save(path: str, image_bgr: np.ndarray) -> None:
    cv2.imwrite(str(path), image_bgr)
