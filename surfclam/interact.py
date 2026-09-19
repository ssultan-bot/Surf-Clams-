"""Interactive 2-point picker (the only human input the pipeline allows).

Opens a window, the user clicks the two ends of the shell (start then end),
ENTER confirms, R resets. Returns the two points in full-resolution image
coordinates. Requires a display; not used in headless runs.
"""

from __future__ import annotations

import cv2
import numpy as np

WIN = "Click START then END (the two tips) | R reset | ENTER confirm"


def pick_two_points(bgr: np.ndarray, win_w: int = 1600, win_h: int = 480) -> list[tuple[float, float]]:
    h, w = bgr.shape[:2]
    scale = min(win_w / w, win_h / h)
    disp_w, disp_h = int(w * scale), int(h * scale)
    base = cv2.resize(bgr, (disp_w, disp_h))
    pts: list[tuple[int, int]] = []

    def render():
        canvas = base.copy()
        for i, (x, y) in enumerate(pts):
            color = (0, 0, 255) if i == 0 else (0, 255, 0)
            cv2.circle(canvas, (x, y), 7, color, -1)
            cv2.putText(canvas, "START" if i == 0 else "END", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(canvas, "Click START then END | R reset | ENTER confirm",
                    (16, disp_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.imshow(WIN, render())

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, on_mouse)
    cv2.imshow(WIN, render())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(pts) == 2:
            break
        if key in (ord("r"), ord("R")):
            pts.clear()
            cv2.imshow(WIN, render())
    cv2.destroyAllWindows()

    return [(x / scale, y / scale) for x, y in pts]
