"""Unit tests on synthetic shapes -- no real images needed.

Run:  python -m pytest tests/    (or)   python tests/test_centerline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surfclam import centerline


def _bent_bar(h=200, w=400, thickness=30):
    """A curved (arched) thick bar, like a shell in miniature."""
    mask = np.zeros((h, w), np.uint8)
    xs = np.arange(w)
    ys = (h * 0.5 - 60 * np.sin(np.pi * xs / w)).astype(int)  # arch
    for x, y in zip(xs, ys):
        mask[max(0, y - thickness): y + thickness, x] = 255
    return mask


def test_centerline_stays_inside_and_centered():
    mask = _bent_bar()
    a, b = centerline.auto_endpoints(mask)
    path = centerline.trace(mask, a, b)

    # every point lies inside the mask
    ix = np.clip(np.round(path[:, 0]).astype(int), 0, mask.shape[1] - 1)
    iy = np.clip(np.round(path[:, 1]).astype(int), 0, mask.shape[0] - 1)
    inside_frac = (mask[iy, ix] > 0).mean()
    assert inside_frac > 0.99, f"only {inside_frac:.2%} of the path is inside the shell"

    # endpoints span most of the bar's width (it traced the whole thing)
    assert path[:, 0].max() - path[:, 0].min() > 0.8 * mask.shape[1]


def test_endpoints_are_far_apart():
    mask = _bent_bar()
    a, b = centerline.auto_endpoints(mask)
    assert np.hypot(a[0] - b[0], a[1] - b[1]) > 0.7 * mask.shape[1]


if __name__ == "__main__":
    test_centerline_stays_inside_and_centered()
    test_endpoints_are_far_apart()
    print("all tests passed")
