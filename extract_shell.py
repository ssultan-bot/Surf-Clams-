"""Cut the shell out of the photo using the SAM mask.

Keeps the original image coordinates (background blacked out, no cropping) so
any point picked here maps straight back to the source photo.

    D:\\Anaconda\\python.exe extract_shell.py
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import config


def clean_mask(mask: np.ndarray, keep_largest: bool = True, fill_holes: bool = False) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if keep_largest:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lab == big).astype(np.uint8)
    if fill_holes:
        from scipy import ndimage as ndi
        m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    return m * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(config.DEFAULT_IMAGE))
    ap.add_argument("--mask", default=str(config.OUTPUT_DIR / "sam_mask1.npy"))
    ap.add_argument("--fill-holes", action="store_true",
                    help="fill the marker-ink holes so the shell is solid")
    ap.add_argument("--keep-largest", action="store_true",
                    help="drop all but the largest piece (off by default: stay faithful "
                         "to the SAM mask as produced)")
    ap.add_argument("--grid", type=int, default=250, help="coordinate grid spacing (px)")
    ap.add_argument("--out-name", default="shell")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    h, w = bgr.shape[:2]
    mask = clean_mask(np.load(args.mask), keep_largest=args.keep_largest,
                      fill_holes=args.fill_holes)

    cut = bgr.copy()
    cut[mask == 0] = 0                       # background -> black, coordinates preserved

    p_cut = config.OUTPUT_DIR / f"{args.out_name}_cut.png"
    p_mask = config.OUTPUT_DIR / f"{args.out_name}_mask.png"
    cv2.imwrite(str(p_cut), cut)
    cv2.imwrite(str(p_mask), mask)

    # a gridded copy, for reading off point coordinates
    grid = cut.copy()
    for x in range(0, w, args.grid):
        cv2.line(grid, (x, 0), (x, h), (60, 140, 60), 1)
        cv2.putText(grid, str(x), (x + 5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (80, 255, 80), 2)
    for y in range(0, h, args.grid):
        cv2.line(grid, (0, y), (w, y), (60, 140, 60), 1)
        cv2.putText(grid, str(y), (8, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (80, 255, 80), 2)
    p_grid = config.OUTPUT_DIR / f"{args.out_name}_grid.png"
    cv2.imwrite(str(p_grid), grid)

    ys, xs = np.where(mask > 0)
    print(f"image      : {w}x{h}")
    print(f"shell mask : {int((mask>0).sum())} px ({100*(mask>0).mean():.1f}% of frame), "
          f"bbox x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]")
    print(f"saved      : {p_cut}")
    print(f"saved      : {p_mask}")
    print(f"saved      : {p_grid}")


if __name__ == "__main__":
    main()
