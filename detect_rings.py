"""Detect growth rings along the automatic centerline and visualise them.

    python detect_rings.py --points 1100,800,4000,380

Saves an overlay (centerline + ring ticks) and a brightness-profile plot, and
writes the ring positions to CSV.
"""

from __future__ import annotations

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")  # headless: save the plot, don't open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import run
from surfclam import imaging, rings, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(config.DEFAULT_IMAGE))
    ap.add_argument("--points", type=run._parse_points, default=None,
                    help="full-res x1,y1,x2,y2 endpoints (else auto)")
    ap.add_argument("--out-name", default="rings")
    args = ap.parse_args()

    # 1) centerline (reuse the trace pipeline)
    mode = "auto" if args.points is None else "points"
    path, mask = run.run(args.image, points=args.points, mode=mode, out_name=f"{args.out_name}_base")

    # 2) sample brightness along it and find the dark bands
    bgr = imaging.load_bgr(args.image)
    gray = imaging.to_gray(bgr)
    s, x, y, nx, ny, prof = rings.profile_along(gray, path)
    valleys, smooth = rings.detect(prof)
    print(f"rings detected: {len(valleys)}")

    # 3) overlay: centerline + a red tick across each ring
    ov = draw.overlay(bgr, path, mask)
    tick = max(14, int(0.02 * max(bgr.shape[:2])))
    for i in valleys:
        p1 = (int(round(x[i] - nx[i] * tick)), int(round(y[i] - ny[i] * tick)))
        p2 = (int(round(x[i] + nx[i] * tick)), int(round(y[i] + ny[i] * tick)))
        cv2.line(ov, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(ov, f"{len(valleys)} rings", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    out_png = config.OUTPUT_DIR / f"{args.out_name}_overlay.png"
    cv2.imwrite(str(out_png), ov)

    # 4) brightness-profile plot
    plt.figure(figsize=(18, 4))
    plt.plot(s, smooth, color="steelblue", lw=1.8, label="brightness along centerline")
    plt.plot(s[valleys], smooth[valleys], "rv", ms=9, label=f"{len(valleys)} rings (dark bands)")
    plt.xlabel("distance along centerline (px)")
    plt.ylabel("gray (CLAHE, 0-255)")
    plt.title("Growth rings = dark valleys in the brightness profile")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_plot = config.OUTPUT_DIR / f"{args.out_name}_profile.png"
    plt.savefig(out_plot, dpi=110)

    # 5) CSV of ring positions
    out_csv = config.OUTPUT_DIR / f"{args.out_name}.csv"
    pd.DataFrame({
        "ring_no": np.arange(1, len(valleys) + 1),
        "x": np.round(x[valleys], 1),
        "y": np.round(y[valleys], 1),
        "dist_along_line_px": np.round(s[valleys], 1),
        "gray": np.round(smooth[valleys], 2),
    }).to_csv(out_csv, index=False)

    print(f"saved: {out_png}")
    print(f"saved: {out_plot}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
