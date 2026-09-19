"""End-to-end: one image -> locate the shell -> trace its centerline.

Examples
--------
Interactive (click the two tips)::

    python run.py --click

Headless with supplied full-res endpoints::

    python run.py --points 1160,300,3860,520

Headless with automatic endpoint detection (for testing)::

    python run.py --auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
from surfclam import imaging, locate, centerline, draw


def run(image_path, points=None, mode="auto", max_side=None, out_name=None):
    max_side = config.WORK_MAX_SIDE if max_side is None else max_side
    image_path = Path(image_path)

    bgr = imaging.load_bgr(image_path)
    gray = imaging.to_gray(bgr)
    small, f = imaging.downscale_gray(gray, max_side)  # work space; full*f -> work

    # --- endpoints (work coords) and shell mask -----------------------------
    if points is not None:
        ep_work = [(points[0] * f, points[1] * f), (points[2] * f, points[3] * f)]
        mask = locate.segment_shell(small, seeds=ep_work)
    elif mode == "click":
        from surfclam import interact
        p_full = interact.pick_two_points(bgr)
        ep_work = [(x * f, y * f) for x, y in p_full]
        mask = locate.segment_shell(small, seeds=ep_work)
    else:  # auto
        mask0 = locate.segment_shell(small, seeds=None)
        a, b = centerline.auto_endpoints(mask0)
        ep_work = [a, b]
        mask = locate.segment_shell(small, seeds=ep_work)

    # --- trace the middle line ---------------------------------------------
    path_work = centerline.trace(mask, ep_work[0], ep_work[1])
    path_full = path_work / f

    # --- outputs ------------------------------------------------------------
    mask_full = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    ov = draw.overlay(bgr, path_full, mask_full)

    stem = out_name or image_path.stem
    out_png = config.OUTPUT_DIR / f"{stem}_centerline.png"
    out_csv = config.OUTPUT_DIR / f"{stem}_centerline.csv"
    draw.save(out_png, ov)
    pd.DataFrame({"x": path_full[:, 0].round(2), "y": path_full[:, 1].round(2)}).to_csv(out_csv, index=False)

    length = float(np.sum(np.hypot(*np.diff(path_full, axis=0).T)))
    shell_px = int((mask_full > 0).sum())
    print(f"image      : {image_path.name}  ({bgr.shape[1]}x{bgr.shape[0]})")
    print(f"work scale : {f:.3f}  ({small.shape[1]}x{small.shape[0]})")
    print(f"endpoints  : start={_fmt(ep_work[0], f)}  end={_fmt(ep_work[1], f)}  (full-res px)")
    print(f"shell mask : {shell_px} px")
    print(f"centerline : {len(path_full)} pts, length {length:.0f} px")
    print(f"saved      : {out_png}")
    print(f"saved      : {out_csv}")
    return path_full, mask_full


def _fmt(pt_work, f):
    return f"({pt_work[0] / f:.0f},{pt_work[1] / f:.0f})"


def _parse_points(s):
    vals = [float(v) for v in s.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("--points needs x1,y1,x2,y2")
    return vals


def main():
    ap = argparse.ArgumentParser(description="Trace a surf clam shell's centerline.")
    ap.add_argument("--image", default=str(config.DEFAULT_IMAGE))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--points", type=_parse_points, help="full-res x1,y1,x2,y2 endpoints")
    g.add_argument("--click", action="store_true", help="pick the two tips in a window")
    g.add_argument("--auto", action="store_true", help="auto-detect the two tips (testing)")
    ap.add_argument("--max-side", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    mode = "click" if args.click else "auto"  # default auto when no flag
    run(args.image, points=args.points, mode=mode, max_side=args.max_side, out_name=args.out_name)


if __name__ == "__main__":
    main()
