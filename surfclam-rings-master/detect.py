"""One entry point for every line-tracing and ring-detection method we have.

    python detect.py --image <path>                       defaults: auto line + raw rings
    python detect.py --image <path> --rings global
    python detect.py --image <path> --line edge --p1 1150,600 --p2 4100,400
    python detect.py --dir <folder> --limit 10            batch
    python detect.py --list                               show all methods

Needs the GPU env for the U-Net / SAM line methods:
    D:\\Anaconda\\envs\\sea\\python.exe detect.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import config

LINE_METHODS = {
    "auto":   "U-Net centreline, no clicks (ported from the earlier repo)",
    "edge":   "shell margin traced between two points, then shifted (needs --p1/--p2)",
    "center": "medial axis of the shell mask between two points (needs --p1/--p2)",
    "saved":  "reuse a line.csv already saved by pick_line.py (--line-csv)",
}
RING_METHODS = {
    "raw":    "peaks/valleys straight off the pixel profile  [DEFAULT, best]",
    "global": "contrast-normalised profile, one global min-distance + prominence",
    "warped": "warp so spacing is uniform, then detect, then unwarp",
    "dp":     "dynamic programming over the whole ring sequence",
    "multi":  "several parallel lines pooled by union, with agreement votes",
}


def _pt(s):
    x, y = s.split(",")
    return float(x), float(y)


# ---------------------------------------------------------------------------
def trace_line(method, gray, bgr, args):
    """Returns (line, shell_mask, extra dict)."""
    from surfclam import autoline

    if method == "auto":
        line, is_real, _ = autoline.centerline(gray, bgr, device=args.device)
        return line, autoline.classical_mask(gray), {"real_fraction": float(is_real.mean())}

    if method == "saved":
        if not args.line_csv:
            raise SystemExit("--line-csv is required for --line saved")
        line = np.loadtxt(args.line_csv, delimiter=",", skiprows=1)
        return line, autoline.classical_mask(gray), {}

    if args.p1 is None or args.p2 is None:
        raise SystemExit(f"--line {method} needs --p1 x,y and --p2 x,y")
    from surfclam import imaging, locate, edge, centerline as cl

    h, w = gray.shape
    f = min(1.0, args.work / max(h, w))
    small = cv2.resize(gray, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    a = (args.p1[0] * f, args.p1[1] * f)
    b = (args.p2[0] * f, args.p2[1] * f)
    mask_s = locate.segment_shell(small, seeds=[a, b])
    mask = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)

    if method == "edge":
        arc = edge.bottom_arc(mask_s, a, b)
        line = edge.offset_up(edge.smooth_axis(arc, tol=2.5) / f, args.shift)
    else:
        line = cl.trace(mask_s, a, b) / f
    return line, mask, {}


def detect_rings(method, gray, line, mask, args):
    """Returns (xy_of_rings, strength_0_to_1, extra dict)."""
    from surfclam import ringdet

    if method == "raw":
        r = ringdet.detect_raw(gray, line, min_dist=args.min_dist,
                               prominence=args.prominence, smooth=args.smooth)
        idx, prom = r["valleys"], r["valley_prom"]
        st = prom / prom.max() if len(prom) else prom
        return r["xy"][idx], st, {"peaks": int(len(r["peaks"])), "xy": r["xy"],
                                  "profile": r["profile"], "peak_idx": r["peaks"]}

    if method == "multi":
        rings = ringdet.detect_multi(gray, line, mask, min_dist=args.min_dist,
                                     prominence=max(args.prominence, 0.02))
        if not rings:
            return np.empty((0, 2)), np.array([]), {}
        xy = np.array([[r["x"], r["y"]] for r in rings])
        st = np.array([r["strength"] for r in rings])
        return xy, st, {"votes": [r["votes"] for r in rings]}

    prof = ringdet.sample_profile(gray, line, mask, half_width=0.0)
    dark = ringdet.normalize(prof["prof"])
    valid = prof["coverage"] >= 0.5
    if method == "global":
        idx = ringdet.detect_global(dark, prof["s"], min_dist=args.min_dist,
                                    prominence=args.prominence, valid=valid)
    elif method == "warped":
        lam, _, _ = ringdet.local_wavelength(dark)
        idx = ringdet.detect_warped(dark, prof["s"], lam, valid=valid)
    elif method == "dp":
        lam, _, _ = ringdet.local_wavelength(dark)
        idx = ringdet.detect_dp(dark, prof["s"], lam, valid=valid)
    else:
        raise SystemExit(f"unknown ring method {method}")
    st = np.ones(len(idx))
    return prof["xy"][idx], st, {"xy": prof["xy"]}


def draw(bgr, line, ring_xy, strength, extra):
    v = bgr.copy()
    cv2.polylines(v, [np.round(line).astype(np.int32)], False, (0, 210, 210), 3, cv2.LINE_AA)
    if len(ring_xy) == 0:
        return v
    xy = extra.get("xy", line)
    tx, ty = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
    tl = np.hypot(tx, ty) + 1e-9
    T = max(22, int(0.030 * max(v.shape[:2])))
    for k, (x, y) in enumerate(ring_xy):
        i = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
        nx, ny = -ty[i] / tl[i], tx[i] / tl[i]
        w = float(strength[k]) if len(strength) else 1.0
        half = T * (0.5 + 0.8 * w)
        cv2.line(v, (int(x - nx * half), int(y - ny * half)),
                 (int(x + nx * half), int(y + ny * half)),
                 (0, int(60 + 150 * (1 - w)), 255), 4, cv2.LINE_AA)
    for i in extra.get("peak_idx", []):                 # bright increments
        nx, ny = -ty[i] / tl[i], tx[i] / tl[i]
        cv2.line(v, (int(xy[i, 0]), int(xy[i, 1])),
                 (int(xy[i, 0] + nx * T * 0.5), int(xy[i, 1] + ny * T * 0.5)),
                 (255, 170, 0), 2, cv2.LINE_AA)
    return v


def run_one(path, args, out_dir):
    from surfclam import imaging
    bgr = imaging.load_bgr(str(path))
    gray = imaging.to_gray(bgr)
    line, mask, linfo = trace_line(args.line, gray, bgr, args)
    ring_xy, strength, extra = detect_rings(args.rings, gray, line, mask, args)

    stem = Path(path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", draw(bgr, line, ring_xy, strength, extra))[1].tofile(
        str(out_dir / f"{stem}.png"))
    np.savetxt(out_dir / f"{stem}_line.csv", line, delimiter=",",
               header="x,y", comments="", fmt="%.2f")
    rows = ["x,y,strength" + (",votes" if "votes" in extra else "")]
    for k, (x, y) in enumerate(ring_xy):
        r = f"{x:.1f},{y:.1f},{float(strength[k]):.3f}"
        if "votes" in extra:
            r += f",{extra['votes'][k]}"
        rows.append(r)
    (out_dir / f"{stem}_rings.csv").write_text("\n".join(rows), encoding="utf-8")
    meta = dict(image=str(path), line_method=args.line, ring_method=args.rings,
                min_dist=args.min_dist, prominence=args.prominence,
                n_rings=int(len(ring_xy)), **linfo,
                **{k: v for k, v in extra.items() if isinstance(v, (int, float))})
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return len(ring_xy), extra.get("peaks")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show available methods and exit")
    ap.add_argument("--image", default=None)
    ap.add_argument("--dir", default=None, help="run over every jpg in this folder")
    ap.add_argument("--limit", type=int, default=0, help="cap how many images in --dir mode")
    ap.add_argument("--line", choices=list(LINE_METHODS), default="auto")
    ap.add_argument("--rings", choices=list(RING_METHODS), default="raw")
    ap.add_argument("--p1", type=_pt, default=None, help="x,y  first endpoint")
    ap.add_argument("--p2", type=_pt, default=None, help="x,y  second endpoint")
    ap.add_argument("--line-csv", default=None, help="line.csv for --line saved")
    ap.add_argument("--shift", type=float, default=120.0,
                    help="--line edge: pixels to move the line inward (negative = down)")
    ap.add_argument("--min-dist", dest="min_dist", type=float, default=5.0,
                    help="minimum pixels between two detections (smaller = more)")
    ap.add_argument("--prominence", type=float, default=0.015,
                    help="how obvious a band must be, as a fraction of profile range "
                         "(smaller = more)")
    ap.add_argument("--smooth", type=float, default=3.0, help="profile smoothing, px")
    ap.add_argument("--work", type=int, default=1600, help="working resolution for --line edge/center")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--out", default=None, help="output folder (default outputs/detect_<line>_<rings>)")
    args = ap.parse_args()

    if args.list:
        print("line methods (--line):")
        for k, v in LINE_METHODS.items():
            print(f"  {k:8s} {v}")
        print("\nring methods (--rings):")
        for k, v in RING_METHODS.items():
            print(f"  {k:8s} {v}")
        print("\ndefault: --line auto --rings raw --min-dist 5 --prominence 0.015")
        return

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:
            args.device = "cpu"

    out_dir = Path(args.out) if args.out else config.OUTPUT_DIR / f"detect_{args.line}_{args.rings}"
    if args.dir:
        files = sorted(p for p in Path(args.dir).glob("*.jpg") if not p.name.startswith("._"))
        if args.limit:
            files = files[:args.limit]
    elif args.image:
        files = [Path(args.image)]
    else:
        files = [config.DEFAULT_IMAGE]

    print(f"line={args.line}  rings={args.rings}  device={args.device}  "
          f"min_dist={args.min_dist}  prominence={args.prominence}")
    for f in files:
        n, npk = run_one(f, args, out_dir)
        print(f"  {f.name[:52]:52s} {n:4d} rings" + (f"  ({npk} bright peaks)" if npk else ""))
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
