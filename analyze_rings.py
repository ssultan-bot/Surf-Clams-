"""Run every ring-detection method on a saved line and compare them.

    D:\\Anaconda\\python.exe analyze_rings.py                 # newest result folder
    D:\\Anaconda\\python.exe analyze_rings.py --folder <name>

Reads results/<folder>/{line.csv, mask.npy, meta.json}, writes the comparison
figure, an overlay per method, and a CSV of ring positions and increments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from surfclam import imaging, ringdet

RESULTS = config.PROJECT_DIR / "results"


def newest_folder() -> Path:
    fs = [p for p in RESULTS.iterdir() if p.is_dir() and (p / "line.csv").exists()]
    if not fs:
        raise SystemExit("results/ 里没有保存过的线,先在 studio.py 里保存一条。")
    return max(fs, key=lambda p: p.stat().st_mtime)


def overlay(bgr, prof, idx, title, color=(0, 0, 255)):
    out = bgr.copy()
    xy = prof["xy"]
    cv2.polylines(out, [np.round(xy).astype(np.int32)], False, (255, 230, 0), 4, cv2.LINE_AA)
    tick = max(20, int(0.022 * max(out.shape[:2])))
    tx = np.gradient(xy[:, 0]); ty = np.gradient(xy[:, 1])
    tl = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tl, tx / tl
    for i in idx:
        p1 = (int(xy[i, 0] - nx[i] * tick), int(xy[i, 1] - ny[i] * tick))
        p2 = (int(xy[i, 0] + nx[i] * tick), int(xy[i, 1] + ny[i] * tick))
        cv2.line(out, p1, p2, color, 3, cv2.LINE_AA)
    cv2.putText(out, f"{title}: {len(idx)} rings", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None)
    ap.add_argument("--half-width", type=float, default=25.0)
    ap.add_argument("--sigma-tensor", type=float, default=40.0,
                    help="structure-tensor scale (px). Must be a decent fraction of the "
                         "ring spacing; too small and the orientation field measures "
                         "surface texture instead of ring bands.")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--min-cos", type=float, default=0.5,
                    help="drop stretches where the line runs along the rings rather "
                         "than across them (the hooks at each tip)")
    ap.add_argument("--offsets", default="0,60,120,180", help="parallel lines for consensus")
    args = ap.parse_args()

    folder = RESULTS / args.folder if args.folder else newest_folder()
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    line = np.loadtxt(folder / "line.csv", delimiter=",", skiprows=1)
    mask = np.load(folder / "mask.npy") if (folder / "mask.npy").exists() else None
    bgr = imaging.load_bgr(meta["image"])
    gray = imaging.to_gray(bgr)
    print(f"folder : {folder.name}")
    print(f"image  : {Path(meta['image']).name}  line pts={len(line)}")

    # ---- profile -----------------------------------------------------------
    prof = ringdet.sample_profile(gray, line, mask, half_width=args.half_width,
                                  sigma_tensor=args.sigma_tensor)
    dark = ringdet.normalize(prof["prof"])
    lam, wl, scal = ringdet.local_wavelength(dark)
    s = prof["s"]
    cov = prof["coverage"]
    # A stretch is measurable only if the band lies inside the shell AND the line
    # actually crosses the rings there. Where the line hooks up around a tip it
    # runs *parallel* to the rings (cos -> 0): no ring is crossed, so nothing can
    # be measured, and any peak found there is spurious.
    cosv = prof["cos_perp"]
    valid = (cov >= args.min_coverage) & (cosv >= args.min_cos)
    usable = float(np.sum(valid)) * (s[1] - s[0]) if len(s) > 1 else 0.0
    print(f"valid  : {100*valid.mean():.1f}% of samples usable "
          f"(coverage>={args.min_coverage:.0%} and cos>={args.min_cos:.2f}) "
          f"-> {usable:.0f}px of {s[-1]:.0f}px")
    print(f"         cos over the usable stretch: mean {cosv[valid].mean():.3f} "
          f"(vs {cosv.mean():.3f} over the whole line, dragged down by the end hooks)")
    print(f"profile: {len(s)} samples over {s[-1]:.0f}px | "
          f"coverage min {cov.min():.2f} (fraction of the band inside the shell)")
    print(f"oblique: cos(angle to ring normal) mean {prof['cos_perp'].mean():.3f} "
          f"min {prof['cos_perp'].min():.3f}  -> along-line spacing overstates the "
          f"true increment by {100*(1/prof['cos_perp'].mean()-1):.1f}% on average")
    print(f"local wavelength: {lam.min():.0f} -> {lam.max():.0f} px "
          f"(ratio {lam.max()/lam.min():.1f}x across the line)")

    # ---- detectors ---------------------------------------------------------
    methods = {
        "A global (old)": ringdet.detect_global(dark, s, valid=valid),
        "B warped": ringdet.detect_warped(dark, s, lam, valid=valid),
        "C dp-sequence": ringdet.detect_dp(dark, s, lam, valid=valid),
    }

    # ---- consensus across parallel lines -----------------------------------
    offs = [float(v) for v in args.offsets.split(",")]
    xs_per_line, per_line_n = [], []
    for dy in offs:
        ln = line.copy(); ln[:, 1] -= dy
        pr = ringdet.sample_profile(gray, ln, mask, half_width=args.half_width,
                                    sigma_tensor=args.sigma_tensor)
        dk = ringdet.normalize(pr["prof"])
        lm, _, _ = ringdet.local_wavelength(dk)
        ix = ringdet.detect_dp(dk, pr["s"], lm,
                               valid=(pr["coverage"] >= args.min_coverage) &
                                     (pr["cos_perp"] >= args.min_cos))
        xs_per_line.append(pr["xy"][ix, 0])
        per_line_n.append(len(ix))
    need = max(2, int(np.ceil(len(offs) * 0.6)))
    cx, votes = ringdet.consensus(xs_per_line, min_votes=need)
    print(f"consensus: per-line counts {per_line_n} -> {len(cx)} rings seen on >={need}/{len(offs)} lines")
    # map consensus x back to indices on the reference line
    cons_idx = np.array([int(np.argmin(np.abs(prof["xy"][:, 0] - x))) for x in cx], dtype=int)
    cons_idx = np.unique(cons_idx)
    methods["D consensus"] = cons_idx

    # ---- report ------------------------------------------------------------
    rows = []
    for name, idx in methods.items():
        inc = ringdet.increments(idx, prof)
        n = len(idx)
        med = float(np.median(inc["perp"])) if n > 1 else float("nan")
        infl = (float(np.mean(inc["along"] / np.maximum(inc["perp"], 1e-9))) - 1) * 100 if n > 1 else 0
        print(f"  {name:16s} rings={n:3d}  median increment {med:6.1f}px "
              f"(oblique correction removed {infl:.1f}%)")
        cv2.imwrite(str(folder / f"rings_{name.split()[0]}.png"), overlay(bgr, prof, idx, name))
        for k, i in enumerate(idx):
            rows.append(dict(method=name, ring_no=k + 1,
                             x=round(float(prof["xy"][i, 0]), 1),
                             y=round(float(prof["xy"][i, 1]), 1),
                             s_along_px=round(float(s[i]), 1),
                             s_perp_px=round(float(prof["s_perp"][i]), 1),
                             increment_perp_px=(round(float(inc["perp"][k - 1]), 1) if k else None),
                             darkness=round(float(dark[i]), 3)))
    pd.DataFrame(rows).to_csv(folder / "rings_all_methods.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(4, 1, figsize=(19, 15),
                           gridspec_kw={"height_ratios": [2, 2, 1.4, 1.6]})
    ax[0].plot(s, dark, lw=1.1, color="0.25")
    ax[0].fill_between(s, -3, 3, where=cov < 0.6, color="orange", alpha=0.25,
                       label="coverage<0.6 (gap: ink/debris)")
    for (name, idx), c in zip(methods.items(), ["tab:red", "tab:blue", "tab:green", "tab:purple"]):
        ax[0].plot(s[idx], dark[idx], "v", ms=7, color=c, label=f"{name} ({len(idx)})")
    ax[0].set_title("normalised darkness along the line (peaks = dark bands = rings)")
    ax[0].set_ylabel("darkness (sd)"); ax[0].legend(ncol=3, fontsize=9); ax[0].grid(alpha=.3)

    ax[1].imshow(scal, aspect="auto", origin="lower", extent=[s[0], s[-1], 0, len(wl) - 1],
                 cmap="magma")
    yt = np.linspace(0, len(wl) - 1, 6).astype(int)
    ax[1].set_yticks(yt); ax[1].set_yticklabels([f"{wl[i]:.0f}" for i in yt])
    ax[1].plot(s, np.interp(lam, wl, np.arange(len(wl))), color="cyan", lw=2, label="dominant λ")
    ax[1].set_ylabel("wavelength (px)"); ax[1].legend(); ax[1].set_title("scalogram: local ring spacing")

    ax[2].plot(s, lam, color="teal", lw=2)
    ax[2].set_ylabel("local λ (px)"); ax[2].grid(alpha=.3)
    ax[2].set_title(f"local ring spacing varies {lam.max()/lam.min():.1f}x along the line "
                    f"-- why one global min-distance cannot work")

    for (name, idx), c in zip(methods.items(), ["tab:red", "tab:blue", "tab:green", "tab:purple"]):
        inc = ringdet.increments(idx, prof)
        if inc["n"] > 1:
            ax[3].plot(prof["s_perp"][idx][1:], inc["perp"], "o-", ms=4, color=c, label=name)
    ax[3].set_xlabel("distance along the line (perpendicular-corrected px)")
    ax[3].set_ylabel("increment (px)"); ax[3].legend(fontsize=9); ax[3].grid(alpha=.3)
    ax[3].set_title("measured increments -- a real growth series should shrink smoothly")

    plt.tight_layout()
    fig.savefig(folder / "rings_compare.png", dpi=100)
    print(f"\nsaved  : {folder / 'rings_compare.png'}")
    print(f"saved  : {folder / 'rings_all_methods.csv'}  + rings_<method>.png")


if __name__ == "__main__":
    main()
