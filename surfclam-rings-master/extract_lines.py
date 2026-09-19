"""Extract the growth lines from a shell photo, in their original form.

    python extract_lines.py                 # default image, auto-locate shell

Produces, inside the shell only:
  *_binary.png      clean black-on-white ring lines
  *_gray.png        the rings' original grayscale pixels on white
  *_darkness.png    grayscale 'darker-than-background' view of the rings
  *_overlay.png     rings highlighted on the original photo (for context)
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import config
from surfclam import imaging, locate, extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(config.DEFAULT_IMAGE))
    ap.add_argument("--method",
                    choices=["adaptive", "ridge", "hyst_sato", "hyst_meijering",
                             "hyst_sato_clean", "hyst_meijering_clean"],
                    default="adaptive")
    ap.add_argument("--out-name", default="lines")
    args = ap.parse_args()

    bgr = imaging.load_bgr(args.image)
    gray = imaging.to_gray(bgr)

    # 1) locate the shell (rough mask, auto) -- confines extraction to the shell
    small, f = imaging.downscale_gray(gray, config.WORK_MAX_SIDE)
    mask_small = locate.segment_shell(small)
    mask = cv2.resize(mask_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2) extract the ring bands
    if args.method == "ridge":
        bw, dark = extract.band_mask_ridge(gray, mask)
    elif args.method == "hyst_sato":
        bw, dark = extract.band_mask_hyst(gray, mask, which="sato")
    elif args.method == "hyst_meijering":
        bw, dark = extract.band_mask_hyst(gray, mask, which="meijering")
    elif args.method in ("hyst_sato_clean", "hyst_meijering_clean"):
        which = "sato" if "sato" in args.method else "meijering"
        bw, dark = extract.band_mask_hyst(
            gray, mask, which=which,
            sigmas=config.CLEAN_SIGMAS, low=config.CLEAN_LOW,
            min_area=config.CLEAN_MIN_AREA, declutter_diag=config.CLEAN_DECLUTTER_DIAG)
    else:
        bw = extract.band_mask(gray, mask)
        dark = extract.darkness(gray, mask)
    gray_on_white = extract.as_grayscale(gray, bw, background=255)
    binary = np.where(bw > 0, 0, 255).astype(np.uint8)  # black lines on white

    overlay = bgr.copy()
    overlay[bw > 0] = (0, 0, 255)

    n_bands = int(cv2.connectedComponents(bw, 8)[0] - 1)
    band_px = int((bw > 0).sum())
    shell_px = int((mask > 0).sum())
    print(f"image      : {args.image}")
    print(f"shell mask : {shell_px} px")
    print(f"ring bands : {n_bands} connected components, {band_px} px "
          f"({100.0 * band_px / max(shell_px, 1):.1f}% of shell)")

    outs = {
        "binary": binary,
        "gray": gray_on_white,
        "darkness": dark,
        "overlay": overlay,
    }
    for tag, img in outs.items():
        p = config.OUTPUT_DIR / f"{args.out_name}_{tag}.png"
        cv2.imwrite(str(p), img)
        print(f"saved      : {p}")


if __name__ == "__main__":
    main()
