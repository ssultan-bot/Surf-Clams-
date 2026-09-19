"""Extract growth lines with DexiNed (direction B: a learned dense-edge model).

Runs kornia's pretrained DexiNed on the GPU. Unlike HED (an object-boundary
detector), DexiNed produces dense edges and tends to capture fine internal
texture -- which is what the growth rings are. Run in the GPU env:

    D:\\Anaconda\\envs\\sea\\python.exe dexined_extract.py --img-size 512 --thresh 0.4
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

import config
from surfclam import imaging, locate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(config.DEFAULT_IMAGE))
    ap.add_argument("--img-size", type=int, default=512, help="DexiNed working size")
    ap.add_argument("--thresh", type=float, default=0.4, help="edge strength -> line")
    ap.add_argument("--erode", type=int, default=10)
    ap.add_argument("--out-name", default="B_dexined")
    args = ap.parse_args()

    from kornia.contrib import EdgeDetectorBuilder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bgr = imaging.load_bgr(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = imaging.to_gray(bgr)
    h, w = gray.shape

    # shell mask (auto), eroded
    small, f = imaging.downscale_gray(gray, config.WORK_MAX_SIDE)
    mask = cv2.resize(locate.segment_shell(small), (w, h), interpolation=cv2.INTER_NEAREST)
    if args.erode:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erode | 1, args.erode | 1))
        mask = cv2.erode(mask, k)

    # DexiNed
    det = EdgeDetectorBuilder.build("dexined", pretrained=True, image_size=args.img_size).to(device).eval()
    t = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
    with torch.inference_mode():
        out = det(t.to(device))
    if isinstance(out, (list, tuple)):
        out = out[-1]  # fused edge map (last side output)
    edge = out.squeeze().float().cpu().numpy()
    if edge.shape != (h, w):
        edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-9)
    edge[mask == 0] = 0.0

    edge_u8 = (edge * 255).astype(np.uint8)
    bw = (edge > args.thresh).astype(np.uint8) * 255
    bw[gray < config.RINGX_ABS_DARK] = 0
    binary = np.where(bw > 0, 0, 255).astype(np.uint8)
    gray_on_white = np.full_like(gray, 255)
    gray_on_white[bw > 0] = gray[bw > 0]
    overlay = bgr.copy()
    overlay[bw > 0] = (0, 0, 255)

    n_bands = int(cv2.connectedComponents(bw, 8)[0] - 1)
    print(f"device     : {device}")
    print(f"DexiNed    : img_size={args.img_size}, thresh={args.thresh}, {n_bands} components, "
          f"{100.0 * (bw > 0).sum() / max((mask > 0).sum(), 1):.1f}% of shell")

    for tag, img in {"edge": edge_u8, "binary": binary, "gray": gray_on_white, "overlay": overlay}.items():
        p = config.OUTPUT_DIR / f"{args.out_name}_{tag}.png"
        cv2.imwrite(str(p), img)
        print(f"saved      : {p}")


if __name__ == "__main__":
    main()
