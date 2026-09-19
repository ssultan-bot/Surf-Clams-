"""Extract growth lines with HED (Holistically-Nested Edge Detection).

Direction 2: a learned edge detector instead of classical filters. Runs the
pretrained HED model through OpenCV's DNN module (no torch needed), confined to
the shell. Kept independent of surfclam.extract so it runs in any env with cv2.

    python hed_extract.py --thresh 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import config
from surfclam import imaging, locate

MODEL_DIR = config.PROJECT_DIR / "models"
PROTO = MODEL_DIR / "hed_deploy.prototxt"
WEIGHTS = MODEL_DIR / "hed.caffemodel"
HED_MEAN = (104.00699, 116.66877, 122.67891)


class CropLayer:
    """HED's Caffe 'Crop' layer, which OpenCV's DNN doesn't implement natively."""

    def __init__(self, params, blobs):
        self.xstart = self.xend = self.ystart = self.yend = 0

    def getMemoryShapes(self, inputs):
        in_shape, target_shape = inputs[0], inputs[1]
        h, w = target_shape[2], target_shape[3]
        self.ystart = (in_shape[2] - h) // 2
        self.xstart = (in_shape[3] - w) // 2
        self.yend, self.xend = self.ystart + h, self.xstart + w
        return [[in_shape[0], in_shape[1], h, w]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def hed_edges(bgr: np.ndarray) -> np.ndarray:
    """Return HED edge strength in [0,1], same H x W as bgr."""
    if not (PROTO.exists() and WEIGHTS.exists()):
        raise FileNotFoundError(f"HED model missing in {MODEL_DIR}")
    try:
        cv2.dnn_registerLayer("Crop", CropLayer)
    except cv2.error:
        pass  # already registered
    net = cv2.dnn.readNetFromCaffe(str(PROTO), str(WEIGHTS))
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0, size=(w, h),
                                 mean=HED_MEAN, swapRB=False, crop=False)
    net.setInput(blob)
    edge = net.forward()[0, 0]
    return np.clip(edge, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(config.DEFAULT_IMAGE))
    ap.add_argument("--thresh", type=float, default=0.3, help="edge strength -> line")
    ap.add_argument("--erode", type=int, default=10, help="shrink shell mask (px)")
    ap.add_argument("--out-name", default="hed")
    args = ap.parse_args()

    bgr = imaging.load_bgr(args.image)
    gray = imaging.to_gray(bgr)
    h, w = gray.shape

    # locate shell (auto), on a downscaled copy, upscaled back
    small, f = imaging.downscale_gray(gray, config.WORK_MAX_SIDE)
    mask = cv2.resize(locate.segment_shell(small), (w, h), interpolation=cv2.INTER_NEAREST)
    if args.erode:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erode | 1, args.erode | 1))
        mask = cv2.erode(mask, k)

    # run HED at working scale for speed, upscale the edge map
    bgr_small = cv2.resize(bgr, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    edge_small = hed_edges(bgr_small)
    edge = cv2.resize(edge_small, (w, h), interpolation=cv2.INTER_LINEAR)
    edge[mask == 0] = 0.0

    # outputs
    edge_u8 = (edge * 255).astype(np.uint8)
    bw = (edge > args.thresh).astype(np.uint8) * 255
    binary = np.where(bw > 0, 0, 255).astype(np.uint8)
    gray_on_white = np.full_like(gray, 255)
    gray_on_white[bw > 0] = gray[bw > 0]
    overlay = bgr.copy()
    overlay[bw > 0] = (0, 0, 255)

    n_bands = int(cv2.connectedComponents(bw, 8)[0] - 1)
    print(f"image      : {args.image}")
    print(f"HED edges  : thresh={args.thresh}, {n_bands} components, "
          f"{int((bw > 0).sum())} px ({100.0 * (bw > 0).sum() / max((mask > 0).sum(), 1):.1f}% of shell)")

    for tag, img in {"edge": edge_u8, "binary": binary, "gray": gray_on_white, "overlay": overlay}.items():
        p = config.OUTPUT_DIR / f"{args.out_name}_{tag}.png"
        cv2.imwrite(str(p), img)
        print(f"saved      : {p}")


if __name__ == "__main__":
    main()
