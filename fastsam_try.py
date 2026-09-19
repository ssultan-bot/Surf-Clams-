"""Run FastSAM over a grid of settings and render every candidate mask.

Purpose: pick, by eye, a setting that segments the shell accurately. Each config
produces one montage showing its largest masks overlaid on the photo, labelled
with area and fill ratio so they can be compared.

    D:\\Anaconda\\envs\\sea\\python.exe fastsam_try.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

import config

MODELS_DIR = config.PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
OUT = config.OUTPUT_DIR

# (tag, weights, imgsz, conf, iou)
CONFIGS = [
    ("cfg1_s512_c40",   "FastSAM-s.pt",  512, 0.40, 0.90),   # the old project's setting
    ("cfg2_s1024_c40",  "FastSAM-s.pt", 1024, 0.40, 0.90),
    ("cfg3_s1024_c15",  "FastSAM-s.pt", 1024, 0.15, 0.70),
    ("cfg4_x1024_c40",  "FastSAM-x.pt", 1024, 0.40, 0.90),
    ("cfg5_x1536_c25",  "FastSAM-x.pt", 1536, 0.25, 0.70),
    ("cfg6_x2048_c25",  "FastSAM-x.pt", 2048, 0.25, 0.70),
]

TOP_N = 12
COLS = 3
TW = 620          # thumbnail width


def get_masks(weights, imgsz, conf, iou, img_path, device):
    from ultralytics import FastSAM
    cwd = os.getcwd()
    os.chdir(MODELS_DIR)          # weights auto-download lands in models/
    try:
        model = FastSAM(weights)
        res = model(img_path, device=device, retina_masks=True, imgsz=imgsz,
                    conf=conf, iou=iou, save=False, verbose=False)
    finally:
        os.chdir(cwd)
    if not res or res[0].masks is None:
        return []
    return res[0].masks.data.cpu().numpy()


def montage(tag, masks, bgr, elapsed):
    h, w = bgr.shape[:2]
    th = int(TW * h / w)
    items = []
    for m in masks:
        mm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        items.append((int(mm.sum()), mm))
    items.sort(key=lambda t: -t[0])
    items = items[:TOP_N]

    rows = (len(items) + COLS - 1) // COLS or 1
    canvas = np.full((rows * th + 60, COLS * TW, 3), 30, np.uint8)
    cv2.putText(canvas, f"{tag}   ({len(masks)} masks, {elapsed:.1f}s)  -- showing largest {len(items)}",
                (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    for i, (area, mm) in enumerate(items):
        r, c = divmod(i, COLS)
        vis = bgr.copy()
        tint = np.zeros_like(vis); tint[mm > 0] = (0, 180, 255)
        vis = cv2.addWeighted(vis, 1.0, tint, 0.45, 0)
        ys, xs = np.where(mm > 0)
        if len(ys):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)
            fill = area / max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
        else:
            fill = 0
        thumb = cv2.resize(vis, (TW, th))
        cv2.putText(thumb, f"#{i}  area={100*area/(h*w):.1f}%  fill={fill:.2f}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        canvas[60 + r*th:60 + (r+1)*th, c*TW:(c+1)*TW] = thumb
    p = OUT / f"fastsam_{tag}.png"
    cv2.imwrite(str(p), canvas)
    return p, items


def main():
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    img_path = str(config.DEFAULT_IMAGE)
    bgr = cv2.imread(img_path)
    print(f"device={device}  image={Path(img_path).name}  {bgr.shape[1]}x{bgr.shape[0]}\n")

    for tag, weights, imgsz, conf, iou in CONFIGS:
        t0 = time.time()
        try:
            masks = get_masks(weights, imgsz, conf, iou, img_path, device)
        except Exception as e:
            print(f"{tag:18s} FAILED: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        if len(masks) == 0:
            print(f"{tag:18s} no masks")
            continue
        p, items = montage(tag, masks, bgr, dt)
        print(f"{tag:18s} {len(masks):3d} masks  {dt:5.1f}s  -> {p.name}")
        # save the single largest mask as npy for later use
        np.save(OUT / f"fastsam_{tag}_top.npy", items[0][1])


if __name__ == "__main__":
    main()
