"""Segment the shell with SAM using positive/negative point prompts.

Unlike FastSAM -- whose point "prompt" only *selects* among masks it already
generated, so a negative point on the debris deletes the whole shell mask -- SAM
feeds the prompts into its mask decoder and *generates* a mask that contains the
positive points and avoids the negative ones. That is what lets us ask for
"the shell, but not the debris / scale bar / marker ink".

    D:\\Anaconda\\envs\\sea\\python.exe sam_prompt.py
    D:\\Anaconda\\envs\\sea\\python.exe sam_prompt.py --pos 2500,400 --neg 3504,819
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import config

MODELS_DIR = config.PROJECT_DIR / "models"
CHECKPOINT = MODELS_DIR / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# Defaults chosen from the cfg4 masks: a point deep inside the shell, and
# negatives on the debris blob and the scale-bar card.
DEFAULT_POS = [(2500, 400), (3829, 521)]
DEFAULT_NEG = [(3504, 819), (3980, 1100)]


def _pts(s):
    if not s:
        return []
    out = []
    for chunk in s.split(";"):
        x, y = chunk.split(",")
        out.append((int(x), int(y)))
    return out


def build_predictor(device="cuda"):
    from segment_anything import sam_model_registry, SamPredictor
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM checkpoint missing: {CHECKPOINT}")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device=device)
    return SamPredictor(sam)


def segment(predictor, rgb, pos, neg, multimask=True):
    predictor.set_image(rgb)
    pts = np.array(list(pos) + list(neg), dtype=np.float32)
    lbl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=lbl, multimask_output=multimask
    )
    return masks, scores


def render(bgr, mask, pos, neg, label):
    vis = bgr.copy()
    tint = np.zeros_like(vis)
    tint[mask > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 1.0, tint, 0.42, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 5)
    for x, y in pos:
        cv2.circle(vis, (x, y), 26, (0, 255, 0), -1)
        cv2.putText(vis, "+", (x - 15, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
    for x, y in neg:
        cv2.circle(vis, (x, y), 26, (0, 0, 255), -1)
        cv2.putText(vis, "-", (x - 13, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 5)
    cv2.putText(vis, label, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 6)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(config.DEFAULT_IMAGE))
    ap.add_argument("--pos", default=None, help="x,y;x,y  positive points")
    ap.add_argument("--neg", default=None, help="x,y;x,y  negative points")
    ap.add_argument("--out-name", default="sam")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = _pts(args.pos) or DEFAULT_POS
    neg = _pts(args.neg) or DEFAULT_NEG

    bgr = cv2.imread(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    print(f"device={device}  image={w}x{h}")
    print(f"positive: {pos}")
    print(f"negative: {neg}")

    predictor = build_predictor(device)
    masks, scores = segment(predictor, rgb, pos, neg)

    rows = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        mm = m.astype(np.uint8)
        area = 100.0 * mm.sum() / (h * w)
        n_pieces = cv2.connectedComponents(mm, 8)[0] - 1
        label = f"SAM #{i}  score={s:.3f}  area={area:.1f}%  pieces={n_pieces}"
        print("  " + label)
        np.save(config.OUTPUT_DIR / f"{args.out_name}_mask{i}.npy", mm)
        rows.append(cv2.resize(render(bgr, mm, pos, neg, label), (1600, int(1600 * h / w))))

    out = config.OUTPUT_DIR / f"{args.out_name}_compare.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print("saved", out)


if __name__ == "__main__":
    main()
