"""Automatic centreline from a trained U-Net -- no clicking required.

Ported from the earlier `Surf-Clam-` repository (`task2_predict_rings_ml/ml/`),
which trained a small U-Net to predict a thin band along the growth-band
centreline and then read a line out of it. The original lives across four files
that import each other by filename (`02_draw_lines`, `ml/model`, `ml/infer`);
this is a single self-contained module using this project's conventions, with
the same weights.

Measured over the 38 reader-comparison images: it produced a line on all 38, a
median 91% of whose points the model claims as real predictions (the rest are
bridged placeholders it flags itself), 95% of the line falling on bright shell,
at ~0.5s per image. Two images came out clearly poor. So it is good enough to
seed a line for a human to accept or correct -- not to trust unattended.

    from surfclam import autoline
    line, is_real, mask = autoline.centerline(gray, bgr)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d

import config

CKPT = config.PROJECT_DIR / "models" / "shell_unet.pt"
SIZE = 512                      # the square the model was trained at
_MODEL = None


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _build():
    """The trained architecture: a deliberately small U-Net (1.95M params),
    kept shallow because the training set was only a few dozen images."""
    import torch.nn as nn
    import torch

    def conv_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class SmallUNet(nn.Module):
        def __init__(self, base=16):
            super().__init__()
            self.enc1 = conv_block(3, base)
            self.enc2 = conv_block(base, base * 2)
            self.enc3 = conv_block(base * 2, base * 4)
            self.enc4 = conv_block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = conv_block(base * 8, base * 16)
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = conv_block(base * 16, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = conv_block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = conv_block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = conv_block(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    return SmallUNet()


def load_model(device: str = "cpu"):
    global _MODEL
    if _MODEL is None:
        import torch
        if not CKPT.exists():
            raise FileNotFoundError(
                f"missing U-Net weights: {CKPT}\n"
                f"copy them from the old repo:\n"
                f'  copy "D:\\Surf-Clam-\\task2_predict_rings_ml\\ml\\shell_unet.pt" "{CKPT}"')
        m = _build()
        m.load_state_dict(torch.load(str(CKPT), map_location="cpu"))
        m.eval().to(device)
        _MODEL = (m, device)
    return _MODEL[0]


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def _letterbox(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, scale, y0, x0


def predict_prob(gray: np.ndarray, bgr: np.ndarray | None = None,
                 device: str = "cpu") -> np.ndarray:
    """Per-pixel confidence in [0,1] at the original resolution, unthresholded.

    The raw probability -- not the binarised mask -- is what gives the
    centreline sub-pixel placement further down.
    """
    import torch
    model = load_model(device)
    h, w = gray.shape[:2]
    rgb = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None
           else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    sq, scale, y0, x0 = _letterbox(rgb, SIZE)
    x = torch.from_numpy(sq).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(prob[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)


def band_mask(gray, bgr=None, thresh: float = 0.5, device: str = "cpu") -> np.ndarray:
    """Binary band the model predicts, largest component only, holes filled.

    Note this is the *centreline band*, roughly 12% of the frame -- not a whole
    shell mask (~30%). The model was trained on thin bands drawn around
    hand-traced centrelines, so it answers "where is the line", not "where is
    the shell".
    """
    m = (predict_prob(gray, bgr, device) > thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = np.uint8(lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return ndi.binary_fill_holes(m > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# band -> line
# ---------------------------------------------------------------------------
def centerline_from_prob(prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Confidence-weighted centroid per column, within that column's band.

    More robust than the midpoint of the thresholded band: pixels near the
    band's fuzzy low-confidence edges barely move the average, so the line
    follows where the model is most confident rather than wherever the 0.5
    cutoff happened to fall.
    """
    xs = np.where(mask.any(axis=0))[0]
    ys_full = np.arange(mask.shape[0], dtype=float)
    mids = np.empty(len(xs), dtype=float)
    for i, x in enumerate(xs):
        col = mask[:, x] > 0
        w = prob[:, x] * col
        s = w.sum()
        if s < 1e-6:
            idx = np.where(col)[0]
            mids[i] = (idx.min() + idx.max()) / 2.0
        else:
            mids[i] = (ys_full * w).sum() / s
    return np.column_stack([xs.astype(float), mids])


def despike(path: np.ndarray, sigma: float = 50, thresh: float = 12.0,
            iters: int = 6) -> np.ndarray:
    """Iteratively reweighted local linear fit; points far off the local trend
    are replaced by interpolation. Removes the occasional column where the
    weighted centroid jumps."""
    n = len(path)
    y = path[:, 1].astype(float)
    x = np.arange(n, dtype=float) - n / 2
    weight = np.ones(n)
    for _ in range(iters):
        S0 = gaussian_filter1d(weight, sigma, mode="nearest")
        S1 = gaussian_filter1d(weight * x, sigma, mode="nearest")
        S2 = gaussian_filter1d(weight * x * x, sigma, mode="nearest")
        Sy = gaussian_filter1d(weight * y, sigma, mode="nearest")
        Sxy = gaussian_filter1d(weight * x * y, sigma, mode="nearest")
        denom = S0 * S2 - S1 * S1
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        slope = (S0 * Sxy - S1 * Sy) / denom
        intercept = (Sy - slope * S1) / np.clip(S0, 1e-9, None)
        weight = (np.abs(y - (intercept + slope * x)) <= thresh).astype(float)
    bad = weight == 0
    out = path.copy()
    if bad.any():
        idx = np.arange(n)
        out[bad, 1] = np.interp(idx[bad], idx[~bad], y[~bad])
    return out


def bridge_gaps(edge: np.ndarray, x_lo: float, x_hi: float, max_gap: int = 3):
    """Fill columns the model said nothing about, and flag them as not real.

    Internal gaps are interpolated between the confident points either side;
    beyond the model's reach the nearest y is held flat out to the reference
    extent. Those filled points are placeholders, not measurements, so
    `is_real` marks them -- a caller that draws them identically to real
    predictions would be passing off a flat guess as a trace.
    """
    edge = edge[np.argsort(edge[:, 0])]
    xs, ys = edge[:, 0], edge[:, 1]
    ox, oy, oreal = [xs[0]], [ys[0]], [True]
    for i in range(1, len(xs)):
        gap = xs[i] - xs[i - 1]
        if gap > max_gap:
            k = int(gap) - 1
            ox.extend(np.linspace(xs[i - 1], xs[i], k + 2)[1:-1])
            oy.extend(np.linspace(ys[i - 1], ys[i], k + 2)[1:-1])
            oreal.extend([False] * k)
        ox.append(xs[i]); oy.append(ys[i]); oreal.append(True)
    xs, ys = np.array(ox), np.array(oy)
    is_real = np.array(oreal, dtype=bool)

    if xs[0] > x_lo + max_gap:
        lead = np.arange(x_lo, xs[0])
        xs = np.concatenate([lead, xs])
        ys = np.concatenate([np.full(len(lead), ys[0]), ys])
        is_real = np.concatenate([np.zeros(len(lead), bool), is_real])
    if xs[-1] < x_hi - max_gap:
        trail = np.arange(xs[-1] + 1, x_hi + 1)
        xs = np.concatenate([xs, trail])
        ys = np.concatenate([ys, np.full(len(trail), ys[-1])])
        is_real = np.concatenate([is_real, np.zeros(len(trail), bool)])
    return np.column_stack([xs, ys]), is_real


def classical_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu shell mask, used only to bound how far the line may plausibly run.

    The component picked is the largest one whose area/bounding-box ratio is
    below 0.9. That ratio test is what keeps the scale-bar card out: it is a
    solid rectangle and so fills its own bounding box almost exactly, whereas a
    curved shell never does. Without it the reference extent runs off to the
    edge of the frame and the line gets bridged across hundreds of columns of
    background as flat placeholder.
    """
    g = cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        boxes = stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]
        shell_like = np.where(areas / np.maximum(boxes, 1) < 0.9)[0]
        pool = shell_like if len(shell_like) else np.arange(len(areas))
        m = np.uint8(lab == 1 + pool[np.argmax(areas[pool])]) * 255
    return ndi.binary_fill_holes(m > 0).astype(np.uint8) * 255


def centerline(gray: np.ndarray, bgr: np.ndarray | None = None,
               shell_mask: np.ndarray | None = None, device: str = "cpu"):
    """Predict -> weighted centreline -> despike -> bridge. Returns
    (line, is_real, band_mask).

    `shell_mask` supplies only the plausible left/right extent of the shell;
    its y values are not used -- that is the model's job. Pass this project's
    SAM mask when there is one, otherwise a plain Otsu mask is derived here.
    """
    band = band_mask(gray, bgr, device=device)
    if band.sum() == 0:
        raise RuntimeError("the U-Net predicted nothing on this image")

    prob = predict_prob(gray, bgr, device=device)
    line = despike(centerline_from_prob(prob, band))

    ref = classical_mask(gray) if shell_mask is None else (shell_mask > 0).astype(np.uint8) * 255

    ref_xs = np.where(ref.any(axis=0))[0]
    x_lo = min(line[:, 0].min(), ref_xs.min()) if len(ref_xs) else line[:, 0].min()
    x_hi = max(line[:, 0].max(), ref_xs.max()) if len(ref_xs) else line[:, 0].max()
    line, is_real = bridge_gaps(line, x_lo, x_hi)
    return line, is_real, band
