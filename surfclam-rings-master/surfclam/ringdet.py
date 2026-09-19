"""Detect growth rings along a traced line, and measure the spacing between them.

The signal is a 1-D brightness profile sampled along the line; each dark band it
crosses is a growth ring. Three things make this harder than "smooth it and call
find_peaks", and this module addresses each:

  * **Ring spacing varies systematically along the line** (wide at one end,
    narrow at the other). A single global min-distance / prominence therefore
    cannot work: whatever value resolves the crowded end over-splits the wide
    end, and vice versa. We estimate the *local* dominant wavelength and let the
    detectors use it.
  * **Oblique crossing inflates spacing.** An increment is defined perpendicular
    to the rings; a line crossing at angle t to the ring normal measures
    spacing/cos(t). We integrate the perpendicular component so reported
    increments are the real ones.
  * **A single line is a thin sample of a 2-D band.** Rings should reappear on
    nearby parallel lines; noise should not. `consensus` exploits that.

Nothing here decides *which* dark bands are annual -- that is a scientific call
that has to be calibrated against the reader-measured data in this dataset.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks, fftconvolve

import config
from .growth_axis import orientation_field


# ---------------------------------------------------------------------------
# 0. the method that works best -- plain peaks and valleys off the pixel values
# ---------------------------------------------------------------------------
def detect_raw(gray: np.ndarray, line: np.ndarray, min_dist: float = 5.0,
               prominence: float = 0.015, smooth: float = 3.0, clahe: bool = True):
    """Peaks and valleys read straight off the grey profile along the line.

    This is the original, simplest approach and on inspection it beats every
    elaboration that followed. Two of those elaborations were actively harmful
    and are kept only for comparison:

      * band-averaging along the estimated ring direction lowered measured band
        contrast at every width tried (11.35 grey levels with no averaging,
        10.59 at the best averaged setting) -- the direction estimate is not
        accurate enough to integrate tens of pixels along;
      * local-contrast normalisation divides by a running standard deviation,
        which equalises contrast along the line and therefore *suppresses* a
        faint band sitting in a stretch that is already busy with bands -- the
        opposite of what maximising recall needs.

    So: CLAHE, light smoothing, find_peaks. Valleys are the dark growth bands;
    peaks are the bright increments between them. Lower `prominence` and
    `min_dist` to detect more.

    Returns dict with s, xy, profile, valleys, peaks and their prominences.
    """
    s, xy = resample(line, 1.0)
    g = (cv2.createCLAHE(config.CLAHE_CLIP, (config.CLAHE_TILE, config.CLAHE_TILE)).apply(gray)
         if clahe else gray)
    p = map_coordinates(g.astype(float), [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
    if smooth:
        p = gaussian_filter1d(p, smooth, mode="nearest")
    rng = float(p.max() - p.min()) or 1.0
    d = max(1, int(min_dist))
    val, vp = find_peaks(-p, distance=d, prominence=rng * prominence)
    pk, pp = find_peaks(p, distance=d, prominence=rng * prominence)
    return dict(s=s, xy=xy, profile=p, valleys=val, peaks=pk,
                valley_prom=vp["prominences"], peak_prom=pp["prominences"])


# ---------------------------------------------------------------------------
# 1. sampling the profile
# ---------------------------------------------------------------------------
def resample(line: np.ndarray, step: float = 1.0):
    """Uniform arc-length resampling. Returns (s, xy)."""
    seg = np.hypot(*np.diff(line, axis=0).T)
    d = np.r_[0.0, np.cumsum(seg)]
    s = np.arange(0.0, float(d[-1]), step)
    x = np.interp(s, d, line[:, 0])
    y = np.interp(s, d, line[:, 1])
    return s, np.column_stack([x, y])


def sample_profile(gray: np.ndarray, line: np.ndarray, mask: np.ndarray | None = None,
                   half_width: float = 25.0, step: float = 1.0,
                   sigma_tensor: float = 40.0, along_ring: bool = True):
    """Band-averaged brightness along the line.

    A ring is a band, so averaging *along the ring's own direction* (from the
    structure tensor) rather than merely perpendicular to the line integrates
    along the feature and is the single biggest signal-to-noise win. Pixels
    outside `mask` (background, marker-ink holes, excised debris) are excluded
    rather than averaged in as if they were shell.

    Returns dict with s, xy, prof, coverage, cos_perp, s_perp.
    """
    s, xy = resample(line, step)
    h, w = gray.shape

    tx = np.gradient(xy[:, 0])
    ty = np.gradient(xy[:, 1])
    tl = np.hypot(tx, ty) + 1e-9
    tx, ty = tx / tl, ty / tl                       # unit tangent of the line

    if along_ring:
        ux, uy = orientation_field(gray, sigma_grad=1.5, sigma_tensor=sigma_tensor, mask=mask)
        ax = map_coordinates(ux, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        ay = map_coordinates(uy, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        al = np.hypot(ax, ay) + 1e-9
        ax, ay = ax / al, ay / al                   # unit vector ALONG the ring
    else:
        ax, ay = -ty, tx                            # simply perpendicular to the line

    # ring normal = growth direction; cos of angle between line tangent and it
    nx, ny = -ay, ax
    cos_perp = np.abs(tx * nx + ty * ny)
    s_perp = np.r_[0.0, np.cumsum(0.5 * (cos_perp[1:] + cos_perp[:-1]) * np.diff(s))]

    # half_width 0 means sample the line itself, one pixel wide. Measured band
    # contrast was highest there (11.35 grey levels) and fell at every averaging
    # width tried, because the ring-direction estimate is not accurate enough for
    # integrating tens of pixels along it to do anything but smear the band.
    offs = np.arange(-half_width, half_width + 1e-9, 1.0) if half_width > 0 else np.array([0.0])
    acc = np.zeros(len(s))
    cnt = np.zeros(len(s))
    for o in offs:
        px = xy[:, 0] + ax * o
        py = xy[:, 1] + ay * o
        val = map_coordinates(gray.astype(np.float64), [py, px], order=1, mode="nearest")
        if mask is not None:
            ok = map_coordinates((mask > 0).astype(np.float64), [py, px],
                                 order=1, mode="constant", cval=0.0) > 0.5
        else:
            ok = np.ones(len(s), dtype=bool)
        acc += np.where(ok, val, 0.0)
        cnt += ok
    prof = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    coverage = cnt / len(offs)

    # fill short gaps so downstream filters behave; long gaps stay flagged
    good = np.isfinite(prof)
    if good.any():
        prof = np.interp(s, s[good], prof[good])

    return dict(s=s, xy=xy, prof=prof, coverage=coverage,
                cos_perp=cos_perp, s_perp=s_perp)


def normalize(prof: np.ndarray, hp_sigma: float = 400.0, ln_sigma: float = 200.0,
              smooth: float = 3.0) -> np.ndarray:
    """High-pass + local contrast normalisation -> 'darkness' signal (ring = high).

    Removes slow illumination drift and makes ring contrast comparable along the
    whole line, so a dim stretch is not silently under-detected relative to a
    bright one.
    """
    p = gaussian_filter1d(prof.astype(float), smooth, mode="nearest")
    base = gaussian_filter1d(p, hp_sigma, mode="nearest")
    hp = p - base
    sd = np.sqrt(gaussian_filter1d(hp ** 2, ln_sigma, mode="nearest")) + 1e-9
    return -(hp / sd)                       # invert: dark bands become peaks


# ---------------------------------------------------------------------------
# 2. local wavelength (how far apart are the rings, here?)
# ---------------------------------------------------------------------------
def scalogram(dark: np.ndarray, wavelengths: np.ndarray, cycles: float = 3.0):
    """|Morlet response| at each wavelength and position -> (len(w), len(dark))."""
    out = np.empty((len(wavelengths), len(dark)))
    x = dark - dark.mean()
    for i, lam in enumerate(wavelengths):
        sigma = cycles * lam / 2.0
        n = int(np.ceil(4 * sigma))
        t = np.arange(-n, n + 1)
        kern = np.exp(2j * np.pi * t / lam) * np.exp(-t ** 2 / (2 * sigma ** 2))
        kern /= np.abs(kern).sum() + 1e-12
        out[i] = np.abs(fftconvolve(x, kern, mode="same"))
    return out


def local_wavelength(dark: np.ndarray, wmin: float = 20.0, wmax: float = 500.0,
                     n: int = 40, smooth: float = 300.0):
    """Dominant wavelength as a function of position. Returns (lam, wavelengths, scal)."""
    wl = np.geomspace(wmin, wmax, n)
    scal = scalogram(dark, wl)
    lam = wl[np.argmax(scal, axis=0)]
    lam = np.exp(gaussian_filter1d(np.log(lam), smooth, mode="nearest"))
    return lam, wl, scal


# ---------------------------------------------------------------------------
# 3. detectors
# ---------------------------------------------------------------------------
def _keep_valid(idx, valid):
    """Drop detections that fall in unusable stretches (ink holes, excised
    debris, line briefly outside the shell). Silently keeping them would turn a
    data gap into a phantom ring."""
    if valid is None or len(idx) == 0:
        return idx
    return idx[valid[idx]]


def detect_global(dark, s, min_dist=15.0, prominence=0.08, valid=None):
    """Baseline: the earlier project's approach -- one min-distance and one
    prominence for the whole line."""
    rng = float(dark.max() - dark.min()) or 1.0
    idx, _ = find_peaks(dark, distance=max(1, int(min_dist)), prominence=rng * prominence)
    return _keep_valid(idx, valid)


def detect_multi(gray, line, mask, offsets=(-30, -15, 0, 15, 30),
                 min_dist=10.0, prominence=0.08, smooth=1.5, tol=16.0):
    """Candidates pooled across several parallel lines -- the recall-first method.

    Two findings drove this shape. Band-averaging along the estimated ring
    direction was measured to *lower* band contrast at every width tried (11.35
    grey levels with no averaging, 10.59 at the best averaged setting), so the
    profile is sampled one pixel wide. And a band that is faint where one line
    crosses it is often obvious 30-60px away, so the lines are pooled by UNION,
    not intersection: the goal is to miss nothing and let a reviewer discard,
    since the sub-annual checks matter in their own right.

    How many of the lines saw a given band becomes a confidence score rather
    than a filter, so nothing is silently dropped.

    Returns a list of dicts sorted along the line, each with x, y, votes (how
    many lines saw it), strength (0-1, best across lines) and s (arc position).
    """
    ux, uy = orientation_field(gray, sigma_grad=1.5, sigma_tensor=40.0, mask=mask)
    per_line = []
    for dy in offsets:
        ln = line.copy()
        ln[:, 1] = ln[:, 1] - dy
        prof = sample_profile(gray, ln, mask, half_width=0.0)
        dark = normalize(prof["prof"], smooth=smooth)
        idx, prom, strength = detect_candidates(dark, prof["s"], min_dist, prominence,
                                                valid=prof["coverage"] >= 0.5)
        for i, st in zip(idx, strength):
            x0, y0 = float(prof["xy"][i, 0]), float(prof["xy"][i, 1])
            # Project the hit back along its own band to where that band crosses
            # the centre line. The bands are strongly tilted, so a detection made
            # 60px above the centre line sits tens of px away in x from the same
            # band's crossing there; matching on raw x made agreement across
            # lines essentially impossible (5-line consensus was always zero).
            if dy:
                ay = float(map_coordinates(uy, [[y0], [x0]], order=1, mode="nearest")[0])
                ax = float(map_coordinates(ux, [[y0], [x0]], order=1, mode="nearest")[0])
                if abs(ay) > 0.2:                     # band not parallel to the line
                    x0 = x0 + (ax / ay) * dy
            per_line.append((x0, float(prof["xy"][i, 1]), float(st), dy,
                             float(prof["s"][i])))
    if not per_line:
        return []
    per_line.sort(key=lambda t: t[0])

    # Group by distance from the cluster's FIRST member, not its last. Comparing
    # against the last member chains: with candidates ~20px apart and a 20px
    # tolerance every point extends the run, and 700 detections collapsed into
    # 12 clusters -- a union that returned fewer rings than a single line.
    out, cur = [], [per_line[0]]
    for p in per_line[1:]:
        if p[0] - cur[0][0] <= tol:
            cur.append(p)
        else:
            out.append(cur); cur = [p]
    out.append(cur)

    rings = []
    for c in out:
        best = max(c, key=lambda t: t[2])
        rings.append(dict(x=float(np.mean([t[0] for t in c])),
                          y=best[1], s=best[4],
                          votes=len({t[3] for t in c}),
                          strength=best[2]))
    return rings


def detect_candidates(dark, s, min_dist=6.0, prominence=0.02, valid=None):
    """High-recall candidate list: every dark band worth a human's glance.

    Deliberately over-detects. The goal here is not a correct ring count -- it
    is to miss nothing, because the sub-annual "false" checks matter in their
    own right (disturbance, spawning, storms) and only a person can say which
    band is which. Precision is the reviewer's job; recall is ours.

    Returns (idx, prominence, strength) where strength is the prominence
    rescaled to 0-1 within this image, so a reviewer can triage: strong bands
    first, faint ones flagged rather than silently dropped.
    """
    rng = float(dark.max() - dark.min()) or 1.0
    idx, props = find_peaks(dark, distance=max(1, int(min_dist)),
                            prominence=rng * prominence)
    prom = props["prominences"]
    if valid is not None and len(idx):
        keep = valid[idx]
        idx, prom = idx[keep], prom[keep]
    if len(prom) == 0:
        return idx, prom, prom
    lo, hi = prom.min(), prom.max()
    strength = (prom - lo) / (hi - lo) if hi > lo else np.ones_like(prom)
    return idx, prom, strength


def detect_warped(dark, s, lam, prominence=0.10, per_lambda=0.55, valid=None):
    """Warp so spacing becomes uniform, detect, warp back.

    If spacing shrinks smoothly, then in the coordinate u = integral(ds/lambda)
    the rings sit at roughly constant intervals, so one min-distance is finally
    legitimate.
    """
    u = np.r_[0.0, np.cumsum(np.diff(s) / lam[:-1])]
    m = max(64, int(u[-1] * 12))
    ug = np.linspace(u[0], u[-1], m)
    dg = np.interp(ug, u, dark)
    du = ug[1] - ug[0]
    rng = float(dg.max() - dg.min()) or 1.0
    idx, _ = find_peaks(dg, distance=max(1, int(per_lambda / du)), prominence=rng * prominence)
    back = np.searchsorted(s, np.interp(ug[idx], u, s))
    return _keep_valid(np.clip(back, 0, len(s) - 1), valid)


def detect_dp(dark, s, lam, ring_cost=0.60, spacing_weight=2.5,
              lo=0.45, hi=2.2, cand_prominence=0.20, valid=None):
    """Pick the whole sequence of rings at once, not each peak independently.

    Score = sum of ring salience - cost per ring - penalty for spacings that
    disagree with the locally expected wavelength. Solved exactly by dynamic
    programming. This is the one detector that uses the biological prior that
    increments form a smooth, gradually shrinking sequence -- exactly the
    information a per-peak threshold cannot use.

    Salience is peak *prominence* scaled by its own 75th percentile, not
    amplitude scaled by the full range: one outlier spike (a data gap at the end
    of the line will produce one) otherwise squashes every real peak's score to
    near zero and the chain collapses to that single spike.
    """
    cand, props = find_peaks(dark, prominence=cand_prominence)
    if valid is not None and len(cand):
        keep = valid[cand]
        cand, props = cand[keep], {k: v[keep] for k, v in props.items()}
    if len(cand) == 0:
        return np.array([], dtype=int)
    prom = props["prominences"]
    amp = np.clip(prom / (np.percentile(prom, 75) + 1e-9), 0.0, 3.0)
    sc = s[cand]

    N = len(cand)
    best = np.full(N, -np.inf)
    prev = np.full(N, -1, dtype=int)
    for j in range(N):
        gain = amp[j] - ring_cost
        best[j] = gain                                    # start a chain at j
        lam_j = lam[cand[j]]
        for i in range(j):
            d = sc[j] - sc[i]
            if d < lo * lam_j or d > hi * lam_j:
                continue
            pen = spacing_weight * (np.log(d / lam_j)) ** 2
            v = best[i] + gain - pen
            if v > best[j]:
                best[j] = v
                prev[j] = i
    j = int(np.argmax(best))
    chain = []
    while j >= 0:
        chain.append(cand[j])
        j = prev[j]
    return np.array(chain[::-1], dtype=int)


# ---------------------------------------------------------------------------
# 4. consensus across parallel lines
# ---------------------------------------------------------------------------
def consensus(xs_per_line: list[np.ndarray], min_votes: int, tol: float = 25.0):
    """Cluster ring x-positions found on several parallel lines; keep clusters
    seen on at least `min_votes` of them. A ring is a 2-D band, so it should
    recur across nearby lines; noise should not."""
    pts = [(x, li) for li, xs in enumerate(xs_per_line) for x in xs]
    if not pts:
        return np.array([]), np.array([])
    pts.sort()
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] <= tol:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    keep_x, votes = [], []
    for c in clusters:
        v = len({li for _, li in c})
        if v >= min_votes:
            keep_x.append(float(np.mean([x for x, _ in c])))
            votes.append(v)
    return np.array(keep_x), np.array(votes)


# ---------------------------------------------------------------------------
# 5. spacing, with the oblique-crossing correction
# ---------------------------------------------------------------------------
def increments(idx: np.ndarray, prof: dict, px_per_mm: float | None = None):
    """True (perpendicular) increments between consecutive rings."""
    if len(idx) < 2:
        return dict(n=len(idx), along=np.array([]), perp=np.array([]), mm=None)
    along = np.diff(prof["s"][idx])
    perp = np.diff(prof["s_perp"][idx])
    out = dict(n=len(idx), along=along, perp=perp, mm=None)
    if px_per_mm:
        out["mm"] = perp / px_per_mm
    return out
