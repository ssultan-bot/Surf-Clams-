# config.py
# ---------------------------------------------------------------------------
# Central settings for the surf-clam centerline project.
# Paths are built relative to this file so the project is portable.
# ---------------------------------------------------------------------------

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# The one image the MVP is validated on first.
DEFAULT_IMAGE = (
    DATA_DIR
    / "Surf Clam hinge images"
    / "NOAA Surf Clam Survey"
    / "1986"
    / "Surf_Clam_1986_Summer_255_2.jpg"
)

# ---------------------------------------------------------------------------
# Processing scale
# ---------------------------------------------------------------------------
# Segmentation and the geodesic run on a downscaled copy for speed; the final
# path is scaled back up to full resolution for display and output. Source
# images range from ~2100 to ~6700 px wide, so we cap the working long side.
WORK_MAX_SIDE = 1600

# ---------------------------------------------------------------------------
# Segmentation (locate the shell)
# ---------------------------------------------------------------------------
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
MORPH_KERNEL = 9        # ellipse kernel size for open/close (working-scale px)
FILL_RATIO_MAX = 0.9    # a component filling <90% of its bbox is "shell-like"
                        # (used only when no click seeds are given)

# ---------------------------------------------------------------------------
# Centerline (draw the middle line)
# ---------------------------------------------------------------------------
CENTER_WEIGHT = 6.0     # additive edge-avoidance: cost = 1 + W*(1 - centeredness).
                        # Length stays dominant (every step costs >=1) so the path
                        # goes roughly straight between the two points; W is how
                        # hard it is pulled toward the medial axis. Too large and it
                        # detours through fat regions; too small and it drifts to an
                        # edge. ~4-8 keeps it centered without detouring.
SMOOTH_TOL_PX = 2.5     # spline smoothing tolerance (working-scale px). Bigger =
                        # smoother/looser; controls rounding of sharp hairpins so
                        # the curled root end doesn't overshoot into a loop.
N_OUTPUT_POINTS = 400   # resample the final centerline to this many points

# ---------------------------------------------------------------------------
# Growth-ring detection (sample brightness along the centerline)
# ---------------------------------------------------------------------------
# Each dark band crossing the centerline is one growth ring. We read a
# perpendicular-averaged brightness profile along the line, smooth it, and take
# the dark valleys as rings. Values below are full-resolution pixels and were
# the tuned defaults for this same image in the earlier project.
RING_HALF_WIDTH = 8     # average gray over +/- this many px perpendicular to the
                        # line (a ring is a band, so averaging across it de-noises)
RING_GRAY_SIGMA = 3     # smooth the 1-D brightness profile before finding valleys
RING_MIN_DIST = 15      # minimum spacing (px along the line) between two rings
RING_PROMINENCE = 0.08  # how deep a dark band must be (fraction of profile range);
                        # smaller = more (fainter) rings, bigger = fewer

# ---------------------------------------------------------------------------
# Growth-line extraction (pull the ring bands out in their original form)
# ---------------------------------------------------------------------------
# Inside the shell, rings are dark bands over the brighter shell body. We flatten
# the slow brightness variation and keep the locally-dark pixels as ring bands.
RINGX_BLOCK = 51        # adaptive-threshold neighborhood (px); ~ a ring's spacing
RINGX_C = 7             # threshold offset; higher = only clearly-darker pixels kept
RINGX_ABS_DARK = 35     # drop pixels darker than this (marker ink / holes / bg bleed
                        # are near-black; real rings are mid-gray, not black)
RINGX_MIN_AREA = 40     # remove connected specks smaller than this (px)

# Ridge-filter variant (direction 1 improvement): more continuous lines via a
# Sato tubeness filter instead of per-pixel thresholding. Pure skimage, no
# download.
RIDGE_SIGMAS = (1, 2, 3)  # line half-widths to detect (px)
RIDGE_THRESH = 0.12       # keep ridge response above this fraction of its max
RIDGE_ERODE = 10          # shrink the shell mask by this many px (drop edge artifacts)
RIDGE_MIN_AREA = 120      # remove connected specks smaller than this (px)
RIDGE_CLOSE = 3           # morphological-close kernel to bridge tiny gaps (px)

# Hysteresis variant (direction A improvement): grow lines from strong ridge
# seeds down to a faint floor, so rings come out both complete AND connected.
RIDGE_SIGMAS_MULTI = (1, 2, 3, 4)  # multi-scale line widths (px)
RIDGE_HYST_LOW = 0.035    # keep faint ridge pixels down to this (fraction of max)...
RIDGE_HYST_HIGH = 0.14    # ...but only if connected to a seed above this

# Mild "declutter" preset: trims the finest capillary-like texture that the ridge
# filters pick up (surface cracks, rib grooves, micro-increments) without
# touching the real annual rings. Gentle by design.
CLEAN_SIGMAS = (2, 3, 4)     # drop sigma=1 -> removes the thinnest threads
CLEAN_LOW = 0.045            # slightly higher floor -> fewer grown tendrils
CLEAN_MIN_AREA = 180         # remove small specks
CLEAN_DECLUTTER_DIAG = 90    # drop components whose bounding-box diagonal < this (px)

# ---------------------------------------------------------------------------
# Ventral-margin ("bottom edge") tracing
# ---------------------------------------------------------------------------
# Morphological close applied to the shell mask before taking its outline.
# Large enough to bridge the notch debris bites out of the lower margin, small
# enough not to merge the shell with the scale-bar card: 41 was the sweet spot
# on the reference image (9 leaves the notch, 81 merges with the scale bar).
EDGE_CLOSE_K = 41
EDGE_OFFSET_FRAC = 0.35   # shift inward by this fraction of local shell thickness
