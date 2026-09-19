"""Draw a line through the *middle* of the shell.

Given the shell mask and two endpoints, we want a smooth curve that runs from
one endpoint to the other while staying centered in the shell band (equidistant
from the top and bottom edges) and following the shell's natural curve.

Method: the distance transform of the mask peaks along the medial axis. We find
the minimum-cost path between the two endpoints on a cost surface that is cheap
at the center and expensive near the edges, so the least-cost route naturally
hugs the medial axis. The raw pixel path is then smoothed to a clean spline.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array

import config


def _snap_inside(inside: np.ndarray, y: int, x: int) -> tuple[int, int]:
    """Return (y, x) if inside the mask, else the nearest inside pixel."""
    h, w = inside.shape
    y = int(np.clip(y, 0, h - 1))
    x = int(np.clip(x, 0, w - 1))
    if inside[y, x]:
        return y, x
    ys, xs = np.where(inside)
    d = (ys - y) ** 2 + (xs - x) ** 2
    i = int(np.argmin(d))
    return int(ys[i]), int(xs[i])


def _cost_surface(mask: np.ndarray) -> np.ndarray:
    """Cheap along the medial axis, dearer near edges, impassable outside.

    Additive form ``1 + W*(1 - centeredness)``: every in-mask step costs at least
    1 so path *length* stays dominant (the route goes roughly straight between
    the endpoints), while the ``W*(1 - centeredness)`` term nudges it onto the
    medial axis. A multiplicative ``1/centeredness`` cost was tried first but
    made travel through fat regions almost free, so the path detoured through
    the widest part of the shell instead of running straight.
    """
    inside = mask > 0
    dt = distance_transform_edt(inside)
    dmax = float(dt.max()) or 1.0
    centeredness = dt / dmax  # 1 on the medial axis, ->0 at the edge
    cost = np.full(mask.shape, np.inf, dtype=np.float64)
    cost[inside] = 1.0 + config.CENTER_WEIGHT * (1.0 - centeredness[inside])
    return cost


def centered_path(mask: np.ndarray, start_xy, end_xy) -> np.ndarray:
    """Raw (x, y) pixel path from start to end through the shell's middle."""
    inside = mask > 0
    cost = _cost_surface(mask)
    sy, sx = _snap_inside(inside, round(start_xy[1]), round(start_xy[0]))
    ey, ex = _snap_inside(inside, round(end_xy[1]), round(end_xy[0]))
    indices, _ = route_through_array(
        cost, (sy, sx), (ey, ex), fully_connected=True, geometric=True
    )
    rc = np.asarray(indices, dtype=float)  # (N, 2) as (row, col) = (y, x)
    return rc[:, ::-1]                      # -> (x, y)


def smooth_path(xy: np.ndarray, n: int = None, tol: float = None) -> np.ndarray:
    """Clean a raw geodesic into a smooth centerline.

    The raw path is jagged (pixel stair-steps) and can hairpin sharply at the
    curled root. We resample it to uniform arc length, then fit an
    *approximating* cubic spline (smoothing factor s > 0) so sharp corners are
    rounded rather than overshot into loops (which an s=0 interpolating spline
    would produce).
    """
    n = config.N_OUTPUT_POINTS if n is None else n
    tol = config.SMOOTH_TOL_PX if tol is None else tol
    keep = np.r_[True, np.any(np.abs(np.diff(xy, axis=0)) > 1e-6, axis=1)]
    xy = xy[keep]
    if len(xy) < 4:
        return xy
    # uniform arc-length resampling
    seg = np.hypot(*np.diff(xy, axis=0).T)
    d = np.r_[0.0, np.cumsum(seg)]
    length = float(d[-1])
    m = max(8, int(length))  # ~1 sample per pixel
    du = np.linspace(0.0, length, m)
    xr = np.interp(du, d, xy[:, 0])
    yr = np.interp(du, d, xy[:, 1])
    # approximating spline; s scales with point count and the px tolerance
    s = m * (tol ** 2)
    tck, _ = splprep([xr, yr], s=s, k=3)
    xs, ys = splev(np.linspace(0, 1, n), tck)
    return np.column_stack([xs, ys])


def trace(mask: np.ndarray, start_xy, end_xy) -> np.ndarray:
    """Full centerline: geodesic through the middle, then smoothed."""
    return smooth_path(centered_path(mask, start_xy, end_xy))


def auto_endpoints(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Estimate the shell's two tips as the endpoints of its longest internal
    (geodesic) path. Used for headless testing and, later, to reduce clicks.

    Returns two (x, y) points.
    """
    inside = mask > 0
    ys, xs = np.where(inside)
    if len(ys) == 0:
        raise ValueError("Empty mask; cannot find endpoints.")
    # Farthest inside pixel from the centroid, then the farthest inside pixel
    # from that one, measured as geodesic (in-mask) distance so the two tips of
    # a curved/arched shell are found rather than two points across a gap.
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    seed = _snap_inside(inside, cy, cx)
    a = _farthest_geodesic(inside, seed)
    b = _farthest_geodesic(inside, a)
    return (a[1], a[0]), (b[1], b[0])  # (x, y)


def _farthest_geodesic(inside: np.ndarray, src_yx: tuple[int, int]) -> tuple[int, int]:
    """The in-mask pixel geodesically farthest from src, via a uniform-cost
    route search over the mask interior."""
    cost = np.where(inside, 1.0, np.inf)
    from skimage.graph import MCP_Geometric

    mcp = MCP_Geometric(cost, fully_connected=True)
    dist, _ = mcp.find_costs([src_yx])
    dist = np.where(np.isfinite(dist), dist, -1.0)
    idx = int(np.argmax(dist))
    y, x = np.unravel_index(idx, dist.shape)
    return int(y), int(x)
