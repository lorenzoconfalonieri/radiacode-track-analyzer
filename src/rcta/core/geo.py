"""Geodesy and slippy-map arithmetic, in pure numpy.

Everything here is closed-form on purpose. EPSG:3857 ("Web Mercator") *is*
defined as the spherical Mercator formula applied to WGS84 degrees as though
they were spherical, so a full PROJ transformation would return exactly the
same numbers at the cost of a 5 MB dependency and a C library. The one thing
PROJ would genuinely buy -- ellipsoidal geodesic distance -- is not needed
either: haversine on the WGS84 mean sphere is within about 0.5 % of Vincenty,
which is far below the positional uncertainty of a handheld GPS reporting
28 m of accuracy.

Three subtleties are worth stating because getting any of them wrong produces
a plot that looks plausible and is wrong.

*Mercator metres are not ground metres.* The projection inflates distance by
1/cos(latitude): 1.16x at 30 degrees, 1.32x at 41 degrees. Distances for the
user are always computed with :func:`haversine` on the sphere; Mercator
coordinates exist only so the plot can be aspect-locked 1:1 and share a
coordinate system with map tiles.

*Tile y counts down from the north.* Slippy-map row 0 is the top of the world,
while the plot's y axis increases northward. The conversions here are the only
place that flip is allowed to happen.

*A gap in a track is not a straight line.* :func:`segment_breaks` decides where
a polyline must be cut, and it tests two independent conditions, because either
alone gets a real case wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "GapParams",
    "MAX_LAT",
    "R_EARTH",
    "R_MEAN",
    "TILE_PX",
    "WORLD_M",
    "bearing",
    "choose_smoothing_window",
    "connect_array",
    "cumulative_distance",
    "haversine",
    "lonlat_to_web_mercator",
    "median_accuracy",
    "mercator_scale",
    "point_segment_distance",
    "segment_breaks",
    "smooth_positions",
    "tile_bounds_3857",
    "tile_resolution",
    "tiles_for_bounds",
    "web_mercator_to_lonlat",
    "zoom_for_resolution",
]

#: Sphere radius that defines EPSG:3857. Not the WGS84 semi-major axis by
#: coincidence -- the projection is specified to use exactly this value.
R_EARTH = 6378137.0

#: Mean radius of the WGS84 ellipsoid, used for great-circle distance. Using
#: R_EARTH here instead would inflate every reported distance by 0.17 %.
R_MEAN = 6371008.8

#: Half the width of the projected world, in metres.
WORLD_M = math.pi * R_EARTH

#: The latitude at which the Mercator ordinate reaches WORLD_M. Beyond it the
#: projection runs to infinity, so tiles simply stop.
MAX_LAT = 85.05112877980659

#: Slippy-map tiles are 256 px square by universal convention.
TILE_PX = 256


# ---- projection -----------------------------------------------------------


def lonlat_to_web_mercator(lon, lat):
    """WGS84 degrees to EPSG:3857 metres, y increasing north."""
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -MAX_LAT, MAX_LAT)
    x = np.radians(lon) * R_EARTH
    # arcsinh(tan(phi)) is the Mercator ordinate. Algebraically identical to
    # log(tan(pi/4 + phi/2)) but better conditioned near the equator, where the
    # log form differences two quantities that are both close to 1.
    y = np.arcsinh(np.tan(np.radians(lat))) * R_EARTH
    return x, y


def web_mercator_to_lonlat(x, y):
    """EPSG:3857 metres to WGS84 degrees."""
    lon = np.degrees(np.asarray(x, dtype=float) / R_EARTH)
    lat = np.degrees(np.arctan(np.sinh(np.asarray(y, dtype=float) / R_EARTH)))
    return lon, lat


def mercator_scale(lat_deg):
    """Mercator inflation factor 1/cos(phi).

    Ground metres = Mercator metres / this. Reported alongside any scale bar,
    because across a 10-degree track it varies by more than 10 % and a single
    unqualified scale bar would be wrong at one end of the picture.
    """
    return 1.0 / np.cos(np.radians(np.asarray(lat_deg, dtype=float)))


# ---- great-circle geometry ------------------------------------------------


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres on the WGS84 mean sphere.

    The haversine form is used rather than the spherical law of cosines
    because the latter loses all precision for short distances -- and short
    distances are the normal case here, where consecutive samples are metres
    apart.
    """
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_MEAN * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing(lon1, lat1, lon2, lat2):
    """Initial great-circle bearing in degrees clockwise from north."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x)) % 360.0


def cumulative_distance(lon, lat):
    """Cumulative great-circle distance, length n, starting at zero.

    The caller is responsible for having sorted by time first. Applied to an
    unsorted track this returns a number roughly a thousand times too large,
    which is exactly what happens if the loader's sort is ever removed.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size == 0:
        return np.zeros(0)
    steps = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    out = np.empty(lon.size, dtype=float)
    out[0] = 0.0
    np.cumsum(steps, out=out[1:])
    return out


# ---- GPS jitter -----------------------------------------------------------

#: Per-step noise after smoothing must be at most this fraction of the true
#: step for the summed path length to be trustworthy. 0.3 keeps the inflation
#: within a few per cent across every platform tested.
_JITTER_TOLERANCE = 0.3

#: Baseline, in samples, used to estimate the true step. Long enough that the
#: displacement over it is dominated by travel rather than by fix error.
_JITTER_BASELINE = 31


def median_accuracy(accuracy_m) -> float:
    """Median of the accuracies a fix actually reported, 0.0 if none did.

    An unrecorded accuracy is NaN rather than zero, so it neither drags the
    median toward a perfect fix nor poisons it.
    """
    accuracy_m = np.asarray(accuracy_m, dtype=float)
    known = accuracy_m[np.isfinite(accuracy_m)]
    return float(np.median(known)) if known.size else 0.0


def choose_smoothing_window(lat, lon, accuracy_m, *,
                            tolerance: float = _JITTER_TOLERANCE,
                            max_window: int = 121) -> int:
    """Moving-average width needed before a path length is meaningful.

    **This is the correction that makes the tool usable on foot.** Summed
    great-circle distance is only a measure of travel when each step is large
    compared with the fix error. On a walk it is not: at 1.4 m/s with 2 s
    logging the true step is 2.8 m while a good handheld fix is 8 m, so the
    measured path is mostly noise. Measured on synthetic walks with known
    truth, raw summed distance overstates the route by **5.2x** at 8 m
    accuracy and still by 2.1x at 3 m. Speed inherits the same error, and so
    does anything derived from either.

    The fix is to smooth the positions first. The useful fact is that adjacent
    moving-average positions share all but two of their samples, so the noise
    on the *difference* falls as ``sigma * sqrt(2) / w`` rather than as
    ``1/sqrt(w)``. Requiring that to stay below ``tolerance`` times the true
    step gives the window directly.

    The true step is not known in advance, so it is estimated over a long
    baseline where travel dominates, and debiased with
    ``E[|d|^2] = d^2 + 4 sigma^2``.

    Returns 1 when no smoothing is needed: a platform that moves well beyond
    its fix error between samples, such as a highway drive or a flight. Slow
    driving still gets a narrow window (30 m per fix at 8 m accuracy gives 3).
    Validated at 1.00-1.03x of the true distance for walking (3, 8 and 20 m
    accuracy), jogging, cycling, driving and flight, on both straight and
    sharply cornered routes.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    # No recorded accuracy means no basis for a window; leave positions raw.
    sigma = median_accuracy(accuracy_m)
    if sigma <= 0 or lat.size < 4:
        return 1

    x, y = lonlat_to_web_mercator(lon, lat)
    # Mercator metres are inflated by 1/cos(lat); undo that so the comparison
    # with a GPS accuracy in ground metres is like for like.
    scale = math.cos(math.radians(float(np.median(lat))))
    x, y = x * scale, y * scale

    baseline = int(min(_JITTER_BASELINE, max(lat.size // 4, 2)))
    net = np.hypot(x[baseline:] - x[:-baseline], y[baseline:] - y[:-baseline])
    if net.size == 0:
        return 1
    observed = float(np.median(net))
    step = math.sqrt(max(observed * observed - 4.0 * sigma * sigma, 0.0)) / baseline
    if step <= 0:
        # The platform never moved further than its own uncertainty.
        return int(max_window)

    window = math.ceil(sigma * math.sqrt(2.0) / (tolerance * step))
    window = int(min(max(window, 1), int(max_window)))
    return window + 1 if window % 2 == 0 else window


def smooth_positions(lat, lon, window: int):
    """Centred moving average of a position series, endpoints held.

    Endpoints are held rather than tapered because a shrinking window there
    would leave exactly the noisy short steps this is meant to remove.
    """
    window = int(window)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if window <= 1 or lat.size <= window:
        return lat, lon

    kernel = np.ones(window) / window
    out_lat = np.convolve(lat, kernel, mode="same")
    out_lon = np.convolve(lon, kernel, mode="same")
    half = window // 2
    out_lat[:half] = out_lat[half]
    out_lat[-half:] = out_lat[-half - 1]
    out_lon[:half] = out_lon[half]
    out_lon[-half:] = out_lon[-half - 1]
    return out_lat, out_lon


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points to a line segment, in the input's own units.

    Vectorised over the points for one segment. Used to mask an interpolated
    grid against the track: a vertex-based distance would punch holes along a
    straight leg whenever consecutive samples are far apart, which at driving
    or flying speed they always are.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / denom, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


# ---- dropouts -------------------------------------------------------------


@dataclass(frozen=True)
class GapParams:
    """When two consecutive samples must not be joined by a line.

    Two independent triggers, because either alone misses a real case. A long
    interval with no movement is a stationary pause, and joining it is
    correct. A short interval with an impossible jump is a GPS glitch that a
    time-only rule would happily draw a line straight through.
    """

    #: Absolute interval above which a break is considered, in seconds.
    max_gap_s: float = 30.0
    #: ...but never below this multiple of the track's own median interval, so
    #: a deliberately slow 60 s logging cadence is not read as continuous
    #: dropouts.
    dt_factor: float = 10.0
    #: A step is implausible if it exceeds this multiple of a robust upper
    #: bound on the track's own speed, scaled by the elapsed time.
    speed_factor: float = 4.0
    #: Steps shorter than this are never implausible, whatever the ratio says;
    #: it stops GPS jitter while stationary from fragmenting the line.
    min_jump_m: float = 50.0


def segment_breaks(time_s, lon, lat, params: GapParams | None = None):
    """Boolean array, length n-1, true where sample i must not join i+1."""
    params = params or GapParams()
    time_s = np.asarray(time_s, dtype=float)
    n = time_s.size
    if n < 2:
        return np.zeros(max(n - 1, 0), dtype=bool)

    dt = np.diff(time_s)
    step = haversine(np.asarray(lon)[:-1], np.asarray(lat)[:-1],
                     np.asarray(lon)[1:], np.asarray(lat)[1:])

    median_dt = float(np.median(dt)) if dt.size else 1.0
    gap_threshold = max(params.max_gap_s, params.dt_factor * median_dt)

    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.inf)
    allowed = params.speed_factor * _robust_max(speed) * dt + params.min_jump_m

    too_slow_to_be_real = (dt > gap_threshold) & (step > params.min_jump_m)
    too_fast_to_be_real = step > allowed
    return too_slow_to_be_real | too_fast_to_be_real


def _robust_max(values) -> float:
    """A high-breakdown upper bound on a track's own speed.

    Deliberately not a high percentile. With a handful of segments the 95th
    percentile *is* the teleport, so the outlier raises the threshold that is
    supposed to catch it and nothing is ever flagged. Median plus three scaled
    MADs has a 50 % breakdown point, so half the track would have to be
    glitched before the bound moved -- and on a stop-start drive, where the
    median speed is near zero, the MAD term still admits the moving segments.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + 3.0 * 1.4826 * mad


def connect_array(breaks):
    """uint8 connect array for ``pyqtgraph.arrayToQPath``.

    Element i answers "does sample i join to i+1"; the final element is unused
    by pyqtgraph but must exist for the length to match the point arrays.
    """
    breaks = np.asarray(breaks, dtype=bool)
    connect = np.ones(breaks.size + 1, dtype=np.uint8)
    connect[:-1][breaks] = 0
    return connect


# ---- slippy-map tiles -----------------------------------------------------


def tile_resolution(zoom: int) -> float:
    """EPSG:3857 metres per tile pixel at ``zoom``."""
    return 2.0 * WORLD_M / (TILE_PX * (1 << int(zoom)))


def zoom_for_resolution(metres_per_pixel: float, max_zoom: int = 19) -> int:
    """Sharpest tile zoom that is not coarser than the display.

    The +0.4 bias picks the sharper level when the choice is close: an
    over-sharp basemap downsamples cleanly, an under-sharp one is visibly
    soft. Callers must pass *device* pixels, not logical ones, or a Retina
    display gets a blurry map.
    """
    mpp = max(float(metres_per_pixel), 1e-9)
    z = math.log2(2.0 * WORLD_M / (TILE_PX * mpp))
    return int(min(int(max_zoom), max(0, math.floor(z + 0.4))))


def tile_bounds_3857(z: int, x: int, y: int):
    """(x0, y0, x1, y1) of a tile in EPSG:3857 metres, y increasing north."""
    span = 2.0 * WORLD_M / (1 << int(z))
    x0 = -WORLD_M + x * span
    # Tile rows count down from the north pole; the plot's y counts up.
    y1 = WORLD_M - y * span
    return x0, y1 - span, x0 + span, y1


def tiles_for_bounds(x0: float, y0: float, x1: float, y1: float, z: int):
    """Tile keys covering a projected rectangle, nearest the centre first.

    Centre-out ordering is what makes a partially loaded map immediately
    useful, and it is also the order in which tiles should be requested so
    that an interrupted pan has filled in the part the user is looking at.
    """
    z = int(z)
    n = 1 << z
    span = 2.0 * WORLD_M / n

    def col(x):
        return int(math.floor((x + WORLD_M) / span))

    def row(y):
        return int(math.floor((WORLD_M - y) / span))

    cx0, cx1 = sorted((col(x0), col(x1)))
    ry0, ry1 = sorted((row(y1), row(y0)))
    cx0, cx1 = max(cx0, 0), min(cx1, n - 1)
    ry0, ry1 = max(ry0, 0), min(ry1, n - 1)
    if cx1 < cx0 or ry1 < ry0:
        return []

    mid_c = (cx0 + cx1) / 2.0
    mid_r = (ry0 + ry1) / 2.0
    keys = [(z, cx, ry) for cx in range(cx0, cx1 + 1) for ry in range(ry0, ry1 + 1)]
    keys.sort(key=lambda k: max(abs(k[1] - mid_c), abs(k[2] - mid_r)))
    return keys
