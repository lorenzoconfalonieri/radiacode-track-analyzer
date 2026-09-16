"""Interpolated dose surfaces and hotspot clustering.

**An interpolated dose surface is the most dangerous plot this program can
produce.** It looks like a measurement of an area when it is a measurement of
a line. A survey drives one road through a valley and the renderer will
cheerfully paint the whole valley, in confident colour, from data that exists
only along a single one-dimensional path. Anyone reading that image will take
the colour thirty metres off the road as seriously as the colour on it. So the
grid here spends more effort deciding where it is *not allowed to have an
opinion* than on the interpolation itself.

The interpolator is a Gaussian-kernel weighted mean (Nadaraya-Watson).

*Not inverse-distance weighting.* IDW has no length scale, so it assigns a
value forty kilometres off-track with the same apparent authority as one on
it, and it is singular at the sample locations, producing a bullseye at every
point. The Gaussian kernel has an explicit physical bandwidth the user can
reason about -- it defaults to three times the median GPS accuracy -- and,
decisively, the same kernel yields the support mask for free: the sum of the
kernel weights in a cell *is* a measure of how much data backs it.

*Not kriging.* A track is a one-dimensional path through two-dimensional
space. The experimental variogram is therefore fully determined along the path
and completely unconstrained across it. A fitted transverse range would be an
artefact of the fitting procedure wearing the clothes of a geostatistical
result, which is worse than an obvious approximation.

Two independent masks are applied and a cell must pass both: a floor on the
kernel weight, and a hard geometric distance to the nearest track *segment*.
Segment rather than vertex matters more than it sounds -- at 233 m/s with 2 s
logging, consecutive samples are 470 m apart, so a vertex-based mask would
punch a row of holes along a perfectly well-surveyed straight leg.

Hotspots are found by clustering *significant samples*, never grid maxima. A
grid maximum is a property of the bandwidth: halve the bandwidth and it moves.
A cluster of samples is a property of the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rcta.core import geo
from rcta.core.statistics import CurrieLimits
from rcta.core.track import Track

__all__ = ["DoseGrid", "GridParams", "Hotspot", "find_hotspots", "interpolate"]


@dataclass(frozen=True)
class GridParams:
    """Configuration for the kernel-smoothed surface."""

    #: Kernel bandwidth in ground metres. None -> max(3 x median accuracy, 50).
    bandwidth_m: float | None = None
    #: Cells along the long axis. Clamped so a cell is at most bandwidth/2.
    grid_size: int = 384
    #: Blank cells whose total kernel weight is below this. One sample at zero
    #: distance contributes 1.0, so 0.05 means "less than a twentieth of one
    #: sample's worth of evidence".
    min_weight: float = 0.05
    #: Hard geometric mask: cells further than this from the nearest track
    #: segment. None -> 2 x bandwidth.
    max_distance_m: float | None = None
    #: "dose" for the absolute field, "excess" for net over background.
    value: str = "dose"
    #: Padding around the track bounding box, in bandwidths.
    padding_bandwidths: float = 2.0

    def __post_init__(self) -> None:
        if self.value not in ("dose", "excess"):
            raise ValueError(f'value must be "dose" or "excess", got {self.value!r}')
        if self.grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if self.bandwidth_m is not None and self.bandwidth_m <= 0:
            raise ValueError("bandwidth_m must be positive")


@dataclass(frozen=True)
class DoseGrid:
    """A two-dimensional field over the track, carrying its own support."""

    values: np.ndarray               #: (ny, nx), NaN where unsupported
    weight: np.ndarray               #: summed kernel weight per cell
    n_effective: np.ndarray          #: W^2 / sum(K^2): samples backing a cell
    distance_to_track_m: np.ndarray
    x_edges: np.ndarray              #: EPSG:3857 metres
    y_edges: np.ndarray
    bandwidth_m: float
    unit: str
    params: GridParams
    warnings: tuple[str, ...] = ()

    @property
    def extent_3857(self) -> tuple[float, float, float, float]:
        return (float(self.x_edges[0]), float(self.y_edges[0]),
                float(self.x_edges[-1]), float(self.y_edges[-1]))

    @property
    def supported(self) -> np.ndarray:
        return np.isfinite(self.values)

    @property
    def coverage(self) -> float:
        return float(self.supported.mean())

    def levels(self, n: int = 6) -> np.ndarray:
        """Contour levels at quantiles of the supported cells.

        Quantiles rather than a linear ramp: on a bimodal field every evenly
        spaced level lands in the empty gap between the two populations, and
        the contour plot comes out blank in the middle and crowded at the ends.
        """
        finite = self.values[self.supported]
        if finite.size == 0:
            return np.zeros(0)
        return np.unique(np.quantile(finite, np.linspace(0.05, 0.95, int(n))))

    def describe(self) -> str:
        finite = self.values[self.supported]
        lines = [
            f"grid       : {self.values.shape[1]} x {self.values.shape[0]} cells, "
            f"bandwidth {self.bandwidth_m:.0f} m",
            f"supported  : {self.coverage * 100.0:.1f} % of cells "
            f"({int(self.supported.sum())} of {self.values.size})",
        ]
        if finite.size:
            lines.append(f"values     : {finite.min():.2f} .. {finite.max():.2f} "
                         f"{self.unit}, median {np.median(finite):.2f}")
        for warning in self.warnings:
            lines.append(f"warning    : {warning}")
        return "\n".join(lines)


def interpolate(track: Track, params: GridParams | None = None, *,
                background=None, progress=None, cancelled=None) -> DoseGrid:
    """Gaussian-kernel weighted mean of the track's readings on a grid.

    ``progress`` and ``cancelled`` are plain callables so the GUI can run this
    on a worker thread without this module importing Qt.
    """
    params = params or GridParams()
    warnings: list[str] = []

    usable = track.position_mask & track.valid_mask
    if usable.sum() < 3:
        raise ValueError("need at least three positioned samples to interpolate")

    x = track.merc_x[usable]
    y = track.merc_y[usable]
    lat = track.lat[usable]

    if params.value == "excess":
        if background is None:
            raise ValueError('value="excess" needs a background estimate')
        values = ((track.count_rate - background.count_rate)[usable]
                  * track.dose_per_count)
        unit = "nSv/h net"
    else:
        values = track.dose_rate[usable]
        unit = "nSv/h"

    # Mercator inflates distance by 1/cos(lat). Working in projected metres
    # keeps cells square on screen and the kernel Euclidean; the bandwidth is
    # converted once, at the grid centre.
    lat_centre = float(np.median(lat))
    inflation = float(geo.mercator_scale(lat_centre))
    lat_span = float(lat.max() - lat.min())
    if lat_span > 5.0:
        warnings.append(
            f"track spans {lat_span:.1f} degrees of latitude; Mercator scale "
            f"varies by {geo.mercator_scale(lat.max()) / geo.mercator_scale(lat.min()):.0%} "
            "across it, so the kernel bandwidth is only exact near the centre")

    bandwidth = params.bandwidth_m
    if bandwidth is None:
        bandwidth = max(3.0 * float(np.median(track.accuracy_m[usable])), 50.0)
    bandwidth_proj = bandwidth * inflation
    max_distance = params.max_distance_m or 2.0 * bandwidth
    max_distance_proj = max_distance * inflation

    pad = params.padding_bandwidths * bandwidth_proj
    x0, x1 = float(x.min()) - pad, float(x.max()) + pad
    y0, y1 = float(y.min()) - pad, float(y.max()) + pad
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)

    # Two competing constraints on the cell size, and a budget that decides
    # between them. Resolving the kernel wants a cell of at most half a
    # bandwidth; covering the extent at that resolution can be absurd. On the
    # bundled 1594 km transect a 69 m bandwidth implies 1.27 *billion* cells,
    # which is not a slow render but a hung program.
    #
    # So the total cell count is capped first, and the bandwidth is reported as
    # under-resolved rather than silently honoured. A continental transect
    # genuinely cannot carry a metre-scale surface; the user needs a coarser
    # bandwidth or a smaller area, and saying so beats freezing.
    budget = float(params.grid_size) ** 2
    cell_from_budget = math.sqrt(max(width * height, 1.0) / budget)
    cell_from_kernel = bandwidth_proj / 2.0
    cell = max(cell_from_kernel, cell_from_budget)
    if cell > cell_from_kernel * 1.001:
        # A kernel narrower than a cell cannot be resolved by that grid, and
        # rendering it anyway produces a blank map: every cell centre falls
        # outside every sample's kernel. Widen the kernel to what the grid can
        # actually carry and say so, rather than returning an empty surface and
        # leaving the user to guess which knob to turn.
        widened = 2.0 * cell
        warnings.append(
            f"bandwidth raised from {bandwidth:.0f} m to "
            f"{widened / inflation:.0f} m: at this extent a grid cell is "
            f"{cell / inflation:.0f} m and a finer kernel cannot be resolved. "
            "Analyse a smaller area for a sharper surface.")
        bandwidth_proj = widened
        bandwidth = widened / inflation
        if params.max_distance_m is None:
            max_distance_proj = 2.0 * bandwidth_proj

    nx = int(np.clip(np.ceil(width / cell), 8, 4096))
    ny = int(np.clip(np.ceil(height / cell), 8, 4096))

    x_edges = np.linspace(x0, x0 + nx * cell, nx + 1)
    y_edges = np.linspace(y0, y0 + ny * cell, ny + 1)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])

    weight = np.zeros((ny, nx))
    weighted = np.zeros((ny, nx))
    weight_sq = np.zeros((ny, nx))

    # Only cells within a few bandwidths of a sample can matter; restricting
    # the update to that box turns an O(cells x samples) product into
    # something linear in the samples.
    reach = 3.0 * bandwidth_proj
    for i in range(x.size):
        if cancelled is not None and (i % 256 == 0) and cancelled():
            raise InterruptedError("interpolation cancelled")
        if progress is not None and i % 512 == 0:
            progress(i / x.size)

        cx0 = np.searchsorted(xc, x[i] - reach, side="left")
        cx1 = np.searchsorted(xc, x[i] + reach, side="right")
        cy0 = np.searchsorted(yc, y[i] - reach, side="left")
        cy1 = np.searchsorted(yc, y[i] + reach, side="right")
        if cx1 <= cx0 or cy1 <= cy0:
            continue

        dx = (xc[cx0:cx1] - x[i]) / bandwidth_proj
        dy = (yc[cy0:cy1] - y[i]) / bandwidth_proj
        k = np.exp(-0.5 * (dy[:, None] ** 2 + dx[None, :] ** 2))
        weight[cy0:cy1, cx0:cx1] += k
        weighted[cy0:cy1, cx0:cx1] += k * values[i]
        weight_sq[cy0:cy1, cx0:cx1] += k * k

    if progress is not None:
        progress(1.0)

    distance = _distance_to_track(track, usable, xc, yc,
                                  max(bandwidth_proj, cell))

    with np.errstate(divide="ignore", invalid="ignore"):
        field = np.where(weight > 0, weighted / np.maximum(weight, 1e-300), np.nan)
        n_eff = np.where(weight_sq > 0,
                         weight ** 2 / np.maximum(weight_sq, 1e-300), 0.0)

    # Both masks, and a cell must pass both. This is the point of the module.
    unsupported = (weight < params.min_weight) | (distance > max_distance_proj)
    field = np.where(unsupported, np.nan, field)

    return DoseGrid(
        values=field, weight=weight, n_effective=n_eff,
        distance_to_track_m=distance / inflation,
        x_edges=x_edges, y_edges=y_edges, bandwidth_m=float(bandwidth),
        unit=unit, params=params, warnings=tuple(warnings),
    )


def _distance_to_track(track: Track, usable, xc, yc, resolution_proj):
    """Distance from every cell centre to the nearest track *segment*.

    Implemented by densifying each segment to half the grid resolution and
    running a KD-tree over the resulting points, which is exact to that
    tolerance and far simpler than analytic point-to-segment over a million
    cells. Segment distance rather than vertex distance is what stops a
    well-surveyed straight leg from being masked into a dotted line whenever
    consecutive fixes are far apart.
    """
    from scipy.spatial import cKDTree

    x = track.merc_x[usable]
    y = track.merc_y[usable]
    # Densifying finer than the grid resolution buys nothing and, on a
    # 1792 km track at a 43 m spacing, would build forty million points.
    spacing = max(resolution_proj / 2.0, 1.0)

    points = [np.column_stack([x, y])]
    dx = np.diff(x)
    dy = np.diff(y)
    length = np.hypot(dx, dy)
    # Do not densify across a dropout: those segments were never travelled, so
    # treating them as surveyed would extend the mask over unvisited ground.
    breaks = geo.segment_breaks(track.time_s[usable], track.lon[usable],
                               track.lat[usable])
    for i in np.flatnonzero((length > spacing) & ~breaks):
        steps = int(min(length[i] / spacing, 4096))
        t = np.linspace(0.0, 1.0, steps + 2)[1:-1]
        points.append(np.column_stack([x[i] + t * dx[i], y[i] + t * dy[i]]))

    tree = cKDTree(np.vstack(points))
    grid = np.column_stack([np.repeat(yc, xc.size), np.tile(xc, yc.size)])
    distance, _ = tree.query(grid[:, ::-1])
    return distance.reshape(yc.size, xc.size)


@dataclass(frozen=True)
class Hotspot:
    """A spatial cluster of significant samples, possibly from several passes."""

    lat: float
    lon: float
    radius_m: float
    n_samples: int
    n_passes: int
    first_time_s: float
    last_time_s: float
    peak_dose_rate: float
    mean_net_dose_rate: float
    net_counts: float
    background_counts: float
    significance: float
    limits: CurrieLimits
    score: float
    #: "strong" (seen on repeat visits), "possible" (one pass), "weak".
    evidence: str
    anomaly_indices: tuple[int, ...]
    flags: tuple[str, ...] = ()

    def describe(self) -> str:
        flags = f"  [{', '.join(self.flags)}]" if self.flags else ""
        return (f"{abs(self.lat):8.5f}{'N' if self.lat >= 0 else 'S'} "
                f"{abs(self.lon):9.5f}{'E' if self.lon >= 0 else 'W'}  "
                f"r={self.radius_m:5.0f} m  {self.n_samples:3d} samples / "
                f"{self.n_passes} pass(es)  "
                f"peak {self.peak_dose_rate:7.1f} nSv/h  "
                f"{self.significance:6.2f} sigma  {self.evidence}{flags}")


def find_hotspots(track: Track, anomalies, background, *,
                  cluster_radius_m: float | None = None,
                  revisit_gap_s: float = 300.0, min_samples: int = 1,
                  bandwidth_m: float | None = None) -> list[Hotspot]:
    """Cluster significant anomalies in space and rank them.

    Ranking is ``net_counts * sqrt(n_passes)``, descending.

    Total excess counts rather than peak dose rate, because a single 2 s
    sample reading four times background is one Poisson fluctuation while a
    sixty-second plateau at a third of that excess is a source. The
    ``sqrt(n_passes)`` factor is corroboration by repetition: a feature seen
    on three separate passes over the same ground is qualitatively better
    evidence than one seen once, and only
    repeat-visit clusters are labelled ``strong``.
    """
    from scipy.spatial import cKDTree

    significant = [a for a in anomalies if a.is_significant]
    if not significant:
        return []

    if cluster_radius_m is None:
        bandwidth = bandwidth_m or max(
            3.0 * float(np.median(track.accuracy_m)), 50.0)
        cluster_radius_m = 3.0 * bandwidth

    lat = np.array([a.lat for a in significant])
    lon = np.array([a.lon for a in significant])
    x, y = geo.lonlat_to_web_mercator(lon, lat)
    inflation = float(geo.mercator_scale(float(np.median(lat))))
    radius_proj = cluster_radius_m * inflation

    # Greedy single linkage. Adequate for the tens to hundreds of significant
    # windows a scan produces, and it needs no extra dependency.
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(radius_proj)
    parent = list(range(len(significant)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    clusters: dict[int, list[int]] = {}
    for i in range(len(significant)):
        clusters.setdefault(find(i), []).append(i)

    hotspots = []
    for members in clusters.values():
        group = [significant[i] for i in members]
        if sum(a.scale for a in group) < min_samples:
            continue
        hotspots.append(_hotspot(track, group, x[members], y[members],
                                 inflation, revisit_gap_s))

    hotspots.sort(key=lambda h: h.score, reverse=True)
    return hotspots


def _hotspot(track, group, xs, ys, inflation, revisit_gap_s) -> Hotspot:
    net = np.array([max(a.net_counts, 0.0) for a in group])
    weight = net.sum()
    if weight > 0:
        cx = float(np.dot(xs, net) / weight)
        cy = float(np.dot(ys, net) / weight)
    else:
        cx, cy = float(xs.mean()), float(ys.mean())
    lon, lat = geo.web_mercator_to_lonlat(cx, cy)

    radius = (float(np.hypot(xs - cx, ys - cy).max()) / inflation
              if xs.size > 1 else 0.0)

    times = np.sort(np.array([a.time_s for a in group]))
    passes = 1 + int((np.diff(times) > revisit_gap_s).sum()) if times.size else 0

    best = max(group, key=lambda a: a.significance)
    total_net = float(sum(a.net_counts for a in group))
    total_background = float(sum(a.background_counts for a in group))
    live = track.effective_integration_time_s
    total_samples = int(sum(a.scale for a in group))

    if passes > 1:
        evidence = "strong"
    elif any("trend" in a.flags for a in group):
        evidence = "weak"
    else:
        evidence = "possible"

    flags = sorted({f for a in group for f in a.flags})
    return Hotspot(
        lat=float(lat), lon=float(lon), radius_m=radius,
        n_samples=total_samples, n_passes=passes,
        first_time_s=float(times.min()), last_time_s=float(times.max()),
        peak_dose_rate=max(a.peak_dose_rate for a in group),
        mean_net_dose_rate=(total_net / (total_samples * live)
                            * track.dose_per_count if total_samples else 0.0),
        net_counts=total_net, background_counts=total_background,
        significance=best.significance, limits=best.limits,
        score=total_net * np.sqrt(max(passes, 1)),
        evidence=evidence,
        anomaly_indices=tuple(a.start_index for a in group),
        flags=tuple(flags),
    )
