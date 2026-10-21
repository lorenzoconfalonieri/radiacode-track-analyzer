"""The authoritative representation of "which samples are selected".

A boolean mask over the track's time-sorted sample index. Never an index
array, and never a time interval -- and the reason is not taste, it is that
the other two representations cannot express what the map can produce.

Draw a rubber band over a road that the survey drove twice in opposite
directions, and the selection is two disjoint runs separated by an hour. A
time interval cannot represent that at all; it would silently widen the
selection to include everything in between, which on a survey means including
several kilometres the user did not select. An index array can represent it,
but then intersecting a spatial selection with a temporal one requires set
logic on sorted integer arrays, whereas with masks it is a single ``&``.

The corollary matters for the GUI: a draggable region item on the time series
is a *control*, not the model. It can only express a contiguous interval, so
when the selection is disjoint the series panel must show the mask some other
way rather than quietly rounding it to its own hull. :attr:`Selection.runs`
exists so the panel can do that.

``source`` and ``interval`` are provenance only. They let a panel restore its
own widget state exactly, and they must never be consulted in place of the
mask -- if they disagree with it, the mask is right.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Selection",
    "select_by_distance",
    "select_by_polygon",
    "select_by_time",
    "select_by_value",
]


@dataclass(frozen=True)
class Selection:
    """A set of samples, plus a note on how the user expressed it."""

    mask: np.ndarray
    #: "map" | "series" | "table" | "value" | "none"
    source: str = "none"
    #: Unix seconds, set only when the user actually dragged a time region.
    interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError(f"mask must be 1-D, got shape {mask.shape}")
        mask = np.ascontiguousarray(mask)
        mask.flags.writeable = False
        object.__setattr__(self, "mask", mask)

    # ---- construction -----------------------------------------------------

    @classmethod
    def empty(cls, n: int) -> Selection:
        return cls(mask=np.zeros(int(n), dtype=bool), source="none")

    @classmethod
    def all(cls, n: int) -> Selection:
        return cls(mask=np.ones(int(n), dtype=bool), source="none")

    @classmethod
    def from_interval(cls, time_s, t0: float, t1: float) -> Selection:
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        time_s = np.asarray(time_s, dtype=float)
        return cls(mask=(time_s >= lo) & (time_s <= hi), source="series",
                   interval=(float(lo), float(hi)))

    @classmethod
    def from_rect(cls, x, y, x0, y0, x1, y1) -> Selection:
        """Samples inside an axis-aligned rectangle in projected metres."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xlo, xhi = sorted((float(x0), float(x1)))
        ylo, yhi = sorted((float(y0), float(y1)))
        inside = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
        return cls(mask=inside & np.isfinite(x) & np.isfinite(y), source="map")

    # ---- properties -------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.mask.size)

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    @property
    def is_empty(self) -> bool:
        return not bool(self.mask.any())

    @property
    def is_interval(self) -> bool:
        """True when the selection is exactly one contiguous run.

        The series panel uses this to decide whether its draggable region can
        represent the selection faithfully or whether it must fall back to an
        overlay.
        """
        return len(self.runs(limit=2)) == 1

    def runs(self, limit: int = 256) -> list[tuple[int, int]]:
        """Contiguous ``[start, stop)`` spans, at most ``limit`` of them."""
        if self.is_empty:
            return []
        padded = np.concatenate(([False], self.mask, [False]))
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        spans = list(zip(edges[0::2].tolist(), edges[1::2].tolist()))
        return spans[:limit]

    def time_span(self, time_s) -> tuple[float, float] | None:
        """(first, last) selected timestamp, or None when nothing is selected.

        This is the *hull* of the selection, and it is not the same thing as
        the selection whenever :attr:`is_interval` is false.
        """
        if self.is_empty:
            return None
        selected = np.asarray(time_s, dtype=float)[self.mask]
        return float(selected.min()), float(selected.max())

    # ---- set operations ---------------------------------------------------

    def __and__(self, other: Selection) -> Selection:
        return replace(self, mask=self.mask & other.mask, interval=None)

    def __or__(self, other: Selection) -> Selection:
        return replace(self, mask=self.mask | other.mask, interval=None)

    def __invert__(self) -> Selection:
        return replace(self, mask=~self.mask, interval=None)

    def without(self, other: Selection) -> Selection:
        return replace(self, mask=self.mask & ~other.mask, interval=None)

    def describe(self) -> str:
        if self.is_empty:
            return "no selection"
        spans = self.runs()
        if len(spans) == 1:
            return f"{self.count} samples in 1 interval"
        return f"{self.count} samples in {len(spans)} intervals"


# ---- convenience selectors ------------------------------------------------


def select_by_time(track, t0: float, t1: float) -> Selection:
    return Selection.from_interval(track.time_s, t0, t1)


def select_by_distance(track, d0: float, d1: float) -> Selection:
    lo, hi = (d0, d1) if d0 <= d1 else (d1, d0)
    d = track.distance_along_m
    return Selection(mask=(d >= lo) & (d <= hi), source="map")


def select_by_polygon(track, polygon) -> Selection:
    """Samples inside a polygon given in EPSG:3857 metres.

    Even-odd ray crossing, vectorised over the samples. Position-masked
    samples are excluded by construction: a fix with 397 m of uncertainty has
    no business being lassoed, because whether it falls inside a 50 m polygon
    is not something the data can answer.
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("polygon must be an (n, 2) array with at least 3 vertices")

    x, y = track.merc_x, track.merc_y
    inside = np.zeros(x.size, dtype=bool)
    px, py = poly[:, 0], poly[:, 1]
    j = poly.shape[0] - 1
    for i in range(poly.shape[0]):
        # Half-open comparison on y, so a vertex is counted once rather than
        # twice; the classic even-odd bug is testing both endpoints inclusively.
        straddles = (py[i] > y) != (py[j] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
        inside ^= straddles & (x < crossing_x)
        j = i
    return Selection(mask=inside & track.position_mask, source="map")


def select_by_value(track, *, dose_range=None, count_range=None,
                    accuracy_max=None) -> Selection:
    """Samples whose readings fall in the given ranges."""
    mask = np.ones(track.n, dtype=bool)
    if dose_range is not None:
        lo, hi = dose_range
        mask &= (track.dose_rate >= lo) & (track.dose_rate <= hi)
    if count_range is not None:
        lo, hi = count_range
        mask &= (track.count_rate >= lo) & (track.count_rate <= hi)
    if accuracy_max is not None:
        mask &= track.accuracy_m <= accuracy_max
    return Selection(mask=mask, source="value")
