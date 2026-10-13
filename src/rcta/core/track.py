"""The track data model: six parallel columns and the geometry derived from them.

Struct-of-arrays, not a list of point objects. A week-long survey at 1 Hz is
600 000 samples; as frozen dataclasses that is roughly half a gigabyte and
every analysis would begin by rebuilding numpy arrays from it. Stored as
columns, haversine, the rolling quantile, the KD-tree query and the kernel
grid are each a single vectorised call over contiguous memory, and pyqtgraph
receives float64 buffers with no copy. The one thing a per-point object buys
is needed in exactly two places -- the map tooltip and the hotspot report --
so :meth:`Track.point` materialises one on demand.

Two invariants are enforced in ``__post_init__`` rather than trusted, because
both have already been observed to fail on real data.

*Sorted by time.* The bundled sample's ``markers`` array is in effectively
random order; its first entry is 5363 s after the recorded start. An unsorted
Track produces a 1.6-million-kilometre polyline, a nonsense speed at every
sample, and a distance that disagrees with the device's own odometer by a
factor of nine hundred. Sorting is cheap and idempotent, so the class simply
refuses to exist in the wrong order.

*Read-only columns.* ``writeable = False`` on every array means a cached
property can never silently go stale, and a Track can be handed to a worker
thread with no lock. That second property is load-bearing, not incidental.

Bad samples are masked, never dropped, and there are separate masks because
the failure modes are independent: a fix with 397 m of GPS accuracy still
carries a perfectly good count rate, so it is hidden from the map alone. A
glitched fix -- one the platform could not have reached -- is also bypassed
when measuring the route; a merely imprecise one is not, because averaging
its noise away is what the position smoothing is for, and bypassing the 475
such fixes in the bundled flight would cut 1.8 km of real curvature. The
sample count in any summary always matches the sample count in the file.

Dose rate is canonically nSv/h everywhere in this package. Every display
conversion lives in :mod:`rcta.core.dose`; nothing else divides by 1000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import cached_property

import numpy as np

from rcta.core import geo

__all__ = ["Period", "Track", "TrackMeta", "TrackPoint"]

#: Bounds within which an inferred per-sample integration time is believable.
#: The upper bound only has to stop a pathological log from being read as
#: minutes of live time per sample -- the *median* is already robust to
#: dropouts, so it does not need to be tight. It was 8 s, which silently
#: clipped a real RadiaCode walking log that samples every 10 s and cost 20 %
#: of the counts behind every detection limit.
_INTEGRATION_BOUNDS = (1.0, 60.0)


def _utc(seconds: float) -> datetime:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)


@dataclass(frozen=True)
class Period:
    """One recording segment, as the device itself reported it.

    ``distance_m`` is the device's own odometer, and it is the only external
    check available on our geometry. On the bundled file it reads
    1 795 151.69 m against 1 792 210 m of date-sorted haversine -- 0.16 %
    apart. Keeping it means :meth:`Track.describe` can report the
    disagreement instead of quietly trusting whichever number was computed
    last.
    """

    distance_m: float
    start: datetime
    end: datetime

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    def describe(self) -> str:
        return (f"{self.start:%Y-%m-%d %H:%M:%S}Z .. {self.end:%H:%M:%S}Z  "
                f"({self.duration_s:.0f} s, {self.distance_m / 1000.0:.2f} km)")


@dataclass(frozen=True)
class TrackMeta:
    """Provenance, and everything the file said about itself."""

    title: str = ""
    devices: tuple[str, ...] = ()
    start_time: datetime | None = None
    source_path: str = ""
    #: "rctrk-json" or "rctrk-tsv" -- the two dialects the RadiaCode app writes.
    source_format: str = ""
    periods: tuple[Period, ...] = ()
    #: The device's "spectrum valid" flag. False in every sample seen so far.
    spectrum_valid: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def declared_distance_m(self) -> float | None:
        """Odometer total, or None for a dialect that does not record one."""
        if not self.periods:
            return None
        return sum(p.distance_m for p in self.periods)

    def describe(self) -> str:
        lines = [f"title      : {self.title or '(untitled)'}"]
        if self.devices:
            lines.append(f"device     : {', '.join(self.devices)}")
        if self.source_format:
            lines.append(f"format     : {self.source_format}")
        return "\n".join(lines)


@dataclass(frozen=True)
class TrackPoint:
    """A single sample, materialised on demand for display."""

    index: int
    time: datetime
    lat: float
    lon: float
    accuracy_m: float
    dose_rate: float
    count_rate: float
    distance_along_m: float
    speed_m_s: float
    leg: int
    valid: bool
    position_valid: bool

    def describe(self, unit: str = "nSv/h") -> str:
        # Imported here: dose imports this module.
        from rcta.core.dose import format_dose_rate
        flag = "" if self.position_valid else "  (position uncertain)"
        reading = (f"{format_dose_rate(self.dose_rate, unit)}  "
                   f"{self.count_rate:.2f} cps"
                   if math.isfinite(self.dose_rate) else "no reading")
        accuracy = (f"+-{self.accuracy_m:.0f} m"
                    if math.isfinite(self.accuracy_m) else "accuracy unknown")
        return (f"#{self.index}  {self.time:%Y-%m-%d %H:%M:%S}Z  "
                f"{abs(self.lat):.6f}{'N' if self.lat >= 0 else 'S'} "
                f"{abs(self.lon):.6f}{'E' if self.lon >= 0 else 'W'}  "
                f"{reading}  {accuracy}{flag}")


@dataclass(frozen=True)
class Track:
    """A GPS radiation track: six parallel columns plus masks."""

    time_s: np.ndarray            #: int64 unix epoch seconds, UTC, non-decreasing
    lat: np.ndarray               #: float64 WGS84 degrees
    lon: np.ndarray               #: float64 WGS84 degrees
    accuracy_m: np.ndarray        #: float64 GPS horizontal accuracy, metres
    dose_rate: np.ndarray         #: float64 nSv/h (canonical unit)
    count_rate: np.ndarray        #: float64 counts per second
    meta: TrackMeta = field(default_factory=TrackMeta)
    valid_mask: np.ndarray | None = None
    position_mask: np.ndarray | None = None
    #: Fixes the route is built from. Narrower than ``position_mask``: an
    #: imprecise fix is hidden from the map but still traces the route, since
    #: the position smoothing exists for exactly that noise, while a glitched
    #: one is bypassed. None means every fix.
    geometry_mask: np.ndarray | None = None
    #: Contiguous recording legs; a new leg starts after every dropout.
    leg_index: np.ndarray | None = None
    #: Detector integration time behind each reported rate. None means infer.
    integration_time_s: float | None = None
    #: Moving-average width applied to positions before measuring distance and
    #: speed. None infers it from the GPS accuracy; 1 disables smoothing.
    position_smoothing: int | None = None

    # ---- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        cols = {}
        for name, dtype in (("time_s", float), ("lat", float), ("lon", float),
                            ("accuracy_m", float), ("dose_rate", float),
                            ("count_rate", float)):
            arr = np.asarray(getattr(self, name), dtype=dtype)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
            cols[name] = arr

        n = cols["time_s"].size
        if n == 0:
            raise ValueError("a track must contain at least one sample")
        for name, arr in cols.items():
            if arr.size != n:
                raise ValueError(
                    f"column length mismatch: time_s has {n}, {name} has {arr.size}")

        masks = {}
        for name in ("valid_mask", "position_mask", "geometry_mask"):
            value = getattr(self, name)
            arr = (np.ones(n, dtype=bool) if value is None
                   else np.asarray(value, dtype=bool))
            if arr.size != n:
                raise ValueError(f"{name} has {arr.size} entries, expected {n}")
            masks[name] = arr

        legs = self.leg_index
        legs = (np.zeros(n, dtype=np.int64) if legs is None
                else np.asarray(legs, dtype=np.int64))
        if legs.size != n:
            raise ValueError(f"leg_index has {legs.size} entries, expected {n}")

        # The invariant. Sorting here rather than asking callers to remember is
        # the whole defence against the unsorted-markers trap; the parser also
        # warns, so the user learns it happened.
        order = np.argsort(cols["time_s"], kind="stable")
        if not np.array_equal(order, np.arange(n)):
            for name in cols:
                cols[name] = cols[name][order]
            for name in masks:
                masks[name] = masks[name][order]
            legs = legs[order]

        for name, arr in cols.items():
            arr = np.ascontiguousarray(arr)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        for name, arr in masks.items():
            arr = np.ascontiguousarray(arr)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        legs = np.ascontiguousarray(legs)
        legs.flags.writeable = False
        object.__setattr__(self, "leg_index", legs)

        if self.integration_time_s is not None and self.integration_time_s <= 0:
            raise ValueError(
                f"integration_time_s must be positive, got {self.integration_time_s}")

    # ---- basic geometry ---------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.time_s.size)

    @cached_property
    def times(self) -> tuple[datetime, ...]:
        """Sample timestamps as timezone-aware datetimes."""
        return tuple(_utc(t) for t in self.time_s)

    @cached_property
    def elapsed_s(self) -> np.ndarray:
        """Seconds since the first sample."""
        return self.time_s - self.time_s[0]

    @cached_property
    def dt_s(self) -> np.ndarray:
        """Inter-sample intervals, length n-1."""
        return np.diff(self.time_s)

    @cached_property
    def merc_x(self) -> np.ndarray:
        """EPSG:3857 easting, metres."""
        return geo.lonlat_to_web_mercator(self.lon, self.lat)[0]

    @cached_property
    def merc_y(self) -> np.ndarray:
        """EPSG:3857 northing, metres, increasing north."""
        return geo.lonlat_to_web_mercator(self.lon, self.lat)[1]

    @cached_property
    def smoothing_window(self) -> int:
        """Moving-average width used for distance and speed. 1 means none.

        Inferred from the GPS accuracy unless the caller pinned it. See
        :func:`rcta.core.geo.choose_smoothing_window` for why this exists: on
        foot, raw summed distance overstates the route several-fold because
        each 2.8 m step is smaller than the 8 m fix error.
        """
        if self.position_smoothing is not None:
            return max(int(self.position_smoothing), 1)
        m = self.trusted_position
        return geo.choose_smoothing_window(self.lat[m], self.lon[m],
                                           self.accuracy_m[m])

    @cached_property
    def trusted_position(self) -> np.ndarray:
        """Fixes the geometry is built from: a reading, and a believable place.

        A masked sample's place was borrowed from a neighbour and a glitched
        fix is somewhere the platform never was; letting either into the
        moving average would smear it over the whole window. Merely
        imprecise fixes stay in: averaging their noise away is what the
        smoothing is for.
        """
        m = (self.valid_mask & self.geometry_mask
             & np.isfinite(self.lat) & np.isfinite(self.lon))
        return m if m.any() else np.ones(self.n, dtype=bool)

    @cached_property
    def _smoothed(self):
        """Positions for geometry: trusted fixes smoothed leg by leg.

        Each leg is smoothed on its own so the average never reaches across a
        dropout to a point a hundred kilometres away. Longitude is unwrapped
        first so a leg crossing the antimeridian does not average to zero.
        Untrusted samples are then placed on the smoothed path by
        interpolation in time, so they add no distance of their own.
        """
        trusted = self.trusted_position
        lat = np.array(self.lat, dtype=float)
        lon = np.array(self.lon, dtype=float)
        starts = np.concatenate(
            ([0], np.flatnonzero(np.diff(self.leg_index)) + 1, [self.n]))
        for lo, hi in zip(starts[:-1], starts[1:]):
            good = np.flatnonzero(trusted[lo:hi]) + lo
            if good.size == 0:
                continue
            unwrapped = np.rad2deg(np.unwrap(np.deg2rad(self.lon[good])))
            s_lat, s_lon = geo.smooth_positions(self.lat[good], unwrapped,
                                                self.smoothing_window)
            t = self.time_s[lo:hi]
            lat[lo:hi] = np.interp(t, self.time_s[good], s_lat)
            lon[lo:hi] = np.interp(t, self.time_s[good], s_lon)
        return lat, (lon + 180.0) % 360.0 - 180.0

    @property
    def smooth_lat(self) -> np.ndarray:
        """Positions used for geometry. Equal to ``lat`` when unsmoothed."""
        return self._smoothed[0]

    @property
    def smooth_lon(self) -> np.ndarray:
        return self._smoothed[1]

    @cached_property
    def segment_length_m(self) -> np.ndarray:
        """Great-circle length of each segment, length n-1, jitter-corrected."""
        if self.n < 2:
            return np.zeros(0)
        lat, lon = self.smooth_lat, self.smooth_lon
        return geo.haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])

    @cached_property
    def smooth_merc_x(self) -> np.ndarray:
        """Jitter-corrected easting, for drawing the route."""
        return geo.lonlat_to_web_mercator(self.smooth_lon, self.smooth_lat)[0]

    @cached_property
    def smooth_merc_y(self) -> np.ndarray:
        return geo.lonlat_to_web_mercator(self.smooth_lon, self.smooth_lat)[1]

    @cached_property
    def raw_segment_length_m(self) -> np.ndarray:
        """Segment lengths over the *unsmoothed* fixes, length n-1.

        Teleport detection must use these: smoothing spreads an abrupt jump
        across the whole averaging window, turning one bad fix into a dozen
        implausible-looking segments.
        """
        if self.n < 2:
            return np.zeros(0)
        return geo.haversine(self.lon[:-1], self.lat[:-1],
                             self.lon[1:], self.lat[1:])

    @cached_property
    def raw_speed_m_s(self) -> np.ndarray:
        """Speed from unsmoothed fixes, length n-1. Noise-dominated on foot,
        which is exactly what makes it the right input for glitch detection."""
        if self.n < 2:
            return np.zeros(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.dt_s > 0, self.raw_segment_length_m / self.dt_s, 0.0)

    @cached_property
    def raw_distance_along_m(self) -> np.ndarray:
        """Cumulative distance over the *unsmoothed* fixes.

        Kept so :meth:`describe` can report how much of the raw path length
        was GPS jitter rather than travel. Never use it as the distance.
        """
        return geo.cumulative_distance(self.lon, self.lat)

    @cached_property
    def distance_along_m(self) -> np.ndarray:
        """Cumulative distance from the first sample, length n.

        Measured on jitter-corrected positions, because the raw sum is not a
        measure of travel whenever the step is comparable to the fix error --
        which on foot it always is.

        This includes ground spanned by dropouts. How much of it was never
        actually observed is reported separately by
        :attr:`rcta.core.quality.QualityReport.gap_distance_m`, because the two
        numbers answer different questions and conflating them overstates
        survey coverage.
        """
        return geo.cumulative_distance(self.smooth_lon, self.smooth_lat)

    @cached_property
    def speed_m_s(self) -> np.ndarray:
        """Length n. Element i is the speed at which sample i was arrived at.

        Element 0 has no preceding segment and repeats element 1, so that the
        array aligns with every other column and plots without a leading gap.

        Segments run between trusted fixes only. An untrusted sample takes the
        speed of the trusted segment it falls inside, rather than defining a
        segment of its own from a place it borrowed or a time it shares.
        """
        trusted = np.flatnonzero(self.trusted_position)
        if trusted.size < 2:
            return np.zeros(self.n)
        dt = np.diff(self.time_s[trusted])
        ds = np.diff(self.distance_along_m[trusted])
        with np.errstate(divide="ignore", invalid="ignore"):
            segment = np.where(dt > 0, ds / dt, 0.0)
        arrival = np.clip(np.searchsorted(trusted, np.arange(self.n)),
                          1, trusted.size - 1)
        return segment[arrival - 1]

    @cached_property
    def bearing_deg(self) -> np.ndarray:
        """Heading at each sample, degrees clockwise from north."""
        if self.n < 2:
            return np.zeros(self.n)
        b = geo.bearing(self.lon[:-1], self.lat[:-1], self.lon[1:], self.lat[1:])
        return np.concatenate(([b[0]], b))

    @cached_property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) over position-valid samples."""
        m = (self.position_mask & self.valid_mask
             & np.isfinite(self.lat) & np.isfinite(self.lon))
        if not m.any():
            m = np.ones(self.n, dtype=bool)
        return (float(self.lon[m].min()), float(self.lat[m].min()),
                float(self.lon[m].max()), float(self.lat[m].max()))

    # ---- counting ---------------------------------------------------------

    @cached_property
    def effective_integration_time_s(self) -> float:
        """Seconds of detector live time behind each reported count rate.

        The file records sample *timestamps*, not integration times, and the
        two are emphatically not the same thing: the bundled track has a
        median spacing of 2 s and a maximum of 763 s. Multiplying 21 cps by
        763 s would claim sixteen thousand counts from a sample that
        integrated for about a second, and every Currie limit downstream would
        be wrong by a factor of twenty.

        So: the user's value when supplied, otherwise the median inter-sample
        interval clipped to a believable range. This number is an assumption,
        and it is the single largest systematic in the detection limits.
        """
        if self.integration_time_s is not None:
            return float(self.integration_time_s)
        if self.dt_s.size == 0:
            return _INTEGRATION_BOUNDS[0]
        return float(np.clip(np.median(self.dt_s), *_INTEGRATION_BOUNDS))

    @cached_property
    def counts(self) -> np.ndarray:
        """Estimated counts per sample."""
        return self.count_rate * self.effective_integration_time_s

    @cached_property
    def dose_per_count(self) -> float:
        """Robust nSv/h per cps, for turning a count-rate MDA into dose units.

        The median over samples with a positive count rate, not a fit: the
        ratio is only meaningful where both channels are actually reading, and
        a least-squares slope through the origin would be dominated by the
        high-dose end. :meth:`describe` flags a wandering ratio, because a
        stable one is the normal case and an unstable one means the two
        channels are not measuring the same thing.
        """
        m = (self.count_rate > 0) & self.valid_mask
        if not m.any():
            # Unknown, not zero: a zero ratio would turn every detection
            # limit into 0 nSv/h, which reads as perfect sensitivity.
            return float("nan")
        return float(np.median(self.dose_rate[m] / self.count_rate[m]))

    # ---- masking and selection -------------------------------------------

    def with_mask(self, *, valid=None, position=None, geometry=None, legs=None,
                  integration_time_s=None, meta=None, smoothing=None) -> "Track":
        """A copy with masks or metadata replaced. Columns are never touched."""
        changes = {}
        if valid is not None:
            changes["valid_mask"] = np.asarray(valid, dtype=bool)
        if position is not None:
            changes["position_mask"] = np.asarray(position, dtype=bool)
        if geometry is not None:
            changes["geometry_mask"] = np.asarray(geometry, dtype=bool)
        if legs is not None:
            changes["leg_index"] = np.asarray(legs, dtype=np.int64)
        if integration_time_s is not None:
            changes["integration_time_s"] = float(integration_time_s)
        if smoothing is not None:
            changes["position_smoothing"] = int(smoothing)
        if meta is not None:
            changes["meta"] = meta
        return replace(self, **changes)

    def select(self, mask) -> "Track":
        """A Track over the selected samples, for per-selection statistics.

        ``distance_along_m`` restarts at zero on the result and is meaningless
        when the selection is not contiguous; :meth:`describe` says so rather
        than letting a plausible-looking number be quoted.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.size != self.n:
            raise ValueError(f"mask has {mask.size} entries, expected {self.n}")
        if not mask.any():
            raise ValueError("cannot select an empty subset of a track")
        return Track(
            time_s=self.time_s[mask], lat=self.lat[mask], lon=self.lon[mask],
            accuracy_m=self.accuracy_m[mask], dose_rate=self.dose_rate[mask],
            count_rate=self.count_rate[mask], meta=self.meta,
            valid_mask=self.valid_mask[mask], position_mask=self.position_mask[mask],
            geometry_mask=self.geometry_mask[mask],
            leg_index=self.leg_index[mask],
            integration_time_s=self.effective_integration_time_s,
            position_smoothing=self.smoothing_window,
        )

    def point(self, i: int) -> TrackPoint:
        """One sample as an object, for a tooltip or a report line."""
        i = int(i)
        return TrackPoint(
            index=i, time=_utc(self.time_s[i]), lat=float(self.lat[i]),
            lon=float(self.lon[i]), accuracy_m=float(self.accuracy_m[i]),
            dose_rate=float(self.dose_rate[i]), count_rate=float(self.count_rate[i]),
            distance_along_m=float(self.distance_along_m[i]),
            speed_m_s=float(self.speed_m_s[i]), leg=int(self.leg_index[i]),
            valid=bool(self.valid_mask[i]),
            position_valid=bool(self.position_mask[i]),
        )

    def mask_description(self) -> str:
        n_excluded = int((~self.valid_mask).sum())
        n_poor = int((~self.position_mask).sum())
        return f"{n_excluded} excluded, {n_poor} with poor position"

    # ---- summary ----------------------------------------------------------

    def describe(self) -> str:
        start, end = _utc(self.time_s[0]), _utc(self.time_s[-1])
        span = float(self.time_s[-1] - self.time_s[0])
        hours, rem = divmod(int(span), 3600)
        minutes, seconds = divmod(rem, 60)

        lines = [self.meta.describe()]
        lines.append(f"samples    : {self.n}  ({self.mask_description()})")
        lines.append(f"span       : {start:%Y-%m-%d %H:%M:%S}Z .. {end:%H:%M:%S}Z"
                     f"   ({hours} h {minutes} m {seconds} s)")

        if self.dt_s.size:
            big = int((self.dt_s > 10).sum())
            lines.append(f"sampling   : median {np.median(self.dt_s):.0f} s, "
                         f"max {self.dt_s.max():.0f} s, {big} dropout(s) over 10 s")

        declared = self.meta.declared_distance_m
        measured = float(self.distance_along_m[-1])
        line = f"distance   : {measured / 1000.0:.2f} km"
        if declared:
            delta = abs(measured - declared) / declared * 100.0
            line += f"   (device says {declared / 1000.0:.2f} km, {delta:.2f} % apart)"
        lines.append(line)

        if self.smoothing_window > 1:
            raw = float(self.raw_distance_along_m[-1])
            inflation = raw / max(measured, 1e-9)
            lines.append(
                f"smoothing  : {self.smoothing_window}-sample position average "
                f"(raw fixes give {raw / 1000.0:.2f} km, {inflation:.1f}x more; "
                "the excess is GPS jitter, not travel)")

        source = ("supplied" if self.integration_time_s is not None
                  else "inferred from median spacing")
        lines.append(f"integration: {self.effective_integration_time_s:.1f} s "
                     f"per sample ({source})")

        ok = self.valid_mask
        if ok.any():
            dose, cps = self.dose_rate[ok], self.count_rate[ok]
            lines.append(f"dose rate  : {dose.min():.2f} .. {dose.max():.2f} "
                         f"nSv/h   median {np.median(dose):.2f}")
            lines.append(f"count rate : {cps.min():.2f} .. {cps.max():.2f} "
                         f"cps      median {np.median(cps):.2f}")
        known = self.accuracy_m[np.isfinite(self.accuracy_m)]
        lines.append(f"accuracy   : median {np.median(known):.0f} m, "
                     f"max {known.max():.0f} m" if known.size
                     else "accuracy   : not recorded")

        for warning in self.meta.warnings:
            lines.append(f"warning    : {warning}")
        return "\n".join(lines)
