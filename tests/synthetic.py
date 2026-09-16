"""A track factory with known answers. Not a test module.

Every assertion about anomalies, hotspots, dose and geometry needs a track
whose truth is known analytically rather than by whatever the code produced
last time. This builds one, and returns the truth alongside it.

The defaults deliberately produce a *messy* track: markers shuffled out of
time order, jittered sampling, and Poisson counting noise, because those are
what the real export does and a factory that only makes clean tracks tests
nothing. Gaps and teleports are opt-in per test.
"""

from __future__ import annotations

import numpy as np

from rcta.core import geo
from rcta.core.track import Track, TrackMeta

#: Roughly the RadiaCode-102's nSv/h per cps, measured on the bundled file.
DOSE_PER_COUNT = 5.13


def _path(kind, n, step_m, origin):
    """Latitude/longitude arrays for a named ground track."""
    lat0, lon0 = origin
    m_per_deg_lat = 111195.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(lat0))

    if kind == "line":
        east = np.zeros(n)
        north = np.arange(n, dtype=float) * step_m
    elif kind == "square":
        # Closes on itself, so distance_along and the start/end coincidence are
        # both checkable, and it exercises revisit detection.
        side = max(n // 4, 1)
        legs = []
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            k = np.arange(side, dtype=float) * step_m
            legs.append((dx * k, dy * k))
        east = np.concatenate([legs[0][0],
                               legs[0][0][-1] + legs[1][0],
                               legs[0][0][-1] + legs[2][0],
                               np.zeros(side)])[:n]
        north = np.concatenate([legs[0][1],
                                legs[1][1],
                                legs[1][1][-1] + legs[2][1],
                                legs[1][1][-1] + legs[3][1]])[:n]
    elif kind == "lawnmower":
        # Parallel passes 30 m apart: the only geometry that exercises repeat
        # visits and the transverse behaviour of the interpolation mask.
        per_pass = max(n // 4, 2)
        east, north = [], []
        for p in range(int(np.ceil(n / per_pass))):
            k = np.arange(per_pass, dtype=float) * step_m
            north.append(k if p % 2 == 0 else k[::-1])
            east.append(np.full(per_pass, p * 30.0))
        east = np.concatenate(east)[:n]
        north = np.concatenate(north)[:n]
    elif kind == "flight":
        east = np.zeros(n)
        north = np.arange(n, dtype=float) * step_m
    else:
        raise ValueError(f"unknown path {kind!r}")

    return lat0 + north / m_per_deg_lat, lon0 + east / m_per_deg_lon


def make(*, n=600, start=1788785639, dt_s=2.0, dt_jitter=0.0, path="line",
         speed_m_s=1.4, origin=(30.2, -97.66), background_cps=18.0,
         dose_per_count=DOSE_PER_COUNT, sources=(), drift=None,
         accuracy_m=8.0, accuracy_outliers=0, gaps=(), teleports=(),
         shuffle=True, poisson=True, seed=0):
    """Build a track with known ground truth.

    ``sources`` is a list of ``(index, peak_cps, width_samples)``: a Gaussian
    bump in count rate centred on that sample. Expressing a source in sample
    space rather than in metres keeps the injected net counts exactly
    computable, which is what the recovery tests compare against.

    ``gaps`` is ``[(after_index, seconds), ...]`` and ``teleports`` is
    ``[(index, metres), ...]``.

    Returns ``(track, truth)`` where ``truth`` carries the exact distance,
    accumulated dose, injected net counts per source, and the indices of every
    injected defect.
    """
    rng = np.random.default_rng(seed)

    dt = np.full(n, float(dt_s))
    if dt_jitter:
        dt = np.clip(dt + rng.normal(0.0, dt_jitter, n), 0.25 * dt_s, None)
    time_s = float(start) + np.concatenate(([0.0], np.cumsum(dt[:-1])))

    step_m = float(speed_m_s) * float(dt_s)
    lat, lon = _path(path, n, step_m, origin)

    # A dropout advances position as well as time: the platform kept moving
    # while the detector was not logging. Shifting only the clock would model
    # a stationary pause, which is a different thing entirely and is correctly
    # *not* treated as a break in the track.
    gap_indices = []
    for after, seconds in gaps:
        time_s[after + 1:] += float(seconds)
        lat[after + 1:] += float(speed_m_s) * float(seconds) / 111195.0
        gap_indices.append(int(after))

    teleport_indices = []
    for index, metres in teleports:
        lat[index:] += metres / 111195.0
        teleport_indices.append(int(index))

    # ---- radiation -------------------------------------------------------
    true_rate = np.full(n, float(background_cps))
    if drift is not None:
        true_rate = np.linspace(float(drift[0]), float(drift[1]), n)

    injected = []
    for index, peak_cps, width in sources:
        k = np.arange(n, dtype=float)
        bump = float(peak_cps) * np.exp(-0.5 * ((k - index) / float(width)) ** 2)
        true_rate = true_rate + bump
        injected.append({
            "index": int(index),
            "peak_cps": float(peak_cps),
            "width_samples": float(width),
            "net_counts": float(bump.sum() * dt_s),
            "lat": float(lat[int(index)]),
            "lon": float(lon[int(index)]),
        })

    if poisson:
        counts = rng.poisson(np.maximum(true_rate, 0.0) * dt_s)
        count_rate = counts / dt_s
    else:
        count_rate = true_rate.copy()
    dose_rate = count_rate * float(dose_per_count)

    accuracy = np.full(n, float(accuracy_m))
    outlier_indices = []
    if accuracy_outliers:
        outlier_indices = rng.choice(n, size=int(accuracy_outliers),
                                     replace=False).tolist()
        accuracy[outlier_indices] = 250.0

    truth = {
        "n": n,
        "distance_m": float(geo.cumulative_distance(lon, lat)[-1]),
        "background_cps": float(background_cps),
        "dose_per_count": float(dose_per_count),
        "true_rate": true_rate,
        "sources": injected,
        "gap_indices": gap_indices,
        "teleport_indices": teleport_indices,
        "accuracy_outlier_indices": sorted(outlier_indices),
        "integration_time_s": float(dt_s),
    }

    order = np.arange(n)
    if shuffle:
        # The real export's markers arrive in effectively random order; a
        # factory that hands over sorted data would never exercise the loader.
        order = rng.permutation(n)

    track = Track(
        time_s=time_s[order], lat=lat[order], lon=lon[order],
        accuracy_m=accuracy[order], dose_rate=dose_rate[order],
        count_rate=count_rate[order],
        meta=TrackMeta(title="synthetic", devices=("RadiaCode-102",),
                       source_format="synthetic"),
        integration_time_s=float(dt_s),
    )
    return track, truth


# ---------------------------------------------------------------------------
# flight trajectories
# ---------------------------------------------------------------------------

#: Altitude of a typical narrow-body cruise, in metres. FL360.
CRUISE_M = 10972.8

#: Scale height of the cosmic dose profile used by :func:`flight_dose`, in
#: metres. Chosen so the fit has a known answer to be checked against, not
#: because 2 km is the truth for any particular latitude or solar cycle.
TRUE_SCALE_HEIGHT_M = 2000.0


def flight_altitude(elapsed_s, *, span_s, ceiling_m=CRUISE_M, field_m=150.0,
                    climb_frac=0.20, descent_frac=0.20):
    """A piecewise climb-cruise-descent altitude profile, in metres.

    Deliberately continuous but not smooth: the corners at top of climb and
    start of descent are exactly where linear interpolation across a waypoint
    gap does the most damage, so a factory that rounded them off would hide
    the failure mode ``max_gap_s`` exists to prevent.
    """
    elapsed_s = np.asarray(elapsed_s, dtype=float)
    climb_end = float(span_s) * float(climb_frac)
    descent_start = float(span_s) * (1.0 - float(descent_frac))

    altitude = np.full(elapsed_s.shape, float(ceiling_m))
    climbing = elapsed_s < climb_end
    altitude[climbing] = field_m + (ceiling_m - field_m) * (
        elapsed_s[climbing] / climb_end)
    descending = elapsed_s > descent_start
    fraction = ((elapsed_s[descending] - descent_start)
                / max(span_s - descent_start, 1.0))
    altitude[descending] = ceiling_m - (ceiling_m - field_m) * fraction
    return np.clip(altitude, field_m, ceiling_m)


def flight_dose(altitude_m, *, ground_cps=1.5, terrestrial_cps=1.2,
                scale_height_m=TRUE_SCALE_HEIGHT_M):
    """Count rate from altitude: a fixed terrestrial part plus an exponential.

    The terrestrial term is why :data:`rcta.core.flight.DEFAULT_FIT_FLOOR_M`
    exists. It does not vary with altitude at all, so including the low bins
    biases the fitted scale height upward -- measured on this factory with the
    Poisson noise off, and with a 2000 m truth:

    ==================  ==============  =================
    terrestrial (cps)   fitted, no floor  fitted, 3000 m floor
    ==================  ==============  =================
    0.0                 1927 m            1915 m
    1.2                 1999 m            1956 m
    2.6                 2082 m            2003 m
    12.0                2551 m            2288 m
    ==================  ==============  =================

    So the floor is worth a few per cent at a realistic pedestal and a great
    deal at a large one, and it never removes the bias entirely. The residual
    3 % low at zero terrestrial is the method's own: a bin mean of a convex
    function exceeds the function at the bin centre.
    """
    altitude_m = np.asarray(altitude_m, dtype=float)
    cosmic = float(ground_cps) * np.exp(altitude_m / float(scale_height_m))
    return float(terrestrial_cps) + cosmic


def flight_trajectory(track, *, cadence_s=30.0, ceiling_m=CRUISE_M,
                      field_m=150.0, climb_frac=0.20, descent_frac=0.20,
                      offset_m=0.0, time_offset_s=0.0, gaps=(),
                      callsign="DAL2618", icao24="a1b2c3",
                      altitude_kind="barometric"):
    """A :class:`~rcta.core.flight.Trajectory` that matches ``track``.

    Positions are interpolated from the track's own fixes, which is what a
    genuine ADS-B path for the same aeroplane looks like to within the GPS
    error. ``offset_m`` displaces the whole path north, which is how the tests
    build a trajectory that is the right shape at the wrong place -- the wrong
    flight, in other words, and the case the position check exists to reject.

    ``gaps`` is ``[(after_index, seconds), ...]`` in waypoint space, for
    testing that altitude is not interpolated across a dropout.
    """
    from rcta.core.flight import Trajectory

    t0, t1 = float(track.time_s[0]), float(track.time_s[-1])
    span = t1 - t0
    time_s = np.arange(t0, t1 + cadence_s, float(cadence_s))

    for after, seconds in gaps:
        time_s = np.concatenate([time_s[:after + 1],
                                 time_s[after + 1:] + float(seconds)])
    time_s = time_s[time_s <= t1]

    lat = np.interp(time_s, track.time_s, track.smooth_lat)
    lon = np.interp(time_s, track.time_s, track.smooth_lon)
    lat = lat + float(offset_m) / 111195.0

    altitude = flight_altitude(time_s - t0, span_s=span, ceiling_m=ceiling_m,
                               field_m=field_m, climb_frac=climb_frac,
                               descent_frac=descent_frac)
    return Trajectory(
        time_s=time_s + float(time_offset_s), lat=lat, lon=lon,
        altitude_m=altitude, altitude_kind=altitude_kind,
        on_ground=altitude <= field_m + 1.0,
        callsign=callsign, icao24=icao24, provider="synthetic",
        source="tests.synthetic.flight_trajectory")


def flight_track(*, n=900, start=1788785639, dt_s=10.0, origin=(30.2, -97.66),
                 destination=(40.79, -111.98), ceiling_m=CRUISE_M,
                 field_m=150.0, climb_frac=0.20, descent_frac=0.20,
                 dose_per_count=51.3, accuracy_m=25.0, poisson=True, seed=0):
    """A flight track whose count rate is a known function of altitude.

    Returns ``(track, trajectory, truth)``. Unlike :func:`make` this one
    builds the two consistently from the same altitude profile, so the
    recovered scale height has an exact target and the join has an exact
    expected coverage.
    """
    from rcta.core.flight import Trajectory
    from rcta.core.track import Track, TrackMeta

    rng = np.random.default_rng(seed)
    time_s = float(start) + np.arange(n, dtype=float) * float(dt_s)
    span = float(time_s[-1] - time_s[0])

    fraction = np.linspace(0.0, 1.0, n)
    lat = origin[0] + (destination[0] - origin[0]) * fraction
    lon = origin[1] + (destination[1] - origin[1]) * fraction

    altitude = flight_altitude(time_s - time_s[0], span_s=span,
                               ceiling_m=ceiling_m, field_m=field_m,
                               climb_frac=climb_frac, descent_frac=descent_frac)
    true_cps = flight_dose(altitude)
    if poisson:
        count_rate = rng.poisson(true_cps * dt_s) / dt_s
    else:
        count_rate = true_cps.copy()

    track = Track(
        time_s=time_s, lat=lat, lon=lon,
        accuracy_m=np.full(n, float(accuracy_m)),
        dose_rate=count_rate * float(dose_per_count), count_rate=count_rate,
        meta=TrackMeta(title="synthetic flight", devices=("RadiaCode-102",),
                       source_format="synthetic"),
        integration_time_s=float(dt_s), position_smoothing=1)

    trajectory = Trajectory(
        time_s=time_s, lat=lat, lon=lon, altitude_m=altitude,
        altitude_kind="barometric", on_ground=altitude <= field_m + 1.0,
        callsign="DAL2618", icao24="a1b2c3", provider="synthetic")

    truth = {
        "n": n,
        "altitude_m": altitude,
        "true_cps": true_cps,
        "scale_height_m": TRUE_SCALE_HEIGHT_M,
        "ceiling_m": float(ceiling_m),
        "dose_per_count": float(dose_per_count),
        "integration_time_s": float(dt_s),
    }
    return track, trajectory, truth
