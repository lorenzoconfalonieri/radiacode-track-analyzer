"""Exporters. Every function returns a string; the caller writes the file.

Keeping I/O out of here means the GUI, the CLI and the tests all exercise the
identical code path, and a test can assert on the exact bytes without a
temporary directory.

Two format decisions are worth stating because the obvious alternatives are
quietly lossy.

*Legs, not one line.* Every geometry export emits one LineString or one
``trkseg`` per recording leg. A single continuous line across a 763 s dropout
would assert a route through 103 km of ground the detector never observed, and
once the file leaves this program nothing downstream can tell that the claim
was invented.

*A namespaced GPX extension, not a borrowed field.* GPX has no element meaning
nanosieverts per hour. The tempting move is to smuggle dose into Garmin's
``TrackPointExtension``, whose fields are heart rate, cadence and temperature
-- so any tool that actually understands that schema would read a radiation
survey as a cardio workout. A documented ``rcta:`` namespace is ignored by
tools that do not know it and unambiguous to those that do.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import numpy as np

from rcta import __version__
from rcta.core.dose import DOSE_UNITS, convert_dose_rate
from rcta.core.flight import (FLIGHT_LEVEL_M, compare_branches,
                              fit_scale_height)

__all__ = [
    "altitude_profile_to_csv",
    "altitude_samples_to_csv",
    "anomalies_to_csv",
    "flight_report_to_json",
    "grid_prj",
    "grid_to_asc",
    "hotspots_to_csv",
    "hotspots_to_geojson",
    "report_to_json",
    "track_to_csv",
    "track_to_geojson",
    "track_to_gpx",
]

GPX_NAMESPACE = "https://github.com/rcta/gpx/v1"


def _finite(value):
    """``value`` with every non-finite float replaced by None, recursively.

    ``NaN`` is not JSON, and Python's encoder writes it anyway; a report that
    a strict reader rejects is worse than one that says "unknown" as null.
    """
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _legs(track, quality):
    """Contiguous index spans, one per recording leg."""
    legs = track.leg_index
    if quality is not None:
        legs = quality.leg_index
    spans = []
    start = 0
    for i in range(1, track.n):
        if legs[i] != legs[i - 1]:
            spans.append((start, i))
            start = i
    spans.append((start, track.n))
    return spans


# ---- tabular --------------------------------------------------------------

_TRACK_COLUMNS = [
    ("index", lambda t, i, ctx: i),
    ("iso_time", lambda t, i, ctx: _iso(t.time_s[i])),
    ("epoch_s", lambda t, i, ctx: f"{t.time_s[i]:.0f}"),
    ("lat", lambda t, i, ctx: f"{t.lat[i]:.7f}"),
    ("lon", lambda t, i, ctx: f"{t.lon[i]:.7f}"),
    ("accuracy_m", lambda t, i, ctx: f"{t.accuracy_m[i]:.1f}"),
    ("dose_rate", lambda t, i, ctx: f"{ctx['dose'][i]:.6g}"),
    ("count_rate", lambda t, i, ctx: f"{t.count_rate[i]:.4f}"),
    ("distance_along_m", lambda t, i, ctx: f"{t.distance_along_m[i]:.2f}"),
    ("speed_m_s", lambda t, i, ctx: f"{t.speed_m_s[i]:.3f}"),
    ("leg", lambda t, i, ctx: int(ctx["legs"][i])),
    ("background_cps", lambda t, i, ctx:
        "" if ctx["background"] is None else f"{ctx['background'][i]:.4f}"),
    ("valid", lambda t, i, ctx: int(t.valid_mask[i])),
    ("position_valid", lambda t, i, ctx: int(t.position_mask[i])),
]


def track_to_csv(track, *, quality=None, background=None, unit="nSv/h") -> str:
    """One row per sample. The dose column carries its unit in the header."""
    if unit not in DOSE_UNITS:
        raise ValueError(f"unknown dose unit {unit!r}")
    ctx = {
        "dose": convert_dose_rate(track.dose_rate, unit),
        "legs": quality.leg_index if quality is not None else track.leg_index,
        "background": None if background is None else background.count_rate,
    }
    header = [name if name != "dose_rate" else f"dose_rate_{unit.replace('/', '_per_')}"
              for name, _ in _TRACK_COLUMNS]
    rows = [",".join(header)]
    for i in range(track.n):
        # A masked sample has no reading; its cells are blank, the same
        # convention the altitude profile uses for an empty bin.
        cells = (str(fn(track, i, ctx)) for _, fn in _TRACK_COLUMNS)
        rows.append(",".join("" if c == "nan" else c for c in cells))
    return "\n".join(rows) + "\n"


_ANOMALY_COLUMNS = [
    ("start_index", lambda a, u: a.start_index),
    ("stop_index", lambda a, u: a.stop_index),
    ("scale_samples", lambda a, u: a.scale),
    ("iso_time", lambda a, u: _iso(a.time_s)),
    ("duration_s", lambda a, u: f"{a.duration_s:.0f}"),
    ("lat", lambda a, u: f"{a.lat:.7f}"),
    ("lon", lambda a, u: f"{a.lon:.7f}"),
    ("length_m", lambda a, u: f"{a.length_m:.1f}"),
    ("gross_counts", lambda a, u: f"{a.gross_counts:.1f}"),
    ("background_counts", lambda a, u: f"{a.background_counts:.1f}"),
    ("net_counts", lambda a, u: f"{a.net_counts:.1f}"),
    ("net_count_rate_cps", lambda a, u: f"{a.net_count_rate:.4f}"),
    ("net_dose_rate", lambda a, u: f"{float(convert_dose_rate(a.net_dose_rate, u)):.6g}"),
    ("peak_dose_rate", lambda a, u: f"{float(convert_dose_rate(a.peak_dose_rate, u)):.6g}"),
    ("significance_sigma", lambda a, u: f"{a.significance:.3f}"),
    ("threshold_sigma", lambda a, u: f"{a.threshold_sigma:.3f}"),
    ("critical_level_counts", lambda a, u: f"{a.limits.critical_level:.1f}"),
    ("detection_limit_counts", lambda a, u: f"{a.limits.detection_limit:.1f}"),
    ("mda_cps", lambda a, u: f"{a.limits.mda_cps:.4f}"),
    ("significant", lambda a, u: int(a.is_significant)),
    ("flags", lambda a, u: " ".join(a.flags)),
]


def anomalies_to_csv(anomalies, *, unit="nSv/h") -> str:
    """Every candidate, significant or not, with the reason it was excluded.

    Exporting only the significant ones would hide the flagged trends, which
    are exactly what a reviewer needs to see to judge whether the flagging was
    right.
    """
    rows = [",".join(name for name, _ in _ANOMALY_COLUMNS)]
    for a in anomalies:
        rows.append(",".join(str(fn(a, unit)) for _, fn in _ANOMALY_COLUMNS))
    return "\n".join(rows) + "\n"


_HOTSPOT_COLUMNS = [
    ("lat", lambda h, u: f"{h.lat:.7f}"),
    ("lon", lambda h, u: f"{h.lon:.7f}"),
    ("radius_m", lambda h, u: f"{h.radius_m:.1f}"),
    ("n_samples", lambda h, u: h.n_samples),
    ("n_passes", lambda h, u: h.n_passes),
    ("first_iso", lambda h, u: _iso(h.first_time_s)),
    ("last_iso", lambda h, u: _iso(h.last_time_s)),
    ("peak_dose_rate", lambda h, u: f"{float(convert_dose_rate(h.peak_dose_rate, u)):.6g}"),
    ("mean_net_dose_rate", lambda h, u:
        f"{float(convert_dose_rate(h.mean_net_dose_rate, u)):.6g}"),
    ("net_counts", lambda h, u: f"{h.net_counts:.1f}"),
    ("significance_sigma", lambda h, u: f"{h.significance:.3f}"),
    ("score", lambda h, u: f"{h.score:.1f}"),
    ("evidence", lambda h, u: h.evidence),
    ("flags", lambda h, u: " ".join(h.flags)),
]


def hotspots_to_csv(hotspots, *, unit="nSv/h") -> str:
    rows = [",".join(name for name, _ in _HOTSPOT_COLUMNS)]
    for h in hotspots:
        rows.append(",".join(str(fn(h, unit)) for _, fn in _HOTSPOT_COLUMNS))
    return "\n".join(rows) + "\n"


# ---- geometry -------------------------------------------------------------


def track_to_geojson(track, *, quality=None, include_points=True,
                     unit="nSv/h", indent=2) -> str:
    """RFC 7946 FeatureCollection: one LineString per leg, plus point samples."""
    dose = convert_dose_rate(track.dose_rate, unit)
    features = []
    omitted = []

    for leg, (start, stop) in enumerate(_legs(track, quality)):
        rows = [i for i in range(start, stop) if track.position_mask[i]]
        if len(rows) < 2:
            # A leg whose fixes were all masked cannot become a LineString.
            # Recorded rather than silently skipped: on the bundled file five
            # of fourteen legs vanish this way, and an export that loses a
            # third of its legs without saying so is a trap.
            omitted.append({"leg": leg, "n_samples": stop - start,
                            "n_positioned": len(rows)})
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                # RFC 7946 order is [lon, lat]. Getting this backwards is the
                # single most common GeoJSON bug and renders in the ocean.
                "coordinates": [[round(float(track.lon[i]), 7),
                                 round(float(track.lat[i]), 7)] for i in rows],
            },
            "properties": {
                "leg": leg,
                "start_time": _iso(track.time_s[rows[0]]),
                "end_time": _iso(track.time_s[rows[-1]]),
                "n_samples": len(rows),
                "dose_rate_unit": unit,
                "mean_dose_rate": round(float(dose[rows].mean()), 6),
                "max_dose_rate": round(float(dose[rows].max()), 6),
            },
        })

    if include_points:
        for i in range(track.n):
            if not track.position_mask[i]:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [round(float(track.lon[i]), 7),
                                             round(float(track.lat[i]), 7)]},
                "properties": {
                    "index": i,
                    "time": _iso(track.time_s[i]),
                    "dose_rate": round(float(dose[i]), 6),
                    "dose_rate_unit": unit,
                    "count_rate_cps": round(float(track.count_rate[i]), 4),
                    "accuracy_m": round(float(track.accuracy_m[i]), 1),
                    "speed_m_s": round(float(track.speed_m_s[i]), 3),
                },
            })

    return json.dumps({
        "type": "FeatureCollection",
        "properties": {
            "title": track.meta.title,
            "devices": list(track.meta.devices),
            "source_format": track.meta.source_format,
            "generator": "rcta",
            "legs_omitted": omitted,
        },
        "features": features,
    }, indent=indent)


def hotspots_to_geojson(hotspots, *, unit="nSv/h", indent=2) -> str:
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(h.lon, 7),
                                                      round(h.lat, 7)]},
        "properties": {
            "radius_m": round(h.radius_m, 1),
            "n_samples": h.n_samples,
            "n_passes": h.n_passes,
            "peak_dose_rate": round(float(convert_dose_rate(h.peak_dose_rate, unit)), 6),
            "dose_rate_unit": unit,
            "net_counts": round(h.net_counts, 1),
            "significance_sigma": round(h.significance, 3),
            "evidence": h.evidence,
            "flags": list(h.flags),
        },
    } for h in hotspots]
    return json.dumps({"type": "FeatureCollection", "features": features},
                      indent=indent)


def track_to_gpx(track, *, quality=None, creator=f"rcta {__version__}") -> str:
    """GPX 1.1 with one ``trkseg`` per leg and dose in an ``rcta:`` extension."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<gpx version="1.1" creator="{escape(creator)}"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        f'     xmlns:rcta="{GPX_NAMESPACE}">',
        "  <trk>",
        f"    <name>{escape(track.meta.title or 'track')}</name>",
    ]
    for start, stop in _legs(track, quality):
        rows = [i for i in range(start, stop) if track.position_mask[i]]
        if not rows:
            continue
        lines.append("    <trkseg>")
        for i in rows:
            lines.append(
                f'      <trkpt lat="{track.lat[i]:.7f}" lon="{track.lon[i]:.7f}">')
            lines.append(f"        <time>{_iso(track.time_s[i])}</time>")
            lines.append(f"        <hdop>{track.accuracy_m[i]:.1f}</hdop>")
            lines.append("        <extensions>")
            lines.append("          <rcta:doseRate units=\"nSv/h\">"
                         f"{track.dose_rate[i]:.4f}</rcta:doseRate>")
            lines.append("          <rcta:countRate units=\"cps\">"
                         f"{track.count_rate[i]:.4f}</rcta:countRate>")
            lines.append("        </extensions>")
            lines.append("      </trkpt>")
        lines.append("    </trkseg>")
    lines += ["  </trk>", "</gpx>"]
    return "\n".join(lines) + "\n"


# ---- the full report ------------------------------------------------------


#: ESRI WKT for EPSG:3857, the CRS the grid is computed in. GIS tools read
#: the sibling .prj; without it an .asc has no coordinate system at all.
_WEB_MERCATOR_WKT = (
    'PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",'
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Mercator_Auxiliary_Sphere"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",0.0],'
    'PARAMETER["Standard_Parallel_1",0.0],'
    'PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]')

_NODATA = -9999


def grid_prj() -> str:
    """The .prj that belongs beside :func:`grid_to_asc` output."""
    return _WEB_MERCATOR_WKT + "\n"


def grid_to_asc(grid, *, unit="nSv/h") -> str:
    """The heatmap as an ESRI ASCII grid in EPSG:3857 metres.

    Unsupported cells are NODATA, never zero: a blank cell is one the track
    did not constrain, and writing 0 there would hand a GIS a low reading
    nobody took. Cells are square in projected metres, so a single
    ``cellsize`` is exact. Rows run north to south, as the format requires.
    """
    if unit not in DOSE_UNITS:
        raise ValueError(f"unknown dose unit {unit!r}")
    values = convert_dose_rate(grid.values, unit)
    ny, nx = values.shape
    cell = float(grid.x_edges[1] - grid.x_edges[0])
    lines = [
        f"ncols {nx}",
        f"nrows {ny}",
        f"xllcorner {float(grid.x_edges[0]):.3f}",
        f"yllcorner {float(grid.y_edges[0]):.3f}",
        f"cellsize {cell:.6f}",
        f"NODATA_value {_NODATA}",
    ]
    for row in values[::-1]:
        lines.append(" ".join(f"{v:.6g}" if np.isfinite(v) else str(_NODATA)
                              for v in row))
    return "\n".join(lines) + "\n"


def report_to_json(analysis, *, indent=2) -> str:
    """The complete analysis snapshot, parameters included.

    The parameters are not decoration. A detection limit is a statement about
    an assumed integration time, an assumed background method and an assumed
    threshold; without them the number is not reproducible and should not be
    quoted.
    """
    track = analysis.track
    params = analysis.params
    applied = analysis.background.params
    unit = params.unit

    return json.dumps(_finite({
        "generator": "rcta",
        "track": {
            "title": track.meta.title,
            "devices": list(track.meta.devices),
            "source_path": track.meta.source_path,
            "source_format": track.meta.source_format,
            "n_samples": track.n,
            "start": _iso(track.time_s[0]),
            "end": _iso(track.time_s[-1]),
            "distance_m": round(float(track.distance_along_m[-1]), 2),
            "declared_distance_m": track.meta.declared_distance_m,
            "integration_time_s": track.effective_integration_time_s,
            "integration_time_inferred": params.integration_time_s is None,
            "dose_per_count_nsv_h_per_cps": round(track.dose_per_count, 4),
            "bbox": [round(v, 6) for v in track.bbox],
        },
        # Every setting that can change a number below, as *applied*: the
        # background window may have been widened from the one requested, and
        # a report that quoted the request could not be reproduced from.
        "parameters": {
            "quality": {
                "max_accuracy_m": params.quality.max_accuracy_m,
                "gap_threshold_s": params.quality.gap_threshold_s,
                "gap_threshold_applied_s": analysis.quality.gap_threshold_s,
                "max_speed_m_s": params.quality.max_speed_m_s,
                "speed_outlier_factor": params.quality.speed_outlier_factor,
                "min_jump_m": params.quality.min_jump_m,
                "smoothing_window_samples": track.smoothing_window,
            },
            "background": {
                "method": applied.method,
                "window_s": applied.window_s,
                "window_s_requested": params.background.window_s,
                "trim_fraction": applied.trim_fraction,
                "guard_s": applied.guard_s,
                "min_samples": applied.min_samples,
                "reference_span": (list(applied.reference_span)
                                   if applied.reference_span else None),
                "scale_separation": applied.scale_separation,
            },
            "scan": {
                "scales": list(params.scan.scales),
                "alpha": params.scan.alpha,
                "bonferroni": params.scan.bonferroni,
                "merge_overlapping": params.scan.merge_overlapping,
                "trend_flag_ratio": params.scan.trend_flag_ratio,
                "poor_gps_fraction": params.scan.poor_gps_fraction,
            },
            "integration_time_s": params.integration_time_s,
            "cluster_radius_m": params.cluster_radius_m,
            "n_boot": params.n_boot,
            "unit": unit,
        },
        "quality": {
            "n_poor_accuracy": analysis.quality.n_poor_accuracy,
            "n_speed_outliers": analysis.quality.n_speed_outliers,
            "n_gaps": analysis.quality.n_gaps,
            "n_legs": analysis.quality.n_legs,
            "gap_time_s": analysis.quality.gap_time_s,
            "gap_distance_m": round(analysis.quality.gap_distance_m, 1),
            "coverage": round(analysis.quality.coverage, 4),
        },
        "dose": {
            "total_nsv": round(analysis.dose.total_nsv, 4),
            "naive_total_nsv": round(analysis.dose.naive_total_nsv, 4),
            "covered_time_s": analysis.dose.covered_time_s,
            "span_time_s": analysis.dose.span_time_s,
            "coverage": round(analysis.dose.coverage, 4),
        },
        "distribution": {
            "threshold": round(analysis.dose_distribution.threshold, 4),
            "ashman_d": round(analysis.dose_distribution.ashman_d, 4),
            "is_bimodal": analysis.dose_distribution.is_bimodal,
            "mu_low": round(analysis.dose_distribution.mu_low, 4),
            "mu_high": round(analysis.dose_distribution.mu_high, 4),
        },
        "speed_dose_correlation": {
            "spearman_rho": round(analysis.speed_dose.spearman_rho, 5),
            "n": analysis.speed_dose.n,
            "n_effective": round(analysis.speed_dose.n_effective, 2),
            "p_naive": analysis.speed_dose.p_naive,
            "p_effective": analysis.speed_dose.p_effective,
            "p_block_bootstrap": analysis.speed_dose.p_block_bootstrap,
            "caveat": analysis.speed_dose.caveat,
        },
        "background": {
            "method": analysis.background.method,
            "median_cps": round(analysis.background.median_count_rate, 4),
        },
        "detection_limits": [
            {"window_samples": w, "window_s": s, "mda_nsv_h": round(m, 3)}
            for w, s, m in analysis.mda_table
        ],
        "anomalies": [{
            "start_index": a.start_index,
            "stop_index": a.stop_index,
            "scale": a.scale,
            "time": _iso(a.time_s),
            "duration_s": a.duration_s,
            "lat": round(a.lat, 7),
            "lon": round(a.lon, 7),
            "net_counts": round(a.net_counts, 2),
            # In parameters.unit, as it always was; the canonical copy beside
            # it follows the rest of the report, where a dose field names its
            # unit in the key, so a script never has to look the unit up.
            "net_dose_rate": round(float(convert_dose_rate(a.net_dose_rate, unit)), 4),
            "net_dose_rate_nsv_h": round(float(a.net_dose_rate), 4),
            "significance_sigma": round(a.significance, 3),
            "threshold_sigma": round(a.threshold_sigma, 3),
            "critical_level_counts": round(a.limits.critical_level, 2),
            "detection_limit_counts": round(a.limits.detection_limit, 2),
            "significant": a.is_significant,
            "flags": list(a.flags),
        } for a in analysis.anomalies],
        "hotspots": [{
            "lat": round(h.lat, 7),
            "lon": round(h.lon, 7),
            "radius_m": round(h.radius_m, 1),
            "n_samples": h.n_samples,
            "n_passes": h.n_passes,
            "net_counts": round(h.net_counts, 2),
            "significance_sigma": round(h.significance, 3),
            "score": round(h.score, 2),
            "evidence": h.evidence,
            "flags": list(h.flags),
        } for h in analysis.hotspots],
        "warnings": list(analysis.warnings),
    }), indent=indent, allow_nan=False)


# ---- flight ---------------------------------------------------------------


def altitude_profile_to_csv(profile, *, unit="nSv/h") -> str:
    """The dose-versus-altitude profile, one row per bin, empty bins included.

    Empty bins are emitted with a blank value rather than skipped, so that a
    reader who plots the file gets the same gaps the program shows. Dropping
    them silently closes the plot across altitudes the aircraft never held,
    which is the same invention that
    :func:`~rcta.core.profile.resample_by_distance` refuses to make in
    distance.

    Both error columns are written. ``counting_sem`` is what Poisson allows;
    ``observed_sem`` is the spread the samples showed. Exporting only one
    leaves the reader unable to tell a counting-limited bin from one dominated
    by real variation, which is the difference between an error bar and a
    disagreement.
    """
    dose = convert_dose_rate(profile.dose_rate, unit)
    counting = convert_dose_rate(profile.counting_sem_cps * profile.dose_per_count,
                                 unit)
    observed = convert_dose_rate(profile.observed_sem_cps * profile.dose_per_count,
                                 unit)

    def num(value, fmt=".6g"):
        return "" if not np.isfinite(value) else format(float(value), fmt)

    rows = [f"altitude_m,flight_level,n_samples,live_time_s,count_rate_cps,"
            f"dose_rate_{unit.replace('/', '_per_')},"
            f"counting_sem,observed_sem"]
    for i in range(profile.altitude_m.size):
        rows.append(",".join([
            f"{profile.altitude_m[i]:.1f}",
            f"{profile.altitude_m[i] / FLIGHT_LEVEL_M:.0f}",
            str(int(profile.n_samples[i])),
            f"{profile.live_time_s[i]:.1f}",
            num(profile.count_rate[i]),
            num(dose[i]),
            num(counting[i]),
            num(observed[i]),
        ]))
    return "\n".join(rows) + "\n"


def flight_report_to_json(join, profile, *, unit="nSv/h", floor_m=3000.0,
                          branch_step_m=1000.0, indent=2) -> str:
    """The whole altitude analysis, with the provenance that makes it checkable.

    The match report and the trajectory's source are included for the same
    reason :func:`report_to_json` includes the analysis parameters: a dose
    profile against altitude is a statement about *which* trajectory was
    joined and how well it agreed with the track's own fixes. Without that a
    reader cannot distinguish a validated join from one that was forced.
    """
    fit = fit_scale_height(profile, floor_m=floor_m)
    branches = compare_branches(join, step_m=branch_step_m)
    trajectory = join.trajectory
    low, high = join.altitude_span_m

    def num(value):
        value = float(value)
        return None if not np.isfinite(value) else round(value, 6)

    return json.dumps({
        "generator": "rcta",
        "unit": unit,
        "trajectory": {
            "provider": trajectory.provider,
            "source": trajectory.source,
            "callsign": trajectory.callsign,
            "icao24": trajectory.icao24,
            "altitude_kind": trajectory.altitude_kind,
            "n_waypoints": trajectory.n,
            "start": _iso(trajectory.time_s[0]),
            "end": _iso(trajectory.time_s[-1]),
            "max_gap_s": num(trajectory.max_gap_s),
            "fetched_at": (trajectory.fetched_at.isoformat()
                           if trajectory.fetched_at else None),
            "warnings": list(trajectory.warnings),
        },
        "join": {
            "n_samples": join.track.n,
            "n_joined": join.n_joined,
            "coverage": round(join.coverage, 6),
            "altitude_min_m": num(low),
            "altitude_max_m": num(high),
            "max_gap_s": num(join.max_gap_s),
            "time_offset_s": num(join.time_offset_s),
            "n_outside_trajectory": join.n_before + join.n_after,
            "n_gap_masked": join.n_gap_masked,
            "widest_bridged_gap_s": num(join.widest_bridged_gap_s),
            "widest_bridged_change_m": num(join.widest_bridged_change_m),
            "match": {
                "matched": bool(join.match.matched),
                "forced": bool(join.match.forced),
                "n_compared": join.match.n_compared,
                "residual_median_m": num(join.match.residual_median_m),
                "residual_p95_m": num(join.match.residual_p95_m),
                "tolerance_m": num(join.match.tolerance_m),
                "clock_offset_s": num(join.match.clock_offset_s),
                "residual_at_offset_m": num(join.match.residual_at_offset_m),
            },
        },
        "profile": {
            "step_m": num(profile.step_m),
            "statistic": profile.statistic,
            "branch": profile.branch,
            "excess_scatter": num(profile.excess_scatter),
            "bins": [
                {
                    "altitude_m": num(profile.altitude_m[i]),
                    "n_samples": int(profile.n_samples[i]),
                    "count_rate_cps": num(profile.count_rate[i]),
                    "dose_rate": num(convert_dose_rate(profile.dose_rate[i],
                                                       unit)),
                    "sem": num(convert_dose_rate(profile.dose_sem[i], unit)),
                }
                for i in range(profile.altitude_m.size)
                if profile.n_samples[i] > 0
            ],
        },
        "scale_height": {
            "constrained": bool(fit.constrained),
            "scale_height_m": num(fit.scale_height_m),
            "scale_height_sd_m": num(fit.scale_height_sd_m),
            "doubling_m": num(fit.doubling_m),
            "reference_altitude_m": num(fit.reference_altitude_m),
            "dose_at_reference_nsv_h": num(fit.dose_at_reference),
            "r_squared": num(fit.r_squared),
            "chi2_per_dof": num(fit.chi2_per_dof),
            "n_bins": fit.n_bins,
            "floor_m": num(fit.floor_m),
            "note": fit.note,
        },
        "climb_vs_descent": {
            "n_shared_bins": branches.n_shared,
            "consistent": bool(branches.consistent),
            "threshold_sigma": num(branches.threshold_sigma),
            "max_abs_z": num(branches.max_abs_z),
            "median_abs_difference": num(branches.median_abs_difference),
            "median_relative_difference": num(
                branches.median_relative_difference),
            "note": branches.note,
        },
    }, indent=indent)


def altitude_samples_to_csv(join, *, unit="nSv/h", branch="all") -> str:
    """Every joined sample, one row, for re-analysis in other software.

    This is the export that exists so nobody has to accept this program's
    choices. :func:`altitude_profile_to_csv` gives the binned profile the GUI
    draws, which is already a decision about bin width, statistic and error
    bar; this gives the samples underneath it, so a reader can bin at 250 m
    instead of 500, fit in log space or linear, weight by live time, or plot a
    raw scatter with no binning at all.

    Everything needed to do that independently is present. ``counts`` is the
    estimated counts behind the sample, so Poisson errors can be recomputed
    from scratch rather than trusted -- and ``integration_time_s`` is repeated
    on every row, because that number is an *assumption* (see
    :attr:`rcta.core.track.Track.effective_integration_time_s`) and an export
    that omitted it would let those errors be recomputed wrongly. ``phase``
    labels climb, level and descent so the two traverses can be separated
    without rederiving a vertical rate.

    Only samples that actually received an altitude are written -- rows with a
    blank altitude cannot be plotted against one, and silently including them
    invites a reader to treat them as sea level. How many were dropped, and
    why, is in :meth:`rcta.core.flight.AltitudeJoin.describe` and in the JSON
    report; it is deliberately not a comment line here, because a ``#`` header
    breaks a plain CSV reader.
    """
    from rcta.core.flight import BRANCH_LABELS, FLIGHT_LEVEL_M

    mask = join.branch_mask(branch)
    track = join.track
    dt = track.effective_integration_time_s
    dose = convert_dose_rate(track.dose_rate, unit)
    unit_column = unit.replace("/", "_per_")

    rows = [",".join([
        "index", "time_s", "utc", "lat", "lon", "accuracy_m",
        "altitude_m", "flight_level", "vertical_rate_m_s", "phase",
        f"dose_rate_{unit_column}", "count_rate_cps", "counts",
        "integration_time_s", "distance_along_m", "speed_m_s",
        "valid", "position_valid",
    ])]

    def num(value, fmt=".6g"):
        value = float(value)
        return "" if not np.isfinite(value) else format(value, fmt)

    for i in np.flatnonzero(mask):
        rows.append(",".join([
            str(int(i)),
            f"{track.time_s[i]:.0f}",
            _iso(track.time_s[i]),
            f"{track.lat[i]:.7f}",
            f"{track.lon[i]:.7f}",
            num(track.accuracy_m[i], ".1f"),
            num(join.altitude_m[i], ".2f"),
            num(join.altitude_m[i] / FLIGHT_LEVEL_M, ".0f"),
            num(join.vertical_rate_m_s[i], ".3f"),
            BRANCH_LABELS.get(int(join.branch[i]), "unknown"),
            num(dose[i]),
            num(track.count_rate[i]),
            num(track.count_rate[i] * dt, ".1f"),
            f"{dt:.1f}",
            num(track.distance_along_m[i], ".1f"),
            num(track.speed_m_s[i], ".2f"),
            "1" if track.valid_mask[i] else "0",
            "1" if track.position_mask[i] else "0",
        ]))
    return "\n".join(rows) + "\n"
