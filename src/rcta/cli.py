"""Command line interface: ``rcta-cli info | analyze | export | flight``.

The CLI is the report generator. Everything it prints comes from a
``describe()`` on a core object, so the text in a terminal, the text in the
GUI log and the numbers in an exported JSON cannot drift apart.

``flight`` is the one subcommand that can reach the network, and it is the
only one whose result depends on something outside the file it was given. It
therefore has a distinct exit code for each way it can honestly fail -- 3 when
the trajectory could not be fetched, 4 when the trajectory that came back does
not describe this track -- so a script can tell "no data" from "wrong flight",
which a single non-zero status cannot express.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rcta import __version__
from rcta.core.analysis import AnalysisParams, analyze_track
from rcta.core.airports import find_airport, nearest_airport
from rcta.core.anomalies import ScanParams
from rcta.core.background import BackgroundParams
from rcta.core.dose import DOSE_UNITS, convert_dose_rate
from rcta.core.flight import (FLIGHT_LEVEL_M, DEFAULT_ALTITUDE_STEP_M,
                              DEFAULT_FIT_FLOOR_M, DEFAULT_MATCH_TOLERANCE_M,
                              DEFAULT_MAX_GAP_S, FlightMatchError,
                              bin_by_altitude, compare_branches,
                              fit_scale_height, join_altitude,
                              looks_like_flight, parse_flight_number,
                              search_window)
from rcta.core.io_export import (altitude_profile_to_csv,
                                 altitude_samples_to_csv, anomalies_to_csv,
                                 flight_report_to_json, grid_prj,
                                 grid_to_asc, hotspots_to_csv,
                                 report_to_json, track_to_csv,
                                 track_to_geojson, track_to_gpx)
from rcta.core.io_flight import load_trajectory, trajectory_to_csv
from rcta.core.io_radiacode import TrackParseError, load_track
from rcta.core.quality import QualityParams, assess
from rcta.core.spatial import GridParams, interpolate
from rcta.net.flights import (FlightFetchError, fetch_trajectory,
                              load_flight_providers)

RULE = "=" * 78
THIN = "-" * 78

#: Ramp for the ASCII sparkline, lightest to heaviest.
SPARK = " .:-=+*#%@"


def _sparkline(values, width: int = 74) -> str:
    """A coarse ASCII profile of a series, for a terminal with no plots."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return ""
    edges = np.linspace(0, values.size, min(width, values.size) + 1).astype(int)
    # Masked samples are NaN. A bin of nothing but masked samples is drawn
    # as "_" -- no reading, and distinct from the lowest level, which is a
    # space -- rather than crashing int(nan).
    binned = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = values[a:b]
        chunk = chunk[np.isfinite(chunk)]
        if b > a:
            binned.append(float(chunk.max()) if chunk.size else None)
    finite = [v for v in binned if v is not None]
    if not finite:
        return "_" * len(binned)
    lo, hi = min(finite), max(finite)
    span = hi - lo
    def mark(v):
        if v is None:
            return "_"
        if span <= 0:
            return SPARK[len(SPARK) // 2]
        return SPARK[min(int((v - lo) / span * (len(SPARK) - 1)), len(SPARK) - 1)]
    return "".join(mark(v) for v in binned)


def _ascii_map(track, width: int = 70, height: int = 20) -> str:
    """A braille-free ASCII plot of the route.

    ``info`` should show the shape of the route, because that is what tells
    you at a glance whether you loaded the file you meant to.
    """
    mask = track.position_mask
    if mask.sum() < 2:
        return "(no positioned samples)"
    lon, lat = track.lon[mask], track.lat[mask]
    lon_span = max(float(lon.max() - lon.min()), 1e-9)
    lat_span = max(float(lat.max() - lat.min()), 1e-9)

    grid = [[" "] * width for _ in range(height)]
    cx = ((lon - lon.min()) / lon_span * (width - 1)).astype(int)
    # Row 0 is the top of the terminal but the north of the map.
    cy = ((lat.max() - lat) / lat_span * (height - 1)).astype(int)
    for x, y in zip(cx, cy):
        grid[y][x] = "o" if grid[y][x] == " " else "O"
    grid[cy[0]][cx[0]] = "S"
    grid[cy[-1]][cx[-1]] = "E"
    return "\n".join("  " + "".join(row) for row in grid)


def _build_params(args) -> AnalysisParams:
    quality = QualityParams(
        max_accuracy_m=args.max_accuracy,
        gap_threshold_s=args.gap_threshold,
        max_speed_m_s=args.max_speed,
    )
    background = BackgroundParams(
        method=args.background,
        window_s=args.bg_window,
        trim_fraction=args.bg_trim,
        reference_span=tuple(args.reference) if args.reference else None,
    )
    scan = ScanParams(
        scales=args.scales,
        alpha=args.alpha,
        bonferroni=not args.no_bonferroni,
    )
    return AnalysisParams(
        quality=quality, background=background, scan=scan,
        integration_time_s=args.integration_time, unit=args.unit,
        n_boot=args.n_boot,
    )


def cmd_info(args) -> int:
    for i, path in enumerate(args.paths):
        if i:
            print()
        _info(path)
    return 0


def _info(path) -> None:
    # Assessed, so the distance and smoothing printed here are the ones
    # `analyze` and the GUI report: per-leg, with glitched fixes bypassed.
    track = assess(load_track(path))
    print(RULE)
    print(f"  {track.meta.title or Path(path).name}")
    print(RULE)
    print(track.describe())
    print()
    print(THIN)
    print("  dose rate against time")
    print(THIN)
    print("  " + _sparkline(track.dose_rate))
    readings = track.dose_rate[track.valid_mask]
    if readings.size:
        print(f"  {readings.min():.1f} .. {readings.max():.1f} nSv/h")
    print()
    print(THIN)
    print("  route")
    print(THIN)
    print(_ascii_map(track))


def cmd_analyze(args) -> int:
    if args.csv and len(args.paths) > 1:
        raise ValueError("--csv writes one track's anomaly table; "
                         "give a single track with it")
    params = _build_params(args)
    reports = []
    for i, path in enumerate(args.paths):
        track = load_track(path)
        analysis = analyze_track(track, params)
        if i:
            print()
        print(RULE)
        print(f"  {track.meta.title or Path(path).name}")
        print(RULE)
        print(analysis.describe())
        reports.append(report_to_json(analysis))

    if args.json:
        # One report for one track, as before; a JSON array for several, in
        # the order given, so a script can zip it with its own file list.
        text = (reports[0] if len(reports) == 1 else
                "[\n" + ",\n".join(r.rstrip() for r in reports) + "\n]\n")
        Path(args.json).write_text(text, encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.csv:
        Path(args.csv).write_text(
            anomalies_to_csv(analysis.anomalies, unit=args.unit),
            encoding="utf-8")
        print(f"wrote {args.csv}")
    return 0


def cmd_export(args) -> int:
    track = load_track(args.path)
    analysis = analyze_track(track, _build_params(args))

    writers = {
        "csv": lambda: track_to_csv(analysis.track, quality=analysis.quality,
                                    background=analysis.background,
                                    unit=args.unit),
        "geojson": lambda: track_to_geojson(analysis.track,
                                            quality=analysis.quality,
                                            unit=args.unit),
        "gpx": lambda: track_to_gpx(analysis.track, quality=analysis.quality),
        "json": lambda: report_to_json(analysis),
        "anomalies": lambda: anomalies_to_csv(analysis.anomalies, unit=args.unit),
        "hotspots": lambda: hotspots_to_csv(analysis.hotspots, unit=args.unit),
    }
    Path(args.output).write_text(writers[args.format](), encoding="utf-8")
    print(f"wrote {args.output} ({args.format})")
    return 0


def cmd_grid(args) -> int:
    track = load_track(args.path)
    analysis = analyze_track(track, _build_params(args))
    grid = interpolate(
        analysis.track,
        GridParams(value=args.value, bandwidth_m=args.bandwidth,
                   grid_size=args.grid_size),
        background=analysis.background)
    print(grid.describe())

    output = Path(args.output)
    output.write_text(grid_to_asc(grid, unit=args.unit), encoding="utf-8")
    prj = output.with_suffix(".prj")
    prj.write_text(grid_prj(), encoding="utf-8")
    print(f"wrote {output} and {prj.name} (EPSG:3857, {args.unit}"
          f"{' net' if args.value == 'excess' else ''}, blank cells NODATA)")
    return 0


# ---- flight ---------------------------------------------------------------


def _ascii_altitude_plot(profile, *, unit="nSv/h", width=64, height=18) -> str:
    """Dose rate against altitude on a log ordinate, for a terminal.

    Log, because the profile is an exponential and on a linear axis the whole
    troposphere below cruise is squashed into the bottom two rows. On a log
    axis a constant scale height is a straight line, so the eye can check the
    fit that :func:`~rcta.core.flight.fit_scale_height` reports.
    """
    m = profile.occupied & np.isfinite(profile.dose_rate) & (profile.dose_rate > 0)
    if m.sum() < 2:
        return "  (too few occupied altitude bins to plot)"

    x = profile.altitude_m[m] / 1000.0
    y = np.log10(convert_dose_rate(profile.dose_rate[m], unit))
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    x_span = max(x1 - x0, 1e-9)
    y_span = max(y1 - y0, 1e-9)

    grid = [[" "] * width for _ in range(height)]
    for xi, yi in zip(x, y):
        col = int((xi - x0) / x_span * (width - 1))
        row = int((y1 - yi) / y_span * (height - 1))
        grid[row][col] = "*" if grid[row][col] == " " else "#"

    label_w = 9
    lines = []
    for r, row in enumerate(grid):
        value = 10.0 ** (y1 - (y1 - y0) * r / (height - 1))
        tick = f"{value:>{label_w}.3g}" if r in (0, height // 2, height - 1) \
            else " " * label_w
        lines.append(f"  {tick} |" + "".join(row))
    lines.append("  " + " " * label_w + " +" + "-" * width)
    axis = f"{x0:.1f}".ljust(width // 2) + f"{x1:.1f}".rjust(width - width // 2)
    lines.append("  " + " " * label_w + "  " + axis)
    lines.append("  " + " " * label_w + "  "
                 + "altitude (km)".center(width))
    lines.insert(0, f"  dose rate ({unit}), log scale")
    return "\n".join(lines)


def _altitude_table(profile, *, unit="nSv/h", bar_width=28) -> str:
    """One row per occupied bin: the numbers behind the plot."""
    m = profile.occupied
    if not m.any():
        return "  (no occupied bins)"
    dose = convert_dose_rate(profile.dose_rate, unit)
    sem = convert_dose_rate(profile.dose_sem, unit)
    peak = float(np.nanmax(dose[m]))
    lines = [f"  {'altitude':>14}  {'n':>5}  {'cps':>7}  "
             f"{'dose (' + unit + ')':>18}",
             "  " + "-" * 62]
    for i in np.flatnonzero(m)[::-1]:
        filled = int(round(dose[i] / peak * bar_width)) if peak > 0 else 0
        lines.append(
            f"  {profile.altitude_m[i]:>8.0f} m FL{profile.altitude_m[i] / FLIGHT_LEVEL_M:>3.0f}  "
            f"{profile.n_samples[i]:>5d}  {profile.count_rate[i]:>7.2f}  "
            f"{dose[i]:>9.3g} +-{sem[i]:<7.3g} " + "#" * filled)
    return "\n".join(lines)


def _departure_airport(track, args):
    """The airport the track left from, resolved the way the fetch will.

    Duplicated here, ahead of the fetch, only because ``--date`` is a local
    calendar day and the departure airport is what says which zone that is.
    Deliberately silent about failure: a code that names no airport and a
    first fix that is not on an apron are both reported properly by
    :func:`rcta.net.flights.fetch_trajectory`, with the hint that goes with
    them, and repeating that here would print it twice.
    """
    if args.departure:
        return find_airport(args.departure)
    return nearest_airport(float(track.lat[0]), float(track.lon[0]))


def _flight_window(track, args) -> tuple[float, float, list[str]]:
    """The provider search window. See :func:`rcta.core.flight.search_window`."""
    airport = _departure_airport(track, args)
    notes = []
    if args.date and airport is not None and not airport.tz:
        notes.append(f"the table has no timezone for {airport.icao}, so "
                     "--date was read as a UTC calendar day")
    begin, end, warnings = search_window(
        float(track.time_s[0]), float(track.time_s[-1]), args.date,
        tz=airport.tz if airport is not None else "")
    return begin, end, notes + list(warnings)


def _obtain_trajectory(track, args) -> tuple[object, list[str]]:
    """Either read a trajectory from disk or fetch one, and say which."""
    notes = []
    if args.trajectory:
        trajectory = load_trajectory(args.trajectory,
                                     altitude_unit=args.altitude_unit,
                                     callsign=args.callsign or "")
        notes.append(f"trajectory read from {args.trajectory}")
        return trajectory, notes

    if not args.flight and not args.callsign:
        raise ValueError("give --flight DL2618 (with --date if the detector "
                         "clock is wrong), or --trajectory FILE to work with "
                         "no network")

    flight = parse_flight_number(args.callsign or args.flight)
    notes.append(flight.describe())
    begin, end, warnings = _flight_window(track, args)
    notes.extend(warnings)

    providers, default = load_flight_providers()
    if args.provider and args.provider not in providers:
        raise ValueError(f"unknown --provider {args.provider!r}; "
                         f"expected one of {sorted(providers)}")
    provider = providers[args.provider or default]
    trajectory = fetch_trajectory(flight, begin_s=begin, end_s=end,
                                  origin=(float(track.lat[0]),
                                          float(track.lon[0])),
                                  departure_icao=args.departure,
                                  provider=provider, offline=args.offline,
                                  margin_s=args.search_margin)
    if provider.attribution:
        notes.append(provider.attribution)
    return trajectory, notes


def cmd_flight(args) -> int:
    track = assess(load_track(args.path), QualityParams(
        max_accuracy_m=args.max_accuracy, max_speed_m_s=args.max_speed))
    # "is not None", not truthiness: 0 is a value the user typed, and the
    # Track refuses it as analyze does, rather than silently inferring one.
    if args.integration_time is not None:
        track = track.with_mask(integration_time_s=args.integration_time)

    print(RULE)
    print(f"  {track.meta.title or Path(args.path).name}")
    print(RULE)

    verdict = looks_like_flight(track)
    print(verdict.describe())
    if not verdict.is_flight:
        print("  ! proceeding anyway: slow and short flights are real, and "
              "ground speed cannot rule one out")

    try:
        trajectory, notes = _obtain_trajectory(track, args)
    except FlightFetchError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"hint : {exc.hint}", file=sys.stderr)
        return 3

    print()
    print(THIN)
    print("  trajectory")
    print(THIN)
    for note in notes:
        print(f"  {note}")
    print(trajectory.describe())

    if args.save_trajectory:
        Path(args.save_trajectory).write_text(trajectory_to_csv(trajectory),
                                              encoding="utf-8")
        print(f"  wrote {args.save_trajectory}")

    print()
    print(THIN)
    print("  join")
    print(THIN)
    try:
        join = join_altitude(track, trajectory, max_gap_s=args.max_gap,
                             tolerance_m=args.tolerance,
                             time_offset_s=args.time_offset,
                             force_match=args.force_match)
    except FlightMatchError as exc:
        print(exc.report.describe())
        print()
        print("refusing to attribute these altitudes to this track. A wrong "
              "altitude axis\nis worse than none: it is smooth, plausible and "
              "entirely fictional.\nPass --force-match to override.",
              file=sys.stderr)
        return 4
    print(join.describe())

    profile = bin_by_altitude(join, step_m=args.step, statistic=args.statistic,
                              branch=args.branch)
    print()
    print(THIN)
    print(f"  dose rate against altitude  ({args.branch})")
    print(THIN)
    print(profile.describe(args.unit))
    print()
    print(_ascii_altitude_plot(profile, unit=args.unit))
    print()
    print(_altitude_table(profile, unit=args.unit))

    print()
    print(THIN)
    print("  the exponential")
    print(THIN)
    print(fit_scale_height(profile, floor_m=args.fit_floor).describe(args.unit))

    print()
    print(THIN)
    print("  control: does the descent reproduce the climb?")
    print(THIN)
    print(compare_branches(join, step_m=args.branch_step).describe(args.unit))

    if args.csv:
        Path(args.csv).write_text(
            altitude_profile_to_csv(profile, unit=args.unit), encoding="utf-8")
        print(f"\nwrote {args.csv}  ({int(profile.occupied.sum())} occupied "
              f"bins of {args.step:.0f} m)")
    if args.samples_csv:
        text = altitude_samples_to_csv(join, unit=args.unit,
                                       branch=args.branch)
        Path(args.samples_csv).write_text(text, encoding="utf-8")
        print(f"wrote {args.samples_csv}  "
              f"({len(text.splitlines()) - 1} samples, one row each)")

    if args.json:
        Path(args.json).write_text(flight_report_to_json(
            join, profile, unit=args.unit, floor_m=args.fit_floor,
            branch_step_m=args.branch_step), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


#: Every default below is read off the frozen dataclass that consumes it, the
#: way the flight subcommand already reads ``DEFAULT_ALTITUDE_STEP_M`` and the
#: rest. Three front ends -- this parser, the GUI's Analysis dock and a direct
#: core call -- have to agree on what the default answer is, and a literal
#: written out in each is a default that drifts silently the first time one of
#: them is tuned. ``--help`` therefore also cannot go stale.
_QUALITY_DEFAULTS = QualityParams()
_BACKGROUND_DEFAULTS = BackgroundParams()
_SCAN_DEFAULTS = ScanParams()
_ANALYSIS_DEFAULTS = AnalysisParams()


def _scales(text: str) -> tuple[int, ...]:
    """--scales as a tuple, or a usage error naming the expected form."""
    try:
        scales = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated sample counts such as 1,2,4,8; "
            f"got {text!r}") from None
    if any(s <= 0 for s in scales):
        raise argparse.ArgumentTypeError(
            f"sample counts must be positive; got {text!r}")
    return scales


def _add_analysis_flags(parser) -> None:
    parser.add_argument("--background", default=_BACKGROUND_DEFAULTS.method,
                        choices=["rolling", "flank", "reference", "global"])
    parser.add_argument("--bg-window", type=float,
                        default=_BACKGROUND_DEFAULTS.window_s,
                        metavar="S", help="background window in seconds")
    parser.add_argument("--bg-trim", type=float,
                        default=_BACKGROUND_DEFAULTS.trim_fraction, metavar="F",
                        help="fraction trimmed from each tail of the window")
    parser.add_argument("--reference", nargs=2, type=float, metavar=("T0", "T1"),
                        help="unix seconds of a span known to be clean")
    parser.add_argument("--integration-time", type=float, metavar="S",
                        help="detector live time per sample; inferred if omitted")
    parser.add_argument("--scales", type=_scales,
                        default=",".join(str(s) for s in _SCAN_DEFAULTS.scales),
                        help="comma-separated scan window widths in samples")
    parser.add_argument("--alpha", type=float, default=_SCAN_DEFAULTS.alpha)
    parser.add_argument("--no-bonferroni", action="store_true",
                        help="do not correct for the number of windows tested")
    parser.add_argument("--max-accuracy", type=float,
                        default=_QUALITY_DEFAULTS.max_accuracy_m, metavar="M")
    parser.add_argument("--gap-threshold", type=float,
                        default=_QUALITY_DEFAULTS.gap_threshold_s, metavar="S")
    parser.add_argument("--max-speed", type=float,
                        default=_QUALITY_DEFAULTS.max_speed_m_s, metavar="M_S")
    parser.add_argument("--unit", default="nSv/h", choices=sorted(DOSE_UNITS))
    parser.add_argument("--n-boot", type=int, default=_ANALYSIS_DEFAULTS.n_boot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rcta-cli",
        description="Analyse RadiaCode .rctrk GPS radiation tracks.")
    parser.add_argument("--version", action="version",
                        version=f"rcta-cli {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="metadata, sampling and route shape")
    info.add_argument("paths", nargs="+", metavar="path")
    info.set_defaults(func=cmd_info)

    analyze = sub.add_parser("analyze", help="full analysis and detection limits")
    analyze.add_argument("paths", nargs="+", metavar="path")
    analyze.add_argument("--json", metavar="OUT",
                         help="write a JSON report; an array of reports when "
                              "several tracks are given")
    analyze.add_argument("--csv", metavar="OUT", help="write the anomaly table")
    _add_analysis_flags(analyze)
    analyze.set_defaults(func=cmd_analyze)

    export = sub.add_parser("export", help="write the track in another format")
    export.add_argument("path")
    export.add_argument("--format", required=True,
                        choices=["csv", "geojson", "gpx", "json",
                                 "anomalies", "hotspots"])
    export.add_argument("-o", "--output", required=True)
    _add_analysis_flags(export)
    export.set_defaults(func=cmd_export)

    grid = sub.add_parser(
        "grid", help="write the interpolated surface as an ESRI ASCII grid")
    grid.add_argument("path")
    grid.add_argument("-o", "--output", required=True,
                      help="the .asc file; a .prj is written beside it")
    grid.add_argument("--value", default="dose", choices=["dose", "excess"],
                      help="absolute dose rate, or net over the background")
    grid.add_argument("--bandwidth", type=float, metavar="M",
                      help="kernel bandwidth in ground metres; "
                           "default max(3 x median accuracy, 50)")
    grid.add_argument("--grid-size", type=int, default=GridParams().grid_size,
                      metavar="N", help="cells along the long axis")
    _add_analysis_flags(grid)
    grid.set_defaults(func=cmd_grid)

    flight = sub.add_parser(
        "flight", help="import altitude for an airborne track and profile "
                       "dose against it")
    flight.add_argument("path")
    source = flight.add_argument_group(
        "where the altitude comes from",
        "The track format carries none. Either name the flight and let the "
        "provider be queried, or hand over a trajectory that is already on "
        "disk -- which needs no account and no network.")
    source.add_argument("--flight", metavar="DL2618",
                        help="IATA flight number, or an ICAO callsign")
    source.add_argument("--callsign", metavar="DAL2618",
                        help="ICAO callsign, when the airline is not in the "
                             "table or the flight has no IATA number")
    source.add_argument("--date", metavar="YYYY-MM-DD",
                        help="the day of the flight, as a calendar day at the "
                             "departure airport -- the date on the boarding "
                             "pass, not the UTC one. Checked against the "
                             "track's own timestamps; if it disagrees, it "
                             "wins and the difference is reported as a "
                             "detector clock error")
    source.add_argument("--trajectory", metavar="FILE",
                        help="read the trajectory from a CSV or JSON file "
                             "instead of fetching it")
    source.add_argument("--altitude-unit", default="auto",
                        choices=["auto", "m", "ft"],
                        help="unit of the altitude column in --trajectory. "
                             "Most flight trackers export feet; OpenSky "
                             "exports metres")
    source.add_argument("--provider", default=None,
                        help="trajectory provider id (default: opensky)")
    source.add_argument("--offline", action="store_true",
                        help="use the response cache only, never the network")
    source.add_argument("--search-margin", type=float, default=3600.0,
                        metavar="S",
                        help="seconds searched either side of the track's own "
                             "span when looking for the flight")
    source.add_argument("--departure", default="", metavar="ICAO",
                        help="airport the flight left from, when the track "
                             "does not start on an apron. Defaults to the "
                             "airport nearest the first fix")
    source.add_argument("--save-trajectory", metavar="OUT",
                        help="write the trajectory as CSV, to re-use offline")

    joining = flight.add_argument_group("joining, and refusing to")
    joining.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP_S,
                         metavar="S",
                         help="widest waypoint gap altitude may be "
                              "interpolated across")
    joining.add_argument("--tolerance", type=float,
                         default=DEFAULT_MATCH_TOLERANCE_M, metavar="M",
                         help="largest median position residual that still "
                              "counts as the same journey")
    joining.add_argument("--time-offset", type=float, default=0.0, metavar="S",
                         help="seconds added to the track's clock before "
                              "joining, for a detector that is not GPS-locked")
    joining.add_argument("--force-match", action="store_true",
                         help="join even when the positions disagree. The "
                              "caveat is then attached to every output")

    shape = flight.add_argument_group("the profile")
    shape.add_argument("--step", type=float, default=DEFAULT_ALTITUDE_STEP_M,
                       metavar="M", help="altitude bin width")
    shape.add_argument("--statistic", default="mean",
                       choices=["mean", "median", "max", "min"])
    shape.add_argument("--branch", default="all",
                       choices=["all", "climb", "level", "cruise", "descent"],
                       help="restrict the profile to one phase of flight")
    shape.add_argument("--branch-step", type=float, default=1000.0, metavar="M",
                       help="bin width for the climb-versus-descent control")
    shape.add_argument("--fit-floor", type=float, default=DEFAULT_FIT_FLOOR_M,
                       metavar="M",
                       help="exclude bins below this from the exponential fit, "
                            "where terrestrial gamma is not negligible")

    flight.add_argument("--csv", metavar="OUT",
                        help="write the binned altitude profile")
    flight.add_argument("--samples-csv", metavar="OUT",
                        help="write one row per joined sample: altitude, dose "
                             "rate, count rate, estimated counts and phase of "
                             "flight, for binning and plotting elsewhere")
    flight.add_argument("--json", metavar="OUT",
                        help="write the full report, provenance included")
    flight.add_argument("--max-accuracy", type=float,
                        default=_QUALITY_DEFAULTS.max_accuracy_m, metavar="M")
    flight.add_argument("--max-speed", type=float,
                        default=_QUALITY_DEFAULTS.max_speed_m_s, metavar="M_S")
    flight.add_argument("--integration-time", type=float, metavar="S")
    flight.add_argument("--unit", default="nSv/h", choices=sorted(DOSE_UNITS))
    flight.set_defaults(func=cmd_flight)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TrackParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FlightFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"hint : {exc.hint}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
