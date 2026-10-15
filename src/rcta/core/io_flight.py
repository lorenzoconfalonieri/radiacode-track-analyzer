"""Readers for an external flight trajectory, and the unit trap in the middle.

A trajectory can arrive from four places and this module accepts all of them,
because the alternative is a feature that only works when an API key is
present: a cached OpenSky ``/tracks/all`` response, a stitched list of OpenSky
state vectors, a generic JSON list of waypoints, or a CSV exported by hand
from any flight tracker. Dispatch is on the first non-whitespace byte and then
on which keys are present, never on the file extension, for the same reason
:func:`rcta.core.io_radiacode.load_track` does it: the extension is whatever
the download happened to be called.

**Altitude units are the one thing here that can silently ruin the result.**
Every flight tracker aimed at pilots and passengers reports feet; OpenSky and
everything aimed at researchers reports metres. The ratio is 3.28, and it
falls entirely on the x-axis of the only plot this feature exists to draw --
so a mistake does not look like a mistake. It moves a cruise altitude from
FL350 to 35 000 m, quadruples the fitted scale height, and produces a smooth
convincing curve throughout. So:

* a column named ``altitude_ft``, ``baro_altitude`` or ``geo_altitude`` states
  its own unit and is believed;
* a bare ``altitude`` whose maximum exceeds 20 000 is feet, because no
  civil flight cruises at 20 km;
* a bare ``altitude`` below that is genuinely ambiguous -- 8 000 could be a
  light aircraft in feet or a turboprop in metres -- and is read as metres
  *with a warning naming the override*, rather than guessed at silently.

The other tolerances are inherited from the track loader, and for the same
reasons. Timestamps arrive as epoch seconds, epoch milliseconds or ISO 8601.
Nulls are kept as NaN rather than zeroed, because a waypoint with no
barometric altitude is not a waypoint at sea level. Recoverable problems
append to :attr:`rcta.core.flight.Trajectory.warnings`; only a file that
cannot be read at all raises.
"""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rcta.core.flight import M_PER_FT, Trajectory
from rcta.core.io_radiacode import TrackParseError

__all__ = ["TrajectoryParseError", "load_trajectory", "parse_opensky_states",
           "parse_opensky_track", "trajectory_to_csv"]


class TrajectoryParseError(TrackParseError):
    """Raised when a trajectory file cannot be interpreted at all.

    Subclasses :class:`~rcta.core.io_radiacode.TrackParseError` so the CLI's
    single ``except`` clause covers both kinds of input file and reports them
    with the same wording.
    """


#: Accepted column names, lower-cased with punctuation stripped. Order matters
#: only within a role: the first match wins, so an explicit ``altitude_m``
#: beats a bare ``altitude`` in a file that carries both.
_FIELDS = {
    "time_s": ("time", "timestamp", "epoch", "unixtime", "date", "datetime",
               "utc", "lastcontact", "timeposition", "postime"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "lng", "long", "longitude"),
    "altitude_m": ("altitudem", "altm", "baroaltitude", "barometricaltitude",
                   "pressurealtitude", "geoaltitude", "geometricaltitude",
                   "altitudeft", "altft", "altitudefeet", "altitude", "alt",
                   "height"),
    "on_ground": ("onground", "ground", "isonground"),
}

#: Altitude column names that declare themselves to be in feet.
_FEET_NAMES = ("altitudeft", "altft", "altitudefeet", "feet")

#: Altitude column names that declare themselves to be in metres. OpenSky's
#: ``baro_altitude`` and ``geo_altitude`` are documented as metres.
_METRE_NAMES = ("altitudem", "altm", "baroaltitude", "barometricaltitude",
                "geoaltitude", "geometricaltitude", "pressurealtitude")

#: Above this, a unit-less altitude cannot be metres: 20 km is twice the
#: cruise altitude of any airliner and above the ceiling of every civil type.
_FEET_IF_ABOVE_M = 20000.0


def _norm(name: str) -> str:
    """Lower-case and strip everything that is not a letter or a digit."""
    return "".join(c for c in str(name).lower() if c.isalnum())


def _parse_time(value) -> float:
    """Unix seconds from an epoch number, epoch milliseconds, or ISO 8601."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number / 1000.0 if number > 1e11 else number
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return number / 1000.0 if number > 1e11 else number

    cleaned = text.replace("/", "-").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                    "%Y-%m-%d %H:%M:%S.%f"):
            try:
                parsed = datetime.strptime(cleaned.replace("T", " "), fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"unrecognised timestamp {value!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _to_float(value) -> float:
    """A float, with anything unreadable or absent becoming NaN.

    NaN rather than an exception because a null altitude is normal in every
    one of these sources -- an aircraft on the ground reports none -- and
    because a trajectory is useful with holes in it. The holes are then masked
    by :func:`rcta.core.flight.join_altitude` instead of being interpolated
    across.
    """
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _resolve_altitude(values: np.ndarray, column: str, unit: str
                      ) -> tuple[np.ndarray, list[str]]:
    """Convert an altitude column to metres, deciding the unit if asked to.

    Returns the converted array and any warnings, which are threaded onto the
    Trajectory rather than logged, so the CLI and the GUI both replay them.
    """
    unit = str(unit).lower()
    if unit in ("m", "metre", "meter", "metres", "meters"):
        return values, []
    if unit in ("ft", "foot", "feet"):
        return values * M_PER_FT, []
    if unit != "auto":
        raise ValueError(f"unknown altitude unit {unit!r}; expected "
                         "'auto', 'm' or 'ft'")

    key = _norm(column)
    if any(name in key for name in _FEET_NAMES):
        return values * M_PER_FT, []
    if any(name in key for name in _METRE_NAMES):
        return values, []

    finite = values[np.isfinite(values)]
    peak = float(finite.max()) if finite.size else 0.0
    if peak > _FEET_IF_ABOVE_M:
        return values * M_PER_FT, [
            f"column {column!r} does not name its unit; a maximum of "
            f"{peak:.0f} cannot be metres, so it was read as feet"]
    return values, [
        f"column {column!r} does not name its unit and its maximum of "
        f"{peak:.0f} is plausible in either; it was read as METRES. If the "
        "export is in feet -- which most flight trackers are -- the altitude "
        "axis is wrong by 3.28x. Pass altitude_unit='ft' to be sure."]


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def load_trajectory(path, *, altitude_unit: str = "auto", callsign: str = "",
                    icao24: str = "") -> Trajectory:
    """Read a trajectory from disk, dispatching on content not extension."""
    path = Path(path)
    try:
        # utf-8-sig: a byte-order mark, which Windows editors and Excel
        # write, would otherwise hide the leading "{" of a JSON file and
        # send it to the delimited parser as a one-column table.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise TrajectoryParseError(f"cannot read trajectory: {exc}",
                                   path=str(path)) from exc
    except UnicodeDecodeError as exc:
        raise TrajectoryParseError(
            f"not a text file (undecodable byte at offset {exc.start}); a "
            "trajectory is JSON, CSV or tab-separated text",
            path=str(path)) from exc
    head = text.lstrip()
    if not head:
        raise TrajectoryParseError("file is empty", path=str(path))

    if head[:1] in "{[":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TrajectoryParseError(f"cannot parse JSON: {exc}",
                                       path=str(path)) from exc
        return _from_json(raw, path=path, altitude_unit=altitude_unit,
                          callsign=callsign, icao24=icao24)
    return _from_delimited(text, path=path, altitude_unit=altitude_unit,
                           callsign=callsign, icao24=icao24)


def _from_json(raw, *, path, altitude_unit, callsign, icao24) -> Trajectory:
    if isinstance(raw, dict):
        if "path" in raw:
            return parse_opensky_track(raw, source=str(path))
        if "states" in raw:
            return parse_opensky_states(raw, source=str(path),
                                        callsign=callsign, icao24=icao24)
        for key in ("waypoints", "positions", "track", "trajectory", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raise TrajectoryParseError(
                "JSON object carries no recognised waypoint list; expected a "
                "'path' (OpenSky track), 'states' (OpenSky state vectors), or "
                "a 'waypoints'/'positions'/'data' array", path=str(path))
    if not isinstance(raw, list):
        raise TrajectoryParseError("expected a JSON array of waypoints",
                                   path=str(path))
    if not raw:
        raise TrajectoryParseError("waypoint list is empty", path=str(path))
    if not isinstance(raw[0], dict):
        raise TrajectoryParseError(
            "waypoints must be objects with named fields; a bare array of "
            "numbers does not say which column is which", path=str(path))

    columns = {_norm(k): k for k in raw[0]}
    chosen = _choose_columns(columns, path=path)
    rows = [{role: entry.get(name) for role, name in chosen.items()}
            for entry in raw]
    return _assemble(rows, chosen, path=path, altitude_unit=altitude_unit,
                     callsign=callsign or str(raw[0].get("callsign", "")).strip(),
                     icao24=icao24 or str(raw[0].get("icao24", "")).strip(),
                     provider="file")


def _from_delimited(text, *, path, altitude_unit, callsign, icao24) -> Trajectory:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise TrajectoryParseError("no header row found", path=str(path))

    columns = {_norm(k): k for k in reader.fieldnames if k}
    chosen = _choose_columns(columns, path=path)
    rows = [{role: entry.get(name) for role, name in chosen.items()}
            for entry in reader]
    if not rows:
        raise TrajectoryParseError("header row but no data", path=str(path))
    return _assemble(rows, chosen, path=path, altitude_unit=altitude_unit,
                     callsign=callsign, icao24=icao24, provider="file")


def _choose_columns(columns: dict[str, str], *, path) -> dict[str, str]:
    """Map each role to the first accepted column name that is present."""
    chosen: dict[str, str] = {}
    for role, accepted in _FIELDS.items():
        for candidate in accepted:
            if candidate in columns:
                chosen[role] = columns[candidate]
                break
    missing = [role for role in ("time_s", "lat", "lon", "altitude_m")
               if role not in chosen]
    if missing:
        raise TrajectoryParseError(
            f"trajectory is missing a column for {', '.join(missing)}; found "
            f"{sorted(columns.values())}", path=str(path))
    return chosen


def _assemble(rows, chosen, *, path, altitude_unit, callsign, icao24,
              provider) -> Trajectory:
    time_s, lat, lon, alt, ground = [], [], [], [], []
    warnings: list[str] = []
    n_bad_time = n_bad_position = 0
    for row in rows:
        try:
            when = _parse_time(row["time_s"])
        except (ValueError, TypeError):
            n_bad_time += 1
            continue
        latitude, longitude = _to_float(row["lat"]), _to_float(row["lon"])
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            n_bad_position += 1
        time_s.append(when)
        lat.append(latitude)
        lon.append(longitude)
        alt.append(_to_float(row["altitude_m"]))
        ground.append(_truthy(row.get("on_ground")))

    if not time_s:
        raise TrajectoryParseError("no waypoint carried a usable timestamp",
                                   path=str(path))
    if n_bad_time:
        warnings.append(f"{n_bad_time} waypoint(s) had an unreadable "
                        "timestamp and were dropped")
    if n_bad_position:
        # Kept, not dropped: a waypoint with an altitude but no position still
        # carries the altitude, and it is only the position check that has to
        # skip it.
        warnings.append(f"{n_bad_position} waypoint(s) carry no position; "
                        "their altitude is still used, but they cannot "
                        "contribute to the position match")

    altitude = np.asarray(alt, dtype=float)
    altitude, unit_warnings = _resolve_altitude(
        altitude, chosen["altitude_m"], altitude_unit)
    warnings.extend(unit_warnings)

    n_missing = int((~np.isfinite(altitude)).sum())
    if n_missing:
        warnings.append(f"{n_missing} waypoint(s) carry no altitude and were "
                        "left as NaN rather than zeroed")

    kind = _altitude_kind(chosen["altitude_m"])
    return Trajectory(
        time_s=np.asarray(time_s, dtype=float),
        lat=np.asarray(lat, dtype=float), lon=np.asarray(lon, dtype=float),
        altitude_m=altitude, altitude_kind=kind,
        on_ground=np.asarray(ground, dtype=bool),
        callsign=callsign, icao24=icao24, provider=provider,
        source=str(path), warnings=tuple(warnings))


def _altitude_kind(column: str) -> str:
    key = _norm(column)
    if "baro" in key or "pressure" in key:
        return "barometric"
    if "geo" in key:
        return "geometric"
    return "unknown"


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


# ---------------------------------------------------------------------------
# the OpenSky payloads
# ---------------------------------------------------------------------------

def parse_opensky_track(raw, *, source: str = "") -> Trajectory:
    """Parse an OpenSky ``/tracks/all`` response.

    The path rows are positional -- ``[time, lat, lon, baro_altitude,
    true_track, on_ground]`` -- so unlike every other reader here this one has
    no choice but to index them, and it checks the row width instead of
    trusting it.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("path"), list):
        raise TrajectoryParseError("not an OpenSky track response",
                                   path=source)
    path_rows = raw["path"]
    if not path_rows:
        raise TrajectoryParseError(
            "OpenSky returned a track with an empty path: it knows of the "
            "aircraft but has no waypoints for that time", path=source)

    time_s, lat, lon, alt, ground = [], [], [], [], []
    short = 0
    for row in path_rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            short += 1
            continue
        try:
            when = _parse_time(row[0])
        except (ValueError, TypeError):
            short += 1
            continue
        time_s.append(when)
        lat.append(_to_float(row[1]))
        lon.append(_to_float(row[2]))
        alt.append(_to_float(row[3]))
        ground.append(_truthy(row[5]) if len(row) > 5 else False)

    if not time_s:
        raise TrajectoryParseError("no usable waypoint in the OpenSky path",
                                   path=source)

    warnings = []
    if short:
        warnings.append(f"{short} malformed waypoint row(s) skipped")
    altitude = np.asarray(alt, dtype=float)
    n_missing = int((~np.isfinite(altitude)).sum())
    if n_missing:
        warnings.append(f"{n_missing} waypoint(s) carry no barometric "
                        "altitude and were left as NaN")

    return Trajectory(
        time_s=np.asarray(time_s, dtype=float),
        lat=np.asarray(lat, dtype=float), lon=np.asarray(lon, dtype=float),
        altitude_m=altitude, altitude_kind="barometric",
        on_ground=np.asarray(ground, dtype=bool),
        callsign=str(raw.get("callsign") or "").strip(),
        icao24=str(raw.get("icao24") or "").strip(),
        provider="opensky", source=source, warnings=tuple(warnings))


#: Column offsets in an OpenSky state vector, from the API documentation.
_STATE_ICAO24, _STATE_CALLSIGN = 0, 1
_STATE_LON, _STATE_LAT, _STATE_BARO = 5, 6, 7
_STATE_ON_GROUND, _STATE_GEO = 8, 13
_STATE_TIME = 4                                  # last_contact


def parse_opensky_states(raw, *, source: str = "", callsign: str = "",
                         icao24: str = "", prefer: str = "barometric"
                         ) -> Trajectory:
    """Parse OpenSky ``/states/all`` rows into one aircraft's trajectory.

    A single ``/states/all`` response is one instant for many aircraft, so a
    file of them is only a trajectory once it has been filtered to one
    airframe. Filtering is mandatory when more than one appears: silently
    keeping the first row of each instant would interleave several aircraft
    into a path that never existed.
    """
    rows = raw.get("states") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise TrajectoryParseError("no state vectors found", path=source)

    wanted_call = callsign.strip().upper()
    wanted_icao = icao24.strip().lower()
    kept, seen = [], set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 9:
            continue
        row_icao = str(row[_STATE_ICAO24] or "").strip().lower()
        row_call = str(row[_STATE_CALLSIGN] or "").strip().upper()
        seen.add(row_icao or row_call)
        if wanted_icao and row_icao != wanted_icao:
            continue
        if wanted_call and row_call != wanted_call:
            continue
        kept.append(row)

    if not kept:
        raise TrajectoryParseError(
            f"no state vector matched callsign {callsign!r} / icao24 "
            f"{icao24!r}; the file holds {len(seen)} distinct aircraft",
            path=source)
    if not (wanted_call or wanted_icao) and len(seen) > 1:
        raise TrajectoryParseError(
            f"the file holds {len(seen)} distinct aircraft; pass a callsign or "
            "icao24 to say which one is the trajectory", path=source)

    baro = prefer == "barometric"
    index = _STATE_BARO if baro else _STATE_GEO
    time_s, lat, lon, alt, ground = [], [], [], [], []
    for row in kept:
        when = row[_STATE_TIME] if row[_STATE_TIME] is not None else row[3]
        try:
            time_s.append(_parse_time(when))
        except (ValueError, TypeError):
            continue
        lat.append(_to_float(row[_STATE_LAT]))
        lon.append(_to_float(row[_STATE_LON]))
        alt.append(_to_float(row[index] if len(row) > index else None))
        ground.append(_truthy(row[_STATE_ON_GROUND]))

    if not time_s:
        raise TrajectoryParseError("no state vector carried a usable time",
                                   path=source)

    first = kept[0]
    return Trajectory(
        time_s=np.asarray(time_s, dtype=float),
        lat=np.asarray(lat, dtype=float), lon=np.asarray(lon, dtype=float),
        altitude_m=np.asarray(alt, dtype=float),
        altitude_kind="barometric" if baro else "geometric",
        on_ground=np.asarray(ground, dtype=bool),
        callsign=callsign or str(first[_STATE_CALLSIGN] or "").strip(),
        icao24=icao24 or str(first[_STATE_ICAO24] or "").strip(),
        provider="opensky", source=source)


# ---------------------------------------------------------------------------
# writing one back out
# ---------------------------------------------------------------------------

def trajectory_to_csv(trajectory: Trajectory) -> str:
    """Serialise a trajectory so a fetched one can be kept and re-used.

    The altitude column is named ``altitude_m`` rather than ``altitude`` on
    purpose: the file states its own unit, so re-reading it can never land in
    the ambiguous branch of :func:`_resolve_altitude`. Writing a column that
    would need guessing on the way back in is a bug waiting to happen.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["time_s", "utc", "lat", "lon", "altitude_m", "on_ground"])
    for i in range(trajectory.n):
        when = datetime.fromtimestamp(float(trajectory.time_s[i]),
                                      tz=timezone.utc)
        altitude = trajectory.altitude_m[i]
        writer.writerow([
            f"{trajectory.time_s[i]:.0f}",
            when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            f"{trajectory.lat[i]:.6f}", f"{trajectory.lon[i]:.6f}",
            # Two decimals, not one: OpenSky reports barometric altitude to
            # the centimetre and a lossy round-trip through this writer would
            # mean the cached CSV and the response it came from disagree.
            "" if not math.isfinite(altitude) else f"{altitude:.2f}",
            "1" if trajectory.on_ground[i] else "0",
        ])
    return out.getvalue()
