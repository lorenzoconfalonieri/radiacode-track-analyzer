"""Readers for the two ``.rctrk`` dialects the RadiaCode app writes.

The parsing doctrine is the one that survives contact with real exports:
recoverable problems append a string to :attr:`TrackMeta.warnings`, which the
GUI replays into its log dock and the CLI prints with a leading ``!``. Only an
unrecoverable problem raises :class:`TrackParseError`. A sample that cannot be
believed is *masked*, never deleted, so the sample count in any summary always
matches the sample count in the file and the user can see what was excluded
and why.

Four tolerances below are not optional refinements; each was measured on a
real export and each silently corrupts the analysis if omitted.

*Key order inside a marker is randomised.* The bundled 2964-sample file
contains 81 distinct key orderings, so every field is looked up by name and
nothing is ever read positionally.

*Numeric fields are sometimes bare integers.* ``doseRate`` is an int in 25
records and ``countRate`` in 32, so both are coerced rather than trusted to
arrive as floats.

*The marker list is not in time order.* Its first entry is 5363 s after the
recorded start. The loader counts the inversions and warns, and
:class:`~rcta.core.track.Track` sorts unconditionally, because file-order
geometry overstates the distance nine-hundred-fold.

*There are two dialects.* One is a single pretty-printed JSON object; the
other is tab-separated with a ``Track:`` preamble. Dispatch is on the first
non-whitespace byte, not on the file extension, because both use ``.rctrk``.

*The stored dose unit is microroentgen per hour, and the file never says so.*
This was established against a real walking log rather than inferred: the app
displays about 0.07 uSv/h outdoors while the file records 7.19, so the stored
number is uR/h and one uR/h is ten nSv/h. Reading it as nSv/h -- which an
earlier version of this parser did -- puts ordinary outdoor background ten
times below the lowest value physically possible, and puts an aircraft at
cruise altitude *below* ground level. Everything downstream is canonically
nSv/h, so the conversion happens exactly once, here, on load.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rcta.core.track import Period, Track, TrackMeta

__all__ = ["RCTRK_DOSE_TO_NSV_H", "RCTRK_DOSE_UNIT", "TrackParseError",
           "load_rctrk_json", "load_rctrk_tsv", "load_track"]

#: Fields every marker must supply for the sample to be usable.
_REQUIRED = ("date", "lat", "lon", "doseRate", "countRate")

#: The file stores microroentgen per hour; the package works in nSv/h.
#: 1 R = 10 mSv for gamma by the conventional survey-meter equivalence, so
#: 1 uR/h = 10 nSv/h. Confirmed against a real log and the app's own display.
RCTRK_DOSE_UNIT = "uR/h"
RCTRK_DOSE_TO_NSV_H = 10.0

#: Header names in the tab-separated dialect, lower-cased, mapped to columns.
_TSV_FIELDS = {
    "timestamp": "time_s",
    "latitude": "lat",
    "longitude": "lon",
    "accuracy": "accuracy_m",
    "doserate": "dose_rate",
    "countrate": "count_rate",
}


class TrackParseError(Exception):
    """Raised when a track file cannot be interpreted at all.

    Carries the location inside the document -- a JSON key or a line number --
    so the message points at the problem instead of surfacing as a bare
    ``KeyError`` from three frames down.
    """

    def __init__(self, message: str, *, path: str = "", key: str = "",
                 line: int = 0) -> None:
        detail = "".join(filter(None, [
            f" in {path}" if path else "",
            f" at key {key!r}" if key else "",
            f" on line {line}" if line else "",
        ]))
        super().__init__(f"{message}{detail}")
        self.path = path
        self.key = key
        self.line = line


def _utc(seconds) -> datetime:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)


def load_track(path) -> Track:
    """Read either ``.rctrk`` dialect, dispatching on content not extension."""
    path = Path(path)
    try:
        head = path.read_bytes()[:4096].lstrip()
    except OSError as exc:
        raise TrackParseError(f"cannot read track: {exc}", path=str(path)) from exc
    if not head:
        raise TrackParseError("file is empty", path=str(path))
    return (load_rctrk_json(path) if head[:1] == b"{" else load_rctrk_tsv(path))


def load_rctrk_json(path) -> Track:
    """Read the RadiaCode app's JSON track export."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackParseError(f"cannot parse JSON: {exc}", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise TrackParseError("top level is not a JSON object", path=str(path))

    markers = raw.get("markers")
    if not isinstance(markers, list) or not markers:
        raise TrackParseError("no markers array", path=str(path), key="markers")

    warnings: list[str] = []
    n = len(markers)
    time_s = np.zeros(n)
    lat = np.zeros(n)
    lon = np.zeros(n)
    accuracy = np.zeros(n)
    dose = np.zeros(n)
    count = np.zeros(n)
    valid = np.ones(n, dtype=bool)
    bad: list[int] = []

    for i, marker in enumerate(markers):
        if not isinstance(marker, dict) or any(k not in marker for k in _REQUIRED):
            valid[i] = False
            bad.append(i)
            continue
        try:
            # Looked up by name, never by position: 81 distinct key orderings
            # appear in a single file.
            time_s[i] = float(marker["date"])
            lat[i] = float(marker["lat"])
            lon[i] = float(marker["lon"])
            # Both rate fields arrive as bare ints in a minority of records.
            # uR/h in the file, nSv/h everywhere else in this package.
            dose[i] = float(marker["doseRate"]) * RCTRK_DOSE_TO_NSV_H
            count[i] = float(marker["countRate"])
            accuracy[i] = float(marker.get("acc", 0.0))
        except (TypeError, ValueError):
            valid[i] = False
            bad.append(i)

    if bad:
        shown = ", ".join(str(i) for i in bad[:8])
        more = f" and {len(bad) - 8} more" if len(bad) > 8 else ""
        warnings.append(
            f"{len(bad)} marker(s) missing or unparseable fields, masked not "
            f"dropped (indices {shown}{more})")
    if not valid.any():
        raise TrackParseError("no usable markers", path=str(path), key="markers")

    # Fill masked rows with a neighbour's timestamp so sorting stays stable and
    # a zero epoch does not drag the track to the Gulf of Guinea in 1970.
    if bad:
        good = np.flatnonzero(valid)
        for i in bad:
            j = good[np.argmin(np.abs(good - i))]
            time_s[i], lat[i], lon[i] = time_s[j], lat[j], lon[j]

    inversions = int((np.diff(time_s[valid]) < 0).sum())
    if inversions:
        warnings.append(
            f"markers were not in time order ({inversions} inversions); "
            "sorted on load")

    periods = []
    for entry in raw.get("periods", []) or []:
        try:
            periods.append(Period(distance_m=float(entry["distance"]),
                                  start=_utc(entry["start"]),
                                  end=_utc(entry["end"])))
        except (KeyError, TypeError, ValueError):
            warnings.append("a period entry was malformed and was ignored")

    devices = tuple(str(d) for d in (raw.get("devices") or []))
    start = raw.get("start")
    meta = TrackMeta(
        title=str(raw.get("title", "")),
        devices=devices,
        start_time=_utc(start) if start is not None else None,
        source_path=str(path),
        source_format="rctrk-json",
        periods=tuple(periods),
        spectrum_valid=bool(raw.get("sv", False)),
        warnings=tuple(warnings),
    )
    return Track(time_s=time_s, lat=lat, lon=lon, accuracy_m=accuracy,
                 dose_rate=dose, count_rate=count, meta=meta, valid_mask=valid)


def load_rctrk_tsv(path) -> Track:
    """Read the RadiaCode app's tab-separated track export.

    The dialect opens with a ``Track: <name>`` line, then a header row. Column
    *names* are matched case-insensitively; the order is never assumed, for the
    same reason the JSON keys are not. This dialect carries no ``periods``, so
    the odometer cross-check is simply unavailable -- reported as absent rather
    than faked.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise TrackParseError(f"cannot read track: {exc}", path=str(path)) from exc

    lines = text.splitlines()
    warnings: list[str] = []
    title = ""
    offset = 0
    if lines and lines[0].lower().startswith("track:"):
        title = lines[0].split(":", 1)[1].strip()
        offset = 1
    if len(lines) <= offset:
        raise TrackParseError("no header row", path=str(path), line=offset + 1)

    reader = csv.reader(io.StringIO("\n".join(lines[offset:])), delimiter="\t")
    try:
        header = next(reader)
    except StopIteration:
        raise TrackParseError("no header row", path=str(path),
                              line=offset + 1) from None

    index = {}
    for position, name in enumerate(header):
        key = _TSV_FIELDS.get(name.strip().lower().replace(" ", ""))
        if key:
            index[key] = position
    missing = {"time_s", "lat", "lon", "dose_rate", "count_rate"} - set(index)
    if missing:
        raise TrackParseError(f"header lacks {sorted(missing)}", path=str(path),
                              line=offset + 1)

    rows: dict[str, list[float]] = {k: [] for k in
                                    ("time_s", "lat", "lon", "accuracy_m",
                                     "dose_rate", "count_rate")}
    valid: list[bool] = []
    bad = 0
    for lineno, row in enumerate(reader, start=offset + 2):
        if not row or all(not cell.strip() for cell in row):
            continue
        parsed = {}
        ok = True
        for key, position in index.items():
            try:
                cell = row[position].strip()
                value = (_parse_timestamp(cell) if key == "time_s"
                         else float(cell.replace(",", ".")))
                if key == "dose_rate":
                    value *= RCTRK_DOSE_TO_NSV_H
                parsed[key] = value
            except (IndexError, ValueError):
                ok = False
                break
        if not ok:
            bad += 1
            continue
        for key in rows:
            rows[key].append(parsed.get(key, 0.0))
        valid.append(True)

    if bad:
        warnings.append(f"{bad} row(s) could not be parsed and were skipped")
    if not valid:
        raise TrackParseError("no usable rows", path=str(path))

    time_s = np.asarray(rows["time_s"], dtype=float)
    inversions = int((np.diff(time_s) < 0).sum())
    if inversions:
        warnings.append(
            f"rows were not in time order ({inversions} inversions); "
            "sorted on load")

    # The uR/h assumption was verified against a real JSON export. No TSV
    # export was available to check, so say so rather than assume silently.
    warnings.append(
        f"dose rate assumed to be {RCTRK_DOSE_UNIT} and scaled to nSv/h; "
        "this was verified for the JSON dialect but not for this one")

    meta = TrackMeta(
        title=title,
        start_time=_utc(time_s.min()),
        source_path=str(path),
        source_format="rctrk-tsv",
        warnings=tuple(warnings),
    )
    return Track(time_s=time_s,
                 lat=np.asarray(rows["lat"], dtype=float),
                 lon=np.asarray(rows["lon"], dtype=float),
                 accuracy_m=np.asarray(rows["accuracy_m"], dtype=float),
                 dose_rate=np.asarray(rows["dose_rate"], dtype=float),
                 count_rate=np.asarray(rows["count_rate"], dtype=float),
                 meta=meta, valid_mask=np.asarray(valid, dtype=bool))


def _parse_timestamp(cell: str) -> float:
    """Unix seconds from either an epoch number or an ISO-ish datetime.

    The app has shipped both, and a bare integer is ambiguous only in the
    sense that milliseconds would put the track 50 000 years in the future --
    which is exactly the check applied here.
    """
    try:
        value = float(cell)
    except ValueError:
        pass
    else:
        # Anything past the year 5000 in seconds is milliseconds.
        return value / 1000.0 if value > 1e11 else value

    text = cell.replace("/", "-").replace("T", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(
                tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp {cell!r}")
