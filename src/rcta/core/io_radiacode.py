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


def _number(value) -> float:
    """A finite float, or NaN for anything absent, non-numeric or infinite.

    ``bool`` is refused explicitly: it is an int in Python, and ``true`` read
    as a dose rate of one is exactly the silent coercion this guards against.
    """
    if value is None or isinstance(value, bool):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _nearest(good: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """For each of ``rows``, the closest index in the sorted array ``good``."""
    right = np.clip(np.searchsorted(good, rows), 0, good.size - 1)
    left = np.clip(right - 1, 0, good.size - 1)
    return np.where(np.abs(good[left] - rows) <= np.abs(good[right] - rows),
                    good[left], good[right])


def _read_text(path: Path) -> str:
    """The file as UTF-8 text, or a TrackParseError saying it is not text.

    UTF-8 explicitly: the default is the locale's encoding, which on Windows
    misreads any non-ASCII track title. A decode failure means the file is
    not a track export at all -- an image, an archive, a .rctrk renamed
    from something else -- and says so instead of leaking a codec error.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise TrackParseError(
            f"not a text file (undecodable byte at offset {exc.start}); a "
            "RadiaCode track export is JSON or tab-separated text",
            path=str(path)) from exc


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
        raw = json.loads(_read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackParseError(f"cannot parse JSON: {exc}", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise TrackParseError("top level is not a JSON object", path=str(path))

    markers = raw.get("markers")
    if not isinstance(markers, list) or not markers:
        raise TrackParseError("no markers array", path=str(path), key="markers")

    warnings: list[str] = []
    n = len(markers)
    # A field that is absent or unreadable is NaN, never zero: a zero dose
    # rate or a zero-metre accuracy is a believable reading, and anything that
    # forgot to consult the mask would average it in without complaint.
    time_s = np.full(n, np.nan)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    accuracy = np.full(n, np.nan)
    dose = np.full(n, np.nan)
    count = np.full(n, np.nan)
    valid = np.ones(n, dtype=bool)
    bad: list[int] = []

    for i, marker in enumerate(markers):
        if not isinstance(marker, dict):
            valid[i] = False
            bad.append(i)
            continue
        # Looked up by name, never by position: 81 distinct key orderings
        # appear in a single file. Both rate fields arrive as bare ints in a
        # minority of records, so every field is coerced.
        field = {key: _number(marker.get(key)) for key in (*_REQUIRED, "acc")}
        time_s[i], lat[i], lon[i] = field["date"], field["lat"], field["lon"]
        # uR/h in the file, nSv/h everywhere else in this package.
        dose[i] = field["doseRate"] * RCTRK_DOSE_TO_NSV_H
        count[i] = field["countRate"]
        accuracy[i] = field["acc"]
        if not all(np.isfinite(field[key]) for key in _REQUIRED):
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

    # A masked row still needs a place in time and space so the track can be
    # sorted and drawn, and nothing a broken marker says can be trusted -- not
    # even the fields that parsed, since a zero epoch parses fine and drags
    # the track to the Gulf of Guinea in 1970. So it borrows its neighbour's
    # place in the file wholesale. Its readings are NaN, so the borrowed place
    # carries none.
    if bad:
        bad_rows = np.asarray(bad)
        source = _nearest(np.flatnonzero(valid), bad_rows)
        for column in (time_s, lat, lon):
            column[bad_rows] = column[source]
        dose[bad_rows] = count[bad_rows] = accuracy[bad_rows] = np.nan

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
        text = _read_text(path)
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
            # Only accuracy is optional, and an unknown accuracy is NaN, not a
            # perfect zero-metre fix.
            rows[key].append(parsed.get(key, np.nan))
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
