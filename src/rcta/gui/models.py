"""Table models for anomalies, hotspots and legs.

Each model is driven by a module-level list of ``(header, accessor)`` pairs, so
adding a column is one line and the CSV exporters in
:mod:`rcta.core.io_export` can mirror the same order.

Excluded candidates are shown, not hidden. A reviewer judging whether the
trend flagging was right needs to see what was flagged and why, and a table
that silently drops everything below threshold makes that impossible.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from PySide6 import QtCore, QtGui

from rcta.core.dose import format_dose_rate
from rcta.gui import theme

Qt = QtCore.Qt

__all__ = ["AnomalyTableModel", "HotspotTableModel", "LegTableModel"]


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime(
        "%H:%M:%S")


def _dose(value: float, unit: str, *, signed: bool = False) -> str:
    """A dose rate in the display unit, without the unit (it is in the header)."""
    if not math.isfinite(value):
        return "-"
    text = format_dose_rate(value, unit).rsplit(" ", 1)[0]
    return f"+{text}" if signed and value >= 0 else text


def _sort_key(text: str):
    """Numbers sort as numbers ("9.2" before "10.5"); "-" and words last."""
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text)


def _mean(values) -> float:
    return float(values.mean()) if values.size else float("nan")


class _BaseModel(QtCore.QAbstractTableModel):
    #: (header, accessor(row, unit)); "{unit}" in a header is the display unit.
    COLUMNS: list = []

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list = []
        self._unit = "nSv/h"
        self._font = theme.mono_font()

    def set_unit(self, unit: str) -> None:
        self.beginResetModel()
        self._unit = unit
        self.endResetModel()

    def set_rows(self, rows) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_object(self, row: int):
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section][0].format(unit=self._unit)
        return section + 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        obj = self._rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[index.column()][1](obj, self._unit)
        if role == Qt.ItemDataRole.UserRole:
            return _sort_key(self.COLUMNS[index.column()][1](obj, self._unit))
        if role == Qt.ItemDataRole.FontRole:
            return self._font
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() > 0:
            return int(Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter)
        return self._decoration(obj, index, role)

    def _decoration(self, obj, index, role):
        return None


class AnomalyTableModel(_BaseModel):
    """Every scan candidate, significant or not, with its exclusion reason."""

    COLUMNS = [
        ("time", lambda a, u: _iso(a.time_s)),
        ("dur (s)", lambda a, u: f"{a.duration_s:.0f}"),
        ("lat", lambda a, u: f"{a.lat:.5f}"),
        ("lon", lambda a, u: f"{a.lon:.5f}"),
        ("net (cps)", lambda a, u: f"{a.net_count_rate:+.2f}"),
        ("net ({unit})", lambda a, u: _dose(a.net_dose_rate, u, signed=True)),
        ("peak ({unit})", lambda a, u: _dose(a.peak_dose_rate, u)),
        ("sigma", lambda a, u: f"{a.significance:.2f}"),
        ("L_C", lambda a, u: f"{a.limits.critical_level:.0f}"),
        ("MDA (cps)", lambda a, u: f"{a.limits.mda_cps:.2f}"),
        ("flags", lambda a, u: " ".join(a.flags)),
    ]

    def _decoration(self, anomaly, index, role):
        if role == Qt.ItemDataRole.ForegroundRole and not anomaly.is_significant:
            return QtGui.QBrush(QtGui.QColor(theme.FLAG_COLORS["trend"]))
        if role == Qt.ItemDataRole.ToolTipRole:
            if anomaly.is_significant:
                return "above the critical level and the corrected threshold"
            if "trend" in anomaly.flags:
                return ("a local straight line already explains this excess; "
                        "it is a trend, not a localised source")
            return (f"below the critical level "
                    f"({anomaly.limits.critical_level:.0f} counts) or the "
                    f"{anomaly.threshold_sigma:.2f} sigma threshold")
        return None


class HotspotTableModel(_BaseModel):
    COLUMNS = [
        ("lat", lambda h, u: f"{h.lat:.5f}"),
        ("lon", lambda h, u: f"{h.lon:.5f}"),
        ("r (m)", lambda h, u: f"{h.radius_m:.0f}"),
        ("samples", lambda h, u: str(h.n_samples)),
        ("passes", lambda h, u: str(h.n_passes)),
        ("peak ({unit})", lambda h, u: _dose(h.peak_dose_rate, u)),
        ("net counts", lambda h, u: f"{h.net_counts:.0f}"),
        ("sigma", lambda h, u: f"{h.significance:.2f}"),
        ("score", lambda h, u: f"{h.score:.0f}"),
        ("evidence", lambda h, u: h.evidence),
    ]

    def _decoration(self, hotspot, index, role):
        if role == Qt.ItemDataRole.ForegroundRole:
            colour = theme.EVIDENCE_COLORS.get(hotspot.evidence)
            if colour:
                return QtGui.QBrush(QtGui.QColor(colour))
        if role == Qt.ItemDataRole.ToolTipRole:
            return {
                "strong": "seen on more than one pass over the same ground",
                "possible": "a single pass; no corroboration from a revisit",
                "weak": "overlaps a feature flagged as a trend",
            }.get(hotspot.evidence)
        return None


class LegTableModel(_BaseModel):
    """One row per contiguous recording leg."""

    COLUMNS = [
        ("leg", lambda r, u: str(r["leg"])),
        ("start", lambda r, u: _iso(r["start_time"])),
        ("end", lambda r, u: _iso(r["end_time"])),
        ("samples", lambda r, u: str(r["n"])),
        ("dur (s)", lambda r, u: f"{r['duration_s']:.0f}"),
        ("dist (km)", lambda r, u: f"{r['distance_m'] / 1000.0:.2f}"),
        ("mean ({unit})", lambda r, u: _dose(r["mean_dose"], u)),
        ("gap after (s)", lambda r, u: "-" if r["gap_after_s"] <= 0
                                    else f"{r['gap_after_s']:.0f}"),
    ]

    @staticmethod
    def rows_from(track, quality):
        """Build the leg rows from a track and its quality report."""
        legs = quality.leg_index if quality is not None else track.leg_index
        rows = []
        start = 0
        for i in range(1, track.n + 1):
            if i == track.n or legs[i] != legs[i - 1]:
                stop = i
                gap = 0.0
                if (quality is not None and stop < track.n
                        and stop - 1 < quality.gap_after.size
                        and quality.gap_after[stop - 1]):
                    gap = float(track.dt_s[stop - 1])
                rows.append({
                    "leg": int(legs[start]),
                    "start_time": float(track.time_s[start]),
                    "end_time": float(track.time_s[stop - 1]),
                    "n": stop - start,
                    "duration_s": float(track.time_s[stop - 1]
                                        - track.time_s[start]),
                    "distance_m": float(track.distance_along_m[stop - 1]
                                        - track.distance_along_m[start]),
                    # Masked samples carry NaN; a leg's mean is over what
                    # was actually read.
                    "mean_dose": _mean(track.dose_rate[start:stop]
                                       [track.valid_mask[start:stop]]),
                    "gap_after_s": gap,
                })
                start = i
        return rows
