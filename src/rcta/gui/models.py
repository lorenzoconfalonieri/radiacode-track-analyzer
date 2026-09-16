"""Table models for anomalies, hotspots and legs.

Each model is driven by a module-level list of ``(header, accessor)`` pairs, so
adding a column is one line and the CSV exporters in
:mod:`rcta.core.io_export` can mirror the same order.

Excluded candidates are shown, not hidden. A reviewer judging whether the
trend flagging was right needs to see what was flagged and why, and a table
that silently drops everything below threshold makes that impossible.
"""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6 import QtCore, QtGui

from rcta.gui import theme

Qt = QtCore.Qt

__all__ = ["AnomalyTableModel", "HotspotTableModel", "LegTableModel"]


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime(
        "%H:%M:%S")


class _BaseModel(QtCore.QAbstractTableModel):
    COLUMNS: list = []

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list = []
        self._font = theme.mono_font()

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
            return self.COLUMNS[section][0]
        return section + 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        obj = self._rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[index.column()][1](obj)
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
        ("time", lambda a: _iso(a.time_s)),
        ("dur (s)", lambda a: f"{a.duration_s:.0f}"),
        ("lat", lambda a: f"{a.lat:.5f}"),
        ("lon", lambda a: f"{a.lon:.5f}"),
        ("net (cps)", lambda a: f"{a.net_count_rate:+.2f}"),
        ("net (nSv/h)", lambda a: f"{a.net_dose_rate:+.1f}"),
        ("peak (nSv/h)", lambda a: f"{a.peak_dose_rate:.1f}"),
        ("sigma", lambda a: f"{a.significance:.2f}"),
        ("L_C", lambda a: f"{a.limits.critical_level:.0f}"),
        ("MDA (cps)", lambda a: f"{a.limits.mda_cps:.2f}"),
        ("flags", lambda a: " ".join(a.flags)),
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
        ("lat", lambda h: f"{h.lat:.5f}"),
        ("lon", lambda h: f"{h.lon:.5f}"),
        ("r (m)", lambda h: f"{h.radius_m:.0f}"),
        ("samples", lambda h: str(h.n_samples)),
        ("passes", lambda h: str(h.n_passes)),
        ("peak (nSv/h)", lambda h: f"{h.peak_dose_rate:.1f}"),
        ("net counts", lambda h: f"{h.net_counts:.0f}"),
        ("sigma", lambda h: f"{h.significance:.2f}"),
        ("score", lambda h: f"{h.score:.0f}"),
        ("evidence", lambda h: h.evidence),
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
        ("leg", lambda r: str(r["leg"])),
        ("start", lambda r: _iso(r["start_time"])),
        ("end", lambda r: _iso(r["end_time"])),
        ("samples", lambda r: str(r["n"])),
        ("dur (s)", lambda r: f"{r['duration_s']:.0f}"),
        ("dist (km)", lambda r: f"{r['distance_m'] / 1000.0:.2f}"),
        ("mean (nSv/h)", lambda r: f"{r['mean_dose']:.1f}"),
        ("gap after (s)", lambda r: "-" if r["gap_after_s"] <= 0
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
                    "mean_dose": float(track.dose_rate[start:stop].mean()),
                    "gap_after_s": gap,
                })
                start = i
        return rows
