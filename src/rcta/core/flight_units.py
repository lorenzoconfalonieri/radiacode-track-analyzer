"""Units, labels and thresholds shared by the flight modules.

The flight analysis is split across three modules -- flight numbers and
search windows, the trajectory join, and the altitude profile -- and all of
them speak about the same quantities: metres and feet, flight levels, the
climb/level/descent labels, the tolerances a join is held to. Defining those
once, here, is what keeps a flight level from meaning 30.48 m in one module
and 30.5 m in another, or a branch label from being spelled two ways.

Nothing here computes anything; it is the vocabulary the other three share.
:mod:`rcta.core.flight` re-exports all of it under its historical names.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

#: Metres per foot, exactly. Altitudes are stored in metres and displayed in
#: both, because a reader who flies thinks in flight levels and a reader who
#: does radiation transport thinks in kilometres of air.
M_PER_FT = 0.3048

#: One flight level is 100 ft of pressure altitude.
FLIGHT_LEVEL_M = 100.0 * M_PER_FT

#: Phase-of-flight names for the codes in :attr:`AltitudeJoin.branch`. Exported
#: alongside every sample so that a reader who bins the data themselves can
#: separate the two traverses without recomputing a vertical rate.
BRANCH_LABELS = {1: "climb", 0: "level", -1: "descent", -128: "unknown"}

#: Vertical rate below which the aircraft is treated as level, in m/s. 1.0 m/s
#: is about 200 ft/min: inside the noise of barometric altitude reported in
#: 25 ft steps, and well below any real climb or descent segment.
_LEVEL_RATE_M_S = 1.0

#: Median ground speed above which a track is confidently airborne. A fast
#: motorway drive peaks near 40 m/s and a high-speed train near 90; 100 m/s
#: sustained as a *median* is not reachable on the ground.
_FLIGHT_MEDIAN_SPEED_M_S = 100.0

#: Ground speed a track must reach at the 95th percentile to be called
#: airborne on the strength of speed alone.
_FLIGHT_P95_SPEED_M_S = 140.0

#: Default tolerance on the median position residual of the join, in metres.
#: Generous on purpose: it exists to catch the wrong flight, the wrong day and
#: the wrong leg -- errors measured in hundreds of kilometres -- and not to
#: audit ADS-B multilateration against a phone GPS. A value tight enough to do
#: the latter would reject good joins over sparse receiver coverage.
DEFAULT_MATCH_TOLERANCE_M = 5000.0

#: Widest waypoint gap that altitude may be interpolated across, in seconds.
DEFAULT_MAX_GAP_S = 300.0

#: Range searched when estimating the detector's clock offset, in seconds.
#: At 233 m/s a one-second offset is 233 m of apparent position residual, so
#: on a fast platform this is worth measuring before blaming the join.
_OFFSET_SEARCH_S = 180.0


def _utc(seconds: float) -> datetime:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)


def _fmt_alt(metres: float) -> str:
    """Metres, with the flight level a pilot would recognise beside it."""
    if not np.isfinite(metres):
        return "n/a"
    return f"{metres:.0f} m (FL{metres / FLIGHT_LEVEL_M:.0f})"


