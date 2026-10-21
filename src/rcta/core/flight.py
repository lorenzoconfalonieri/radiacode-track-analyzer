"""Altitude for a track that has none, and the dose-versus-altitude profile.

The ``.rctrk`` format carries no altitude (see ``docs/FORMAT.md``), which is
awkward, because on an airborne track altitude is the *only* variable that
explains the count rate. Cosmic secondaries are attenuated by the air above
the detector, so the dose rate at cruise is an order of magnitude above the
ground value and every ground-survey assumption in this program -- a stable
background, a localised source -- is false. This module imports the missing
column from an external trajectory and turns it into the one plot that makes
an airborne track interpretable.

Nothing here reaches the network; that is :mod:`rcta.net.flights`, and the
separation is enforced by ``tests/test_architecture.py``. What arrives here is
a :class:`Trajectory`, whether it came from an API, a cached response or a CSV
a user exported by hand, and every check below applies identically to all
three.

Four decisions carry the correctness of the result, and each is the one an
obvious implementation gets wrong.

*The join must be validated against position, not trusted.* A flight number is
a user-supplied string, and the failure mode of getting it wrong is silent: any
trajectory joins onto any track by timestamp and produces a smooth, plausible,
entirely fictional plot. So the trajectory's own latitude and longitude are
interpolated onto the track's clock and compared with the fixes the detector
recorded. Disagreement beyond a tolerance raises :class:`FlightMatchError`
rather than returning a profile, because a wrong altitude axis is worse than
no altitude axis.

*Altitude is never extrapolated and never interpolated across a long gap.*
Outside the trajectory's own span the altitude is NaN, and a sample landing in
a waypoint gap wider than ``max_gap_s`` is NaN too. A ten-minute gap spanning a
top-of-climb, interpolated linearly, invents a smooth ramp through a level-off
and assigns dose from cruise to altitudes the aircraft passed through once.

*Ascent and descent are a control, not a nuisance.* Most of a flight is spent
at one altitude, so a whole-flight scatter is a dense blob at cruise with thin
tails, and any curve through it is set by the tails. Binning the climb and the
descent separately gives two independent measurements of the same profile
hours apart; where they disagree, the disagreement is the error bar that a
single fit does not show. :func:`compare_branches` reports it.

*A poor GPS fix does not invalidate an altitude.* Altitude here comes from
ADS-B, not from the phone, so the 16 % of fixes with accuracy worse than 100 m
still receive a good altitude and still contribute their count rate. They are
excluded from the position check and from nothing else.

Altitude is canonically metres everywhere in this package, as dose rate is
canonically nSv/h. Feet appear only in :meth:`AltitudeProfile.describe` and in
the flight-level labels, because that is how aircrew and every flight tracker
quote it.
"""


from __future__ import annotations

from rcta.core.flight_join import (  # noqa: F401
    AltitudeJoin,
    FlightMatchError,
    FlightVerdict,
    MatchReport,
    Trajectory,
    join_altitude,
    looks_like_flight,
)
from rcta.core.flight_numbers import (  # noqa: F401
    FlightNumber,
    load_airlines,
    parse_flight_number,
    search_window,
)
from rcta.core.flight_profile import (  # noqa: F401
    DEFAULT_ALTITUDE_STEP_M,
    DEFAULT_FIT_FLOOR_M,
    AltitudeProfile,
    BranchComparison,
    ScaleHeight,
    bin_by_altitude,
    compare_branches,
    fit_scale_height,
)

# The implementation is split by concern; this module keeps the public
# names where every caller and the documentation expect them.
from rcta.core.flight_units import (  # noqa: F401
    BRANCH_LABELS,
    DEFAULT_MATCH_TOLERANCE_M,
    DEFAULT_MAX_GAP_S,
    FLIGHT_LEVEL_M,
    M_PER_FT,
)

__all__ = [
    "AltitudeJoin",
    "AltitudeProfile",
    "BRANCH_LABELS",
    "BranchComparison",
    "DEFAULT_ALTITUDE_STEP_M",
    "DEFAULT_FIT_FLOOR_M",
    "DEFAULT_MATCH_TOLERANCE_M",
    "DEFAULT_MAX_GAP_S",
    "FLIGHT_LEVEL_M",
    "FlightMatchError",
    "FlightNumber",
    "FlightVerdict",
    "M_PER_FT",
    "MatchReport",
    "ScaleHeight",
    "Trajectory",
    "bin_by_altitude",
    "compare_branches",
    "fit_scale_height",
    "join_altitude",
    "load_airlines",
    "looks_like_flight",
    "parse_flight_number",
    "search_window",
]
