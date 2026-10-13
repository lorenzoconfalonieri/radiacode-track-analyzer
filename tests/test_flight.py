"""Altitude import, the join, and the profile fitted to it.

The theme of these tests is that the module must refuse. A dose-versus-altitude
plot is easy to produce and almost impossible to sanity-check by eye, so most
of what follows checks that a wrong input is rejected rather than rendered.
"""

from __future__ import annotations

import numpy as np
import pytest

import synthetic
from rcta.core import flight as F


# ---- flight numbers -------------------------------------------------------

@pytest.mark.parametrize("text,callsign", [
    ("DL2618", "DAL2618"),
    ("dl2618", "DAL2618"),
    ("DL 2618", "DAL2618"),
    ("dl-2618", "DAL2618"),
    ("BA117", "BAW117"),
    ("U21234", "EZY1234"),
    ("DAL2618", "DAL2618"),
])
def test_flight_numbers_resolve_to_callsigns(text, callsign):
    assert F.parse_flight_number(text).callsign == callsign


def test_the_iata_to_icao_mapping_is_not_guessable():
    """The reason the table exists: no rule generates these."""
    assert F.parse_flight_number("DL1").airline_icao == "DAL"
    assert F.parse_flight_number("AA1").airline_icao == "AAL"
    assert F.parse_flight_number("BA1").airline_icao == "BAW"
    assert F.parse_flight_number("WN1").airline_icao == "SWA"


def test_a_three_letter_prefix_is_taken_as_icao():
    """The escape hatch for a carrier the table does not list."""
    parsed = F.parse_flight_number("XYZ42")
    assert parsed.known_airline and parsed.callsign == "XYZ42"


def test_an_unknown_iata_code_is_reported_not_guessed():
    parsed = F.parse_flight_number("QQ100")
    assert not parsed.known_airline
    assert parsed.candidates == ()
    assert "not in the table" in parsed.describe()


def test_zero_padding_is_tried_both_ways():
    """Operators are not consistent, so one spelling is not enough."""
    assert F.parse_flight_number("DL218").candidates == ("DAL218", "DAL0218")
    assert F.parse_flight_number("DL2618").candidates == ("DAL2618",)


@pytest.mark.parametrize("bad", ["", "not a flight", "12345678", "D"])
def test_malformed_flight_numbers_raise(bad):
    with pytest.raises(ValueError):
        F.parse_flight_number(bad)


# ---- is this a flight -----------------------------------------------------

def test_a_flight_is_recognised(flight_case):
    track, _, _ = flight_case
    assert F.looks_like_flight(track).is_flight


def test_a_walk_is_not_called_a_flight():
    track, _ = synthetic.make(n=300, speed_m_s=1.4, dt_s=2.0)
    assert not F.looks_like_flight(track).is_flight


def test_one_gps_glitch_does_not_make_a_walk_a_flight():
    """The median is the statistic precisely so that this cannot happen."""
    track, _ = synthetic.make(n=300, speed_m_s=1.4, teleports=[(150, 5000.0)])
    verdict = F.looks_like_flight(track)
    assert not verdict.is_flight
    assert verdict.speed_median_m_s < 10.0


def test_a_negative_verdict_says_it_is_not_evidence():
    """Asymmetry is the whole design; it has to be stated."""
    track, _ = synthetic.make(n=200, speed_m_s=1.4)
    assert "not evidence against" in F.looks_like_flight(track).describe()


# ---- the search window ----------------------------------------------------

def test_a_matching_date_leaves_the_window_alone(flight_case):
    track, _, _ = flight_case
    begin, end, warnings = F.search_window(
        track.time_s[0], track.time_s[-1], "2026-09-07")
    assert (begin, end) == (track.time_s[0], track.time_s[-1])
    assert warnings == ()


def test_a_disagreeing_date_shifts_the_window_and_says_so(flight_case):
    track, _, _ = flight_case
    begin, _, warnings = F.search_window(
        track.time_s[0], track.time_s[-1], "2026-09-09")
    assert begin == pytest.approx(track.time_s[0] + 2 * 86400.0)
    assert "clock was wrong" in warnings[0]


def test_a_malformed_date_raises(flight_case):
    track, _, _ = flight_case
    with pytest.raises(ValueError, match="expected"):
        F.search_window(track.time_s[0], track.time_s[-1], "7 Sept 2026")


# ---- ...in the departure airport's own time -------------------------------
#
# DL2728 is the case these exist for. It leaves Salt Lake City at 18:06 on
# Thursday 10 September 2026 and is airborne from 00:06Z on the Friday, so
# the day on the boarding pass and the day in UTC are different days. The
# window itself still comes from the track; the zone only decides which
# calendar day the track is taken to have begun on.

#: First fix of a DL2728 track: 2026-09-11 00:06:24Z, 2026-09-10 18:06 MDT.
DL2728_BEGIN = 1789085184.0
DL2728_END = 1789093469.0


def test_the_date_on_the_boarding_pass_is_the_one_that_works():
    begin, end, warnings = F.search_window(
        DL2728_BEGIN, DL2728_END, "2026-09-10", tz="America/Denver")
    assert (begin, end) == (DL2728_BEGIN, DL2728_END)
    assert warnings == ()


def test_the_utc_date_is_now_a_day_out_and_is_reported_as_one():
    """The behaviour that changed, stated as a test so it cannot drift back."""
    begin, _, warnings = F.search_window(
        DL2728_BEGIN, DL2728_END, "2026-09-11", tz="America/Denver")
    assert begin == pytest.approx(DL2728_BEGIN + 86400.0)
    assert "2026-09-10 at America/Denver" in warnings[0]


def test_the_warning_names_the_zone_it_judged_the_day_in():
    """Without it a timezone surprise is indistinguishable from a bad clock."""
    _, _, warnings = F.search_window(
        DL2728_BEGIN, DL2728_END, "2026-09-12", tz="America/Denver")
    assert "America/Denver" in warnings[0] and "clock was wrong" in warnings[0]


def test_no_zone_means_utc_and_the_old_behaviour():
    """An airport the table has no zone for must still be usable."""
    begin, end, warnings = F.search_window(
        DL2728_BEGIN, DL2728_END, "2026-09-11")
    assert (begin, end) == (DL2728_BEGIN, DL2728_END)
    assert warnings == ()


def test_an_unusable_zone_falls_back_to_utc_and_says_which(monkeypatch):
    """A missing zone database is a Windows deployment, not a crash.

    It must be *said*, though: falling back to UTC silently is the exact
    behaviour the tz argument exists to remove.
    """
    monkeypatch.setattr(F, "ZoneInfo", _raising_zoneinfo)
    begin, _, warnings = F.search_window(
        DL2728_BEGIN, DL2728_END, "2026-09-11", tz="America/Denver")
    assert begin == DL2728_BEGIN                       # judged in UTC
    assert "tzdata" in warnings[0]


def _raising_zoneinfo(name):
    raise F.ZoneInfoNotFoundError(name)


def test_the_shift_is_whole_days_of_86400_s_not_calendar_days():
    """The model being corrected is a detector clock that lost whole days.

    Such a clock knows nothing about a 23-hour spring-forward, so shifting by
    a calendar day across a DST boundary would move the window by an hour
    that no clock error ever produced.
    """
    # 2026-03-05 12:00 MST, four days before US clocks go forward.
    begin = 1772737200.0
    shifted, _, _ = F.search_window(begin, begin + 3600.0, "2026-03-09",
                                    tz="America/Denver")
    assert shifted == pytest.approx(begin + 4 * 86400.0)


def test_a_zone_that_is_a_day_ahead_of_utc_is_handled_too():
    """Auckland at 10:00 on the 11th is 22:00Z on the 10th: the same problem,
    the other way round, and the one a longitude guess gets backwards."""
    begin = 1789077600.0                               # 2026-09-11 10:00 NZST
    same, _, warnings = F.search_window(begin, begin + 3600.0, "2026-09-11",
                                        tz="Pacific/Auckland")
    assert (same, warnings) == (begin, ())
    shifted, _, _ = F.search_window(begin, begin + 3600.0, "2026-09-10",
                                    tz="Pacific/Auckland")
    assert shifted == pytest.approx(begin - 86400.0)


def test_the_departure_zone_comes_from_the_airport_table():
    """The two halves joined up: this is what the CLI and the dock do."""
    from rcta.core.airports import nearest_airport

    airport = nearest_airport(40.788, -111.979)        # KSLC, the first fix
    begin, _, warnings = F.search_window(
        DL2728_BEGIN, DL2728_END, "2026-09-10", tz=airport.tz)
    assert (begin, warnings) == (DL2728_BEGIN, ())


# ---- the trajectory -------------------------------------------------------

def test_a_trajectory_sorts_itself_and_drops_duplicate_times():
    trajectory = F.Trajectory(
        time_s=[30.0, 10.0, 20.0, 20.0], lat=[1, 2, 3, 4], lon=[1, 2, 3, 4],
        altitude_m=[300.0, 100.0, 200.0, 999.0])
    assert list(trajectory.time_s) == [10.0, 20.0, 30.0]
    assert list(trajectory.altitude_m) == [100.0, 200.0, 300.0]
    assert "duplicate" in trajectory.warnings[0]


def test_trajectory_columns_are_read_only():
    trajectory = F.Trajectory(time_s=[1.0, 2.0], lat=[0, 0], lon=[0, 0],
                              altitude_m=[0.0, 1.0])
    with pytest.raises(ValueError):
        trajectory.altitude_m[0] = 5.0


def test_a_missing_altitude_stays_nan_rather_than_zero():
    """Zero would put an aircraft at sea level in Denver."""
    trajectory = F.Trajectory(time_s=[1.0, 2.0], lat=[0, 0], lon=[0, 0],
                              altitude_m=[np.nan, 1000.0])
    assert not trajectory.has_altitude[0]
    assert trajectory.altitude_m.min() != 0.0


def test_longitude_is_unwrapped_across_the_antimeridian():
    trajectory = F.Trajectory(
        time_s=[1.0, 2.0], lat=[0.0, 0.0], lon=[179.9, -179.9],
        altitude_m=[10000.0, 10000.0])
    assert trajectory.unwrapped_lon[1] == pytest.approx(180.1)


def test_an_empty_trajectory_is_refused():
    with pytest.raises(ValueError, match="at least one waypoint"):
        F.Trajectory(time_s=[], lat=[], lon=[], altitude_m=[])


# ---- the join -------------------------------------------------------------

def test_a_matching_trajectory_joins_every_sample(flight_case):
    track, trajectory, truth = flight_case
    join = F.join_altitude(track, trajectory)
    assert join.n_joined == track.n
    assert join.altitude_m == pytest.approx(truth["altitude_m"])
    assert join.match.matched


def test_altitude_is_never_extrapolated(flight_case):
    """Outside the trajectory's own span there is no altitude, only NaN."""
    track, trajectory, _ = flight_case
    clipped = F.Trajectory(
        time_s=trajectory.time_s[100:-100], lat=trajectory.lat[100:-100],
        lon=trajectory.lon[100:-100], altitude_m=trajectory.altitude_m[100:-100])
    join = F.join_altitude(track, clipped)
    assert join.n_before == 100 and join.n_after == 100
    assert not np.isfinite(join.altitude_m[:100]).any()
    assert not np.isfinite(join.altitude_m[-100:]).any()


def test_altitude_is_not_interpolated_across_a_long_gap(flight_case):
    """A ramp drawn through a top of climb is an invention."""
    track, trajectory, _ = flight_case
    keep = np.ones(trajectory.n, dtype=bool)
    keep[150:230] = False                       # 800 s hole through the climb
    holed = F.Trajectory(
        time_s=trajectory.time_s[keep], lat=trajectory.lat[keep],
        lon=trajectory.lon[keep], altitude_m=trajectory.altitude_m[keep])

    permissive = F.join_altitude(track, holed, max_gap_s=1e9)
    strict = F.join_altitude(track, holed, max_gap_s=300.0)
    assert strict.n_gap_masked > 0
    assert strict.n_joined < permissive.n_joined
    assert not np.isfinite(strict.altitude_m[160:220]).any()


def test_the_widest_bridged_gap_is_reported(flight_case):
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    assert join.widest_bridged_gap_s <= join.max_gap_s
    assert "bridged" in join.describe()


def test_a_poor_gps_fix_still_receives_an_altitude(flight_case):
    """Altitude comes from ADS-B, so a bad phone fix does not invalidate it."""
    track, trajectory, _ = flight_case
    position = np.ones(track.n, dtype=bool)
    position[10:20] = False
    track = track.with_mask(position=position)
    join = F.join_altitude(track, trajectory)
    assert np.isfinite(join.altitude_m[10:20]).all()
    assert join.match.n_compared == track.n - 10


# ---- the refusal ----------------------------------------------------------

def test_a_displaced_trajectory_is_refused(flight_case):
    track, _, _ = flight_case
    wrong = synthetic.flight_trajectory(track, offset_m=300_000.0)
    with pytest.raises(F.FlightMatchError) as excinfo:
        F.join_altitude(track, wrong)
    report = excinfo.value.report
    assert not report.matched
    assert report.residual_median_m == pytest.approx(300_000.0, rel=0.02)
    assert "not the flight" in report.describe()


def test_the_refusal_can_be_overridden_and_is_then_labelled(flight_case):
    track, _, _ = flight_case
    wrong = synthetic.flight_trajectory(track, offset_m=300_000.0)
    join = F.join_altitude(track, wrong, force_match=True)
    assert join.match.forced
    assert "overridden" in join.match.describe()


def test_a_clock_offset_is_measured_and_named(flight_case):
    """At 200 m/s a clock error looks exactly like the wrong flight."""
    track, _, _ = flight_case
    skewed = synthetic.flight_trajectory(track, time_offset_s=45.0)
    with pytest.raises(F.FlightMatchError) as excinfo:
        F.join_altitude(track, skewed)
    report = excinfo.value.report
    assert report.clock_offset_s == pytest.approx(45.0, abs=2.0)
    assert report.offset_matters
    assert "--time-offset" in report.describe()


def test_applying_the_offset_rescues_the_join(flight_case):
    track, _, _ = flight_case
    skewed = synthetic.flight_trajectory(track, time_offset_s=45.0)
    join = F.join_altitude(track, skewed, time_offset_s=45.0)
    assert join.match.matched
    assert join.match.residual_median_m < 100.0


def test_the_offset_is_never_applied_on_its_own_initiative(flight_case):
    """Sliding the clock until the data fits is how a wrong flight is sold."""
    track, _, _ = flight_case
    skewed = synthetic.flight_trajectory(track, time_offset_s=45.0)
    with pytest.raises(F.FlightMatchError):
        F.join_altitude(track, skewed)


def test_a_trajectory_with_one_altitude_cannot_be_joined(flight_case):
    track, _, _ = flight_case
    thin = F.Trajectory(time_s=[track.time_s[0], track.time_s[-1]],
                        lat=[track.lat[0], track.lat[-1]],
                        lon=[track.lon[0], track.lon[-1]],
                        altitude_m=[np.nan, 10000.0])
    with pytest.raises(ValueError, match="fewer than two"):
        F.join_altitude(track, thin)


# ---- the profile ----------------------------------------------------------

def test_the_profile_bins_and_never_interpolates(flight_case):
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    profile = F.bin_by_altitude(join, step_m=500.0)
    assert profile.n_samples.sum() == join.n_joined
    empty = ~profile.occupied
    assert not np.isfinite(profile.dose_rate[empty]).any()


def test_bin_edges_are_anchored_so_branches_align(flight_case):
    """Climb and descent must share a lattice to be comparable bin for bin."""
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    climb = F.bin_by_altitude(join, step_m=1000.0, branch="climb")
    descent = F.bin_by_altitude(join, step_m=1000.0, branch="descent")
    shared = np.intersect1d(climb.altitude_m, descent.altitude_m)
    assert shared.size >= 5
    assert np.allclose(np.mod(shared - 500.0, 1000.0), 0.0)


def test_both_error_bars_are_carried(flight_case):
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    profile = F.bin_by_altitude(join, step_m=500.0)
    m = profile.occupied
    assert (profile.counting_sem_cps[m] > 0).all()
    assert np.allclose(profile.sem_cps[m],
                       np.fmax(profile.counting_sem_cps[m],
                               profile.observed_sem_cps[m]))


def test_underdispersion_is_reported_as_a_clock_problem(flight_case):
    """Below-Poisson scatter means the assumed integration time is too short."""
    track, trajectory, _ = flight_case
    # Claim a tenth of the real live time: the counting error inflates and the
    # observed spread does not.
    track = track.with_mask(integration_time_s=1.0)
    join = F.join_altitude(track, trajectory)
    profile = F.bin_by_altitude(join, step_m=500.0)
    assert profile.excess_scatter < 0.7
    assert "too short" in profile.scatter_note


def test_a_branch_with_no_samples_gives_an_empty_profile(flight_case):
    track, trajectory, _ = flight_case
    level_only = F.Trajectory(
        time_s=trajectory.time_s, lat=trajectory.lat, lon=trajectory.lon,
        altitude_m=np.full(trajectory.n, 10000.0))
    join = F.join_altitude(track, level_only)
    assert F.bin_by_altitude(join, branch="climb").altitude_m.size == 0


@pytest.mark.parametrize("bad", [{"step_m": 0.0}, {"statistic": "rms"},
                                 {"branch": "taxi"}])
def test_bad_profile_parameters_raise(flight_case, bad):
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    with pytest.raises(ValueError):
        F.bin_by_altitude(join, **bad)


# ---- the fit --------------------------------------------------------------

def test_the_known_scale_height_is_recovered(flight_case):
    track, trajectory, truth = flight_case
    join = F.join_altitude(track, trajectory)
    fit = F.fit_scale_height(F.bin_by_altitude(join, step_m=500.0))
    assert fit.constrained
    assert fit.scale_height_m == pytest.approx(truth["scale_height_m"], rel=0.10)
    assert fit.doubling_m == pytest.approx(
        truth["scale_height_m"] * np.log(2.0), rel=0.10)


def test_the_fit_reports_its_own_goodness(flight_case):
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    fit = F.fit_scale_height(F.bin_by_altitude(join, step_m=500.0))
    assert 0.9 < fit.r_squared <= 1.0
    assert np.isfinite(fit.chi2_per_dof)


def test_a_flat_profile_is_not_constrained():
    """A track that never changed altitude cannot measure a scale height."""
    track, trajectory, _ = synthetic.flight_track(climb_frac=0.001,
                                                  descent_frac=0.001)
    level = F.Trajectory(time_s=trajectory.time_s, lat=trajectory.lat,
                         lon=trajectory.lon,
                         altitude_m=np.full(trajectory.n, 10000.0))
    join = F.join_altitude(track, level)
    fit = F.fit_scale_height(F.bin_by_altitude(join, step_m=500.0))
    assert not fit.constrained
    assert "not constrained" in fit.describe()


def test_a_falling_profile_is_refused_rather_than_fitted(flight_case):
    """No cosmic profile falls with altitude; a negative slope means a bug."""
    track, trajectory, _ = flight_case
    inverted = F.Trajectory(
        time_s=trajectory.time_s, lat=trajectory.lat, lon=trajectory.lon,
        altitude_m=trajectory.altitude_m.max() - trajectory.altitude_m + 200.0)
    join = F.join_altitude(track, inverted)
    fit = F.fit_scale_height(F.bin_by_altitude(join, step_m=500.0))
    assert not fit.constrained
    assert "falls with altitude" in fit.note


def test_the_fit_floor_reduces_the_terrestrial_bias():
    """Documented in DEFAULT_FIT_FLOOR_M; pinned here so it stays true."""
    import synthetic as syn
    original = syn.flight_dose
    syn.flight_dose = lambda a, **k: original(a, terrestrial_cps=12.0, **k)
    try:
        track, trajectory, truth = syn.flight_track(poisson=False)
    finally:
        syn.flight_dose = original
    join = F.join_altitude(track, trajectory)
    profile = F.bin_by_altitude(join, step_m=500.0)
    truth_h = truth["scale_height_m"]
    without = F.fit_scale_height(profile, floor_m=0.0).scale_height_m
    with_floor = F.fit_scale_height(profile, floor_m=3000.0).scale_height_m
    assert abs(with_floor - truth_h) < abs(without - truth_h)


# ---- the control ----------------------------------------------------------

def test_climb_and_descent_agree_on_a_consistent_flight(flight_case):
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    comparison = F.compare_branches(join, step_m=1000.0)
    assert comparison.n_shared >= 5
    assert comparison.consistent
    assert "consistent" in comparison.describe()


def test_a_disagreement_between_branches_is_detected():
    """Halve the dose after the top of climb and the control must notice."""
    track, trajectory, _ = synthetic.flight_track(poisson=False)
    midpoint = track.n // 2
    counts = np.array(track.count_rate, dtype=float)
    counts[midpoint:] *= 0.5
    broken = track.__class__(
        time_s=track.time_s, lat=track.lat, lon=track.lon,
        accuracy_m=track.accuracy_m, dose_rate=counts * 51.3,
        count_rate=counts, integration_time_s=track.effective_integration_time_s,
        position_smoothing=1)
    join = F.join_altitude(broken, trajectory)
    comparison = F.compare_branches(join, step_m=1000.0)
    assert not comparison.consistent
    assert "not a function of altitude alone" in comparison.describe()


def test_the_branch_threshold_is_bonferroni_corrected(flight_case):
    """Twenty bins at 2 sigma manufactures a disagreement from clean data."""
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    coarse = F.compare_branches(join, step_m=2000.0)
    fine = F.compare_branches(join, step_m=500.0)
    assert fine.n_shared > coarse.n_shared
    assert fine.threshold_sigma > coarse.threshold_sigma


def test_the_branch_threshold_is_two_sided(flight_case):
    """Either branch reading high is a disagreement, so alpha splits across tails."""
    from scipy import stats

    track, trajectory, _ = flight_case
    comparison = F.compare_branches(F.join_altitude(track, trajectory),
                                    step_m=1000.0, alpha=0.05)
    n = comparison.n_shared
    assert comparison.threshold_sigma == pytest.approx(
        stats.norm.isf(0.05 / (2 * n)))


def test_a_non_integer_step_still_matches_bins(flight_case):
    """1000 ft bins have float centres that float equality would not match."""
    track, trajectory, _ = flight_case
    join = F.join_altitude(track, trajectory)
    feet = F.compare_branches(join, step_m=304.8 * 4)
    assert feet.n_shared >= 3


def test_a_missing_branch_is_reported_not_crashed():
    track, trajectory, _ = synthetic.flight_track()
    level = F.Trajectory(time_s=trajectory.time_s, lat=trajectory.lat,
                         lon=trajectory.lon,
                         altitude_m=np.full(trajectory.n, 10000.0))
    join = F.join_altitude(track, level)
    comparison = F.compare_branches(join)
    assert comparison.n_shared == 0
    assert comparison.consistent          # nothing to disagree about
    assert "no shared altitude bin" in comparison.describe()


def test_a_sample_landing_on_a_waypoint_is_not_gap_masked(flight_case):
    """It is not interpolated, so the interval after it is irrelevant.

    Both clocks are integer seconds, so this is the common case rather than a
    corner one, and masking these would discard the join's best altitudes.
    """
    track, trajectory, _ = flight_case
    keep = np.ones(trajectory.n, dtype=bool)
    keep[151:230] = False              # a wide hole starting right after 150
    holed = F.Trajectory(
        time_s=trajectory.time_s[keep], lat=trajectory.lat[keep],
        lon=trajectory.lon[keep], altitude_m=trajectory.altitude_m[keep])

    join = F.join_altitude(track, holed, max_gap_s=300.0)
    on_waypoint = np.flatnonzero(track.time_s == trajectory.time_s[150])
    assert on_waypoint.size == 1
    assert np.isfinite(join.altitude_m[on_waypoint[0]])
    assert join.altitude_m[on_waypoint[0]] == pytest.approx(
        trajectory.altitude_m[150])
    # The samples strictly inside the hole are still masked.
    assert not np.isfinite(join.altitude_m[on_waypoint[0] + 5])
