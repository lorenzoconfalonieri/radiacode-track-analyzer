"""Dose units and gap-aware integration."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core.dose import DOSE_UNITS, accumulated_dose, convert_dose_rate, format_dose_rate
from rcta.core.quality import assess_with_report
from rcta.core.track import Track


def _track(n=100, dt=2.0, rate=90.0):
    return Track(time_s=np.arange(n) * dt, lat=np.full(n, 30.0),
                 lon=np.full(n, -97.0), accuracy_m=np.full(n, 8.0),
                 dose_rate=np.full(n, float(rate)), count_rate=np.full(n, 18.0))


# ---- units ---------------------------------------------------------------


def test_the_canonical_unit_is_the_identity():
    assert DOSE_UNITS["nSv/h"] == 1.0
    assert convert_dose_rate(90.0, "nSv/h") == pytest.approx(90.0)


def test_microsievert_conversion():
    assert float(convert_dose_rate(90.89, "uSv/h")) == pytest.approx(0.09089)


def test_microroentgen_uses_the_survey_meter_equivalence():
    """1 R = 10 mSv for gamma, so 1 uR/h = 10 nSv/h."""
    assert float(convert_dose_rate(100.0, "uR/h")) == pytest.approx(10.0)


def test_an_unknown_unit_raises_rather_than_silently_passing_through():
    with pytest.raises(ValueError, match="unknown dose unit"):
        convert_dose_rate(1.0, "banana")


def test_formatting_keeps_significant_figures_in_small_units():
    """0.09 uSv/h rounded to two places would throw away the reading."""
    assert format_dose_rate(90.89, "uSv/h") == "0.0909 uSv/h"
    assert format_dose_rate(90.89, "nSv/h") == "90.89 nSv/h"


def test_conversion_is_vectorised():
    out = convert_dose_rate(np.array([1000.0, 2000.0]), "uSv/h")
    assert out.tolist() == [1.0, 2.0]


# ---- integration ---------------------------------------------------------


def test_a_constant_rate_integrates_to_rate_times_hours():
    """3600 s at 90 nSv/h is 90 nSv. If this is wrong, everything is."""
    t = _track(n=3601, dt=1.0, rate=90.0)
    assert accumulated_dose(t).total_nsv == pytest.approx(90.0, rel=1e-9)


def test_a_linear_ramp_integrates_to_the_mean_rate():
    n = 3601
    rate = np.linspace(0.0, 100.0, n)
    t = Track(time_s=np.arange(n, dtype=float), lat=np.full(n, 30.0),
              lon=np.full(n, -97.0), accuracy_m=np.full(n, 8.0),
              dose_rate=rate, count_rate=np.full(n, 18.0))
    assert accumulated_dose(t).total_nsv == pytest.approx(50.0, rel=1e-6)


def test_a_dropout_separates_the_two_totals():
    n = 100
    time_s = np.arange(n, dtype=float) * 2.0
    time_s[50:] += 600.0
    t = Track(time_s=time_s, lat=np.full(n, 30.0), lon=np.full(n, -97.0),
              accuracy_m=np.full(n, 8.0), dose_rate=np.full(n, 90.0),
              count_rate=np.full(n, 18.0))
    summary = accumulated_dose(t)
    assert summary.n_gaps_excluded == 1
    assert summary.total_nsv < summary.naive_total_nsv
    assert summary.coverage < 1.0


def test_the_observed_total_never_exceeds_the_naive_one():
    """An invariant: excluding intervals cannot add dose."""
    t = _track()
    summary = accumulated_dose(t)
    assert summary.total_nsv <= summary.naive_total_nsv + 1e-12


def test_coverage_is_one_without_dropouts():
    assert accumulated_dose(_track()).coverage == pytest.approx(1.0)


def test_integration_ignores_the_position_mask():
    """A bad GPS fix still measured a real dose rate; dropping it would
    understate exposure for an unrelated reason."""
    t = _track(n=50)
    masked = t.with_mask(position=np.zeros(50, dtype=bool))
    assert (accumulated_dose(masked).total_nsv
            == pytest.approx(accumulated_dose(t).total_nsv))


def test_a_selection_mask_restricts_the_integration():
    t = _track(n=101, dt=1.0, rate=3600.0)
    mask = np.zeros(101, dtype=bool)
    mask[:51] = True
    assert accumulated_dose(t, mask=mask).total_nsv < accumulated_dose(t).total_nsv


def test_a_single_sample_selection_raises():
    t = _track()
    mask = np.zeros(t.n, dtype=bool)
    mask[0] = True
    with pytest.raises(ValueError, match="at least two"):
        accumulated_dose(t, mask=mask)


def test_describe_reports_both_totals_and_refuses_to_choose():
    n = 100
    time_s = np.arange(n, dtype=float) * 2.0
    time_s[50:] += 600.0
    t = Track(time_s=time_s, lat=np.full(n, 30.0), lon=np.full(n, -97.0),
              accuracy_m=np.full(n, 8.0), dose_rate=np.full(n, 90.0),
              count_rate=np.full(n, 18.0))
    text = accumulated_dose(t).describe()
    assert "observed" in text and "naive" in text
    assert "Neither is 'the' dose" in text


# ---- goldens against the real sample -------------------------------------


def test_sample_dose_totals(sample):
    _, quality = assess_with_report(sample)
    summary = accumulated_dose(sample, quality=quality)
    assert summary.naive_total_nsv == pytest.approx(1619.00, rel=1e-5)
    assert summary.total_nsv == pytest.approx(1234.50, rel=1e-5)
    assert summary.coverage == pytest.approx(6126.0 / 9754.0)
    assert summary.n_gaps_excluded == 13


def test_sample_totals_differ_by_about_a_quarter(sample):
    """The number that makes reporting only one of them indefensible."""
    _, quality = assess_with_report(sample)
    summary = accumulated_dose(sample, quality=quality)
    spread = (summary.naive_total_nsv - summary.total_nsv) / summary.naive_total_nsv
    assert 0.20 < spread < 0.28


# ---- one block, one unit -------------------------------------------------


def test_the_summary_does_not_mix_units_within_one_block():
    """A total in nanosieverts beside a mean in microsieverts is the exact
    failure this module exists to prevent."""
    from rcta.core.dose import DoseSummary

    summary = DoseSummary(total_nsv=1234.499, naive_total_nsv=1619.004,
                          covered_time_s=6126.0, span_time_s=9754.0,
                          n_gaps_excluded=13, max_dose_rate=95.2,
                          unit="uSv/h")
    text = summary.describe()
    assert "uSv over" in text and "uSv integrating" in text
    assert "nSv" not in text


def test_the_default_unit_prints_exactly_what_it_always_did():
    from rcta.core.dose import DoseSummary

    text = DoseSummary(total_nsv=1234.499, naive_total_nsv=1619.004,
                       covered_time_s=6126.0, span_time_s=9754.0,
                       n_gaps_excluded=13, max_dose_rate=95.2).describe()
    assert "observed   : 1234.499 nSv over 6126 s actually logged" in text
    assert "naive      : 1619.004 nSv integrating straight across 13" in text


def test_the_accumulated_unit_drops_the_per_hour():
    from rcta.core.dose import DOSE_UNITS, accumulated_unit

    assert accumulated_unit("nSv/h") == "nSv"
    assert accumulated_unit("uR/h") == "uR"
    # Every rate unit must have a dose counterpart, or describe() breaks on it.
    for unit in DOSE_UNITS:
        assert accumulated_unit(unit) and not accumulated_unit(unit).endswith("/h")


def test_an_unknown_unit_is_refused_rather_than_mangled():
    from rcta.core.dose import accumulated_unit

    with pytest.raises(ValueError, match="unknown dose unit"):
        accumulated_unit("banana")


def test_a_dose_total_converts_by_the_same_factor_as_a_rate():
    """1 uR = 10 nSv, so the roentgen total must scale by 0.1 like the rate."""
    from rcta.core.dose import format_accumulated_dose

    assert format_accumulated_dose(1000.0, "uSv/h") == "1.00000 uSv"
    assert format_accumulated_dose(1000.0, "uR/h") == "100.0000 uR"


def test_a_track_with_nothing_observed_has_no_mean_rate_not_a_zero_one():
    """One sample spans no interval: the mean is undefined, not 0 nSv/h."""
    track = Track(time_s=np.array([0.0]), lat=np.array([40.0]),
                  lon=np.array([-111.9]), accuracy_m=np.array([8.0]),
                  dose_rate=np.array([70.0]), count_rate=np.array([5.0]))
    summary = accumulated_dose(track)
    assert np.isnan(summary.mean_dose_rate)
    text = summary.describe()
    assert "undefined, nothing observed" in text
    assert "mean rate  : 0.00" not in text


def test_signed_dose_rate_always_carries_a_sign_and_converts():
    from rcta.core.dose import format_signed_dose_rate
    assert format_signed_dose_rate(81.51) == "+81.51 nSv/h"
    assert format_signed_dose_rate(-3.2) == "-3.20 nSv/h"
    assert format_signed_dose_rate(0.0) == "+0.00 nSv/h"
    assert format_signed_dose_rate(81.51, "uSv/h") == "+0.0815 uSv/h"


def test_a_selection_of_runs_does_not_integrate_the_time_between_them():
    """Two runs 22 s apart on the flight summed to 45.1 nSv over 176 s, but
    together reported 51.2 nSv over 198 s at 100 % coverage: the unselected
    stretch counted as observed."""
    track = _track(n=400, dt=2.0)
    a = np.zeros(track.n, bool)
    a[100:150] = True
    b = np.zeros(track.n, bool)
    b[160:200] = True
    one, two, both = (accumulated_dose(track, mask=m) for m in (a, b, a | b))
    assert both.total_nsv == pytest.approx(one.total_nsv + two.total_nsv)
    assert both.covered_time_s == pytest.approx(one.covered_time_s
                                                + two.covered_time_s)
    assert both.naive_total_nsv == pytest.approx(one.naive_total_nsv
                                                 + two.naive_total_nsv)
    assert both.coverage < 1.0
    # A single contiguous run is unchanged.
    whole = accumulated_dose(track, mask=a)
    assert whole.coverage == pytest.approx(1.0)
