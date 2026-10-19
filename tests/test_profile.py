"""Summaries, profiles, distributions and correlations."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core import profile as pf
from rcta.core.quality import assess_with_report
from rcta.core.track import Track


def _track(n=200, dt=2.0, speed_m_s=1.4):
    lat = 30.0 + np.arange(n) * speed_m_s * dt / 111195.0
    return Track(time_s=np.arange(n) * dt, lat=lat, lon=np.full(n, -97.0),
                 accuracy_m=np.full(n, 8.0), dose_rate=np.full(n, 90.0),
                 count_rate=np.full(n, 18.0))


# ---- summary -------------------------------------------------------------


def test_summary_counts_match_the_track():
    s = pf.summarise(_track(n=50))
    assert s.n == 50 and s.n_valid == 50


def test_summary_of_a_selection_is_smaller():
    t = _track(n=100)
    mask = np.zeros(100, dtype=bool)
    mask[:40] = True
    assert pf.summarise(t, mask=mask).n == 40


def test_a_disjoint_selection_declines_to_quote_a_distance():
    """Summing unrelated jumps would produce a confident, meaningless number."""
    t = _track(n=100)
    mask = np.zeros(100, dtype=bool)
    mask[:10] = True
    mask[80:] = True
    s = pf.summarise(t, mask=mask)
    assert s.contiguous is False
    assert np.isnan(s.distance_m)
    assert "not contiguous" in s.describe()


def test_summarising_nothing_raises():
    with pytest.raises(ValueError, match="empty selection"):
        pf.summarise(_track(), mask=np.zeros(200, dtype=bool))


# ---- distance profile ----------------------------------------------------


def test_profile_bins_do_not_interpolate_across_an_empty_cell():
    """A NaN is the honest value for a bin with no samples in it."""
    t = _track(n=20, dt=2.0, speed_m_s=100.0)   # 200 m between samples
    p = pf.resample_by_distance(t, step_m=10.0)
    assert p.n_empty > 0
    assert np.isnan(p.dose_rate[p.n_samples == 0]).all()


def test_profile_recovers_a_constant():
    t = _track(n=200)
    p = pf.resample_by_distance(t, step_m=50.0)
    occupied = p.n_samples > 0
    assert np.allclose(p.dose_rate[occupied], 90.0)


def test_profile_statistic_is_honoured():
    n = 100
    dose = np.tile([10.0, 90.0], n // 2)
    lat = 30.0 + np.arange(n) * 1.4 * 2.0 / 111195.0
    t = Track(time_s=np.arange(n) * 2.0, lat=lat, lon=np.full(n, -97.0),
              accuracy_m=np.full(n, 8.0), dose_rate=dose,
              count_rate=np.full(n, 18.0))
    hi = pf.resample_by_distance(t, step_m=100.0, statistic="max")
    lo = pf.resample_by_distance(t, step_m=100.0, statistic="min")
    occupied = hi.n_samples > 0
    assert np.nanmax(hi.dose_rate[occupied]) > np.nanmax(lo.dose_rate[occupied])


@pytest.mark.parametrize("bad", [0.0, -10.0])
def test_non_positive_step_raises(bad):
    with pytest.raises(ValueError, match="step_m"):
        pf.resample_by_distance(_track(), step_m=bad)


def test_unknown_statistic_raises():
    with pytest.raises(ValueError, match="unknown statistic"):
        pf.resample_by_distance(_track(), statistic="mode")


# ---- distribution and bimodality -----------------------------------------


def test_histogram_totals_the_input():
    h = pf.distribution(np.random.default_rng(0).normal(size=500), unit="nSv/h")
    assert h.counts.sum() == 500
    assert h.centres.size == h.counts.size


def test_histogram_of_nothing_is_empty_not_an_error():
    assert pf.distribution([]).counts.size == 0


def test_bimodality_finds_a_clean_two_population_split():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(10.0, 1.0, 500),
                             rng.normal(90.0, 1.0, 500)])
    b = pf.bimodality(values)
    assert b.is_bimodal
    assert b.mu_low == pytest.approx(10.0, abs=0.3)
    assert b.mu_high == pytest.approx(90.0, abs=0.3)
    assert 40.0 < b.threshold < 60.0


def test_a_single_population_is_not_called_bimodal():
    """2-means always splits, so this is the test that stops the diagnostic
    reporting bimodality for every distribution it is ever shown."""
    values = np.random.default_rng(2).normal(50.0, 5.0, 2000)
    b = pf.bimodality(values)
    assert not b.is_bimodal
    assert b.ashman_d == pytest.approx(pf.UNIMODAL_ASHMAN_D, abs=0.1)


def test_the_unimodal_floor_is_the_analytic_value():
    """Splitting a Gaussian at its mean gives half-normal lobes and D = 2.647.
    Measured here so the constant cannot drift away from its derivation."""
    values = np.random.default_rng(9).normal(0.0, 1.0, 200_000)
    assert pf.bimodality(values).ashman_d == pytest.approx(
        pf.UNIMODAL_ASHMAN_D, abs=0.02)


def test_bimodality_is_deterministic():
    """No seed, no initialisation: a diagnostic that moves is not a diagnostic."""
    values = np.random.default_rng(3).normal(size=1000)
    assert pf.bimodality(values) == pf.bimodality(values)


def test_bimodality_handles_a_tiny_sample():
    b = pf.bimodality([1.0, 2.0])
    assert b.ashman_d == 0.0


# ---- correlation ---------------------------------------------------------


def test_correlation_reports_all_three_p_values():
    rng = np.random.default_rng(4)
    t = np.arange(0.0, 1000.0, 2.0)
    x = rng.normal(size=t.size)
    y = rng.normal(size=t.size)
    c = pf.correlate(x, y, t, n_boot=200)
    for p in (c.p_naive, c.p_effective, c.p_block_bootstrap):
        assert 0.0 <= p <= 1.0
    assert "use this one" in c.describe()


def test_correlation_of_independent_noise_is_not_significant():
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 2000.0, 2.0)
    c = pf.correlate(rng.normal(size=t.size), rng.normal(size=t.size), t,
                     n_boot=200)
    assert c.p_effective > 0.05


def test_speed_dose_correlation_carries_its_confounding_caveat():
    """The number reliably invites the wrong conclusion, so the object says so."""
    c = pf.correlate_speed_dose(_track(n=100), n_boot=100)
    assert "confounding, not causation" in c.caveat
    assert "caveat" in c.describe()


# ---- goldens against the real sample -------------------------------------


def test_sample_summary(sample):
    _, quality = assess_with_report(sample)
    s = pf.summarise(sample, quality=quality)
    assert s.n == 2964
    assert s.dose_mean == pytest.approx(742.796, rel=1e-4)
    assert s.distance_m == pytest.approx(1_792_210.09, rel=1e-6)
    assert s.accumulated_dose_nsv == pytest.approx(1234.50, rel=1e-5)


def test_sample_dose_is_two_populations_eighty_apart(sample):
    """Ground versus cruise altitude. Any estimator that assumes one
    population is wrong on this file, which is why background defaults to a
    local method rather than a global one."""
    b = pf.bimodality(sample.dose_rate)
    assert b.ashman_d == pytest.approx(7.99, abs=0.05)
    assert b.is_bimodal
    assert b.n_low == 673 and b.n_high == 2291
    assert b.mu_low == pytest.approx(101.4, abs=0.2)
    assert b.mu_high == pytest.approx(931.2, abs=0.2)
    assert b.threshold == pytest.approx(518.1, abs=0.5)


def test_sample_speed_dose_correlation(sample):
    c = pf.correlate_speed_dose(sample, n_boot=200)
    assert c.spearman_rho == pytest.approx(0.6387, abs=1e-3)
    assert c.n_effective == pytest.approx(307.0, rel=0.05)
    assert c.p_naive < 1e-100
    assert c.p_effective > c.p_naive, "the correction must always cost something"


def test_too_few_readings_are_not_described_as_a_split():
    from rcta.core.profile import bimodality
    text = bimodality(np.array([74.0, 76.9])).describe()
    assert text.startswith("too few readings to judge (2;")
    assert "n=0" not in text
    assert bimodality(np.zeros(0)).describe().startswith("too few readings")
    assert "split at" in bimodality(np.array([1.0, 2.0, 9.0, 10.0])).describe()


def test_a_multi_run_selection_summary_counts_only_its_runs(sample):
    """The Selection dock integrated dose across the unselected stretch and
    printed "nan km" for the distance."""
    from rcta.core.quality import assess
    from rcta.core.profile import summarise
    track = assess(sample)
    a = np.zeros(track.n, bool); a[1000:1050] = True
    b = np.zeros(track.n, bool); b[1060:1100] = True
    one, two, both = (summarise(track, mask=m) for m in (a, b, a | b))
    assert both.accumulated_dose_nsv == pytest.approx(
        one.accumulated_dose_nsv + two.accumulated_dose_nsv)
    assert both.coverage < 1.0
    assert "nan" not in both.describe()


def test_a_selection_measures_the_same_distance_as_the_track(walk_path):
    """The summary re-smoothed the selection on its own and reported a
    shorter path (256 m) than the track gives for those samples (313 m)."""
    from rcta.core.io_radiacode import load_track
    from rcta.core.quality import assess
    from rcta.core.profile import summarise
    track = assess(load_track(walk_path))
    mask = np.zeros(track.n, bool); mask[20:60] = True
    s = summarise(track, mask=mask)
    along = track.distance_along_m
    assert s.distance_m == pytest.approx(along[59] - along[20])
    assert s.speed_median == pytest.approx(np.median(track.speed_m_s[mask]))
    # The whole track is unchanged.
    assert summarise(track).distance_m == pytest.approx(along[-1])
