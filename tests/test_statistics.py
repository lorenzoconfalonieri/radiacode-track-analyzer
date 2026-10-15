"""Counting statistics, including the tests that keep phantom significance dead."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core import statistics as st


# ---- Currie limits -------------------------------------------------------


def test_the_familiar_95_percent_coefficients_appear():
    """L_D = 2.71 + 3.29 sigma_0 is the textbook form; if the algebra drifts,
    this is where it shows."""
    limits = st.currie_limits(100.0, 1, None)
    sigma = 10.0
    assert limits.critical_level == pytest.approx(st.K_95 * sigma)
    assert limits.detection_limit == pytest.approx(2.706 + 3.2897 * sigma, rel=1e-3)


def test_a_short_reference_window_costs_sensitivity():
    """sigma_0^2 = B(1 + n_p/n_b): the whole point of tracking n_b."""
    wide = st.currie_limits(100.0, 10, 1000)
    narrow = st.currie_limits(100.0, 10, 10)
    assert narrow.critical_level > wide.critical_level
    assert narrow.detection_limit > wide.detection_limit


def test_known_background_is_the_optimistic_bound():
    exact = st.currie_limits(100.0, 10, None)
    estimated = st.currie_limits(100.0, 10, 20)
    assert exact.critical_level < estimated.critical_level


def test_zero_background_still_has_a_positive_detection_limit():
    """Even with no background you need 2.71 counts; a limit of zero would
    make every empty window a detection."""
    limits = st.currie_limits(0.0, 1, 10)
    assert limits.critical_level == 0.0
    assert limits.detection_limit == pytest.approx(2.706, rel=1e-3)


def test_mda_scales_inversely_with_live_time():
    short = st.currie_limits(100.0, 10, 100, live_time=10.0)
    long = st.currie_limits(100.0, 10, 100, live_time=100.0)
    assert short.mda_cps == pytest.approx(10.0 * long.mda_cps)


def test_mda_converts_to_dose_with_the_tracks_own_ratio():
    limits = st.currie_limits(100.0, 10, 100, live_time=20.0)
    assert limits.mda_dose_rate(5.129) == pytest.approx(limits.mda_cps * 5.129)


@pytest.mark.parametrize("kwargs", [
    dict(background_counts=-1.0, n_peak_channels=1),
    dict(background_counts=1.0, n_peak_channels=0),
    dict(background_counts=1.0, n_peak_channels=1, n_background_channels=0),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        st.currie_limits(**kwargs)


def test_currie_limits_rejects_non_positive_live_time():
    with pytest.raises(ValueError, match="live_time"):
        st.currie_limits(10.0, 1, live_time=0.0)


# ---- significance --------------------------------------------------------


def test_significance_is_zero_when_gross_equals_background():
    assert st.net_significance(100.0, 100.0, 10, 100) == pytest.approx(0.0)


def test_significance_is_vectorised():
    out = st.net_significance(np.array([100.0, 200.0]), np.array([100.0, 100.0]),
                              np.array([10, 10]), np.array([100, 100]))
    assert out.shape == (2,)
    assert out[1] > out[0]


def test_significance_is_not_net_over_sqrt_net():
    """The naive form overstates a weak excess; this is the classic error."""
    gross, background = 130.0, 100.0
    naive = (gross - background) / np.sqrt(gross - background)
    assert st.net_significance(gross, background, 10, 100) < naive


def test_false_positive_rate_matches_the_nominal_alpha():
    """Monte Carlo over pure background: L_C must mean what it says.

    If a refactor ever changes the variance model,
    the realised false-positive rate moves and this catches it, where a
    tolerance test on a single number would not.
    """
    rng = np.random.default_rng(7)
    trials, n_peak, n_bg, true_b = 4000, 8, 80, 12.0
    false_positives = 0
    for _ in range(trials):
        gross = rng.poisson(true_b * n_peak)
        side = rng.poisson(true_b * n_bg)
        b_hat = side * n_peak / n_bg
        limits = st.currie_limits(b_hat, n_peak, n_bg)
        if gross - b_hat > limits.critical_level:
            false_positives += 1
    rate = false_positives / trials
    assert 0.03 < rate < 0.075, rate


# ---- rolling quantile ----------------------------------------------------


def test_rolling_quantile_tracks_a_step():
    t = np.arange(200.0)
    v = np.where(t < 100, 10.0, 30.0)
    q, _ = st.rolling_quantile(v, t, window_s=20.0, q=0.5)
    assert q[10] == pytest.approx(10.0)
    assert q[190] == pytest.approx(30.0)


def test_rolling_window_is_measured_in_seconds_not_samples():
    """The dropout trap: an index window would reach across the gap and pull
    the background from the other side of it."""
    t = np.concatenate([np.arange(50.0), np.arange(50.0) + 5000.0])
    v = np.concatenate([np.full(50, 10.0), np.full(50, 40.0)])
    q, counts = st.rolling_quantile(v, t, window_s=20.0, q=0.5, min_samples=3)
    assert q[25] == pytest.approx(10.0)
    assert q[75] == pytest.approx(40.0)
    assert counts[25] <= 21


def test_a_low_quantile_ignores_a_hotspot_the_median_would_absorb():
    """The median of a window that is half hotspot is the hotspot."""
    t = np.arange(100.0)
    v = np.full(100, 10.0)
    v[40:90] = 100.0                       # half the window is elevated
    q25, _ = st.rolling_quantile(v, t, window_s=100.0, q=0.25)
    q50, _ = st.rolling_quantile(v, t, window_s=100.0, q=0.5)
    assert q25[65] < q50[65]


def test_undersupported_windows_fall_back_and_report_their_count():
    t = np.array([0.0, 1000.0, 2000.0])
    v = np.array([1.0, 2.0, 3.0])
    q, counts = st.rolling_quantile(v, t, window_s=10.0, q=0.5, min_samples=5)
    assert counts.tolist() == [1, 1, 1]
    assert np.allclose(q, np.quantile(v, 0.5))


def test_rolling_quantile_handles_an_empty_series():
    q, counts = st.rolling_quantile([], [], window_s=10.0)
    assert q.size == 0 and counts.size == 0


# ---- autocorrelation and the bootstrap -----------------------------------


def test_white_noise_is_uncorrelated_and_keeps_its_sample_size():
    rng = np.random.default_rng(1)
    v = rng.normal(size=4000)
    r, n_eff = st.autocorr_time(v)
    assert abs(r) < 0.05
    assert n_eff > 3000


def test_a_smooth_series_collapses_the_effective_sample_size():
    v = np.cumsum(np.random.default_rng(2).normal(size=3000))
    r, n_eff = st.autocorr_time(v)
    assert r > 0.99
    assert n_eff < 50


def test_autocorr_of_a_constant_series_is_defined():
    r, n_eff = st.autocorr_time(np.ones(100))
    assert r == 0.0 and n_eff == 100.0


def test_effective_p_does_not_call_two_random_walks_significant():
    """Two independent random walks correlate strongly by construction. This
    is the spurious-regression trap, and effective_p is what avoids it."""
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=1000))
    y = np.cumsum(rng.normal(size=1000))
    rho = st.spearman_rho(x, y)
    assert abs(rho) > 0.5, "the walks really are strongly correlated"
    assert st.effective_p(rho, st.bartlett_effective_n(x, y)) > 0.05


def test_the_block_bootstrap_is_anticonservative_on_a_unit_root():
    """Pins a known limitation so nobody later promotes the bootstrap.

    Block resampling preserves dependence only up to the block length, so a
    long-range trend is destroyed and the surrogates are far less correlated
    than the data. On two independent random walks it reports significance
    where effective_p does not -- which is exactly why Correlation leads with
    effective_p and carries this one only for comparison.
    """
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 2000.0, 2.0)
    x = np.cumsum(rng.normal(size=t.size))
    y = np.cumsum(rng.normal(size=t.size))
    bootstrap = st.block_bootstrap_p(x, y, t, block_s=200.0, n_boot=400, seed=0)
    honest = st.effective_p(st.spearman_rho(x, y), st.bartlett_effective_n(x, y))
    assert bootstrap < honest


def test_effective_p_is_one_when_nothing_survives_the_correction():
    assert st.effective_p(0.9, 2.0) == 1.0
    assert st.effective_p(0.9, 1.0) == 1.0


def test_bartlett_penalises_two_smooth_series_hardest():
    rng = np.random.default_rng(11)
    n = 2000
    rough = rng.normal(size=n)
    smooth = np.cumsum(rng.normal(size=n))
    assert st.bartlett_effective_n(rough, rough) > 0.9 * n
    assert st.bartlett_effective_n(smooth, smooth) < 0.05 * n


def test_block_bootstrap_still_finds_a_real_association():
    rng = np.random.default_rng(4)
    t = np.arange(0.0, 2000.0, 2.0)
    x = np.cumsum(rng.normal(size=t.size))
    y = 3.0 * x + rng.normal(scale=0.1, size=t.size)
    p = st.block_bootstrap_p(x, y, t, block_s=200.0, n_boot=400, seed=0)
    assert p < 0.05


def test_bootstrap_never_reports_exactly_zero():
    """A p of 0 is not something 2000 resamples can justify."""
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 1000.0, 2.0)
    x = np.arange(t.size, dtype=float)
    y = x + rng.normal(scale=1e-6, size=t.size)
    assert st.block_bootstrap_p(x, y, t, n_boot=100, seed=0) > 0.0


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(6)
    x = rng.normal(size=200)
    y = x ** 3 + rng.normal(size=200)
    assert st.spearman_rho(x, y) == pytest.approx(spearmanr(x, y).statistic)


def test_spearman_handles_ties():
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 2.0, 3.0, 3.0])
    assert -1.0 <= st.spearman_rho(x, y) <= 1.0


# ---- goldens against the real sample -------------------------------------


def test_sample_count_rate_is_almost_perfectly_autocorrelated(sample):
    r, n_eff = st.autocorr_time(sample.count_rate)
    assert r == pytest.approx(0.9985, abs=1e-4)
    assert n_eff < 5, "2964 samples carry the information of about two"


def test_sample_dose_rate_is_a_near_unit_root(sample):
    r, _ = st.autocorr_time(sample.dose_rate)
    assert r == pytest.approx(0.9973, abs=1e-4)


def test_sample_speed_dose_correlation_survives_correction_but_is_confounded(sample):
    """The measured outcome, which is not the intuitive one.

    Speed is only mildly autocorrelated (r = 0.81) even though dose rate is
    nearly a unit root, so the Bartlett effective size is about 307 rather
    than the ~2 that the count-rate series alone would suggest. The
    correlation therefore does survive the serial-dependence correction. What
    it does not survive is physics: on a climbing aircraft both speed and dose
    rate are driven by altitude, so this is confounding, not causation, and
    Correlation.describe() has to say so in words because no p-value can.
    """
    speed, dose = sample.speed_m_s[1:], sample.dose_rate[1:]
    rho = st.spearman_rho(speed, dose)
    assert rho == pytest.approx(0.6387, abs=1e-3)

    n_eff = st.bartlett_effective_n(speed, dose)
    assert n_eff == pytest.approx(307.0, rel=0.05)
    assert n_eff < 0.15 * sample.n, "the nominal sample size is still a fiction"
    assert st.effective_p(rho, n_eff) < 0.01


# ---- rolling trimmed mean --------------------------------------------------


def test_the_trimmed_mean_drops_the_same_count_from_each_tail():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], float)
    # 0.2 of 10 is 2 from each end: the mean of 3..8.
    assert st._trimmed_mean(values, 0.2) == pytest.approx(5.5)
    assert st._trimmed_mean(values, 0.0) == pytest.approx(values.mean())


def test_trimming_never_empties_a_tiny_window():
    # round(0.49 * 2) = 1 from each end would leave nothing; all are kept.
    assert st._trimmed_mean(np.array([2.0, 4.0]), 0.49) == pytest.approx(3.0)


def test_the_window_is_centred_and_inclusive_at_both_ends():
    time_s = np.arange(10, dtype=float)
    values = time_s ** 2
    out, counts = st.rolling_trimmed_mean(values, time_s, window_s=4.0,
                                          trim_fraction=0.0, min_samples=1)
    # Sample 5 sees t = 3..7: five samples, mean of their squares.
    assert counts[5] == 5
    assert out[5] == pytest.approx(np.mean([9, 16, 25, 36, 49]))
    # At the edge the window is truncated, not padded.
    assert counts[0] == 3
    assert out[0] == pytest.approx(np.mean([0, 1, 4]))


def test_the_window_is_in_time_not_samples():
    """Across a dropout the neighbours by index are not neighbours in time."""
    time_s = np.array([0, 1, 2, 3, 4, 1000, 1001, 1002, 1003, 1004], float)
    values = np.array([10] * 5 + [50] * 5, float)
    out, counts = st.rolling_trimmed_mean(values, time_s, window_s=20.0,
                                          trim_fraction=0.0, min_samples=1)
    assert (counts == 5).all()
    np.testing.assert_allclose(out, values)


def test_thin_windows_fall_back_to_the_global_trimmed_mean():
    time_s = np.array([0, 1, 2, 3, 4, 500], float)
    values = np.array([10, 10, 10, 10, 10, 90], float)
    out, counts = st.rolling_trimmed_mean(values, time_s, window_s=10.0,
                                          trim_fraction=0.25, min_samples=3)
    assert counts[-1] == 1
    assert out[-1] == pytest.approx(st._trimmed_mean(values, 0.25))
    assert out[0] == pytest.approx(10.0)


def test_rolling_trimmed_mean_validates_and_handles_empty_input():
    out, counts = st.rolling_trimmed_mean(np.zeros(0), np.zeros(0), window_s=10)
    assert out.size == 0 and counts.size == 0
    with pytest.raises(ValueError):
        st.rolling_trimmed_mean(np.ones(5), np.arange(5.0), window_s=10,
                                trim_fraction=0.5)
