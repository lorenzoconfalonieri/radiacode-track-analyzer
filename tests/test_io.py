"""Parsing, including every tolerance the real export forced on us."""

from __future__ import annotations

import json
import textwrap
from datetime import timezone

import numpy as np
import pytest

from rcta.core.io_radiacode import (TrackParseError, load_rctrk_tsv, load_track)


# ---- the real file -------------------------------------------------------


def test_sample_sample_count(sample):
    assert sample.n == 2964


def test_sample_metadata(sample):
    assert sample.meta.title == "Track 07 Sep 2026 07:53:59"
    assert sample.meta.devices == ("RadiaCode-102",)
    assert sample.meta.source_format == "rctrk-json"
    assert sample.meta.spectrum_valid is False
    assert sample.meta.start_time.astimezone(timezone.utc).timestamp() == 1788785639


def test_sample_has_one_period_with_the_device_odometer(sample):
    assert len(sample.meta.periods) == 1
    assert sample.meta.declared_distance_m == pytest.approx(1795151.69)


def test_sample_is_sorted_on_load_and_says_so(sample):
    """The file's markers are shuffled; silence here would be the bug."""
    assert np.all(np.diff(sample.time_s) >= 0)
    assert any("not in time order" in w for w in sample.meta.warnings)


def test_sample_time_bounds(sample):
    assert sample.time_s[0] == 1788785639
    assert sample.time_s[-1] == 1788795393


def test_rate_columns_are_float_despite_integer_literals(sample):
    """25 doseRate and 32 countRate values are bare ints in the JSON."""
    assert sample.dose_rate.dtype == np.float64
    assert sample.count_rate.dtype == np.float64


def test_no_samples_were_dropped(sample):
    assert sample.valid_mask.all()


# ---- tolerances ----------------------------------------------------------


def _write(tmp_path, payload):
    path = tmp_path / "t.rctrk"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _marker(date, **kw):
    base = {"date": date, "lat": 30.0, "lon": -97.0, "acc": 5,
            "doseRate": 90.0, "countRate": 18.0}
    base.update(kw)
    return base


def test_marker_keys_are_read_by_name_not_position(tmp_path):
    """81 distinct key orderings appear in one real file."""
    a = {"lat": 30.0, "doseRate": 90.0, "date": 100, "acc": 5,
         "countRate": 18.0, "lon": -97.0}
    b = {"countRate": 19.0, "date": 102, "lon": -97.1, "lat": 30.1,
         "acc": 6, "doseRate": 95.0}
    track = load_track(_write(tmp_path, {"markers": [a, b]}))
    assert track.dose_rate.tolist() == [900.0, 950.0]   # uR/h -> nSv/h
    assert track.count_rate.tolist() == [18.0, 19.0]


def test_integer_rates_are_coerced(tmp_path):
    track = load_track(_write(tmp_path, {
        "markers": [_marker(100, doseRate=91, countRate=18),
                    _marker(102, doseRate=90.5, countRate=17.6)]}))
    assert track.dose_rate[0] == pytest.approx(910.0)
    assert track.count_rate[0] == pytest.approx(18.0)


def test_shuffled_markers_are_sorted_and_counted(tmp_path):
    track = load_track(_write(tmp_path, {
        "markers": [_marker(300), _marker(100), _marker(200)]}))
    assert track.time_s.tolist() == [100.0, 200.0, 300.0]
    assert any("1 inversions" in w for w in track.meta.warnings)


def test_a_marker_missing_a_field_is_masked_not_dropped(tmp_path):
    """The sample count in the summary must still match the file."""
    bad = {"date": 150, "lat": 30.0, "lon": -97.0}
    track = load_track(_write(tmp_path, {
        "markers": [_marker(100), bad, _marker(200)]}))
    assert track.n == 3
    assert int((~track.valid_mask).sum()) == 1
    assert any("masked not dropped" in w for w in track.meta.warnings)


def test_a_masked_marker_does_not_teleport_the_track_to_1970(tmp_path):
    """A zero epoch would drag the bbox to the Gulf of Guinea."""
    bad = {"date": 150, "lat": 30.0}
    track = load_track(_write(tmp_path, {
        "markers": [_marker(1788785639), bad, _marker(1788785641)]}))
    assert track.time_s.min() > 1e9
    assert abs(track.lat).max() < 90


def test_missing_accuracy_defaults_to_zero(tmp_path):
    marker = {"date": 100, "lat": 30.0, "lon": -97.0,
              "doseRate": 90.0, "countRate": 18.0}
    track = load_track(_write(tmp_path, {"markers": [marker]}))
    assert track.accuracy_m.tolist() == [0.0]


def test_a_malformed_period_is_ignored_with_a_warning(tmp_path):
    track = load_track(_write(tmp_path, {
        "markers": [_marker(100)], "periods": [{"distance": "?"}]}))
    assert track.meta.periods == ()
    assert any("period" in w for w in track.meta.warnings)


# ---- errors --------------------------------------------------------------


@pytest.mark.parametrize("payload,message", [
    ({}, "no markers"),
    ({"markers": []}, "no markers"),
    ({"markers": [{"date": 1}]}, "no usable markers"),
])
def test_unrecoverable_inputs_raise(tmp_path, payload, message):
    with pytest.raises(TrackParseError, match=message):
        load_track(_write(tmp_path, payload))


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.rctrk"
    path.write_text("")
    with pytest.raises(TrackParseError, match="empty"):
        load_track(path)


def test_missing_file_raises_track_parse_error(tmp_path):
    with pytest.raises(TrackParseError, match="cannot read"):
        load_track(tmp_path / "nope.rctrk")


def test_top_level_array_raises(tmp_path):
    path = tmp_path / "a.rctrk"
    path.write_text("[1, 2, 3]")
    with pytest.raises(TrackParseError):
        load_track(path)


def test_parse_error_carries_its_location(tmp_path):
    try:
        load_track(_write(tmp_path, {}))
    except TrackParseError as exc:
        assert exc.key == "markers"
        assert "t.rctrk" in exc.path
    else:
        pytest.fail("expected TrackParseError")


# ---- the tab-separated dialect -------------------------------------------


TSV = textwrap.dedent("""\
    Track: Walk 1
    Timestamp\tLatitude\tLongitude\tAccuracy\tDoseRate\tCountRate\tComment
    1788785639\t30.100000\t-97.700000\t8\t95.0\t18.5\t
    1788785641\t30.100100\t-97.700100\t9\t96.5\t18.9\tmarker
    1788785643\t30.100200\t-97.700200\t7\t94.0\t18.1\t
    """)


def _tsv(tmp_path, text=TSV):
    path = tmp_path / "walk.rctrk"
    path.write_text(text)
    return path


def test_tsv_dialect_is_detected_from_content_not_extension(tmp_path):
    track = load_track(_tsv(tmp_path))
    assert track.meta.source_format == "rctrk-tsv"
    assert track.n == 3
    assert track.meta.title == "Walk 1"


def test_tsv_columns_are_matched_by_name_not_order(tmp_path):
    reordered = textwrap.dedent("""\
        Track: Reordered
        CountRate\tDoseRate\tLongitude\tLatitude\tTimestamp\tAccuracy
        18.5\t95.0\t-97.7\t30.1\t1788785639\t8
        """)
    track = load_rctrk_tsv(_tsv(tmp_path, reordered))
    assert track.dose_rate.tolist() == [950.0]
    assert track.count_rate.tolist() == [18.5]
    assert track.lat.tolist() == [30.1]


def test_tsv_has_no_odometer_and_says_so_by_absence(tmp_path):
    """Reported as unavailable, never faked from our own geometry."""
    track = load_track(_tsv(tmp_path))
    assert track.meta.periods == ()
    assert track.meta.declared_distance_m is None


def test_tsv_iso_timestamps_are_accepted(tmp_path):
    iso = textwrap.dedent("""\
        Track: ISO
        Timestamp\tLatitude\tLongitude\tAccuracy\tDoseRate\tCountRate
        2026-09-07 12:53:59\t30.1\t-97.7\t8\t95.0\t18.5
        2026-09-07 12:54:01\t30.2\t-97.8\t8\t96.0\t18.7
        """)
    track = load_rctrk_tsv(_tsv(tmp_path, iso))
    assert track.time_s[0] == 1788785639


def test_tsv_millisecond_epochs_are_detected(tmp_path):
    ms = textwrap.dedent("""\
        Track: MS
        Timestamp\tLatitude\tLongitude\tAccuracy\tDoseRate\tCountRate
        1788785639000\t30.1\t-97.7\t8\t95.0\t18.5
        """)
    track = load_rctrk_tsv(_tsv(tmp_path, ms))
    assert track.time_s[0] == pytest.approx(1788785639)


def test_tsv_unparseable_row_is_skipped_with_a_warning(tmp_path):
    broken = TSV + "not\ta\trow\n"
    track = load_rctrk_tsv(_tsv(tmp_path, broken))
    assert track.n == 3
    assert any("could not be parsed" in w for w in track.meta.warnings)


def test_tsv_without_required_columns_raises(tmp_path):
    bad = "Track: X\nTimestamp\tLatitude\n1\t2\n"
    with pytest.raises(TrackParseError, match="header lacks"):
        load_rctrk_tsv(_tsv(tmp_path, bad))
