"""Exporters: round-tripping, leg splitting, and unit honesty."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import synthetic

from rcta.core.analysis import AnalysisParams, analyze_track
from rcta.core.io_export import (GPX_NAMESPACE, anomalies_to_csv,
                                 hotspots_to_csv, hotspots_to_geojson,
                                 report_to_json, track_to_csv,
                                 track_to_geojson, track_to_gpx)
from rcta.core.quality import assess_with_report


@pytest.fixture(scope="module")
def gapped():
    """A track with three dropouts, so leg splitting is actually exercised."""
    track, _ = synthetic.make(n=400, path="line", speed_m_s=20.0,
                              shuffle=False, seed=1,
                              gaps=[(100, 600.0), (200, 900.0), (300, 400.0)])
    return assess_with_report(track)


# ---- CSV -----------------------------------------------------------------


def test_track_csv_has_one_row_per_sample(gapped):
    track, quality = gapped
    lines = track_to_csv(track, quality=quality).strip().splitlines()
    assert len(lines) == track.n + 1


def test_track_csv_names_its_dose_unit_in_the_header(gapped):
    """A bare "dose_rate" column is how a nanosievert file gets read as
    microsieverts three months later."""
    track, quality = gapped
    assert "dose_rate_nSv_per_h" in track_to_csv(track, quality=quality)
    assert "dose_rate_uSv_per_h" in track_to_csv(track, quality=quality,
                                                 unit="uSv/h")


def test_track_csv_converts_the_values_with_the_header(gapped):
    track, quality = gapped
    rows = track_to_csv(track, quality=quality, unit="uSv/h").splitlines()
    header = rows[0].split(",")
    column = header.index("dose_rate_uSv_per_h")
    assert float(rows[1].split(",")[column]) == pytest.approx(
        track.dose_rate[0] / 1000.0, rel=1e-6)


def test_track_csv_rejects_an_unknown_unit(gapped):
    track, quality = gapped
    with pytest.raises(ValueError, match="unknown dose unit"):
        track_to_csv(track, quality=quality, unit="rem/h")


def test_anomaly_csv_includes_excluded_candidates_and_their_reason():
    """A reviewer needs to see what was flagged, not just what survived."""
    track, _ = synthetic.make(n=1200, drift=(5.0, 60.0), shuffle=False, seed=2)
    a = analyze_track(track, AnalysisParams(n_boot=50))
    text = anomalies_to_csv(a.anomalies)
    assert "significant" in text and "flags" in text
    assert len(text.strip().splitlines()) == len(a.anomalies) + 1


def test_hotspot_csv_header_is_stable():
    assert hotspots_to_csv([]).strip() == (
        "lat,lon,radius_m,n_samples,n_passes,first_iso,last_iso,"
        "peak_dose_rate,mean_net_dose_rate,net_counts,significance_sigma,"
        "score,evidence,flags")


# ---- GeoJSON -------------------------------------------------------------


def test_geojson_parses_and_has_one_linestring_per_leg(gapped):
    track, quality = gapped
    data = json.loads(track_to_geojson(track, quality=quality))
    lines = [f for f in data["features"]
             if f["geometry"]["type"] == "LineString"]
    assert len(lines) == quality.n_legs == 4


def test_no_linestring_spans_a_dropout(gapped):
    """The claim a single continuous line would make is one the data cannot
    support: the ground inside a dropout was never observed."""
    track, quality = gapped
    data = json.loads(track_to_geojson(track, quality=quality,
                                       include_points=False))
    gap_indices = set(np.flatnonzero(quality.gap_after).tolist())
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"]
        start = feature["properties"]["n_samples"]
        assert start == len(coords)
    # Every leg boundary must correspond to a recorded gap.
    assert len(data["features"]) == len(gap_indices) + 1


def test_geojson_uses_lon_lat_order(gapped):
    """RFC 7946 order. Reversed, the whole survey renders in the ocean."""
    track, quality = gapped
    data = json.loads(track_to_geojson(track, quality=quality))
    lon, lat = data["features"][0]["geometry"]["coordinates"][0]
    assert lon == pytest.approx(float(track.lon[0]), abs=1e-6)
    assert lat == pytest.approx(float(track.lat[0]), abs=1e-6)
    assert -180 <= lon <= 180 and -90 <= lat <= 90


def test_geojson_points_can_be_omitted(gapped):
    track, quality = gapped
    with_points = json.loads(track_to_geojson(track, quality=quality))
    without = json.loads(track_to_geojson(track, quality=quality,
                                          include_points=False))
    assert len(with_points["features"]) > len(without["features"])


def test_geojson_omits_position_masked_samples(gapped):
    track, quality = gapped
    masked = track.with_mask(position=np.zeros(track.n, dtype=bool))
    data = json.loads(track_to_geojson(masked, quality=quality))
    assert data["features"] == []


def test_hotspot_geojson_is_a_valid_feature_collection():
    data = json.loads(hotspots_to_geojson([]))
    assert data["type"] == "FeatureCollection" and data["features"] == []


# ---- GPX -----------------------------------------------------------------


def test_gpx_parses_and_declares_its_namespace(gapped):
    track, quality = gapped
    root = ET.fromstring(track_to_gpx(track, quality=quality))
    assert root.tag.endswith("gpx")
    assert root.attrib["version"] == "1.1"


def test_gpx_has_one_trkseg_per_leg(gapped):
    track, quality = gapped
    root = ET.fromstring(track_to_gpx(track, quality=quality))
    segs = root.findall(".//{http://www.topografix.com/GPX/1/1}trkseg")
    assert len(segs) == quality.n_legs == 4


def test_gpx_puts_dose_in_its_own_namespace_not_a_borrowed_field(gapped):
    """Garmin's TrackPointExtension has no field meaning nSv/h; a tool that
    understood that schema would read a survey as a cardio workout."""
    track, quality = gapped
    xml = track_to_gpx(track, quality=quality)
    assert GPX_NAMESPACE in xml
    assert "TrackPointExtension" not in xml
    assert "hr>" not in xml

    root = ET.fromstring(xml)
    dose = root.findall(f".//{{{GPX_NAMESPACE}}}doseRate")
    assert dose and dose[0].attrib["units"] == "nSv/h"


def test_gpx_escapes_a_hostile_title():
    track, _ = synthetic.make(n=20, shuffle=False)
    from dataclasses import replace
    track = track.with_mask(meta=replace(track.meta, title="a & b <c>"))
    xml = track_to_gpx(track)
    assert "a &amp; b &lt;c&gt;" in xml
    ET.fromstring(xml)


# ---- the JSON report -----------------------------------------------------


def test_report_records_every_parameter_used():
    track, _ = synthetic.make(n=600, shuffle=False, seed=3)
    a = analyze_track(track, AnalysisParams(n_boot=50))
    data = json.loads(report_to_json(a))
    params = data["parameters"]
    assert params["background"]["method"] == "rolling"
    assert params["background"]["trim_fraction"] == 0.25
    assert params["scan"]["bonferroni"] is True
    assert params["scan"]["scales"] == [1, 2, 4, 8, 16, 32, 64]


def test_report_flags_that_the_integration_time_was_inferred():
    track, _ = synthetic.make(n=600, shuffle=False, seed=4)
    data = json.loads(report_to_json(analyze_track(track,
                                                   AnalysisParams(n_boot=50))))
    assert data["track"]["integration_time_inferred"] is True


def test_report_carries_both_dose_totals():
    track, _ = synthetic.make(n=600, shuffle=False, seed=5,
                              gaps=[(300, 900.0)])
    data = json.loads(report_to_json(analyze_track(track,
                                                   AnalysisParams(n_boot=50))))
    assert data["dose"]["total_nsv"] < data["dose"]["naive_total_nsv"]
    assert 0.0 < data["dose"]["coverage"] < 1.0


def test_report_includes_the_detection_limits_even_with_no_detections():
    track, _ = synthetic.make(n=1200, shuffle=False, seed=6)
    data = json.loads(report_to_json(analyze_track(track,
                                                   AnalysisParams(n_boot=50))))
    assert data["anomalies"] == [] or all(
        not a["significant"] for a in data["anomalies"])
    assert len(data["detection_limits"]) == 7


def test_report_is_valid_json_for_the_real_file(sample):
    a = analyze_track(sample, AnalysisParams(n_boot=50))
    data = json.loads(report_to_json(a))
    assert data["track"]["n_samples"] == 2964
    assert data["quality"]["n_legs"] == 14
    assert data["track"]["declared_distance_m"] == pytest.approx(1795151.69)
    assert data["warnings"]


def test_geojson_records_legs_it_could_not_export(sample):
    """A leg whose fixes were all masked cannot be a LineString, and dropping
    it silently would hide a third of the track on the bundled file."""
    from rcta.core.quality import assess_with_report

    track, quality = assess_with_report(sample)
    data = json.loads(track_to_geojson(track, quality=quality,
                                       include_points=False))
    omitted = data["properties"]["legs_omitted"]
    assert len(data["features"]) + len(omitted) == quality.n_legs == 14
    assert omitted, "this file does have unexportable legs"
    assert all(entry["n_positioned"] < 2 for entry in omitted)


# ---- altitude samples -----------------------------------------------------


@pytest.fixture()
def join(sample_trajectory_path, assessed):
    from rcta.core.flight import join_altitude
    from rcta.core.io_flight import load_trajectory
    return join_altitude(assessed, load_trajectory(sample_trajectory_path))


def _rows(text):
    return [line.split(",") for line in text.strip().splitlines()]


def test_the_samples_export_has_one_row_per_joined_sample(join):
    from rcta.core.io_export import altitude_samples_to_csv
    rows = _rows(altitude_samples_to_csv(join))
    assert len(rows) - 1 == join.n_joined


def test_the_samples_export_carries_what_a_reanalysis_needs(join):
    """The point of it: someone else's binning, fitting and error bars."""
    from rcta.core.io_export import altitude_samples_to_csv
    header = _rows(altitude_samples_to_csv(join))[0]
    for column in ("altitude_m", "dose_rate_nSv_per_h", "count_rate_cps",
                   "counts", "integration_time_s", "phase", "utc"):
        assert column in header, column


def test_the_assumed_integration_time_is_on_every_row(join):
    """It is an assumption, and Poisson errors recomputed without it are wrong."""
    from rcta.core.io_export import altitude_samples_to_csv
    rows = _rows(altitude_samples_to_csv(join))
    column = rows[0].index("integration_time_s")
    values = {row[column] for row in rows[1:]}
    assert values == {f"{join.track.effective_integration_time_s:.1f}"}


def test_counts_are_consistent_with_the_rate_and_the_live_time(join):
    from rcta.core.io_export import altitude_samples_to_csv
    rows = _rows(altitude_samples_to_csv(join))
    head = rows[0]
    cps = head.index("count_rate_cps")
    counts = head.index("counts")
    dt = head.index("integration_time_s")
    for row in rows[1:50]:
        assert float(row[counts]) == pytest.approx(
            float(row[cps]) * float(row[dt]), abs=0.1)


def test_the_samples_export_never_writes_a_row_without_an_altitude(join):
    """A blank altitude cannot be plotted against one, and reads as sea level."""
    from rcta.core.io_export import altitude_samples_to_csv
    rows = _rows(altitude_samples_to_csv(join))
    column = rows[0].index("altitude_m")
    assert all(row[column] not in ("", "nan") for row in rows[1:])


@pytest.mark.parametrize("branch", ["climb", "descent", "level"])
def test_the_samples_export_can_be_restricted_to_one_phase(join, branch):
    from rcta.core.io_export import altitude_samples_to_csv
    rows = _rows(altitude_samples_to_csv(join, branch=branch))
    column = rows[0].index("phase")
    assert len(rows) > 1
    assert {row[column] for row in rows[1:]} == {branch}
    assert len(rows) - 1 == int(join.branch_mask(branch).sum())


def test_the_phases_partition_the_join(join):
    from rcta.core.io_export import altitude_samples_to_csv
    total = sum(len(_rows(altitude_samples_to_csv(join, branch=b))) - 1
                for b in ("climb", "descent", "level"))
    assert total == join.n_joined


def test_the_samples_export_honours_the_unit(join):
    from rcta.core.io_export import altitude_samples_to_csv
    rows = _rows(altitude_samples_to_csv(join, unit="uSv/h"))
    assert "dose_rate_uSv_per_h" in rows[0]
    nsv = _rows(altitude_samples_to_csv(join))
    column_a = rows[0].index("dose_rate_uSv_per_h")
    column_b = nsv[0].index("dose_rate_nSv_per_h")
    assert float(rows[1][column_a]) == pytest.approx(
        float(nsv[1][column_b]) / 1000.0, rel=1e-3)


def test_every_flight_csv_parses_as_plain_csv(join):
    """No comment lines, no preamble: these are read by other software."""
    import csv
    import io

    from rcta.core.flight import bin_by_altitude
    from rcta.core.io_export import (altitude_profile_to_csv,
                                     altitude_samples_to_csv)
    from rcta.core.io_flight import trajectory_to_csv
    for text in (altitude_samples_to_csv(join),
                 altitude_profile_to_csv(bin_by_altitude(join)),
                 trajectory_to_csv(join.trajectory)):
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert rows and all(None not in row for row in rows)
        assert not text.lstrip().startswith("#")
