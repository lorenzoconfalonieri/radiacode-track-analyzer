"""The CLI, driven through main() and checked on its exit codes and output."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from rcta.cli import main


def test_info_prints_the_summary_and_the_route(sample_path, capsys):
    assert main(["info", str(sample_path)]) == 0
    out = capsys.readouterr().out
    assert "RadiaCode-102" in out
    assert "1792.21 km" in out
    assert "0.16 % apart" in out
    assert "route" in out
    assert "S" in out and "E" in out          # start and end markers


def test_info_warns_about_the_unsorted_markers(sample_path, capsys):
    main(["info", str(sample_path)])
    assert "not in time order" in capsys.readouterr().out


def test_analyze_reports_the_null_result_and_the_limits(sample_path, capsys):
    assert main(["analyze", str(sample_path), "--n-boot", "50"]) == 0
    out = capsys.readouterr().out
    assert "no anomalies above L_C" in out
    assert "minimum detectable dose rate" in out
    assert "excluded by flags" in out


def test_analyze_prints_both_dose_totals(sample_path, capsys):
    main(["analyze", str(sample_path), "--n-boot", "50"])
    out = capsys.readouterr().out
    assert "1234.499 nSv" in out
    assert "1619.004 nSv" in out
    assert "Neither is 'the' dose" in out


def test_analyze_never_prints_a_bare_zero_p_value(sample_path, capsys):
    """Underflow is not certainty."""
    main(["analyze", str(sample_path), "--n-boot", "50"])
    out = capsys.readouterr().out
    assert "p (naive)  : < 1e-300" in out
    assert "p (naive)  : 0\n" not in out


def test_analyze_honours_the_unit_flag(sample_path, capsys):
    main(["analyze", str(sample_path), "--unit", "uSv/h", "--n-boot", "50"])
    assert "uSv/h" in capsys.readouterr().out


def test_analyze_writes_a_json_report(sample_path, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["analyze", str(sample_path), "--json", str(out),
                 "--n-boot", "50"]) == 0
    data = json.loads(out.read_text())
    assert data["track"]["n_samples"] == 2964
    assert data["parameters"]["scan"]["bonferroni"] is True


def test_analyze_writes_an_anomaly_csv(sample_path, tmp_path):
    out = tmp_path / "anomalies.csv"
    main(["analyze", str(sample_path), "--csv", str(out), "--n-boot", "50"])
    assert out.read_text().startswith("start_index,stop_index")


def test_no_bonferroni_changes_the_verdict(sample_path, capsys):
    """The correction is what makes the null result a null result."""
    main(["analyze", str(sample_path), "--no-bonferroni", "--n-boot", "50"])
    uncorrected = capsys.readouterr().out
    main(["analyze", str(sample_path), "--n-boot", "50"])
    corrected = capsys.readouterr().out
    assert "no anomalies above L_C" in corrected
    assert "no anomalies above L_C" not in uncorrected


@pytest.mark.parametrize("fmt,check", [
    ("csv", lambda t: t.startswith("index,iso_time")),
    ("geojson", lambda t: json.loads(t)["type"] == "FeatureCollection"),
    ("gpx", lambda t: ET.fromstring(t).tag.endswith("gpx")),
    ("json", lambda t: json.loads(t)["generator"] == "rcta"),
    ("anomalies", lambda t: t.startswith("start_index")),
    ("hotspots", lambda t: t.startswith("lat,lon")),
])
def test_export_formats(sample_path, tmp_path, fmt, check):
    out = tmp_path / f"out.{fmt}"
    assert main(["export", str(sample_path), "--format", fmt, "-o", str(out),
                 "--n-boot", "50"]) == 0
    assert check(out.read_text())


# ---- error handling ------------------------------------------------------


def test_a_missing_file_exits_two(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nope.rctrk")]) == 2
    assert "error:" in capsys.readouterr().err


def test_a_malformed_file_exits_two(tmp_path, capsys):
    path = tmp_path / "bad.rctrk"
    path.write_text("{not json")
    assert main(["info", str(path)]) == 2


def test_a_bad_parameter_exits_one(sample_path, capsys):
    """A ValueError from the core is a usage problem, not a parse problem."""
    assert main(["analyze", str(sample_path), "--bg-trim", "0.9"]) == 1
    assert "trim_fraction" in capsys.readouterr().err


def test_an_unknown_background_method_is_rejected_by_argparse(sample_path):
    with pytest.raises(SystemExit):
        main(["analyze", str(sample_path), "--background", "magic"])


# ---- flight ---------------------------------------------------------------


def test_flight_profiles_dose_against_altitude(sample_path,
                                               sample_trajectory_path, capsys):
    assert main(["flight", str(sample_path), "--flight", "DL2618",
                 "--trajectory", str(sample_trajectory_path)]) == 0
    out = capsys.readouterr().out
    assert "airborne" in out
    assert "position match : ok" in out
    assert "dose rate against altitude" in out
    assert "altitude (km)" in out               # the ASCII plot
    assert "FL360" in out or "FL35" in out      # the flight-level column


def test_flight_reports_the_controls_on_the_bundled_sample(
        sample_path, sample_trajectory_path, capsys):
    """The bundled trajectory's altitude axis is fabricated.

    So no altitude profile reconciles the climb with the descent, and the
    point of shipping this pair is that the controls say so instead of
    printing a tidy scale height. The real trajectory disagrees too, by less
    -- see docs/FLIGHT.md. If this ever starts agreeing, the controls have
    broken.
    """
    main(["flight", str(sample_path), "--flight", "DL2618",
          "--trajectory", str(sample_trajectory_path)])
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out
    assert "not a function of altitude alone" in out
    assert "single exponential does not describe this profile" in out


def test_flight_needs_a_flight_or_a_trajectory(sample_path, capsys):
    assert main(["flight", str(sample_path)]) == 1
    assert "--trajectory" in capsys.readouterr().err


@pytest.fixture()
def wrong_flight(tmp_path):
    """A well-formed trajectory over England, for a track across Texas."""
    path = tmp_path / "elsewhere.json"
    path.write_text(json.dumps({
        "icao24": "ffffff", "callsign": "XXX1 ",
        "path": [[1788785639, 51.5, -0.1, 150.0, 0.0, True],
                 [1788790000, 52.0, 0.5, 10668.0, 0.0, False],
                 [1788795393, 52.5, 1.0, 200.0, 0.0, False]]}))
    return path


def test_flight_refuses_a_trajectory_that_does_not_match(sample_path,
                                                         wrong_flight, capsys):
    """Exit 4 is reserved for this, so a script can tell it from 'no data'."""
    assert main(["flight", str(sample_path),
                 "--trajectory", str(wrong_flight)]) == 4
    captured = capsys.readouterr()
    assert "MISMATCH" in captured.out
    assert "refusing" in captured.err.lower()


def test_flight_can_be_forced_and_labels_the_result(sample_path, wrong_flight,
                                                    capsys):
    assert main(["flight", str(sample_path), "--trajectory", str(wrong_flight),
                 "--force-match"]) == 0
    assert "MISMATCH, overridden" in capsys.readouterr().out


def test_flight_without_credentials_exits_three_and_says_how(sample_path,
                                                             tmp_path, capsys,
                                                             monkeypatch):
    monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("RCTA_FLIGHT_CREDENTIALS", str(tmp_path / "none.json"))
    monkeypatch.setenv("RCTA_FLIGHT_CACHE", str(tmp_path / "cache"))
    assert main(["flight", str(sample_path), "--flight", "DL2618",
                 "--offline"]) == 3
    err = capsys.readouterr().err
    assert "not in the cache" in err or "anonymous" in err
    assert "--trajectory" in err


def test_flight_writes_the_profile_and_the_report(sample_path,
                                                  sample_trajectory_path,
                                                  tmp_path, capsys):
    csv_path = tmp_path / "altitude.csv"
    json_path = tmp_path / "flight.json"
    assert main(["flight", str(sample_path), "--flight", "DL2618",
                 "--trajectory", str(sample_trajectory_path),
                 "--csv", str(csv_path), "--json", str(json_path)]) == 0

    header = csv_path.read_text().splitlines()[0]
    assert "altitude_m" in header and "counting_sem" in header
    report = json.loads(json_path.read_text())
    assert report["join"]["match"]["matched"] is True
    assert report["trajectory"]["altitude_kind"] == "barometric"
    assert report["climb_vs_descent"]["consistent"] is False


def test_flight_saves_a_trajectory_for_reuse(sample_path,
                                             sample_trajectory_path, tmp_path):
    out = tmp_path / "saved.csv"
    assert main(["flight", str(sample_path), "--flight", "DL2618",
                 "--trajectory", str(sample_trajectory_path),
                 "--save-trajectory", str(out)]) == 0
    assert "altitude_m" in out.read_text().splitlines()[0]
    # And it reads back, which is what "for reuse" has to mean.
    assert main(["flight", str(sample_path), "--trajectory", str(out)]) == 0


def test_flight_honours_the_unit_flag(sample_path, sample_trajectory_path,
                                      capsys):
    main(["flight", str(sample_path), "--flight", "DL2618", "--unit", "uSv/h",
          "--trajectory", str(sample_trajectory_path)])
    assert "dose rate (uSv/h)" in capsys.readouterr().out


def test_flight_warns_when_the_track_is_not_a_flight(tmp_path, capsys):
    """A slow flight is real, so this warns and proceeds rather than refusing."""
    import synthetic
    track, _ = synthetic.make(n=200, speed_m_s=1.4, dt_s=2.0, shuffle=False)
    lines = ["Track: walk",
             "Timestamp\tLatitude\tLongitude\tAccuracy\tDoseRate\tCountRate"]
    for i in range(track.n):
        lines.append(f"{int(track.time_s[i])}\t{track.lat[i]:.6f}\t"
                     f"{track.lon[i]:.6f}\t8\t{track.dose_rate[i] / 10:.2f}\t"
                     f"{track.count_rate[i]:.2f}")
    walk = tmp_path / "walk.rctrk"
    walk.write_text("\n".join(lines))

    trajectory = tmp_path / "t.csv"
    trajectory.write_text(
        "time,lat,lon,altitude_m\n"
        + "\n".join(f"{int(track.time_s[i])},{track.lat[i]:.6f},"
                    f"{track.lon[i]:.6f},{100 + i * 5}"
                    for i in range(0, track.n, 10)))
    assert main(["flight", str(walk), "--trajectory", str(trajectory)]) == 0
    assert "proceeding anyway" in capsys.readouterr().out


def test_flight_rejects_a_malformed_date(sample_path, sample_trajectory_path,
                                         capsys):
    assert main(["flight", str(sample_path), "--flight", "DL2618",
                 "--date", "7 Sept 2026"]) == 1
    assert "YYYY-MM-DD" in capsys.readouterr().err


# ---- --date is a calendar day at the departure airport --------------------

def _slc_evening_track():
    """A track that pushes back from Salt Lake City at 18:06 local.

    The UTC date is already the 11th; the date on the boarding pass is the
    10th. This is the shape that made the two readings disagree.
    """
    import synthetic
    track, _, _ = synthetic.flight_track(
        n=200, start=1789085184, origin=(40.7889, -111.9799),
        destination=(30.1975, -97.662))
    return track


def test_the_date_is_read_at_the_departure_airport(monkeypatch):
    import types

    from rcta.cli import _flight_window

    track = _slc_evening_track()
    args = types.SimpleNamespace(date="2026-09-10", departure="")
    begin, _, notes = _flight_window(track, args)
    assert begin == track.time_s[0]                    # the ticket date works
    assert notes == []


def test_the_utc_date_is_reported_as_a_day_out():
    import types

    from rcta.cli import _flight_window

    track = _slc_evening_track()
    args = types.SimpleNamespace(date="2026-09-11", departure="")
    begin, _, notes = _flight_window(track, args)
    assert begin == pytest.approx(track.time_s[0] + 86400.0)
    assert "America/Denver" in notes[0]


def test_an_explicit_departure_supplies_the_zone_too():
    """--departure exists for a track that does not start on an apron, and
    such a track has no zone of its own to fall back on."""
    import types

    from rcta.cli import _flight_window

    track = _slc_evening_track()
    args = types.SimpleNamespace(date="2026-09-10", departure="KSLC")
    begin, _, notes = _flight_window(track, args)
    assert (begin, notes) == (track.time_s[0], [])


def test_flight_writes_a_per_sample_csv(sample_path, sample_trajectory_path,
                                        tmp_path, capsys):
    out = tmp_path / "samples.csv"
    assert main(["flight", str(sample_path), "--flight", "DL2618",
                 "--trajectory", str(sample_trajectory_path),
                 "--samples-csv", str(out)]) == 0
    assert "one row each" in capsys.readouterr().out
    rows = out.read_text().strip().splitlines()
    assert len(rows) - 1 == 2961
    for column in ("altitude_m", "dose_rate_nSv_per_h", "count_rate_cps",
                   "counts", "integration_time_s", "phase"):
        assert column in rows[0]


def test_the_samples_csv_honours_the_branch_flag(sample_path,
                                                 sample_trajectory_path,
                                                 tmp_path):
    out = tmp_path / "climb.csv"
    main(["flight", str(sample_path), "--trajectory",
          str(sample_trajectory_path), "--branch", "climb",
          "--samples-csv", str(out)])
    rows = out.read_text().strip().splitlines()
    phase = rows[0].split(",").index("phase")
    assert {row.split(",")[phase] for row in rows[1:]} == {"climb"}


def test_flight_can_write_every_export_at_once(sample_path,
                                               sample_trajectory_path,
                                               tmp_path):
    """The whole point: four files, then make your own figures."""
    args = ["flight", str(sample_path), "--trajectory",
            str(sample_trajectory_path),
            "--csv", str(tmp_path / "profile.csv"),
            "--samples-csv", str(tmp_path / "samples.csv"),
            "--save-trajectory", str(tmp_path / "trajectory.csv"),
            "--json", str(tmp_path / "report.json")]
    assert main(args) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "profile.csv", "report.json", "samples.csv", "trajectory.csv"]
    for name in ("profile.csv", "samples.csv", "trajectory.csv"):
        assert (tmp_path / name).read_text().count(",") > 10


def test_every_exported_csv_is_readable_by_a_plain_csv_reader(
        sample_path, sample_trajectory_path, tmp_path):
    """These files exist to be opened by other software."""
    import csv
    main(["flight", str(sample_path), "--trajectory",
          str(sample_trajectory_path),
          "--csv", str(tmp_path / "profile.csv"),
          "--samples-csv", str(tmp_path / "samples.csv"),
          "--save-trajectory", str(tmp_path / "trajectory.csv")])
    for name in ("profile.csv", "samples.csv", "trajectory.csv"):
        with open(tmp_path / name, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, name
        assert all(None not in row for row in rows), name


# ---- defaults that must not drift ----------------------------------------


def test_the_parser_defaults_are_the_core_defaults():
    """Three front ends have to agree on what the default answer is.

    ``--help`` is a published statement about how the program behaves when
    told nothing, so a literal copied into the parser is a lie waiting for
    somebody to tune the dataclass and not the CLI.
    """
    from rcta.cli import build_parser
    from rcta.core.analysis import AnalysisParams
    from rcta.core.anomalies import ScanParams
    from rcta.core.background import BackgroundParams
    from rcta.core.quality import QualityParams

    args = build_parser().parse_args(["analyze", "TRACK"])
    assert args.background == BackgroundParams().method
    assert args.bg_window == BackgroundParams().window_s
    assert args.bg_trim == BackgroundParams().trim_fraction
    assert args.alpha == ScanParams().alpha
    assert args.scales == ",".join(str(s) for s in ScanParams().scales)
    assert args.max_accuracy == QualityParams().max_accuracy_m
    assert args.max_speed == QualityParams().max_speed_m_s
    assert args.gap_threshold == QualityParams().gap_threshold_s
    assert args.n_boot == AnalysisParams().n_boot


def test_the_flight_subcommand_shares_the_same_quality_defaults():
    from rcta.cli import build_parser
    from rcta.core.quality import QualityParams

    args = build_parser().parse_args(["flight", "TRACK"])
    assert args.max_accuracy == QualityParams().max_accuracy_m
    assert args.max_speed == QualityParams().max_speed_m_s


def test_parsing_nothing_reproduces_the_default_analysis_params():
    """The assembled params must equal a bare AnalysisParams, not merely look
    like it: this is what pins the two together end to end."""
    from rcta.cli import _build_params, build_parser
    from rcta.core.analysis import AnalysisParams

    args = build_parser().parse_args(["analyze", "TRACK"])
    assert _build_params(args) == AnalysisParams()


def test_version_is_reported(capsys):
    from rcta import __version__
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_an_unknown_provider_is_an_error_not_a_traceback(sample_path, capsys):
    code = main(["flight", str(sample_path), "--flight", "DL2618",
                 "--provider", "nonesuch", "--offline"])
    assert code == 1
    assert "unknown --provider 'nonesuch'" in capsys.readouterr().err


def test_the_json_report_records_the_applied_background_window(tmp_path):
    """The default window is widened on a 10 s walk; the report must say so."""
    from pathlib import Path
    walk = Path(__file__).resolve().parents[1] / "sample-data" \
        / "Track 07 Sep 2026 14-30-57.rctrk"
    if not walk.exists():
        pytest.skip("walk sample not present")
    out = tmp_path / "walk.json"
    assert main(["analyze", str(walk), "--n-boot", "20",
                 "--json", str(out)]) == 0
    background = json.loads(out.read_text(encoding="utf-8"))[
        "parameters"]["background"]
    assert background["window_s_requested"] == 900.0
    assert background["window_s"] > 900.0


def test_grid_writes_an_asc_and_its_projection(sample_path, tmp_path, capsys):
    out = tmp_path / "surface.asc"
    assert main(["grid", str(sample_path), "-o", str(out), "--grid-size", "64",
                 "--n-boot", "10"]) == 0
    assert out.read_text().startswith("ncols ")
    assert "Mercator_Auxiliary_Sphere" in (tmp_path / "surface.prj").read_text()
    printed = capsys.readouterr().out
    assert "supported" in printed and "NODATA" in printed


def test_grid_excess_needs_no_extra_input(sample_path, tmp_path):
    out = tmp_path / "net.asc"
    assert main(["grid", str(sample_path), "-o", str(out), "--value", "excess",
                 "--grid-size", "32", "--n-boot", "10"]) == 0
    assert out.exists()


def _walk_path():
    from pathlib import Path
    walk = (Path(__file__).resolve().parents[1] / "sample-data"
            / "Track 07 Sep 2026 14-30-57.rctrk")
    if not walk.exists():
        pytest.skip("walk sample not present")
    return walk


def test_info_takes_several_tracks_as_the_readme_shows(sample_path, capsys):
    """`rcta-cli info sample-data/Track*.rctrk` used to be a usage error."""
    assert main(["info", str(sample_path), str(_walk_path())]) == 0
    out = capsys.readouterr().out
    assert "2964" in out and "348" in out


def test_analyze_writes_an_array_of_reports_for_several_tracks(
        sample_path, tmp_path):
    out = tmp_path / "reports.json"
    assert main(["analyze", str(sample_path), str(_walk_path()),
                 "--json", str(out), "--n-boot", "10"]) == 0
    reports = json.loads(out.read_text())
    assert isinstance(reports, list) and len(reports) == 2
    assert [r["track"]["n_samples"] for r in reports] == [2964, 348]


def test_analyze_refuses_one_anomaly_csv_for_several_tracks(sample_path,
                                                            tmp_path, capsys):
    assert main(["analyze", str(sample_path), str(_walk_path()),
                 "--csv", str(tmp_path / "a.csv"), "--n-boot", "10"]) == 1
    assert "single track" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["PySide6", "pyqtgraph"])
def test_the_gui_without_its_extra_explains_how_to_install_it(
        monkeypatch, capsys, missing):
    """A bare `pip install rcta` used to answer `rcta` with a traceback."""
    import sys
    from rcta.gui import app
    monkeypatch.setitem(sys.modules, missing, None)
    assert app.main(["rcta"]) == 1
    err = capsys.readouterr().err
    assert "pip install 'rcta[gui]'" in err and missing in err
    assert "rcta-cli" in err


def test_the_version_is_defined_once():
    """pyproject, the GPX creator and the About box used to repeat it."""
    from pathlib import Path
    import rcta
    from rcta.core.io_export import track_to_gpx
    import inspect
    root = Path(__file__).resolve().parents[1]
    assert 'dynamic = ["version"]' in (root / "pyproject.toml").read_text()
    default = inspect.signature(track_to_gpx).parameters["creator"].default
    assert default == f"rcta {rcta.__version__}"


def test_sparkline_survives_masked_readings():
    """Masked samples are NaN; int(nan) used to crash `rcta-cli info`."""
    import numpy as np
    from rcta.cli import _sparkline
    values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 5.0])
    line = _sparkline(values, width=6)
    assert len(line) == 6
    assert line[1] == "_" and line[3] == "_"
    assert _sparkline(np.full(4, np.nan)) == "____"


def test_info_runs_on_a_track_with_masked_markers(tmp_path, capsys):
    out = tmp_path / "broken.rctrk"
    walk = _walk_path()
    data = json.loads(walk.read_text())
    for marker in data["markers"][::50]:
        marker.pop("doseRate", None)
    out.write_text(json.dumps(data))
    assert main(["info", str(out)]) == 0
    assert "nan" not in capsys.readouterr().out.lower()


def test_info_reports_the_same_distance_as_analyze(capsys):
    """info used to describe the track before quality assessment."""
    walk = str(_walk_path())
    assert main(["info", walk]) == 0
    info = capsys.readouterr().out
    assert main(["analyze", walk, "--n-boot", "10"]) == 0
    analyze = capsys.readouterr().out
    line = next(l for l in info.splitlines() if l.startswith("distance"))
    assert line.split(":")[1].split("(")[0].strip() == "2.99 km"
    assert "2.99 km" in analyze


# ---- every command survives masked markers --------------------------------


def _with_masked_markers(source, tmp_path, every=50):
    """A copy of a sample track with every Nth marker's dose rate removed."""
    data = json.loads(source.read_text())
    for marker in data["markers"][::every]:
        marker.pop("doseRate", None)
    out = tmp_path / f"masked-{source.stem.replace(' ', '_')}.rctrk"
    out.write_text(json.dumps(data))
    return str(out)


_COMMANDS = {
    "info": lambda t, d: ["info", t],
    "analyze": lambda t, d: ["analyze", t, "--n-boot", "10",
                             "--json", f"{d}/r.json", "--csv", f"{d}/a.csv"],
    "export-csv": lambda t, d: ["export", t, "--format", "csv",
                                "-o", f"{d}/t.csv"],
    "export-geojson": lambda t, d: ["export", t, "--format", "geojson",
                                    "-o", f"{d}/t.geojson"],
    "export-gpx": lambda t, d: ["export", t, "--format", "gpx",
                                "-o", f"{d}/t.gpx"],
    "export-hotspots": lambda t, d: ["export", t, "--format", "hotspots",
                                     "-o", f"{d}/h.csv"],
    "grid": lambda t, d: ["grid", t, "-o", f"{d}/g.asc", "--grid-size", "32",
                          "--n-boot", "10"],
    "grid-excess": lambda t, d: ["grid", t, "-o", f"{d}/x.asc", "--value",
                                 "excess", "--grid-size", "32", "--n-boot", "10"],
}


@pytest.mark.parametrize("command", sorted(_COMMANDS))
def test_every_command_survives_masked_markers(command, tmp_path, capsys):
    """Masked readings are NaN; `info` once crashed on int(nan)."""
    import re
    track = _with_masked_markers(_walk_path(), tmp_path)
    assert main(_COMMANDS[command](track, tmp_path)) == 0
    captured = capsys.readouterr()
    assert not re.search(r"\bnan\b", captured.out, re.I)
    for path in tmp_path.iterdir():
        if path.suffix != ".rctrk":
            assert not re.search(r"\bnan\b", path.read_text(), re.I), path.name


def test_flight_survives_masked_markers(sample_path, sample_trajectory_path,
                                        tmp_path, capsys):
    import re
    track = _with_masked_markers(sample_path, tmp_path, every=40)
    assert main(["flight", track, "--trajectory", str(sample_trajectory_path),
                 "--csv", str(tmp_path / "p.csv"),
                 "--samples-csv", str(tmp_path / "s.csv"),
                 "--json", str(tmp_path / "f.json")]) == 0
    assert not re.search(r"\bnan\b", capsys.readouterr().out, re.I)
    json.loads((tmp_path / "f.json").read_text(),
               parse_constant=lambda c: pytest.fail(f"JSON constant {c}"))


def test_analyze_quotes_every_dose_rate_in_the_requested_unit(sample_path,
                                                              capsys):
    """--unit used to reach only the dose section; the track, background,
    detection limits, anomalies and hotspots stayed in nSv/h."""
    assert main(["analyze", str(sample_path), "--unit", "uSv/h",
                 "--n-boot", "10"]) == 0
    out = capsys.readouterr().out
    assert "nSv" not in out
    assert "1.1600 uSv/h" in out          # the 2 s MDA, 1159.97 nSv/h
