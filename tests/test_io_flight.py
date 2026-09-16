"""Trajectory readers, and the altitude unit that can quietly ruin the plot."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rcta.core.flight import M_PER_FT
from rcta.core.io_flight import (TrajectoryParseError, load_trajectory,
                                 parse_opensky_states, parse_opensky_track,
                                 trajectory_to_csv)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


OPENSKY_TRACK = {
    "icao24": "a1b2c3", "callsign": "DAL2618 ",
    "startTime": 1788785639, "endTime": 1788785759,
    "path": [
        [1788785639, 30.2, -97.66, None, 12.0, True],
        [1788785679, 30.3, -97.70, 1500.12, 12.0, False],
        [1788785719, 30.4, -97.74, 3000.25, 12.0, False],
        [1788785759, 30.5, -97.78, 4500.5, 12.0, False],
    ],
}


# ---- the OpenSky payloads -------------------------------------------------

def test_an_opensky_track_is_parsed():
    trajectory = parse_opensky_track(OPENSKY_TRACK)
    assert trajectory.n == 4
    assert trajectory.callsign == "DAL2618"
    assert trajectory.icao24 == "a1b2c3"
    assert trajectory.altitude_kind == "barometric"
    assert trajectory.provider == "opensky"


def test_a_null_altitude_on_the_ground_stays_nan():
    trajectory = parse_opensky_track(OPENSKY_TRACK)
    assert not trajectory.has_altitude[0]
    assert trajectory.on_ground[0]
    assert "no barometric altitude" in " ".join(trajectory.warnings)


def test_an_empty_opensky_path_is_an_error_not_an_empty_track():
    with pytest.raises(TrajectoryParseError, match="empty path"):
        parse_opensky_track({"icao24": "x", "path": []})


def test_a_malformed_waypoint_row_is_skipped_and_counted():
    payload = dict(OPENSKY_TRACK)
    payload["path"] = OPENSKY_TRACK["path"] + [[1788785799, 30.6]]
    trajectory = parse_opensky_track(payload)
    assert trajectory.n == 4
    assert "malformed" in " ".join(trajectory.warnings)


def test_state_vectors_need_an_aircraft_named_when_several_are_present():
    payload = {"states": [
        ["a1b2c3", "DAL2618 ", "US", 1, 1788785639, -97.6, 30.2, 9000.0, False],
        ["ffffff", "SWA100  ", "US", 1, 1788785639, -97.6, 30.2, 8000.0, False],
    ]}
    with pytest.raises(TrajectoryParseError, match="pass a callsign"):
        parse_opensky_states(payload)
    trajectory = parse_opensky_states(payload, callsign="DAL2618")
    assert trajectory.n == 1 and trajectory.altitude_m[0] == 9000.0


def test_a_callsign_that_is_absent_says_how_many_aircraft_were_there():
    payload = {"states": [
        ["a1b2c3", "DAL2618 ", "US", 1, 1788785639, -97.6, 30.2, 9000.0, False]]}
    with pytest.raises(TrajectoryParseError, match="1 distinct aircraft"):
        parse_opensky_states(payload, callsign="BAW117")


# ---- dispatch -------------------------------------------------------------

def test_json_and_csv_are_dispatched_on_content_not_extension(tmp_path):
    """Both arrive named whatever the download was called."""
    a = write(tmp_path, "trajectory.csv", json.dumps(OPENSKY_TRACK))
    b = write(tmp_path, "trajectory.json",
              "time,lat,lon,altitude_m\n1,30.0,-97.0,1000\n2,30.1,-97.1,2000\n")
    assert load_trajectory(a).n == 4
    assert load_trajectory(b).n == 2


def test_an_empty_file_is_refused(tmp_path):
    with pytest.raises(TrajectoryParseError, match="empty"):
        load_trajectory(write(tmp_path, "t.csv", "   "))


def test_a_missing_column_names_what_it_wanted_and_what_it_found(tmp_path):
    path = write(tmp_path, "t.csv", "time,lat,lon\n1,30.0,-97.0\n")
    with pytest.raises(TrajectoryParseError) as excinfo:
        load_trajectory(path)
    assert "altitude_m" in str(excinfo.value)
    assert "'lat'" in str(excinfo.value)


def test_a_bare_json_array_of_numbers_is_refused(tmp_path):
    """Positional data does not say which column is which."""
    path = write(tmp_path, "t.json", json.dumps([[1, 30.0, -97.0, 1000.0]]))
    with pytest.raises(TrajectoryParseError, match="which column is which"):
        load_trajectory(path)


@pytest.mark.parametrize("header", ["time,lat,lon,altitude_m",
                                    "Timestamp;Latitude;Longitude;Altitude_m",
                                    "utc\tlat\tlon\tbaro_altitude"])
def test_column_names_are_matched_loosely(tmp_path, header):
    sep = ";" if ";" in header else ("\t" if "\t" in header else ",")
    rows = [header, sep.join(["1788785639", "30.0", "-97.0", "1000"]),
            sep.join(["1788785699", "30.1", "-97.1", "2000"])]
    trajectory = load_trajectory(write(tmp_path, "t.csv", "\n".join(rows)))
    assert trajectory.n == 2
    assert trajectory.altitude_m[1] == 2000.0


@pytest.mark.parametrize("stamp,expected", [
    ("1788785639", 1788785639.0),
    ("1788785639000", 1788785639.0),
    ("2026-09-07T12:53:59Z", 1788785639.0),
    ("2026-09-07 12:53:59", 1788785639.0),
])
def test_timestamps_arrive_in_four_shapes(tmp_path, stamp, expected):
    rows = ["time,lat,lon,altitude_m", f"{stamp},30.0,-97.0,1000",
            "1788785999,30.1,-97.1,2000"]
    trajectory = load_trajectory(write(tmp_path, "t.csv", "\n".join(rows)))
    assert trajectory.time_s[0] == pytest.approx(expected)


# ---- the unit trap --------------------------------------------------------

def test_a_column_naming_feet_is_converted(tmp_path):
    rows = ["time,lat,lon,altitude_ft", "1,30.0,-97.0,35000",
            "2,30.1,-97.1,36000"]
    trajectory = load_trajectory(write(tmp_path, "t.csv", "\n".join(rows)))
    assert trajectory.altitude_m[0] == pytest.approx(35000 * M_PER_FT)
    assert trajectory.warnings == ()


def test_a_column_naming_metres_is_believed(tmp_path):
    rows = ["time,lat,lon,baro_altitude", "1,30.0,-97.0,10668",
            "2,30.1,-97.1,10700"]
    trajectory = load_trajectory(write(tmp_path, "t.csv", "\n".join(rows)))
    assert trajectory.altitude_m[0] == 10668.0
    assert trajectory.altitude_kind == "barometric"


def test_an_unnamed_column_above_twenty_kilometres_must_be_feet(tmp_path):
    """No civil flight cruises at 35 km, so the unit is not ambiguous."""
    rows = ["time,lat,lon,altitude", "1,30.0,-97.0,35000",
            "2,30.1,-97.1,36000"]
    trajectory = load_trajectory(write(tmp_path, "t.csv", "\n".join(rows)))
    assert trajectory.altitude_m[0] == pytest.approx(35000 * M_PER_FT)
    assert "read as feet" in " ".join(trajectory.warnings)


def test_a_genuinely_ambiguous_column_warns_instead_of_guessing(tmp_path):
    """8000 is a light aircraft in feet or a turboprop in metres."""
    rows = ["time,lat,lon,altitude", "1,30.0,-97.0,8000", "2,30.1,-97.1,7500"]
    trajectory = load_trajectory(write(tmp_path, "t.csv", "\n".join(rows)))
    assert trajectory.altitude_m[0] == 8000.0
    warning = " ".join(trajectory.warnings)
    assert "read as METRES" in warning and "3.28x" in warning


def test_the_ambiguous_case_can_be_overridden(tmp_path):
    rows = ["time,lat,lon,altitude", "1,30.0,-97.0,8000", "2,30.1,-97.1,7500"]
    path = write(tmp_path, "t.csv", "\n".join(rows))
    trajectory = load_trajectory(path, altitude_unit="ft")
    assert trajectory.altitude_m[0] == pytest.approx(8000 * M_PER_FT)
    assert trajectory.warnings == ()


def test_an_unknown_unit_is_rejected(tmp_path):
    rows = ["time,lat,lon,altitude", "1,30.0,-97.0,8000", "2,30.1,-97.1,7500"]
    with pytest.raises(ValueError, match="unknown altitude unit"):
        load_trajectory(write(tmp_path, "t.csv", "\n".join(rows)),
                        altitude_unit="fathoms")


# ---- round trip -----------------------------------------------------------

def test_a_written_trajectory_names_its_own_unit(tmp_path):
    """So re-reading it can never land in the ambiguous branch."""
    original = parse_opensky_track(OPENSKY_TRACK)
    text = trajectory_to_csv(original)
    assert "altitude_m" in text.splitlines()[0]

    path = write(tmp_path, "again.csv", text)
    again = load_trajectory(path)
    assert again.n == original.n
    assert again.time_s == pytest.approx(original.time_s)
    finite = np.isfinite(original.altitude_m)
    assert again.altitude_m[finite] == pytest.approx(original.altitude_m[finite])
    assert not np.isfinite(again.altitude_m[~finite]).any()


def test_the_bundled_sample_trajectory_reads(sample_trajectory_path):
    trajectory = load_trajectory(sample_trajectory_path)
    assert trajectory.callsign == "DAL2618"
    assert trajectory.altitude_kind == "barometric"
    assert trajectory.altitude_ceiling_m == pytest.approx(10972.8, abs=1.0)
    assert trajectory.n > 300
