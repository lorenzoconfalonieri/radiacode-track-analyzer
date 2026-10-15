# Improvements

Small, one-commit improvements (under ~100 lines each), worked one per
iteration. Larger directions live in IDEAS.md. Rotate categories:
correctness, tests, GUI usability, performance (measured), code quality.

## Backlog

- **Correctness:** a one-sample track reports "integration: 1.0 s per sample
  (inferred from median spacing)" when there is no spacing to infer from.
  It should say the time is unknown, or fall back explicitly.
- **Tests:** `geo.smooth_positions` and `geo.choose_smoothing_window` have no
  direct tests; only end-to-end distance tests cover them.
- **Tests:** `background.rolling_trimmed_mean` has no direct test of its
  trimming against a hand-computed window.
- **Tests:** `flight_report_to_json` and `map_export.compose_attribution`
  are untested.
- **GUI:** opening a file by dragging it onto the window (`.rctrk`, or a
  trajectory once a track is loaded).
- **GUI:** File > Open recent, via QSettings.
- **GUI:** longitude tick labels overlap on small extents, e.g. the bundled
  walk (`GeoAxisItem`).
- **GUI:** the Track and Selection dock text boxes clip their content at the
  default dock height.
- **GUI:** running `rcta` without the `[gui]` extra should print an install
  hint, not an ImportError traceback.
- **Performance:** `rolling_trimmed_mean` sorts once per sample. Measure on
  the flight, then try a sliding-window approach.
- **Code quality:** the version string is defined in three places
  (`pyproject.toml`, `io_export.py`, `main_window.py`).

## Done

- 2026-10-15 **Correctness:** `DoseSummary.mean_dose_rate` returned 0.0
  when no interval was observed, and the summary printed "mean rate 0.00
  nSv/h" for a one-sample track. It is now NaN, and the summary says
  "undefined, nothing observed".

## Rejected

(none yet)
