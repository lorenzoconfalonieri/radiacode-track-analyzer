# Improvements

Small, one-commit improvements (under ~100 lines each), worked one per
iteration. Larger directions live in IDEAS.md. Rotate categories:
correctness, tests, GUI usability, performance (measured), code quality.

## Backlog

- **Correctness:** `format_degree` can print `111°52'60.0"W`: seconds are
  rounded after the minutes are split off, so 59.96 s shows as 60.0 without
  carrying. Round to the step's precision first, then split.
- **Tests:** `flight_report_to_json` and `map_export.compose_attribution`
  are untested.
- **GUI:** File > Open recent, via QSettings.
- **GUI:** the Track and Selection dock text boxes clip their content at the
  default dock height.
- **GUI:** running `rcta` without the `[gui]` extra should print an install
  hint, not an ImportError traceback.
- **GUI:** the status-bar hint and the empty map say nothing about what to
  do; a banner over the empty map ("Open a .rctrk track (Ctrl+O) or drop it
  here") would use the theme's banner style.
- **Tests:** `synthetic.make()` sets `integration_time_s`, which real files
  never do, so most analysis tests skip the inference path. Consider a
  `from_file=True` option that clears it.
- **Code quality:** the version string is defined in three places
  (`pyproject.toml`, `io_export.py`, `main_window.py`).

## Done

- 2026-10-15 **Correctness:** `DoseSummary.mean_dose_rate` returned 0.0
  when no interval was observed, and the summary printed "mean rate 0.00
  nSv/h" for a one-sample track. It is now NaN, and the summary says
  "undefined, nothing observed".

- 2026-10-15 **Tests:** 8 direct tests for `geo.smooth_positions` and
  `geo.choose_smoothing_window`: identity at window 1, centring on a
  straight line, endpoint holding, noise reduction, odd windows only, walk
  vs fast platform, NaN accuracy, and a stationary platform.

- 2026-10-15 **GUI:** drag and drop. A dropped `.rctrk` opens as the track;
  any other file dropped onto a loaded track goes to the Flight dock as its
  trajectory. A trajectory dropped before any track says why it can't be
  used. 3 tests.

- 2026-10-15 **Code quality:** the `choose_smoothing_window` docstring no
  longer claims that every vehicle gets window 1; slow driving gets 3.

- 2026-10-15 **Correctness:** the integration-time text says how the time was
  actually obtained: supplied, inferred, inferred then clipped to 1-60 s, or
  assumed for a single sample. The "inferred" warning no longer fires for a
  Track that carries its own integration time; two tests had relied on that
  false warning.

- 2026-10-15 **Tests:** 6 direct tests for `rolling_trimmed_mean` and
  `_trimmed_mean`, all against hand-computed values: symmetric trimming, a
  tiny window never trimmed empty, a centred window inclusive at both ends,
  truncation at the edges, windowing in time across a dropout, fallback to
  the global value, and input validation.

- 2026-10-15 **GUI:** longitude labels no longer overlap. The bottom axis
  coarsens its step until the measured label widths fit, and minor ticks are
  left unlabelled. Checked on the walk and the flight.

## Rejected

- 2026-10-15 **Performance:** `rolling_trimmed_mean` vectorisation. Measured
  19 ms on the 2964-sample flight, of a 118 ms warm analysis, and 0.55 s at
  100k samples: not worth a rewrite. The real first-analysis cost is
  importing `scipy.stats` (0.7 s, once per process); moving it to startup
  only moves the wait.
