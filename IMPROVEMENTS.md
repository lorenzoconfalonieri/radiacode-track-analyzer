# Improvements

Small, one-commit improvements (under ~100 lines each), worked one per
iteration. Larger directions live in IDEAS.md. Rotate categories:
correctness, tests, GUI usability, performance (measured), code quality.

## Backlog

- **Tests:** `map_export.compose_attribution` is untested: it should
  enlarge nothing and draw the plate inside the image.
- **GUI:** the Track and Selection dock text boxes clip their content at the
  default dock height.
- **GUI:** the status-bar hint and the empty map say nothing about what to
  do; a banner over the empty map ("Open a .rctrk track (Ctrl+O) or drop it
  here") would use the theme's banner style.
- **Tests:** `synthetic.make()` sets `integration_time_s`, which real files
  never do, so most analysis tests skip the inference path. Consider a
  `from_file=True` option that clears it.

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

- 2026-10-15 **Correctness:** map degree labels round to their printed
  precision before splitting, so 59.96 s carries to the next minute instead
  of printing `52'60.0"` (likewise `40°60'` becomes `41°00'`).

- 2026-10-15 **GUI:** running `rcta` without the `[gui]` extra prints which
  package is missing, `pip install 'rcta[gui]'`, and a pointer to `rcta-cli`,
  then exits 1, instead of an ImportError traceback.

- 2026-10-15 **Code quality:** the version is defined once, in
  `rcta/__init__.py`. `pyproject.toml` reads it dynamically, and the GPX
  `creator` and the About box format it in.

- 2026-10-15 **Tests:** 3 direct tests for `flight_report_to_json` on the
  known-answer synthetic flight: scale height within 5 % of truth, strict
  JSON, only populated bins, unit conversion of the bins, provenance.

- 2026-10-15 **GUI:** File > Open recent. The last 8 tracks, numbered,
  with the file name shown and the full path as the tooltip. Vanished files
  are left out, and there is a Clear list action. The list persists through
  QSettings, which the app injects; tests use an ini file or memory, never
  the user's settings.

## Rejected

- 2026-10-15 **Blocked on the owner, not rejected:** the `USER_AGENT`
  contact URL. It points at `github.com/rcta/...`, a placeholder that may
  belong to someone else; this is the suite's one xfail. The real
  repository is private, so a tile-server operator could not use it
  either. It needs a public contact (make the repo public, an address or a
  project page), which is the owner's call. The GPX namespace URL is an
  identifier and need not resolve; changing it would orphan files already
  exported.
- 2026-10-15 **Performance:** `rolling_trimmed_mean` vectorisation. Measured
  19 ms on the 2964-sample flight, of a 118 ms warm analysis, and 0.55 s at
  100k samples: not worth a rewrite. The real first-analysis cost is
  importing `scipy.stats` (0.7 s, once per process); moving it to startup
  only moves the wait.
