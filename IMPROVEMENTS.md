# Improvements

Small, one-commit improvements (under ~100 lines each), worked one per
iteration. Larger directions live in IDEAS.md. Rotate categories:
correctness, tests, GUI usability, performance (measured), code quality.

## Backlog

- **GUI:** the Track summary (13 lines) shows about 9 in its dock. Instead
  of forcing height (rejected below), move it to a "Summary" tab in the
  bottom stack, or let the Track dock collapse to a one-line caption with
  the full text in a tooltip.

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

- 2026-10-15 **Correctness and tests:** the export attribution plate sized
  its font from the image height only, so a tall narrow PNG clipped the
  licence credit at the right edge. The font now shrinks, in a loop, until
  the widest line fits. 4 tests cover three image shapes, the image size
  being left unchanged, and no lines drawing nothing.

- 2026-10-15 **Correctness:** `rcta-cli info` crashed ("cannot convert
  float NaN to integer") on any track with masked markers, a regression from
  masked readings becoming NaN. The sparkline skips NaN and draws an
  all-masked bin as `_`. `info` also assesses quality first, so it reports
  the same 2.99 km as `analyze` instead of 2.97 km. 3 tests.

- 2026-10-15 **Tests:** every CLI command (info, analyze, four exports,
  grid, grid excess, flight) now runs on sample tracks with masked markers.
  The tests check exit code 0, no `nan` in stdout or any output file, and
  strict flight JSON. Against the pre-fix CLI the info case fails, so the
  test would have caught that crash. A manual sweep found no other command
  affected.

- 2026-10-15 **GUI:** a banner over the empty map ("No track loaded. Open a
  RadiaCode .rctrk file with File > Open track (Ctrl+O), or drop it onto
  this window.") in the theme's banner style; hidden once a track loads.

- 2026-10-15 **Correctness:** the MDA table quoted detection limits for
  scales at which every window held a masked sample, which the scan never
  tested. A two-reading track listed limits up to 64 samples. Such scales
  are now omitted. Bundled-track output is byte-identical.

- 2026-10-15 **Correctness:** with fewer than 4 readings the distribution
  line says "too few readings to judge (n; the split needs at least 4)"
  instead of describing its placeholder as a split of identical readings.
  Text only; the JSON already reported `is_bimodal: false`.

- 2026-10-15 **Tests:** `synthetic.make(from_file=True)` builds a track
  without an integration time, as real files load. The two tests that had
  cleared it by hand now use it. A new test at 2, 5 and 10 s cadence, with
  jitter and a 900 s dropout, checks the inferred time recovers the cadence
  within 5 % and the MDA table matches a track that states it.

- 2026-10-15 **GUI:** the Series and Profile panels show a muted placeholder
  saying what will appear, instead of empty axes with 0.1 ticks, until a
  track is open. A shared `theme.empty_state()` puts it in the plots' own
  slot. The map keeps its graticule, under the no-track banner.

- 2026-10-15 **Code quality:** removed two unused definitions. One was
  `Track.bearing_deg`, which nothing called and which computed headings
  from raw positions, glitched and masked fixes included. The other was
  `BasemapItem.set_lighten`, superseded by `set_recession`. A scan for
  other unreferenced code found only Qt overrides and small public
  conveniences.

- 2026-10-15 **Correctness:** loading a binary file as a track (an image
  renamed `.rctrk`) raised an uncaught UnicodeDecodeError out of
  `MainWindow.load()`; the CLI printed a raw codec error. Both readers now
  read UTF-8 explicitly (the locale default misreads non-ASCII titles on
  Windows) and raise TrackParseError("not a text file ..."). 3 tests.

- 2026-10-15 **Correctness:** `load_trajectory` reads `utf-8-sig`. A JSON
  trajectory saved with a byte-order mark had been refused with a
  misleading "missing a column" error, because the BOM hid its `{`. A binary
  file now says "not a text file" instead of a raw codec error. 2 tests.

## Rejected

- 2026-10-15 **GUI:** minimum heights on the Track and Selection text boxes
  (8 lines). The Selection box showed all its lines, but the window's
  minimum height grew from 950 to 1108 px, taller than many laptop screens.
  Reverted.
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
