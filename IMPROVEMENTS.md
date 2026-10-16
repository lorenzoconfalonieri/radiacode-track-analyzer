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

- 2026-10-15 **Correctness:** every text read and write now names its
  encoding: the providers file, flight providers, airlines and airports
  data, flight credentials, and the trajectory cache. On Windows (cp1252)
  every basemap attribution had come out as "Â© OpenStreetMap
  contributors", including in exported images. An architecture test scans
  the source for `read_text`/`write_text`/`open` without an encoding; it
  fails on the old code.

- 2026-10-15 **GUI:** choosing the "reference" background with no span
  marked now says, in the log and status bar, that the rolling estimator
  is used until a span is marked, and how to mark one. Before, the combo
  read "reference" while the rolling results were shown. 2 tests.

- 2026-10-15 **Correctness:** when the default background window is
  widened for a slow cadence (900 s to 3200 s on the walk), the analysis
  now warns, with the reason and how to override it. The CLI summary and
  the GUI showed only the default; the JSON had been the only record.
  A follow-up made the GUI log it too: it logged only the background
  estimator's warnings, not the analysis-level ones. 3 tests.

- 2026-10-15 **GUI:** a muted note under the Window box shows the applied
  window when it differs ("applied: 3200 s, widened to fit the widest scan
  window at this cadence"). The box keeps the requested value, so the
  automatic choice stays automatic.

- 2026-10-15 **GUI:** Fetch with an empty flight-number field now says what
  to type (or to load a file) and focuses the field. It used to do nothing
  at all.

- 2026-10-15 **GUI:** the Anomalies, Hotspots and Legs tables sort by any
  column (click the header), numerically ("9.2" before "10.5"). They open
  in time order. Row selection maps through the sort proxy, so a sorted
  row still selects its own samples. A test sorts by sigma and checks the
  top row selects the strongest candidate.

- 2026-10-15 **GUI:** double-clicking an Anomalies or Hotspots row zooms the
  map to its samples; the tables' tooltip says so.

- 2026-10-15 **GUI:** selecting Legs rows selects every sample of those
  legs, on the map and the series; double-click zooms to them, as in the
  other two tables.

- 2026-10-15 **Correctness:** `rcta-cli analyze --unit` now reaches every
  dose rate in the summary. It had converted only the dose section; the
  track range, background, detection limits, anomalies and hotspots stayed
  in nSv/h. The MDA lines keep their nSv/h text byte for byte; the README's
  anomaly line now shows two decimals (+81.51).

- 2026-10-15 **Correctness:** `rcta-cli flight --unit` now reaches the
  altitude profile, the scale-height fit and the climb/descent lines, which
  had stayed in nSv/h. In nSv/h they now carry two decimals (582.03, not
  582); FLIGHT.md's rounded table is unaffected.

- 2026-10-15 **GUI:** the Track dock summary follows the display unit. A
  sweep of every label, text box, table tooltip, axis and the status bar
  after switching to uSv/h found it was the last place still showing nSv/h.
  The Flight panel already followed the unit.

- 2026-10-15 **GUI:** keyboard shortcuts. Ctrl+0 zooms to the track, and a
  new View > Clear selection is on Esc (greyed with nothing selected). The
  README's navigation paragraph lists them and the table double-click.

- 2026-10-15 **Correctness:** JSON anomalies gain `net_dose_rate_nsv_h`.
  `net_dose_rate` is in the display unit while every other dose field is
  canonical and names its unit in the key; a script reading the JSON alone
  could not tell. Additive, so no existing reader breaks.

- 2026-10-15 **Correctness:** JSON hotspots carry `peak_dose_rate_nsv_h`
  and `mean_net_dose_rate_nsv_h`; the report had no hotspot dose rate at
  all, though the CSV and the table show the peak.

- 2026-10-15 **Correctness:** the anomaly and hotspot CSVs name the unit in
  their dose-rate headers (`peak_dose_rate_nSv_per_h`), as the track CSV
  always has. The values followed `--unit` but the header did not say so.
  Deliberate schema change: two columns renamed in each file; the
  header-stability test was updated.

- 2026-10-15 **Performance:** the block bootstrap ranked x and y again for
  each of its 500 surrogates, which was 60 % of a GUI load. Shuffling
  blocks only reorders the ranks, so Spearman now ranks once and does one
  dot product per surrogate. Bootstrap 0.085 s to 0.004 s; flight load
  0.188 s to 0.071 s; test suite 61 s to 51 s. P-values identical in 122
  cases with ties; CLI output byte-identical. A test pins the fast path to
  the general one.

- 2026-10-15 **Performance:** the heatmap's distance-to-track KD-tree query
  (0.33 s of 0.36 s on the flight) runs on all cores. Flight grid 0.355 s
  to 0.062 s; the exported .asc is byte-identical; suite 51 s to 45 s.

- 2026-10-16 **Tests:** direct tests for the last untested public core
  helpers: `geo.median_accuracy` (NaN, inf, empty), `format_signed_dose_rate`
  (sign, zero, unit) and `load_airlines` (the bundled table, a custom path,
  the schema check). Also confirmed that GPX and GeoJSON leave out masked
  samples rather than writing borrowed values.

- 2026-10-16 **Correctness:** a hotspot's radius is measured to its
  farthest positioned sample, not its farthest anomaly centre. The walk's
  single-pass hotspot had read "r=0 m, 8 samples" (now 15 m), and the
  two-pass one 92 m (now 117 m). Radius is descriptive only: no score or
  detection changes. The audit also found the constant-count, zero-count
  and no-descent cases clean, with no RuntimeWarnings.

- 2026-10-16 **GUI:** auto re-analysis waits 250 ms after the last parameter
  edit, so a burst of spin-box clicks runs one analysis instead of one per
  click (about 1.2 s each on a 100k-sample track). Synchronous windows, as
  in the tests, still analyse at once. A 100k track loads in 1.8 s overall;
  the remaining cost is the already-rejected `rolling_trimmed_mean` item.

- 2026-10-16 **Performance:** the series curves paint a peak-preserving
  downsample of the visible range with 1 px pens; Qt strokes wider lines
  as polygons. On a 100k-sample track, selecting a span went from 4.0 s to
  0.09 s and clearing from 3.3 s to 0.007 s. Spikes keep their full height.
  Line width goes from 1.2 px to 1 px, like the significance trace.

## Rejected

- 2026-10-15 **Investigated, no change:** the Flight caption's "scatter
  inside bins is 0.2x the counting error -- below Poisson" on the bundled
  flight (0.02x at 50 m bins). Suspected single-sample bins, but they
  number 2 of 180 even at 50 m; the bundled sample's scatter really is
  below Poisson. It is synthetic, and the note's diagnosis, an
  understated integration time, may well be right for it.
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
