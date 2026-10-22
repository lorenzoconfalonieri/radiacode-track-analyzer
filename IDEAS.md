# Ideas

Where rcta could go next. Some of these are a weekend, some are a research
project, and a few are deliberately out of reach. They are grouped by what they
would change about the *answer* the program gives, not by how hard they are,
because the principle in the README still decides priority: **a feature is
worth building if it makes an honest claim stronger or a dishonest one
harder.**

Each idea says why it matters, roughly what it involves, and what could go
wrong. Nothing here is a commitment.

| # | Idea | Size | Why it is first in its group |
| --- | --- | --- | --- |
| 1.1 | Characterise the detector's count-rate filter | weeks | the shipped statistics may rest on it |
| 1.3 | A detection-limit *map* | days | the honest map is the MDA map |
| 2.1 | `rcta record`: acquire directly from the detector | months | removes the largest systematic outright |
| 3.1 | Split ground dose into cosmic + terrestrial | weeks | makes the geology question answerable |
| 3.2 | A physical flight model, not a free exponential | weeks | turns "climb and descent disagree" into a number with a cause |
| 4.2 | Dose per street, from every pass | months | turns single walks into a survey |
| 5.1 | rcta in the browser | weeks | zero install, and the file never leaves the machine |
| 6.1 | CI, and quoted numbers that check themselves | days | the README has drifted once already |

---

## 1. Measurement truth first

### 1.1 Characterise the detector's count-rate filter

**This is the most important item in the file.** A quick look at the two
bundled tracks suggests the `countRate` the app writes is not a raw count over
the sample interval. It looks like a smoothed rate:

| | walk | flight |
| --- | --- | --- |
| median sample spacing | 10 s | 2 s |
| mean count rate | 5.3 cps | 18.3 cps |
| lag-1 autocorrelation, after detrending | 0.60 | 0.85 |
| sample-to-sample variance vs. Poisson at that spacing | ~0.5× | ~0.0025× |
| integration time implied by the variance | ~19 s | hundreds of seconds |

*(Method: subtract a 31-sample running mean, then take half the variance of
first differences. It is crude: the detrend and real trends both bias it. But
an effect of 400× is not going to come from the method.)*

If the device reports something like an exponential or adaptive moving average,
three things follow:

- **Per-sample Poisson errors are wrong.** Counts are being estimated as
  `rate × T`, with `T` inferred from the spacing. If the rate is really an
  average over a longer and possibly variable window, those counts and their
  errors are off by that ratio.
- **The windowed scan is too confident on long windows.** Summing *n* smoothed
  samples does not shrink the error by √n, because neighbouring samples share
  counts. The Currie limit for a 64-sample window could be optimistic, which
  is the dangerous direction.
- **Bonferroni is counting the wrong number of tests.** Overlap within a
  window and smoothing between samples both cut the number of independent
  tests (see 1.4).

**What to do.** Run two bench experiments with an ordinary household check
source. Potassium chloride salt substitute is the classic one: safe, cheap and
legal everywhere.

1. *Stationary run.* Leave the detector logging for an hour or more with no
   source. Its autocorrelation and dispersion measure the filter with no trend
   to confuse things.
2. *Step response.* Bring the KCl up to the detector abruptly, then take it
   away. The shape of the rise and fall *is* the filter's impulse response, and
   tells you whether the window adapts to the count rate. Many handhelds
   shorten it when the rate jumps.

Then put the filter in `core/statistics.py`. Model the reported rate as a
filtered Poisson process and derive effective independent counts per sample
and per window from it. Carry the filter model in `AnalysisParameters` the way
the integration time is carried now, as an assumption reported with every
result. If the filter can be inverted stably, offer a deconvolved count rate
too, clearly labelled.

**Risk.** The filter may depend on firmware version and on device settings.
Record both, and have the program refuse to guess for a firmware it has not
seen characterised. Also: the README's walking-survey example assumes 2 s
logging, but the bundled walk logs every 10 s (median). Worth a sentence there
either way.

### 1.2 Infer the integration time from the data, not the clock

This falls out of 1.1. For a Poisson process the variance of the rate is
`mean / T`. On a stretch with flat background, the scatter of the count rate
therefore *measures* the effective integration time. Fit variance against mean
over many stretches and you get `T`, with an error bar, from the track itself.
That replaces "inferred from the median spacing, clipped, and warned about" —
currently the largest systematic in the program — with a measurement. Keep the
spacing-based value as a cross-check and report any disagreement.

### 1.3 A detection-limit map

The MDA table is already the result on the flight. Draw it as a map. At every
point on the track there is a minimum excess the survey could have detected
there. It depends on the local background, the speed (dwell time), and the
window widths that fit between that point and the nearest dropout. A map
coloured by that number answers the question a dose map cannot: **"if there
were something here, would I have seen it?"**

It should sit next to the dose-rate layer as an equal, not hide in a submenu.
For a negative survey (this park is clean) it is the only defensible figure.
Fast walking, poor GPS and dropouts all show up as patches of weak
sensitivity, which tells the user where to walk again.

### 1.4 A calibrated threshold, not just a safe one

Bonferroni treats all 20,748 tests as independent, but each window overlaps its
neighbours and every width contains the next one down. So it is conservative,
and on a walk that means real weak features are thrown away. Two options, both
of which keep Bonferroni as the reported upper bound:

- **A max-statistic by block permutation.** Circularly shift the residuals in
  blocks longer than the autocorrelation time. Rerun the full multi-scale scan
  on each shift and keep the largest significance. The 95th percentile of that
  maximum is a family-wise threshold that *is* calibrated for this track's own
  correlation structure. `block_bootstrap_p` already does half of this.
- **False discovery rate** (Benjamini–Hochberg or a scan-aware variant), as an
  explicitly labelled mode for *mapping*. There the question is "which of
  these are real" rather than "is anything real at all".

The forty-synthetic-tracks regression test should then pin both properties:
the calibrated threshold must still return zero on clean data, and it must
return more of the true sources than Bonferroni does in an inject-and-recover
test (1.5).

### 1.5 Measure the sensitivity, and publish it

Inject synthetic point sources into real background tracks across a grid of
strength, standoff distance and walking speed, then report the recovery rate.
That gives detection-probability curves (an ROC per configuration) of the kind
you would expect in an instrument paper. They also turn the README's
qualitative claims into figures that can be checked. Run it in CI at a small
size, and in full for a release.

---

## 2. Better data in

### 2.1 `rcta record`: acquire directly from the detector

Almost every hard problem in this codebase exists because the `.rctrk` export
throws information away: no live time, no spectra, no altitude, and a smoothed
count rate. The community-maintained
[`radiacode`](https://github.com/cdump/radiacode) Python library talks to
RadiaCode devices over USB and Bluetooth. A `rcta record` command, plus a
matching GUI mode, could log:

- raw counts over *known* intervals, which removes 1.1 and 1.2 as problems
  rather than solving them;
- spectra at whatever cadence the device allows (see 2.2);
- GPS from a USB receiver or a paired phone, with altitude;
- firmware version, temperature and device settings, as provenance.

Write it to a documented, lossless format: newline-delimited JSON with a
header record is simplest to repair by hand, and Parquet is better for
week-long logs. Keep `.rctrk` reading forever, but make the native format the
one the program is happiest with.

Live mode earns its own statistics: a **sequential probability ratio test**
that alarms while you are still standing next to the wall. Its false-alarm
rate is stated per hour, which is the unit a person walking around actually
experiences.

**Risk.** It depends on a third-party, reverse-engineered protocol, and the
library may lag new firmware. Keep it in `net/` or a new `acq/` package so the
architecture test still holds `core/` pure, and pin the versions it has been
tested against.

### 2.2 Spectra along the track

The RadiaCode app can record spectra separately from the track. Joining a
spectrogram recording to the track by timestamp, which is the same problem as
the flight join and needs the same validation, would let hotspots carry a
spectrum. That opens the door to:

- K-40 / U-series / Th-series window ratios per segment, and therefore *which*
  geology a granite wall is made of;
- an anomaly class that the count rate alone cannot give, such as "same rate,
  different shape";
- a bridge to the companion spectrum tool that `statistics.py` already keeps
  character-identical Currie algebra with.

**Without any spectra at all**, the file already holds a weak hardness
indicator. `doseRate / countRate` is 13 nSv/h per cps on the ground walk and
51 at cruise, and it ranges 10.5–16.2 within the walk alone. A CsI detector's
dose response is energy-dependent, so that ratio tracks spectral hardness. Map
it. A spot that is elevated in count rate but *not* in hardness is a different
claim from one elevated in both, and that is free information the tool
currently discards.

### 2.3 Altitude for any track, not just flights

Generalise the flight join into "import a trajectory". It could be a GPX from a
hiking app, a phone's barometric log, or a car's OBD log. The position
validation, clock-offset measurement and refusal logic already exist and do
not care what was moving. With no external file, fall back to a terrain model
(USGS 3DEP, Copernicus GLO-30) for ground tracks, read from a user-supplied
tile rather than a live server. That feeds 3.1 directly.

---

## 3. Physics models instead of free fits

### 3.1 Split ground dose into cosmic + terrestrial

A walk in Salt Lake City at 1,300 m gets noticeably more cosmic dose than the
same walk at sea level would, and the cosmic part does not care about geology.
Subtract a modelled cosmic component, using elevation (2.3) and geomagnetic
cutoff rigidity at the location. What is left is the **terrestrial** dose rate,
the part the README says the tool is for.

Then compare it with a prediction. National airborne radiometric surveys
publish potassium, equivalent-uranium and equivalent-thorium grids; for the US
that is the USGS NURE-derived compilation. Turn concentrations into dose rate
with the standard UNSCEAR coefficients (0.0417, 0.462 and 0.604 nGy/h per
Bq/kg for K-40, U-238 and Th-232) and plot **measured minus predicted**. Bedrock
explains most of the map. Building materials, road aggregate and shelter show
up as the residual, and that residual is the scientifically interesting
picture. It also gives an external sanity check on the dose calibration.

**Risk.** Airborne grids have kilometre resolution and the walk has metre
resolution. The comparison is only meaningful after smoothing the walk to the
grid's scale, and the program has to say so, not paint metre-scale residuals
against kilometre-scale predictions.

### 3.2 A physical flight model, not a free exponential

The scale-height fit is honest, and it reports its own failure. The next step
is to fit a model whose parameters mean something:

- use **atmospheric depth** (g/cm²) rather than altitude, from a standard
  atmosphere or, better, reanalysis pressure along the route;
- add **geomagnetic cutoff rigidity** along the route from IGRF. The bundled
  climb and descent are about 10° of latitude apart, which the README already
  suspects is the cause of their disagreement;
- add **solar modulation** for the day from the neutron-monitor database
  (NMDB).

Then compare with an established aircrew dose model such as CARI-7 or
PARMA/EXPACS, either as a reference curve or as a fitted normalisation. The
climb/descent control stops being "disagree by 12.5 %, 4.07 σ". It becomes
"disagree by 12.5 %, of which the rigidity difference predicts *x*, leaving *y*
unexplained". That answer is more useful, and harder to wave away.

### 3.3 Rain

Rain scavenges radon progeny out of the air, and after the onset of rainfall
ground-level dose rate can rise noticeably for an hour or two. A repeat survey
done after a shower can therefore look like a change in the ground when it was
the weather. Join hourly precipitation for the track's time and place from a
keyless historical archive such as Open-Meteo. Flag affected stretches the way
`trend` is flagged now: shown and explained, never hidden.

### 3.4 Source localisation, honestly

Limitation 1 in the README. When a track passes a compact source at several
distances, the count-rate excess along the path constrains where the source is
and how strong it is: 1/r², air attenuation, and the detector's angular
response. A likelihood over (x, y, activity) given the track geometry yields a
confidence region. The honest part is reporting when that region is
**unconstrained**. One straight pass cannot tell which side of the road the
source is on, and the program should draw both lobes rather than pick one.
Distributed sources (a granite wall, a gravel bed) do not fit a point model,
and a goodness-of-fit test should say so.

---

## 4. Many tracks, not one

### 4.1 A survey library

Open a folder, not a file. Index every track, deduplicate re-exports of the
same acquisition (same device, overlapping timestamps), and let the analysis
run across them. Most of what follows depends on this.

### 4.2 Dose per street, from every pass

Map-match each track to street and footpath segments from a user-supplied
OpenStreetMap extract. That means an offline file, not a public API, in the
same spirit as the tile policy section. Then estimate dose rate **per segment**
from every pass ever walked along it, with a Poisson error that shrinks as
passes accumulate. That turns a pile of separate walks into an actual survey,
and gives three new results for free:

- **repeat-pass consistency** as validation: the same street on two days
  should agree within errors, or the disagreement is a finding (see 3.3);
- **change detection** between surveys months apart;
- a much better hotspot ranking. The `× √passes` rule already rewards being
  seen on several passes, and segment-level aggregation makes "pass" a
  precise notion instead of a clustering heuristic.

### 4.3 Cross-detector calibration

`devices` is a list for a reason. Where two detectors cover the same segments,
their ratio is a cross-calibration. Carry it as a per-device factor with an
uncertainty, and warn when a library mixes uncalibrated devices.

### 4.4 A personal dose ledger

Accumulated dose across every track in the library, by day and by year, with
flights and walks broken out. Report both totals, gap-integrated and observed,
exactly as `DoseSummary` does now, because the coverage problem is worse
across a year than across a walk.

---

## 5. Reach

### 5.1 rcta in the browser

`core/` is pure numpy and scipy, never touches the network, and never imports
Qt. That is exactly the profile that [Pyodide](https://pyodide.org) runs
unchanged. A static page on GitHub Pages could take a dropped `.rctrk` and run
the real `analyze_track` locally in the browser. The file never leaves the
user's machine, which is worth saying loudly for location data. The
architecture test is what makes this cheap: it has been preserving this option
all along.

### 5.2 Interoperability

- **Safecast.** Export in the bGeigie log format, and fetch Safecast's existing
  measurements along the route as a comparison layer. That gives an
  independent dataset to agree or disagree with, and a path to contributing
  data back if the user wants to.
- **QGIS.** A thin plugin around `core/` so surveyors can use the analysis
  inside the GIS they already work in. GeoPackage export would serve the same
  audience.
- **KML** for people who just want to look at a walk in Google Earth.

### 5.3 A report that proves where it came from

`rcta-cli report TRACK.rctrk -o report.html` writes one self-contained file:
map, series, detections, MDA table, warnings, and a provenance block with the
input's SHA-256, the rcta version, and every parameter. Put the same
provenance block in every export: CSV header columns, GeoJSON `properties`, a
GPX extension. A number that leaves the program should always be traceable
back to the file and the settings that produced it.

---

## 6. Engineering

### 6.1 CI, and quoted numbers that check themselves

There is no CI yet. A GitHub Actions matrix (Linux, macOS, Windows; Python
3.10–3.13; headless Qt via `QT_QPA_PLATFORM=offscreen`) is a day of work and
protects 860+ tests.

More interesting: **the README and the paper quote numbers, and those numbers
have drifted before** (1160.02 vs 1159.97). Mark every quoted figure in the
docs, run the CLI in CI, and fail the build when the docs and the program
disagree. The paper's README already lists where each number comes from; turn
that list into a test.

### 6.2 Property-based tests

The parser's doctrine (masked, never deleted; units converted once; any key
order; ints and floats both accepted) is a set of invariants, which is what
[Hypothesis](https://hypothesis.readthedocs.io) is for. Generate thousands of
malformed but plausible `.rctrk` files and assert the invariants, rather than
pinning the eight cases someone thought of.

### 6.3 Scale

The `Track` docstring talks about week-long 600,000-sample surveys. Benchmark
one: load, analyse, grid and render. The rolling background and the
multi-scale scan are the likely costs. Know the limits before a user finds
them.

### 6.4 Distribution

Publish to PyPI and conda-forge, and build signed desktop bundles (Briefcase or
PyInstaller) so a non-programmer with a RadiaCode can use the GUI without
conda. Most RadiaCode owners are not Python users, so this decides who the
audience actually is.

### 6.5 From the October 2026 code review

Fixed: masked samples leaking zeros into the statistics, GPS glitches in
distance and speed, a one-sided climb/descent test, the heatmap that was
computed but never drawn, a PNG export that carried tiles its provider
forbids, a Display unit most panels ignored, overlapping longitude labels,
and the drifted README coverage figure. The paper's quoted figures are now
pinned by tests, which covers the walk and flight part of 6.1.

Built from the review's feature list (details in IMPROVEMENTS.md and git):
stop-and-measure dwells (core, report, Dwells panel); `analyze
--fail-on-detection` (exit 5) with a batch summary; a gps-transition flag
for shelter and vehicles; a detector consistency check; legs split at the
device's recording periods; a combined significance across a hotspot's
passes; QSettings for layout, recent files and basemap; drag and drop;
sortable tables; a UTC clock-time axis; selection export; session files.

Engineering, also done: ruff in `dev` and in CI; layer-direction rules in the
architecture test (no relative imports, `core` imports nothing above it, `net`
never imports the GUI); CI on Linux, macOS and Windows under Python 3.10 and
3.12 (the CI part of 6.1); and `core/flight.py` split into flight-number
parsing, the join and the profile behind a re-exporting facade. Earlier:
the version defined once, an install hint when the GUI extra is missing,
`rcta` parsed with argparse (`--offline`, `--version`, Qt options passed
through), and direct tests for `smooth_positions`,
`choose_smoothing_window`, `rolling_trimmed_mean`, `flight_report_to_json`
and `compose_attribution`. Of the review's performance items the bootstrap
now ranks once; `rolling_trimmed_mean`, `spatial.interpolate` and the scan
were measured at 19-26 ms, 24 ms and 16 ms warm on the flight and left
alone (IMPROVEMENTS.md has the numbers).

Leg distances on smoothed tracks, fixed: per-leg smoothing held each leg's
end samples, so the walk's legs (2.64 km) and dropouts (0.09 km) fell short
of its 2.99 km total. Ends are now point-reflected; the walk is 3.15 km and
its legs and dropouts add up to it. Held ends had also hidden a residual
2-5 % overstatement on straight walks, which the README now states, with the
corner cutting that pulls the other way.

Closed windows, fixed: signal connections kept a closed `MainWindow` alive
with its track and plots. It now sets `WA_DeleteOnClose`, and a test checks
that a closed window is freed.

Measured and not adopted:

- **An exact Poisson tail.** The scan's normal approximation divides by the
  root of the gross counts and is never above the exact tail wherever it
  reaches 1 sigma, so switching would only add detections. A test pins this.
- **A local dose-per-count ratio.** In the bundled data the ratio does not
  drift within a track; the walk's per-sample ratio is noisy, and a 31-sample
  median moved its background by 6 %. Revisit with a real flight that has a
  ground phase.

The User-Agent contact, resolved: it names the repository, which has been
public since 2026-10-22.

---

## 7. Publication and community

- ~~Make the repository public~~ done 2026-10-22, with the v0.1.0 desktop
  release and the website at
  https://lorenzoconfalonieri.github.io/radiacode-track-analyzer/.
- **Archive a release on Zenodo** for a citable DOI.
- **Submit to [JOSS](https://joss.theoj.org) as well.** It reviews the code,
  not just the paper, and its checklist (statement of need, tests,
  documentation, an OSI licence) is already mostly met. Its peer review would
  carry more weight than an arXiv listing alone.
- **Turn `docs/FORMAT.md` into a standalone, versioned specification** and ask
  other owners for sample files. The tab-separated dialect's dose unit is
  still unverified, and other models and firmware versions are entirely
  untested. A small corpus of donated, anonymised tracks would harden the
  parser more than any amount of reasoning. Ship a `rcta-cli anonymise`
  command that shifts positions and timestamps so people can contribute
  safely.

---

## Moonshots

Out of reach for now, but worth writing down because they decide what the
pieces above should not rule out.

- **A crowd-sourced background atlas.** Thousands of rcta-processed walks,
  each one terrestrial-only (3.1), detection-limit-aware (1.3) and
  provenance-stamped (5.3), aggregated per street (4.2). Done honestly it
  would be a map whose blank areas mean "not measured", not "zero".
- **Drone surveys.** A RadiaCode on a small UAV is a flight track at 50 m
  instead of 11 km. The altitude join, source localisation and the MDA map
  combine into something close to a real aerial radiometric survey.
- **Real-time spectral identification** on top of `rcta record`, with the
  same refusal to claim what the counts cannot support.
