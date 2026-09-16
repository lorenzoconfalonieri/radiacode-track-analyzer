# Altitude for an airborne track

The `.rctrk` format carries **no altitude** (`docs/FORMAT.md`, "What the format
does not contain"). On a flight that is the whole problem: altitude is the only
variable that explains the count rate, and without it a two-hour cruise looks
like a two-hour elevated background of unknown cause.

So `rcta` imports altitude from an external trajectory, matched on time, and
plots dose rate against it.

```sh
# with a trajectory already on disk -- no account, no network
rcta-cli flight TRACK.rctrk --trajectory flight.csv

# or look the flight up (needs free OpenSky credentials, see below)
rcta-cli flight TRACK.rctrk --flight DL2618 --date 2026-09-07

# ...naming the departure airport, if the track does not start on an apron
rcta-cli flight TRACK.rctrk --flight DL2618 --departure KAUS
```

In the GUI it is the **Flight** dock: type the flight number, pick the date,
press Fetch — or `File ▸ Open flight trajectory…` for a local file. **Clear
cache** beside them deletes every cached response; it asks first, and names
the size, because a settled flight is cached indefinitely and clearing it
means paying those metered requests again.

When a fetch fails, the dock reports the reason *and* the next step, with any
sign-up link live and the full failure in the Log dock. The CLI prints the
same two lines. "No credentials found" and "a credentials file I could not
use" are deliberately different messages: the first means register, the
second means fix a key name.

## The one thing that can go badly wrong

A flight number is a string a human typed, and **any** trajectory joins onto
**any** track by timestamp. Get the flight, the day or the leg wrong and you
get a smooth, plausible, entirely fictional curve — with no symptom, because
the fabricated altitude axis is monotonic and the dose really does rise along
it.

The defence is that the trajectory carries positions too. `rcta` interpolates
them onto the track's clock, compares them with the fixes the detector
recorded, and **refuses** past a tolerance:

```
position match : MISMATCH
  median residual 300.00 km, p95 300.00 km, max 300.00 km
  over 898 compared sample(s), tolerance 5.0 km
  this is not the flight, the day or the leg that produced this track
```

`rcta-cli flight` exits **4** for this, distinct from **3** for "could not
fetch", so a script can tell a wrong flight number from an unreachable API.
`--force-match` overrides it, and then every output says `MISMATCH, overridden`.

### Clock offsets look exactly like the wrong flight

At 233 m/s a one-second clock error is 233 m of apparent residual, so a
detector whose clock is not GPS-locked fails the position check for a reason
that has nothing to do with the flight. The offset that would minimise the
residual is therefore always measured and reported — but never applied on its
own initiative, because a clock search that is free to slide until the data
fits will absorb a genuine mismatch:

```
  a +45 s clock offset would reduce the median to 0.00 km; the detector clock
  is probably not GPS-locked. Re-run with --time-offset +45
```

### `--date` is a local date

`--date` is the calendar day **at the departure airport** — the date printed on
the boarding pass, not the UTC one. Any evening departure makes those different
days, and the failure that follows is not an empty result. It is a wrong one.

DL2728 left Salt Lake City at 18:34 on Friday 11 September 2026. The detector
was switched on at the gate seventeen minutes earlier:

```
DL2728  SLC → AUS
  Fri 11 Sep 2026  18:34 MDT  →  21:48 CDT      the boarding pass
  Sat 12 Sep 2026  00:34Z     →  02:48Z         UTC, and OpenSky
```

Asked for `--date 2026-09-11` — the correct date, the one on the ticket — a
UTC comparison reads the track as beginning on the **12th**, concludes the
detector's clock is a day fast, and shifts the window back a day. There is a
DL2728 in that window: Thursday's. It is the same flight number, the same
airports, a different aeroplane (`a399e4`, not `a38c66`), and it is fetched
and reported without a murmur — a real trajectory, correctly parsed, wrong.

What stops it is the position check, and only just. The two are a day apart,
so they share no instant at which a position could be compared at all:

```
position match : NOT CHECKED
  no sample had both a usable fix and a trajectory position to compare it with
```

Having nothing to compare is itself treated as a refusal rather than a pass —
had the check quietly approved what it could not test, the run would have
produced a clean-looking profile from the wrong aircraft. That is the second
defence catching the first one's miss, and it is the whole argument for
reading the date locally: a daily flight number always has a plausible wrong
answer sitting exactly one day away.

Read as a local date, the ticket is the thing that works and no shift happens
at all. When a date genuinely does disagree, the warning names the zone it
judged the day in, so a timezone surprise cannot be mistaken for a clock that
lost a day:

```
  the track begins on 2026-09-11 at America/Denver but the flight date given
  is 2026-09-10: searching -1 day(s) from the recorded time. If the flight is
  found there, the detector's clock was wrong by that much and so is every
  timestamp in the file
```

If you are unsure which date to give, the track's own title already has it.
The Radiacode app writes the title in **local time** while every `date` field
in the file is UTC (`FORMAT.md`), so a track called
`Track 11 Sep 2026 18:17:52 SLC-AUS flight` began at 00:17:52Z on the 12th and
its `--date` is `2026-09-11`.

The zone is a column in the bundled airport table, sourced from
[OpenFlights](https://openflights.org/data) (ODbL) and joined on by code. It is
never inferred from longitude: China spans five solar hours in a single zone
and Spain keeps Berlin's, so a longitude guess is out by an hour or two in
exactly the places people fly from — and an hour is the whole margin at a date
boundary. 3,908 of the 4,053 airports carry one; for the rest the column is
empty, `--date` is read as UTC, and the report says so. A shift is always a
whole 86400 s day, because the thing it models is a detector clock that lost
days, and such a clock knows nothing about a 23-hour spring-forward.

Only `--date` is local. Everything printed stays UTC and keeps its `Z`.

## Getting OpenSky credentials

Historical data needs an account. **Anonymous callers are refused**: every
history endpoint answers `HTTP 403 You cannot access historical flights`, and
that is not a rate limit, it is the access rule. There is no keyless
alternative — adsb.lol, adsb.fi and airplanes.live serve live aircraft only and
return an empty set for a flight that has landed; ADS-B Exchange refuses direct
access to its history archive.

1. Register at <https://opensky-network.org> (free, non-commercial/research).
2. On your account page, create an **API client** — this is not the website
   login; it issues a separate client id and secret.
3. Export them:

```sh
export OPENSKY_CLIENT_ID=your-client-id
export OPENSKY_CLIENT_SECRET=your-client-secret
```

Or put them in `~/.config/rcta/flight_credentials.json`:

```json
{"opensky": {"client_id": "...", "client_secret": "..."}}
```

The file OpenSky gives you on download is flat and camelCase, and is read as
it comes — save it at that path unedited if you prefer:

```json
{"clientId": "...", "clientSecret": "..."}
```

Prefer the file to the environment if you launch the GUI from Finder or the
Dock, which inherits nothing your shell exports. It is re-read on every fetch,
so an edit takes effect on the next **Fetch** with no restart.

OpenSky asks to be cited in publications; the citation is in
`src/rcta/data/flight_providers.json` and is printed with every fetch.

### Living inside the request budget

The budget is metered per account, and three mechanisms respect that.

- **The query is keyed by the track's own departure airport.** OpenSky has no
  callsign search. The obvious route, `/flights/all` over the track's time
  window, does not work: it returns only flights contained *entirely* within
  the requested interval and caps that interval at two hours, so **no flight
  longer than two hours can ever be found by it** — measured on real
  responses, 29,000 flights across six windows and not one above 1.99 h. The
  failure was silent, and the hint blamed the flight number.
  `/flights/departure` has no containment rule and allows seven days, so one
  request finds a flight of any length. It needs an airport, and the track's
  first fix is on the apron of the one it left from, resolved against a
  bundled table of airports with scheduled service — 4,053 of them, a trimmed
  extract of [OurAirports](https://ourairports.com), which is public domain,
  carrying codes, coordinates and an [IANA
  timezone](#--date-is-a-local-date) from
  [OpenFlights](https://openflights.org/data) (ODbL).
  A match must be within 10 km, so a track that does not start at an airport
  says so rather than querying one 200 km away and returning a confident,
  wrong flight. `--departure ICAO` overrides it for a track that starts away
  from the terminal, and `--date` remains a *check* rather than the search
  key. A typical flight now costs one request plus one for the waypoint path.
- **Every response is cached, with a TTL that depends on the query's age.**
  Settled history is immutable and cached indefinitely — unlike a map tile it
  cannot go stale. A window inside the last two days gets an hour, because
  OpenSky backfills as receivers report.
- **A rate limit stops the fetch.** One retry, `Retry-After` honoured, and a
  hard ceiling of eight requests per invocation so a mistaken window cannot
  spend a day's budget in one command.

`--offline` uses the cache and never the network.

## Trajectory files

Any of these, dispatched on content rather than on the file extension:

| Shape | Notes |
| --- | --- |
| OpenSky `/tracks/all` response | The primary format. A cached response is one of these. |
| OpenSky `/states/all` rows | Needs `--callsign` when the file holds more than one aircraft. |
| JSON array of objects | Named fields: `time`/`lat`/`lon`/`altitude`. |
| CSV / TSV / semicolon | Header matched case-insensitively; column order not assumed. |

Timestamps may be epoch seconds, epoch milliseconds, or ISO 8601.
`--save-trajectory OUT.csv` writes what was fetched, so a flight need only be
looked up once.

### Altitude units are the trap

Every flight tracker aimed at pilots and passengers reports **feet**; OpenSky
and everything aimed at researchers reports **metres**. The ratio is 3.28 and
it falls entirely on the x-axis of the only plot this feature exists to draw,
so a mistake does not look like a mistake — it moves cruise from FL350 to
35 000 m and quadruples the fitted scale height, smoothly and convincingly.

- A column named `altitude_ft`, `baro_altitude` or `geo_altitude` states its
  own unit and is believed.
- A bare `altitude` whose maximum exceeds 20 000 is feet: no civil flight
  cruises at 20 km.
- A bare `altitude` below that is genuinely ambiguous — 8 000 is a light
  aircraft in feet or a turboprop in metres — and is read as metres **with a
  warning naming the override**. Pass `--altitude-unit ft`.

Files written by `--save-trajectory` name the unit in the column header, so a
round trip can never land in the ambiguous branch.

## Getting the data out

The plot in the dock is not meant to be your final figure. It embeds three
choices — bin width, statistic, error bar — and anyone preparing something
for print will want their own. So the **Export** row hands over the numbers
with those decisions unmade, and the `Phase` selector beside it scopes the two
per-sample exports to `all`, `climb`, `descent` or `level`.

| Button | CLI | What it is |
| --- | --- | --- |
| **Samples CSV** | `--samples-csv` | One row per joined sample. The one to take. |
| **Profile CSV** | `--csv` | The binned profile as drawn, both errors per bin. |
| **Trajectory CSV** | `--save-trajectory` | The flight path, re-readable offline. |
| **Report JSON** | `--json` | Everything numeric, provenance included. |

```sh
rcta-cli flight TRACK.rctrk --trajectory flight.csv \
    --samples-csv samples.csv --csv profile.csv \
    --save-trajectory traj.csv --json report.json

# the climb and the descent as separate files, which is usually the point
rcta-cli flight TRACK.rctrk --trajectory flight.csv --branch climb \
    --samples-csv climb.csv
rcta-cli flight TRACK.rctrk --trajectory flight.csv --branch descent \
    --samples-csv descent.csv
```

`samples.csv` carries everything needed to redo the analysis from scratch:

```
index,time_s,utc,lat,lon,accuracy_m,altitude_m,flight_level,
vertical_rate_m_s,phase,dose_rate_nSv_per_h,count_rate_cps,counts,
integration_time_s,distance_along_m,speed_m_s,valid,position_valid
```

Three columns are there for specific reasons:

- **`counts`** is the estimated counts behind the sample, so Poisson errors can
  be recomputed rather than taken on trust.
- **`integration_time_s`** is repeated on every row, because it is an
  *assumption* (`docs/FORMAT.md` trap 5) and errors recomputed without it are
  wrong by its square root. An export that omitted it would invite exactly that.
- **`phase`** labels climb, level and descent, so the two traverses can be
  separated without rederiving a vertical rate.

Only samples that actually received an altitude are written: a row with a blank
altitude cannot be plotted against one, and including it invites a reader to
treat it as sea level. How many were dropped, and why, is in the join summary
and in the JSON report. There are deliberately **no comment lines and no
preamble** in any of these files — a `#` header breaks a plain CSV reader, and
these exist to be opened by other software. The dose-rate column names its unit,
and follows the display unit, so `--unit uSv/h` gives
`dose_rate_uSv_per_h`.

The exports read back the panel's own controls, so a file can never disagree
with the plot on screen. All four buttons stay greyed out until there is
something to write — except **Trajectory CSV**, which is enabled as soon as a
trajectory is loaded even if the position match *refused* it, because opening
it in something else is how you work out why it did not match.

## What the profile does and does not claim

**Barometric altitude is the right variable, not a compromise.** It is a
pressure measurement, and the cosmic dose rate is set by the mass of air above
the detector rather than by geometric height.

**Altitude is never extrapolated, and never interpolated across a long gap.**
Outside the trajectory's span the altitude is NaN; a sample landing in a
waypoint gap wider than `--max-gap` (default 300 s) is NaN too. A ten-minute
gap spanning a top of climb, interpolated linearly, invents a ramp through a
level-off and assigns cruise dose to altitudes the aircraft passed through
once.

**A poor GPS fix does not invalidate an altitude.** Altitude comes from ADS-B,
so the 16 % of fixes worse than 100 m accuracy still get a good altitude and
still contribute their count rate. They are excluded from the position check
and from nothing else.

**The fit is to the bins, not to the samples.** Most of a flight is spent
within a few hundred metres of one cruise altitude, so a per-sample fit is
weighted by dwell time: thousands of points at FL350 against a few dozen
through the climb. The slope is then a two-point measurement wearing the error
bar of ten thousand samples. One point per occupied altitude bin weights the
profile by altitude coverage, which is the question being asked.

**`--fit-floor` excludes the low bins.** Below about 3 km the terrestrial gamma
from soil and building materials is comparable to the cosmic component and does
not vary with altitude at all, so a single exponential there is fitting two
processes with one parameter. Measured on a synthetic profile with a 2000 m
truth, the floor moves the recovered scale height from 4 % high to within
0.2 %; with a large terrestrial pedestal, from 28 % to 14 %. It reduces the
bias and does not abolish it. The method is also biased about 3 % low even
with no terrestrial term, because a bin mean of a convex function exceeds the
function at the bin centre. **Quote two significant figures.**

**Two error bars are carried and the difference between them is the point.**
The counting error is what Poisson allows from the counts in the bin; the
observed error is the spread the samples actually showed. A 500 m bin at cruise
also spans an hour of flying and a thousand kilometres of geomagnetic latitude,
so the second is normally larger, and the plotted bar is the larger of the two.

A ratio well *below* 1 is the interesting case: samples cannot scatter less
than Poisson allows, so it means the assumed per-sample integration time is too
short — the same systematic `docs/FORMAT.md` trap 5 warns about, showing up
somewhere it can be measured.

**Climb versus descent is the honest error bar.** The aircraft traverses the
same altitudes twice, hours apart and a continent away. Agreement is evidence
that the profile is a function of altitude; disagreement bounds how much of it
is something else — geomagnetic latitude, a solar event, or a join that is
quietly wrong at one end. The threshold is Bonferroni-corrected over the number
of shared bins, for the same reason the anomaly scan is.

## The bundled example, and why it disagrees

**The bundled track is a real acquisition** — a RadiaCode carried on DL2618,
Austin to Salt Lake City, 2026-09-07. Its positions and its count rate are
both measured.

`sample-data/DL2618-2026-09-07-synthetic-trajectory.json`, which accompanies
it, is **not**: its altitude is fabricated. It exists so the whole path can be
exercised with no account and no network, and it was built because the lookup
could not fetch the real trajectory — a flight of 2 h 43 min was invisible to
the endpoint then in use (see "Living inside the request budget"). That is
fixed, and the real trajectory now fetches in one request and matches to a
**0.15 km median residual over 2,286 samples**.

The synthetic file's positions are the track's own smoothed fixes, so the
position match passes trivially. Its altitude is a straight
climb–cruise–descent whose four breakpoints were read off the track's
dose-rate crossings — the dose reaches 10 % of its cruise level at 1645 s and
95 % at 3226 s, and falls back through the same levels at 7715 s and 8820 s.
Nothing else was tuned. **Its altitude axis is therefore a guess**, and any
scale height fitted through it is a description of that guess.

![the control rejecting a profile](screenshot-flight.png)

**Both trajectories report an inconsistent profile, and both are right to:**

|  | synthetic trajectory | real trajectory |
| --- | --- | --- |
| scale height | 7.52 ± 0.20 km | 3.33 ± 0.04 km |
| R², χ²/dof | 0.783, 31.2 | 0.962, 20.8 |
| shared bins | 5 | 6 |
| median difference | 101.1 nSv/h (33.4 %) | 20.5 nSv/h (12.5 %) |
| worst bin | 6.03 σ vs 2.33 σ | 4.07 σ vs 2.39 σ |

The fabricated altitude axis inflates the disagreement, which is what a
fabricated axis should be expected to do. It does not create it: on the real
trajectory the climb still yields 2.99 km against the descent's 3.46 km, and
**the descent sits above the climb in 5 of the 6 shared bins**, worst at
7500 m.

That direction is the interesting part, because it is the one the geometry
predicts. The climb happens over Austin at 30.2° N and the descent over Salt
Lake City at 40.8° N, and cosmic dose rate at altitude rises toward the
geomagnetic poles as the cutoff rigidity falls. Ten degrees of latitude
between the two traverses is a confound of exactly this sign, and separating
altitude from geomagnetic latitude is what this control exists to bound in
the first place. A single flight cannot settle it — that would need two
traverses at the *same* latitude — which is precisely why the tool refuses to
call 3.33 km a measurement of the atmosphere.

A regression test pins the synthetic pair: if it ever starts agreeing, the
control has broken.

For a case with a known answer, `tests/synthetic.flight_track()` builds a track
and a trajectory from one altitude profile with a 2000 m scale height, which
the fit recovers to within 3 %:

![a profile with a known answer](screenshot-flight-fit.png)
