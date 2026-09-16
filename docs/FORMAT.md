# The `.rctrk` track format

Reverse-engineered from a RadiaCode-102 export. There is no published
specification, so everything here was measured against a real file and is
enforced by tests in `tests/test_io.py`.

The RadiaCode app writes **two different formats under the same extension**.
Dispatch on the first non-whitespace byte, never on the filename: `{` means
JSON, anything else means the tab-separated variant.

## JSON dialect

One pretty-printed JSON object. The separators are `" : "` (space-colon-space),
which is Jackson's `DefaultPrettyPrinter` — a hint that the writer is the
Android app. Do not attempt to parse it with regular expressions.

```json
{
  "periods" : [ { "distance" : 1795151.69, "start" : 1788785639, "end" : 1788795474 } ],
  "start" : 1788785639,
  "sv" : false,
  "title" : "Track 07 Sep 2026 07:53:59",
  "devices" : [ "RadiaCode-102" ],
  "markers" : [ { "lat" : 36.153351, "doseRate" : 90.89, "countRate" : 19.76,
                  "acc" : 15, "date" : 1788791002, "lon" : -104.537687 } ]
}
```

| Key | Type | Meaning |
| --- | --- | --- |
| `periods` | list of object | Recording segments. `distance` is the device's own odometer in **metres**; `start`/`end` are unix epoch **seconds**. |
| `start` | int | Unix epoch seconds of the first sample. |
| `sv` | bool | Spectrum-valid flag. False in every file seen so far; the format carries no spectra. |
| `title` | str | Display name. Note it is **local time** while `date` fields are UTC. |
| `devices` | list of str | e.g. `["RadiaCode-102"]`. |
| `markers` | list of object | The samples. Despite the name these are not user waypoints. |

Each marker has exactly six keys:

| Field | Type | Unit |
| --- | --- | --- |
| `date` | int | Unix epoch **seconds**, UTC |
| `lat`, `lon` | float | WGS84 degrees |
| `acc` | int | GPS horizontal accuracy, **metres** |
| `doseRate` | float *or* int | **µR/h** (see below) |
| `countRate` | float *or* int | **counts per second** |

### Dose units

The file does not record the unit. It is **µR/h**, and `rcta` multiplies by 10
on load to get nSv/h (1 R = 10 mSv for gamma by the conventional survey-meter
equivalence).

This was settled against a real walking log plus the app's own display, not by
inference. The evidence:

| | walk (real, ground) | flight (cruise) |
| --- | --- | --- |
| stored median value | 7.19 | 90.56 |
| read as nSv/h | 0.0072 µSv/h | 0.091 µSv/h |
| read as µR/h | **0.072 µSv/h** | **0.91 µSv/h** |

Normal outdoor background is 0.05–0.15 µSv/h and the app displays about
0.07 µSv/h for that walk, so µR/h is right. Reading the values as nSv/h — which
an earlier version of this parser did — puts ordinary background ten times
below anything physically possible, and puts an aircraft at cruise altitude
*below* ground level.

Note the nSv/h-per-cps ratio differs between the two files (13 on the ground,
51 at altitude). That is expected: a CsI detector's dose response is
energy-dependent, and cosmic secondaries at altitude are far harder than
terrestrial gammas. `rcta` computes the ratio per track rather than assuming
one.

The conversion happens exactly once, in the loader. Everything downstream is
canonically nSv/h and converts only for display. The µR/h assumption is
verified for the JSON dialect; no tab-separated export was available to check,
so that path applies the same conversion and says so in a warning.

## Traps

Each of these was observed in the bundled 2,964-sample file and each silently
corrupts an analysis if ignored.

1. **`markers` is not sorted by `date`.** Consecutive deltas reach −8,360 s.
   Summing haversine in file order gives 1,626,923 km; date-sorted gives
   1,792.2 km. Sort before computing anything, and cross-check against
   `periods[0].distance` — the two agree to 0.16 %.
2. **Key order inside a marker is randomised.** 81 distinct orderings appear in
   one file. Look every field up by name.
3. **Numeric fields are sometimes bare integers.** `doseRate` is an int in 25
   records and `countRate` in 32. Coerce both.
4. **Sampling is irregular.** Median 2 s, but 21 gaps exceed 10 s and the
   longest is 763 s. Any windowed calculation must be defined in *seconds*, not
   in samples, or a window will silently span a dropout.
5. **The file records timestamps, not integration times.** There is no way to
   know the detector live time behind each reported rate. `rcta` infers it from
   the median spacing, clips it to a believable range, and warns — because
   every detection limit scales directly with it.
6. **GPS accuracy has a long tail.** Median 28 m, maximum 397 m; 16 % of fixes
   exceed 100 m. A bad fix still carries a good count rate, so position and
   value must be masked independently.
7. **On a walking track the fix error exceeds the step.** At 1.4 m/s with 2 s
   logging the step is 2.8 m against an 8 m fix, so summed raw distance is
   dominated by noise and overstates the route roughly fivefold. This is not a
   property of the format but of how the format is normally used, and it is
   the single most important thing to correct before quoting any distance,
   speed, or distance-based profile. The bundled sample is a flight and hides
   it completely.

## Tab-separated dialect

```
Track: <name>
Timestamp	Latitude	Longitude	Accuracy	DoseRate	CountRate	Comment
1788785639	30.100000	-97.700000	8	95.0	18.5
```

Column *names* are matched case-insensitively; the order is not assumed. This
dialect carries no `periods`, so the odometer cross-check is unavailable —
`rcta` reports it as absent rather than substituting its own geometry.
Timestamps may be epoch seconds, epoch milliseconds, or `YYYY-MM-DD HH:MM:SS`.

## What the format does not contain

No spectra, no channel data, no calibration, no altitude, and no detector live
time. Nuclide identification is therefore impossible from a track alone: it
needs the channel spectrum, which the device records separately and this
format does not carry.

The missing altitude matters enough on an airborne track that `rcta` imports
one from an external flight trajectory and validates it against the recorded
fixes — see `docs/FLIGHT.md`. It is an import, not a measurement, and the
program keeps saying so.
