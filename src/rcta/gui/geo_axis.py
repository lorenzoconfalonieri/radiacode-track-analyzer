"""Degree ticks and a ground-metre scale bar on axes whose units are metres.

The map plots in EPSG:3857 metres, because that is what makes tiles line up
and what lets the aspect ratio be locked 1:1 correctly. But projected metres
are a physically meaningless quantity to print on an axis: nobody can state a
position in them, and a distance read off them is wrong by 1/cos(latitude).

So the axes are relabelled in degrees and a separate bar reports true ground
distance. Both are needed. A scientist reading a survey wants to state where
something is (degrees) *and* judge how far apart two things are (metres), and
neither alone will do.

The distortion is not a rounding detail. Across the bundled track's ten
degrees of latitude the Mercator scale factor runs from 1.16 at the south end
to 1.32 at the north -- a thirteen per cent difference between the top and
bottom of one picture. A single unqualified scale bar would be wrong at one
end, so :class:`GroundScaleBar` recomputes for the view centre and, when the
view spans enough latitude for it to matter, labels the latitude it is valid
at rather than implying it applies everywhere.
"""

from __future__ import annotations

import math

import pyqtgraph as pg

from rcta.core.geo import lonlat_to_web_mercator, mercator_scale, web_mercator_to_lonlat

__all__ = ["GeoAxisItem", "GroundScaleBar", "format_degree", "nice_distance"]

#: Tick steps that land on whole degrees, then whole minutes, then seconds.
_DEG_STEPS = [30.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 1 / 6, 1 / 12, 1 / 30,
              1 / 60, 1 / 120, 1 / 360, 1 / 720, 1 / 1800, 1 / 3600]


def _pick_step(target: float) -> float:
    """Largest tabulated step not exceeding ``target``."""
    for step in _DEG_STEPS:
        if step <= target:
            return step
    return _DEG_STEPS[-1]


def format_degree(value: float, step: float, is_lat: bool) -> str:
    """A coordinate as degrees/minutes/seconds, at a precision matching ``step``.

    Hemisphere suffix rather than a minus sign: ``104 deg 32' W`` is
    unambiguous where ``-104.5`` invites a reader to wonder about the
    convention.
    """
    hemi = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    value = abs(value)
    if step >= 1.0:
        return f"{value:.0f}°{hemi}"
    if step >= 0.1:
        return f"{value:.2f}°{hemi}"

    degrees = int(value)
    minutes = (value - degrees) * 60.0
    if step >= 1 / 60:
        return f"{degrees}°{minutes:02.0f}'{hemi}"
    whole_minutes = int(minutes)
    seconds = (minutes - whole_minutes) * 60.0
    return f"{degrees}°{whole_minutes:02d}'{seconds:04.1f}\"{hemi}"


def nice_distance(target: float) -> float:
    """Round a distance down to a 1/2/5 x 10^n value."""
    if target <= 0:
        return 1.0
    exponent = math.floor(math.log10(target))
    base = 10.0 ** exponent
    for factor in (5.0, 2.0, 1.0):
        if factor * base <= target:
            return factor * base
    return base


def format_distance(metres: float) -> str:
    if metres >= 1000.0:
        return f"{metres / 1000.0:g} km"
    return f"{metres:g} m"


class GeoAxisItem(pg.AxisItem):
    """Degree ticks on an axis whose data units are EPSG:3857 metres.

    Latitude ticks are not evenly spaced in metres, so the positions are
    generated in degrees and then projected -- never the other way round,
    which would put the labels at irregular fractions of a degree.
    """

    def __init__(self, orientation, **kwargs) -> None:
        super().__init__(orientation, **kwargs)
        self._is_lat = orientation in ("left", "right")
        self.enableAutoSIPrefix(False)

    def _to_degrees(self, metres: float) -> float:
        if self._is_lat:
            return float(web_mercator_to_lonlat(0.0, metres)[1])
        return float(web_mercator_to_lonlat(metres, 0.0)[0])

    def _to_metres(self, degrees: float) -> float:
        if self._is_lat:
            return float(lonlat_to_web_mercator(0.0, degrees)[1])
        return float(lonlat_to_web_mercator(degrees, 0.0)[0])

    def tickValues(self, minVal, maxVal, size):
        try:
            lo, hi = sorted(self._to_degrees(v) for v in (minVal, maxVal))
        except (ValueError, OverflowError):
            return []
        span = hi - lo
        if not math.isfinite(span) or span <= 0:
            return []

        # About 90 device pixels per major tick keeps labels from colliding.
        n_major = max(2.0, float(size) / 90.0)
        major = _pick_step(span / n_major)
        minor = _pick_step(span / (n_major * 5.0))

        out = []
        for step, is_major in ((major, True), (minor, False)):
            first = math.ceil(lo / step) * step
            values = []
            v = first
            # Guard against a pathological zoom producing millions of ticks.
            while v <= hi and len(values) < 512:
                if is_major or abs(v / major - round(v / major)) > 1e-9:
                    values.append(v)
                v += step
            out.append((step, [self._to_metres(v) for v in values]))
        return out

    def tickStrings(self, values, scale, spacing):
        # ``spacing`` is whatever tickValues declared as the step, and this
        # axis declares it in *degrees*. Using it directly is both correct and
        # cheaper than re-deriving a step from the values, which mixes the
        # major and minor levels together and picks the wrong precision --
        # printing a whole-degree tick as "41.00 deg N".
        step = float(spacing) if spacing else 1.0
        return [format_degree(self._to_degrees(v), step, self._is_lat)
                for v in values]


class GroundScaleBar(pg.ScaleBar):
    """A scale bar in true ground metres, valid at the view-centre latitude."""

    def __init__(self, **kwargs) -> None:
        super().__init__(size=1000.0, suffix=None, **kwargs)
        self._caption = ""

    def update_for(self, view_rect) -> None:
        """Resize and relabel for the current view."""
        centre_y = view_rect.center().y()
        try:
            lat = float(web_mercator_to_lonlat(0.0, centre_y)[1])
        except (ValueError, OverflowError):
            return
        inflation = float(mercator_scale(lat))
        if not math.isfinite(inflation) or inflation <= 0:
            return

        ground = nice_distance(view_rect.width() / inflation * 0.25)
        self.size = ground * inflation

        label = format_distance(ground)
        # Only qualify the bar when the view spans enough latitude for the
        # distortion to be visible; on a walking survey it is noise.
        if view_rect.height() > 2.0e5:
            hemi = "N" if lat >= 0 else "S"
            label += f"  at {abs(lat):.1f}°{hemi}"
        self.text.setText(label)
        self.updateBar()
