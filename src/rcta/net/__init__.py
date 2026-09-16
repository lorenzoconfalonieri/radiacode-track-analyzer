"""Outbound requests that are not map tiles, and the fence around them.

:mod:`rcta.core` may never import a network library -- an estimator whose
answer depends on whether a server was reachable is not reproducible, and
``tests/test_architecture.py`` enforces that absolutely. Basemap tiles are
already excluded from the core by living in ``gui/tiles.py``, but they get to
be Qt-based because the map is inherently interactive. A flight trajectory is
not: it is fetched once, by the CLI as often as by the GUI, so it cannot
depend on a Qt event loop existing.

Hence this package. Everything here is plain-stdlib and blocking, uses no
threads of its own, and returns core objects that carry no trace of how they
arrived. When the GUI needs a fetch off the main thread it hands the call to
``gui/worker.py``, which is the one module allowed to know about concurrency.
"""

from rcta.net.flights import (FlightCache, FlightFetchError, FlightProvider,
                              OpenSkyClient, credentials_from_env,
                              fetch_trajectory, load_flight_providers)

__all__ = [
    "FlightCache",
    "FlightFetchError",
    "FlightProvider",
    "OpenSkyClient",
    "credentials_from_env",
    "fetch_trajectory",
    "load_flight_providers",
]
