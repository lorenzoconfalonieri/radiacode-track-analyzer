"""Radiacode Track Analyzer: GPS radiation track analysis."""

__version__ = "0.1.0"

#: Sent by every outbound request this program makes, whether for map tiles or
#: for a flight trajectory. It lives here rather than in either subsystem
#: because both need it and neither should import the other: the basemap
#: fetcher is Qt-based and lives in ``gui/``, the flight client is stdlib and
#: lives in ``net/``, and a shared constant is the only thing they have in
#: common. A generic agent such as "python-requests" is explicitly forbidden by
#: the OpenStreetMap tile policy and is poor manners towards every other
#: volunteer-funded service, so this names the application and offers a
#: contact URL.
USER_AGENT = (f"RadiacodeTrackAnalyzer/{__version__} "
              "(+https://github.com/rcta/radiacode-track-analyzer)")
