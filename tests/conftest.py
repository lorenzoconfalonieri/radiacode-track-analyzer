import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rcta.core.io_radiacode import load_track
from rcta.core.quality import assess

SAMPLE = (Path(__file__).resolve().parents[1] / "sample-data"
          / "Track 07 Sep 2026 07-53-59.rctrk")


@pytest.fixture(scope="session")
def sample_path():
    if not SAMPLE.exists():
        pytest.skip(f"sample data not present at {SAMPLE}")
    return SAMPLE


WALK = SAMPLE.with_name("Track 07 Sep 2026 14-30-57.rctrk")


@pytest.fixture(scope="session")
def walk_path():
    """The bundled 2 h walking survey: 348 samples at a 10 s cadence."""
    if not WALK.exists():
        pytest.skip(f"walk sample not present at {WALK}")
    return WALK


@pytest.fixture(scope="session")
def sample(sample_path):
    return load_track(sample_path)


@pytest.fixture(scope="session")
def assessed(sample):
    """The sample with GPS quality masks and legs applied."""
    return assess(sample)


SAMPLE_TRAJECTORY = (Path(__file__).resolve().parents[1] / "sample-data"
                     / "DL2618-2026-09-07-synthetic-trajectory.json")


@pytest.fixture(scope="session")
def sample_trajectory_path():
    """The synthetic trajectory shipped alongside the real sample track.

    The *track* is a real acquisition; this trajectory is not. Its positions
    are the track's own fixes and its altitude axis is fabricated -- not
    fetched from anywhere, see the ``_note`` inside the file. It exists so
    that the whole flight path -- read, join, validate, bin, fit -- can be
    exercised with no account and no network.
    """
    if not SAMPLE_TRAJECTORY.exists():
        pytest.skip(f"sample trajectory not present at {SAMPLE_TRAJECTORY}")
    return SAMPLE_TRAJECTORY


@pytest.fixture()
def flight_case():
    """``(track, trajectory, truth)`` with a known scale height."""
    import synthetic
    return synthetic.flight_track()
