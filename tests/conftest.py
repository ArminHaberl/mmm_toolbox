"""Shared fixtures for mmm_toolbox tests — load MATLAB-generated reference data."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from scipy.special import jn_zeros

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture(scope="session")
def bessel_zeros():
    """Full array of Bessel J1 zeros (200 values + leading zero)."""
    return np.concatenate([[0.0], jn_zeros(1, 199)])


# ---------------------------------------------------------------------------
# Integration / end-to-end test data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def horncoords_mat():
    return _load("horncoords.mat")


@pytest.fixture(scope="session")
def init_mat():
    return _load("init_data.mat")


@pytest.fixture(scope="session")
def zrad_mat():
    return _load("zrad.mat")


@pytest.fixture(scope="session")
def calculate_mat():
    return _load("calculate.mat")


@pytest.fixture(scope="session")
def prad_mat():
    return _load("prad.mat")


@pytest.fixture(scope="session")
def di_mat():
    return _load("di.mat")


# ---------------------------------------------------------------------------
# Isolated unit test data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def makefmat_expanding_mat():
    return _load("makefmat_expanding.mat")


@pytest.fixture(scope="session")
def makefmat_contracting_mat():
    return _load("makefmat_contracting.mat")


@pytest.fixture(scope="session")
def makefmat_equal_mat():
    return _load("makefmat_equal.mat")


@pytest.fixture(scope="session")
def makekm_mat():
    return _load("makekm.mat")


@pytest.fixture(scope="session")
def eigenfunctions_mat():
    return _load("eigenfunctions.mat")


@pytest.fixture(scope="session")
def prad_rayleigh_mat():
    return _load("prad_rayleigh.mat")


@pytest.fixture(scope="session")
def pressure_dist_mat():
    return _load("pressure_dist.mat")


@pytest.fixture(scope="session")
def pressure_dist_nf_mat():
    return _load("pressure_dist_nearfield.mat")


@pytest.fixture(scope="session")
def di_gerzon_mat():
    return _load("di_gerzon.mat")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    d = scipy.io.loadmat(str(TEST_DATA_DIR / name))
    # Remove MATLAB metadata keys so the dict only has data variables
    return {k: v for k, v in d.items() if not k.startswith("__")}


def _scalar(d: dict, key: str) -> float:
    """Extract a MATLAB scalar from a loaded .mat dict."""
    val = d[key]
    if isinstance(val, np.ndarray):
        return val.item()
    return val
