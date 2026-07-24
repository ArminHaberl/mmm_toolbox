"""Shared fixtures for hornsim tests — load MATLAB-generated reference data."""

import pytest
import scipy.io
import numpy as np
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
MATLAB_DIR = Path(__file__).parent.parent / "matlab"


@pytest.fixture(scope="session")
def bessel_zeros():
    """Full array of Bessel J1 zeros from MMM_besselzeros.mat."""
    d = scipy.io.loadmat(str(MATLAB_DIR / "MMM_besselzeros.mat"))
    return d["bz"].flatten()


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
