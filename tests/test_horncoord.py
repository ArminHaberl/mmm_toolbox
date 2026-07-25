"""Test MMM_1Dhorncoord — horn contour generation for all 9 types."""

from pathlib import Path

import numpy as np
import scipy.io

from hornsim.geometry import horn_coord_1d

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
_RTH = np.sqrt(10e-4 / np.pi)
_RM = np.sqrt(500e-4 / np.pi)
_RM_SPH = np.sqrt(2000e-4 / np.pi)


_VAR_NAMES = {
    "flared_conical": "horncoords_flared",
}


def _load_expected(htype: str) -> np.ndarray:
    safe = htype.replace(" ", "_")
    d = scipy.io.loadmat(str(TEST_DATA_DIR / f"horncoords_{safe}.mat"))
    varname = _VAR_NAMES.get(htype, f"horncoords_{safe}")
    return d[varname]


def _assert_horn(htype: str, rth: float, rm: float, **kwargs):
    expected = _load_expected(htype)
    result = horn_coord_1d(htype, rth, rm,
                           length=0.3, tn=1.0, dz=0.0012, **kwargs)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_conical():
    _assert_horn("conical", _RTH, _RM)


def test_exponential():
    _assert_horn("exponential", _RTH, _RM)


def test_hypex():
    _assert_horn("hypex", _RTH, _RM)


def test_oswg():
    _assert_horn("oswg", _RTH, _RM)


def test_bessel():
    _assert_horn("bessel", _RTH, _RM)


def test_spherical():
    _assert_horn("spherical", _RTH, _RM_SPH)


def test_tractrix():
    _assert_horn("tractrix", _RTH, _RM)


def test_radius():
    _assert_horn("radius", _RTH, 0.0, add_radius=False,
                 th_fta=45.0, rad_r=0.2, rad_fta=70.0)


def test_flared_conical():
    _assert_horn("flared_conical", _RTH, _RM)
