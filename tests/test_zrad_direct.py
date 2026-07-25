"""Test MMM_ASbaffledradzmatrix — direct numerical integration."""

from pathlib import Path

import numpy as np
from scipy.special import j1 as besselj1
from scipy.special import jn_zeros

from mmm_toolbox.radiation import (
    _struve_h1,
    baffled_rad_zmatrix_axi,
    baffled_rad_zmatrix_direct_axi,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


def _load_bz():
    return np.concatenate([[0.0], jn_zeros(1, 199)])


def test_direct_vs_analytical_fundamental():
    """Fundamental mode (0,0) must match analytical R00 + i*X00."""
    bz = _load_bz()
    k = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    a = np.sqrt(1.0 / np.pi)
    kR = k * a

    R00 = 1.0 - besselj1(2.0 * kR) / kR
    X00 = 2.0 * _struve_h1(2.0 * kR) / (2.0 * kR)
    expected_00 = R00 + 1j * X00

    Zmat = baffled_rad_zmatrix_direct_axi(
        k, 1.0, 1.0, 1.0, 3, bz, use_hf_approx=False,
    )

    np.testing.assert_allclose(Zmat[0, 0, :], expected_00, atol=1e-10)


def test_direct_symmetry():
    """Radiation impedance matrix must be symmetric: Z[i,j] == Z[j,i]."""
    bz = _load_bz()
    k = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    Zmat = baffled_rad_zmatrix_direct_axi(
        k, 1.0, 1.0, 1.0, 4, bz, use_hf_approx=False,
    )

    for i in range(4):
        for j in range(4):
            np.testing.assert_allclose(
                Zmat[i, j, :], Zmat[j, i, :], atol=1e-8,
            )


def test_direct_agrees_with_interpolation():
    """Direct integration should agree with interpolation within 2%."""
    bz = _load_bz()
    k = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    max_modes = 5

    Zmat_direct = baffled_rad_zmatrix_direct_axi(
        k, 1.0, 1.0, 1.0, max_modes, bz, use_hf_approx=False,
    )

    Zmat_intp = baffled_rad_zmatrix_axi(
        k, 1.0, 1.0, 1.0, max_modes,
        filename=str(TEST_DATA_DIR / "ZradAS32.mat"),
    )

    np.testing.assert_allclose(
        np.abs(Zmat_direct), np.abs(Zmat_intp), rtol=0.02, atol=0.0,
    )
