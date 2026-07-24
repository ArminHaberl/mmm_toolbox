"""Test MMM_ASbaffledradzmatrix — direct numerical integration + precomputation."""

import os
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
from scipy.special import j1 as besselj1

from hornsim.radiation import (
    baffled_rad_zmatrix_axi,
    baffled_rad_zmatrix_direct_axi,
    precompute_rad_zmatrix,
    _struve_h1,
)

MATLAB_DIR = Path(__file__).parent.parent / "matlab"


def _load_bz():
    d = scipy.io.loadmat(str(MATLAB_DIR / "MMM_besselzeros.mat"))
    return d["bz"].flatten()


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
    """Direct integration should agree with interpolation within ~1%."""
    bz = _load_bz()
    k = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    max_modes = 5

    Zmat_direct = baffled_rad_zmatrix_direct_axi(
        k, 1.0, 1.0, 1.0, max_modes, bz, use_hf_approx=False,
    )

    Zmat_intp = baffled_rad_zmatrix_axi(
        k, 1.0, 1.0, 1.0, max_modes,
        str(MATLAB_DIR / "ZradAS32.mat"),
    )

    np.testing.assert_allclose(
        np.abs(Zmat_direct), np.abs(Zmat_intp), rtol=0.02, atol=0.0,
    )


def test_precompute_small():
    """Precompute a small lookup table and verify structure."""
    bz = _load_bz()
    max_modes = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "ZradAS5_test.mat"
        result_path = precompute_rad_zmatrix(
            max_modes=max_modes,
            bessel_zeros_path=str(MATLAB_DIR / "MMM_besselzeros.mat"),
            output_path=str(out),
            progress_report=False,
        )

        assert os.path.exists(result_path)

        d = scipy.io.loadmat(result_path)
        ka = d["ka"].flatten()
        Zmat = d["Zmat"]

        assert Zmat.shape[:2] == (max_modes, max_modes)
        assert len(ka) == Zmat.shape[2]
        assert np.all(Zmat[0, 0, :].real > 0.0)
