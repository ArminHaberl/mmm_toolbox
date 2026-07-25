"""Test MMM_ASbaffledradzmatrix — direct numerical integration + table generation."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from scipy.interpolate import CubicSpline
from scipy.special import j1 as besselj1
from scipy.special import jn_zeros

from mmm_toolbox.radiation import (
    _build_lookup_table,
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


def _interp_onto(ka_src, Z_src, ka_tgt):
    M = Z_src.shape[0]
    out = np.empty((M, M, len(ka_tgt)), dtype=complex)
    for m in range(M):
        for n in range(M):
            cs_r = CubicSpline(ka_src, Z_src[m, n, :].real)
            cs_x = CubicSpline(ka_src, Z_src[m, n, :].imag)
            out[m, n, :] = cs_r(ka_tgt) + 1j * cs_x(ka_tgt)
    return out


@pytest.mark.slow
def test_built_table_matches_baseline():
    """Auto-generated 32-mode table must agree with git-tracked baseline."""
    ka_new, Z_new = _build_lookup_table(32, n_quad=2000)

    base = scipy.io.loadmat(str(TEST_DATA_DIR / "ZradAS32.mat"))
    ka_base = base["ka"].flatten()
    Z_base = base["Zmat"]

    ka = np.linspace(
        max(ka_new[0], ka_base[0]),
        min(ka_new[-1], ka_base[-1]),
        2000,
    )
    Z_new = _interp_onto(ka_new, Z_new, ka)
    Z_base = _interp_onto(ka_base, Z_base, ka)

    abs_diff = np.abs(Z_new - Z_base)
    significant = np.abs(Z_base) > 1e-4
    ok = np.count_nonzero(
        significant & (abs_diff <= 0.02 * np.abs(Z_base)),
    )
    n_sig = np.count_nonzero(significant)
    pct = 100.0 * ok / n_sig

    assert pct >= 99.5, (
        f"Only {pct:.2f}% of significant elements within 2% "
        f"(need >= 99.5%, {ok}/{n_sig})"
    )
