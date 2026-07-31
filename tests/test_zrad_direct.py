"""Test MMM_ASbaffledradzmatrix — direct numerical integration + table generation."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from scipy.interpolate import CubicSpline
from scipy.special import j1 as besselj1
from scipy.special import jn_zeros, roots_legendre

from mmm_toolbox.radiation import (
    _build_lookup_table,
    _compute_zmat_fixed_quad,
    _get_lookup_table,
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


def test_zrad_max_modes_error():
    """Requesting more modes than available must raise ValueError."""
    with pytest.raises(ValueError, match="Higher number of modes"):
        baffled_rad_zmatrix_axi(
            np.array([1.0]), 1.0, 1.0, 1.0, 40,
            filename=str(TEST_DATA_DIR / "ZradAS32.mat"),
        )


def test_direct_with_hf_approx():
    """Direct integration with HF asymptotic must agree with interpolation."""
    bz = _load_bz()
    k = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    max_modes = 5

    Zmat_direct = baffled_rad_zmatrix_direct_axi(
        k, 1.0, 1.0, 1.0, max_modes, bz, use_hf_approx=True,
    )

    Zmat_intp = baffled_rad_zmatrix_axi(
        k, 1.0, 1.0, 1.0, max_modes,
        filename=str(TEST_DATA_DIR / "ZradAS32.mat"),
    )

    np.testing.assert_allclose(
        np.abs(Zmat_direct), np.abs(Zmat_intp), rtol=0.05, atol=0.0,
    )


def test_cache_superset_reuse():
    """A smaller mode-count request must reuse a larger cached table."""
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "mmm_toolbox"
    q = 2000

    # Clean up: remove any cached tables at the test n_quad
    for cf in cache_dir.glob(f"ZradAS*_q{q}.mat"):
        cf.unlink()

    # Build 24-mode table
    _get_lookup_table(24, q)

    # Remove the smaller 8-mode cache file if it happened to exist
    small = cache_dir / f"ZradAS8_q{q}.mat"
    if small.exists():
        small.unlink()

    # Request 8-mode — must reuse 24-mode, NOT rebuild
    _get_lookup_table(8, q)

    # No 8-mode file should have been created
    assert not small.exists(), (
        "8-mode request should have reused the existing 24-mode "
        "cache instead of building a new table"
    )


def test_singularity_correction_resistance():
    """L'Hopital limit fires cleanly when denom -> 0 (resistance loop).

    Constructs a quadrature node tau and sets kR so that
    (bz[1] / kR)^2 == tau^2 *exactly*, guaranteeing ``abs(denom) < 1e-12``
    inside the resistance quadrature.  Verifies the output is finite and
    the Zmat is symmetric — the function would produce NaN/inf without the
    vectorised singularity correction.
    """
    bz = np.concatenate([[0.0], jn_zeros(1, 199)])
    n_quad = 50

    nodes, _weights = roots_legendre(n_quad)
    t = 0.5 * (nodes + 1.0) * (np.pi / 2.0)
    tau = np.sin(t[n_quad // 2])
    assert tau != 0.0

    kR = np.array([bz[1] / tau])
    k = kR.copy()
    a = 1.0
    M = 3

    Zmat = _compute_zmat_fixed_quad(
        k, kR, a, bz, M, use_hf_approx=False, n_quad=n_quad,
    )

    assert np.all(np.isfinite(Zmat)), "NaN or inf in Zmat"
    for i in range(M):
        for j in range(i + 1, M):
            np.testing.assert_allclose(
                Zmat[i, j, :], Zmat[j, i, :], atol=1e-10,
            )


def test_singularity_correction_reactance():
    """L'Hopital limit fires cleanly when denom -> 0 (reactance loop).

    Mirrors the resistance test but targets the reactance quadrature
    where tau = cosh(t_x) >= 1, ensuring *both* vectorised
    singular-value fallbacks are exercised.
    """
    bz = np.concatenate([[0.0], jn_zeros(1, 199)])
    n_quad = 50

    nodes_x, _weights_x = roots_legendre(n_quad)
    t_x = 0.5 * (nodes_x + 1.0) * 10.0
    tau = np.cosh(t_x[n_quad // 2])
    assert tau > 1.0

    kR = np.array([bz[1] / tau])
    k = kR.copy()
    a = 1.0
    M = 3

    Zmat = _compute_zmat_fixed_quad(
        k, kR, a, bz, M, use_hf_approx=False, n_quad=n_quad,
    )

    assert np.all(np.isfinite(Zmat)), "NaN or inf in Zmat"
    for i in range(M):
        for j in range(i + 1, M):
            np.testing.assert_allclose(
                Zmat[i, j, :], Zmat[j, i, :], atol=1e-10,
            )
