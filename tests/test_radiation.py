"""Test MMM_ASbaffledradzmatrixIntp — modal radiation impedance matrix."""

from pathlib import Path

import numpy as np
import pytest

from mmm_toolbox.radiation import baffled_rad_zmatrix_axi

DATA_DIR = Path(__file__).parent.parent / "test_data"


def test_radiation_zmatrix(zrad_mat, init_mat):
    """Full radiation impedance: 8 modes, 200 wavenumbers."""
    expected = zrad_mat["Zrad"]
    k = init_mat["kvec"].flatten()
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()
    Sm = init_mat["Sm"].item()
    n_modes = int(init_mat["nModes"].item())

    result = baffled_rad_zmatrix_axi(k, rho, c, Sm, n_modes, str(DATA_DIR / "ZradAS32.mat"))

    assert result.shape == (8, 8, 200)
    np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.slow
def test_radiation_zmatrix_cached(init_mat):
    """Auto-cache interpolated Z must match baseline-table interpolation.

    Exercises the default (no *filename*) path where the lookup table
    is built on first call by our fixed Gauss-Legendre quadrature and
    cached to disk.  Uses a small set of moderate k values so that
    spline interpolation error does not dominate the comparison.
    """
    k = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()
    Sm = init_mat["Sm"].item()
    max_modes = 5

    result_cache = baffled_rad_zmatrix_axi(k, rho, c, Sm, max_modes)
    result_baseline = baffled_rad_zmatrix_axi(
        k, rho, c, Sm, max_modes,
        filename=str(DATA_DIR / "ZradAS32.mat"),
    )

    assert result_cache.shape == (5, 5, len(k))
    np.testing.assert_allclose(
        np.abs(result_cache), np.abs(result_baseline), rtol=0.02, atol=0.0,
    )


def test_pos_id_extrapolation():
    """Frequencies above the lookup-table ka range (kain > ka_max).

    Exercises the ``pos_id`` branch where HF asymptotic formulas are used
    for both resistance and reactance instead of spline interpolation.
    """
    S = 0.01
    k = np.array([5000.0, 8000.0])
    a = np.sqrt(S / np.pi)
    kain = k * a
    assert np.all(kain > 245.4), (
        f"kain = {kain} must exceed ZradAS32 ka_max ≈ 245.4"
    )

    Zmat = baffled_rad_zmatrix_axi(
        k, 1.2, 344.0, S, 3, filename=str(DATA_DIR / "ZradAS32.mat"),
    )

    assert Zmat.shape == (3, 3, 2)
    assert np.all(np.isfinite(Zmat)), "NaN or inf in extrapolated Zmat"


def test_low_x_id_extrapolation():
    """Frequencies below the lookup-table ka range (kain < ka[0]).

    Exercises the ``low_x_id`` branch where the imaginary part is
    extrapolated linearly from (0, 0) through the first table point.
    """
    S = 0.01
    k = np.array([0.05, 0.3])
    a = np.sqrt(S / np.pi)
    kain = k * a
    assert np.all(kain < 0.1), (
        f"kain = {kain} must be below ZradAS32 ka[0] = 0.1"
    )

    Zmat = baffled_rad_zmatrix_axi(
        k, 1.2, 344.0, S, 3, filename=str(DATA_DIR / "ZradAS32.mat"),
    )

    assert Zmat.shape == (3, 3, 2)
    assert np.all(np.isfinite(Zmat)), "NaN or inf in extrapolated Zmat"
