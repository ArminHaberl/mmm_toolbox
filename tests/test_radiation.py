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
