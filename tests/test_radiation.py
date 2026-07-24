"""Test MMM_ASbaffledradzmatrixIntp — modal radiation impedance matrix."""

from pathlib import Path

import numpy as np

from hornsim.radiation import baffled_rad_zmatrix_axi

MATLAB_DIR = Path(__file__).parent.parent / "matlab"


def test_radiation_zmatrix(zrad_mat, init_mat):
    """Full radiation impedance: 8 modes, 200 wavenumbers."""
    expected = zrad_mat["Zrad"]
    k = init_mat["kvec"].flatten()
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()
    Sm = init_mat["Sm"].item()
    n_modes = int(init_mat["nModes"].item())

    result = baffled_rad_zmatrix_axi(k, rho, c, Sm, n_modes, str(MATLAB_DIR / "ZradAS32.mat"))

    assert result.shape == (8, 8, 200)
    np.testing.assert_allclose(result, expected, atol=1e-10)
