"""Test MMM_ASgeteigenfunctions — Bessel-mode eigenfunction evaluation."""

import numpy as np

from hornsim.axi import get_eigenfunctions_axi


def test_eigenfunctions(eigenfunctions_mat, init_mat):
    """8 modes at 20 radial points, normalized."""
    expected = eigenfunctions_mat["phi"]
    R = eigenfunctions_mat["R_mouth"].item()
    rcoords = eigenfunctions_mat["r_test"].flatten()
    eigen_vals = init_mat["eigenValues"].flatten()[:8]

    result = get_eigenfunctions_axi(R, rcoords, eigen_vals, normalize=True)

    np.testing.assert_allclose(result, expected, atol=1e-12)
