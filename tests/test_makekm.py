"""Test MMM_ASmakekm — modal wavenumber computation."""

import numpy as np

from hornsim.axi import make_km_axi


def test_makekm(makekm_mat, init_mat, bessel_zeros):
    """Single k, first duct step, 8 modes."""
    expected = np.squeeze(makekm_mat["km"])
    k = init_mat["kvec"].flatten()[0]
    coord = init_mat["steppedCoords"][0, :]
    n_modes = 8

    result = make_km_axi(k, coord, n_modes, bessel_zeros)

    np.testing.assert_allclose(result, expected, atol=1e-12)
