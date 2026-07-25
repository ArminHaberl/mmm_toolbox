"""Test MMM_ASmakefmat — F coupling matrix for axisymmetric discontinuities."""

import numpy as np

from mmm_toolbox.axi import make_fmat_axi


def test_makefmat_expanding(makefmat_expanding_mat, bessel_zeros):
    """R1 < R2: expanding duct."""
    expected = makefmat_expanding_mat["F_expand"]
    bz5 = bessel_zeros[:5]
    coord1 = np.array([0.0, 0.01])
    coord2 = np.array([0.0, 0.02])

    result = make_fmat_axi(5, coord1, coord2, bz5)

    assert result.shape == (5, 5)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_makefmat_contracting(makefmat_contracting_mat, bessel_zeros):
    """R1 > R2: contracting duct."""
    expected = makefmat_contracting_mat["F_contract"]
    bz5 = bessel_zeros[:5]
    coord1 = np.array([0.0, 0.02])
    coord2 = np.array([0.0, 0.01])

    result = make_fmat_axi(5, coord1, coord2, bz5)

    assert result.shape == (5, 5)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_makefmat_equal(makefmat_equal_mat, bessel_zeros):
    """R1 == R2: should return identity matrix."""
    expected = makefmat_equal_mat["F_equal"]
    bz5 = bessel_zeros[:5]
    coord1 = np.array([0.0, 0.01])
    coord2 = np.array([0.0, 0.01])

    result = make_fmat_axi(5, coord1, coord2, bz5)

    assert result.shape == (5, 5)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(result, np.eye(5), atol=1e-12)
