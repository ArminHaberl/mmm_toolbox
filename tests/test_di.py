"""Test MMM_ASgetDI — directivity index calculation."""

import numpy as np

from hornsim.plotting import get_di_axi


def test_di(di_mat, prad_mat):
    """Directivity index from 181 angles, 200 frequencies."""
    expected = di_mat["DI"].flatten()
    angles = di_mat["Angext"].flatten()
    prad = prad_mat["pRad"]

    data = {"pRad": prad, "nfreq": prad.shape[1]}
    data = get_di_axi(data, angles)

    np.testing.assert_allclose(data["DI"], expected, atol=1e-10)
