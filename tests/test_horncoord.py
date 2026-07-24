"""Test MMM_1Dhorncoord — horn contour generation."""

import numpy as np
from hornsim.geometry import horn_coord_1d


def test_exponential_horn(horncoords_mat):
    """Exponential horn contour: 251 points, throat 10 cm² → mouth 500 cm²."""
    expected = horncoords_mat["horncoords"]
    htype = str(horncoords_mat["HornType"][0])
    rth = horncoords_mat["rth"].item()
    rm = horncoords_mat["rm"].item()
    L = horncoords_mat["Lh"].item()
    T = horncoords_mat["T"].item()
    dz = horncoords_mat["dz"].item()

    assert htype == "exponential"

    result = horn_coord_1d(htype, rth, rm, L, T, dz)

    np.testing.assert_allclose(result, expected, atol=1e-12)
