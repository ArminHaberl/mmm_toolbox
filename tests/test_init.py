"""Test MMM_init — data structure initialization (includes makesteps + bigfmat)."""

import numpy as np
import pytest

from mmm_toolbox.core import init_horn_data


def test_init(init_mat, horncoords_mat):
    """Full init: exponential horn, 8 modes, 200 frequencies."""
    freq = init_mat["freq"].flatten()
    n_modes = int(init_mat["nModes"].item())
    horncoords = horncoords_mat["horncoords"]
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()

    data = init_horn_data(freq, n_modes, horncoords, "axi", rho, c)

    # Check shapes and key values
    assert data["geometry"] == "axi"
    assert data["n_modes"] == 8
    assert data["nfreq"] == 200

    np.testing.assert_allclose(data["raw_coords"], horncoords, atol=1e-12)
    np.testing.assert_allclose(data["stepped_coords"], init_mat["steppedCoords"], atol=1e-12)
    np.testing.assert_allclose(data["eigen_values"], init_mat["eigenValues"].flatten(), atol=1e-12)
    np.testing.assert_allclose(data["big_f"], init_mat["bigF"], atol=1e-12)
    np.testing.assert_allclose(data["k"], init_mat["kvec"].flatten(), atol=1e-12)
    np.testing.assert_allclose(data["S"], init_mat["S"].flatten(), atol=1e-12)
    assert pytest.approx(data["Sm"]) == init_mat["Sm"].item()
    assert pytest.approx(data["St"]) == init_mat["St"].item()
